import asyncio
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from PIL import Image
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.comfy_client import PictureError, build_workflow
from app.config import ImageSettings
from app.database import build_engine, migrate_database
from app.models import LessonImageRecord, PictureJobRecord, SavedLessonRecord
from app.picture_generation import PictureObservation, ScenePlan, prepare_picture
from app.picture_worker import process_job, recover_jobs
from tests.helpers import make_lesson


def observation():
    return PictureObservation.model_validate({
        "caption": "A cup, a plate and a menu are on a table.",
        "objects": [{"name": name, "evidence": f"The {name} is on the table."} for name in ("cup", "plate", "menu")],
        "questions": [{"question": f"Where is the {name}?", "answer": "On the table.",
                       "evidence": f"The {name} rests on the table.", "object_names": [name]} for name in ("cup", "plate", "menu")],
    })


def picture_bytes():
    output = BytesIO()
    Image.new("RGB", (256, 256), "blue").save(output, "JPEG")
    return output.getvalue()


class PictureGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_inspection_is_blind_and_missing_objects_have_one_retry(self):
        plan = ScenePlan(scene="A simple cafe table beside a window.", required_objects=["cup", "plate", "menu"])
        wrong = observation().model_copy(update={"objects": observation().objects[:2]})
        progress = AsyncMock()
        with patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
             patch("app.picture_generation.plan_scene", new_callable=AsyncMock, return_value=plan), \
             patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
             patch("app.picture_generation.inspect_picture", new_callable=AsyncMock, side_effect=[wrong, observation()]) as inspect:
            _, metadata = await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), progress)
        self.assertEqual(render.await_count, 2)
        self.assertEqual(metadata["seed"], 43)
        self.assertEqual(metadata["attempts"], 2)
        for call in inspect.await_args_list:
            self.assertEqual(call.args[1], "A2")
            self.assertEqual(len(call.args), 3)  # Pixels, CEFR and server settings; no requested objects.

    async def test_two_unsuitable_images_fail_instead_of_inventing_answer_key(self):
        plan = ScenePlan(scene="A simple cafe table beside a window.", required_objects=["cup", "plate", "menu"])
        unsuitable = observation().model_copy(update={"objects": []})
        with patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
             patch("app.picture_generation.plan_scene", new_callable=AsyncMock, return_value=plan), \
             patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
             patch("app.picture_generation.inspect_picture", new_callable=AsyncMock, return_value=unsuitable):
            with self.assertRaisesRegex(PictureError, "two attempts"):
                await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), AsyncMock())
        self.assertEqual(render.await_count, 2)

    def test_question_cannot_reference_an_unobserved_object(self):
        data = observation().model_dump()
        data["questions"][0]["object_names"] = ["umbrella"]
        with self.assertRaises(ValidationError):
            PictureObservation.model_validate(data)

    def test_workflows_have_fixed_models_and_core_nodes_only(self):
        for preset in ("flux2-klein-4b", "z-image-turbo"):
            workflow = build_workflow("A simple cafe.", 42, preset)
            self.assertEqual(workflow["9"]["class_type"], "PreviewImage")
            self.assertEqual(workflow["6"]["inputs"]["width"], 1024)
            self.assertFalse(any("http" in str(node["inputs"]) for node in workflow.values()))


class PictureWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.engine = build_engine(f"sqlite:///{self.directory.name}/jobs.db")
        migrate_database(self.engine)
        self.settings = ImageSettings(storage_dir=Path(self.directory.name) / "images")
        self.app = SimpleNamespace(state=SimpleNamespace(generation_slots=asyncio.Semaphore(1)))
        self.original = make_lesson().model_dump()
        self.original.update(images=[{"url": "old.jpg"}], primary_image={"url": "old.jpg"})
        self.job = {"id": str(uuid4()), "lesson_id": str(uuid4()), "lesson": self.original,
                    "level": "A2", "theme": "Cafe", "style": "photo", "seed": 42, "model": "flux2-klein-4b"}
        with Session(self.engine) as session:
            session.add(SavedLessonRecord(id=self.job["lesson_id"], saved_at=datetime.now(UTC), favorite=True,
                                          level="A2", duration=30, theme="Cafe", mode="live", lesson=self.original))
            session.add(PictureJobRecord(id=self.job["id"], lesson_id=self.job["lesson_id"], status="queued", style="photo",
                                         model=self.job["model"], seed=42, created_at=datetime.now(UTC), updated_at=datetime.now(UTC)))
            session.commit()

    def tearDown(self):
        self.engine.dispose()
        self.directory.cleanup()

    async def test_failure_keeps_previous_picture_and_lesson(self):
        with patch("app.picture_worker.prepare_picture", new_callable=AsyncMock, side_effect=PictureError("Try again.")), \
             patch("app.picture_worker.free_comfy", new_callable=AsyncMock), patch("app.picture_worker.unload_ollama", new_callable=AsyncMock):
            await process_job(self.app, self.engine, self.job, self.settings)
        with Session(self.engine) as session:
            self.assertEqual(session.get(SavedLessonRecord, self.job["lesson_id"]).lesson, self.original)
            self.assertEqual(session.get(PictureJobRecord, self.job["id"]).status, "failed")

    async def test_worker_waits_for_text_generation_and_publishes_image_with_questions_atomically(self):
        metadata = {"model": self.job["model"], "vision_model": "test", "seed": 42, "style": "photo", "observation": observation().model_dump()}
        await self.app.state.generation_slots.acquire()
        with patch("app.picture_worker.prepare_picture", new_callable=AsyncMock, return_value=(picture_bytes(), metadata)) as prepare, \
             patch("app.picture_worker.free_comfy", new_callable=AsyncMock), patch("app.picture_worker.unload_ollama", new_callable=AsyncMock):
            task = asyncio.create_task(process_job(self.app, self.engine, self.job, self.settings))
            await asyncio.sleep(0)
            prepare.assert_not_awaited()
            self.app.state.generation_slots.release()
            await task
        with Session(self.engine) as session:
            lesson = session.get(SavedLessonRecord, self.job["lesson_id"])
            job = session.get(PictureJobRecord, self.job["id"])
            self.assertEqual(job.status, "ready")
            self.assertTrue(lesson.favorite)
            self.assertEqual(lesson.lesson["reading_text"], self.original["reading_text"])
            self.assertEqual(lesson.lesson["visual_activity"]["asset_id"], lesson.lesson["primary_image"]["asset_id"])
            self.assertEqual(lesson.lesson["visual_activity"]["questions"], metadata["observation"]["questions"])
            self.assertEqual(session.get(LessonImageRecord, job.asset_id).metadata_json["seed"], 42)

    def test_restart_requeues_interrupted_jobs(self):
        with Session(self.engine) as session:
            session.get(PictureJobRecord, self.job["id"]).status = "checking"
            session.commit()
        recover_jobs(self.engine)
        with Session(self.engine) as session:
            self.assertEqual(session.get(PictureJobRecord, self.job["id"]).status, "queued")
