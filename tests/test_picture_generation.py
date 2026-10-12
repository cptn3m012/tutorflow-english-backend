import asyncio
import json
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
from app.picture_generation import (
    PictureObservation, ScenePlan, TopicRelevanceReview, inspect_picture, plan_scene, prepare_picture,
    review_scene_topic,
)
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


def topic_approval():
    return TopicRelevanceReview(faithful=True, missing_objects=[], issues=[], repair_instructions="")


class PictureGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_uses_topic_objects_without_a_global_vocabulary_whitelist(self):
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock, return_value=json.dumps({
            "scene": "Visitors place flowers and light candles beside graves in a Polish cemetery.",
            "required_objects": ["grave", "candle", "flower"],
            "excluded_objects": ["sofa", "bowl", "spoon"],
            "source_evidence": "Visitors put flowers and candles on graves.",
        })) as model, patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock,
                            return_value=topic_approval()):
            plan = await plan_scene({"reading_text": "Visitors put flowers and candles on graves."}, "A2", "All Saints' Day in Poland", ImageSettings())
        fields = model.await_args.args[1]["properties"]
        self.assertNotIn("enum", fields["required_objects"]["items"])
        self.assertIn("Visitors put flowers and candles on graves.", fields["source_evidence"]["enum"])
        self.assertEqual(plan.required_objects, ["grave", "candle", "flower"])

    async def test_incongruous_plan_is_repaired_before_rendering(self):
        bad = {"scene": "A sofa and a bowl stand beside graves in a cemetery.",
               "required_objects": ["couch", "bowl", "grave"], "excluded_objects": [], "source_evidence": "People visit graves."}
        good = {"scene": "People stand beside graves and place flowers on them.",
                "required_objects": ["grave", "flower"], "excluded_objects": ["sofa"], "source_evidence": "People visit graves."}
        rejection = TopicRelevanceReview(faithful=False, missing_objects=[], issues=["A sofa and bowl do not belong in a cemetery."],
                                        repair_instructions="Show cemetery graves and flowers with no household furniture.")
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   side_effect=[json.dumps(bad), json.dumps(good)]) as model, \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock,
                   side_effect=[rejection, topic_approval()]):
            plan = await plan_scene({"reading_text": "People visit graves."}, "A2", "All Saints' Day", ImageSettings())
        self.assertEqual(plan.required_objects, ["grave", "flower"])
        self.assertEqual(model.await_count, 2)
        self.assertIn("no household furniture", model.await_args.args[0])

    async def test_invented_plan_grounding_exhausts_bounded_retries(self):
        invalid = {"scene": "Visitors stand beside graves in an autumn cemetery.",
                   "required_objects": ["grave", "candle"], "excluded_objects": [], "source_evidence": "Invented sentence."}
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock, return_value=json.dumps(invalid)) as model, \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock) as review:
            with self.assertRaisesRegex(PictureError, "plan did not match"):
                await plan_scene({"reading_text": "People visit graves."}, "A2", "All Saints' Day", ImageSettings())
        self.assertEqual(model.await_count, 5)
        review.assert_not_awaited()

    def test_scene_cannot_request_an_excluded_object_or_its_alias(self):
        for description, excluded in (
            ("A person calculates savings with a laptop and coins on a desk.", "laptop"),
            ("Visitors place candles beside graves and a grey couch.", "sofa"),
            ("A group works with laptops at a table with coins.", "laptop"),
        ):
            with self.subTest(excluded=excluded), self.assertRaisesRegex(ValidationError, "describes excluded objects"):
                ScenePlan(scene=description, required_objects=["person", "coin"], excluded_objects=[excluded],
                          source_evidence="People manage their money.")

    async def test_contradictory_finance_plan_is_repaired_before_image_generation(self):
        invalid = {"scene": "A person works with a laptop and coins to calculate savings.",
                   "required_objects": ["person", "coin"], "excluded_objects": ["laptop"],
                   "source_evidence": "People manage their money."}
        valid = {**invalid, "excluded_objects": []}
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   side_effect=[json.dumps(invalid), json.dumps(valid)]) as model, \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, return_value=topic_approval()) as review:
            result = await plan_scene({"reading_text": "People manage their money."}, "B2", "Finance", ImageSettings())
        self.assertEqual(result.excluded_objects, [])
        self.assertIn("describes excluded objects: laptop", model.await_args.args[0])
        review.assert_awaited_once()

    def test_missing_or_contradictory_topic_verdict_never_approves(self):
        for data in ({}, {"faithful": True, "missing_objects": [], "issues": ["Sofa in a cemetery"], "repair_instructions": ""},
                     {"faithful": False, "missing_objects": [], "issues": [], "repair_instructions": ""},
                     {"faithful": "true", "missing_objects": [], "issues": [], "repair_instructions": ""},
                     {"faithful": True, "missing_objects": ["grave"], "issues": [], "repair_instructions": ""}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                TopicRelevanceReview.model_validate(data)

    async def test_questions_are_constrained_to_blindly_observed_objects(self):
        observed = observation().model_dump(exclude={"questions", "sentence_frames"})
        with patch("app.picture_generation.vision_json", new_callable=AsyncMock, side_effect=[
            json.dumps(observed), json.dumps({"questions": observation().model_dump()["questions"]}),
        ]) as vision:
            result = await inspect_picture(picture_bytes(), "A2", ImageSettings())
        initial, questions = vision.await_args_list
        self.assertNotIn("questions", initial.args[2]["properties"])
        enum = questions.args[2]["properties"]["questions"]["items"]["properties"]["object_names"]["items"]["enum"]
        self.assertEqual(enum, ["cup", "plate", "menu"])
        self.assertEqual(result.questions[0].object_names, ["cup"])

    async def test_observed_sofa_in_cemetery_cannot_be_rubber_stamped_by_model(self):
        plan = ScenePlan(scene="Visitors light candles beside graves in a cemetery.", required_objects=["grave", "candle"],
                         excluded_objects=["sofa", "bowl"], source_evidence="People visit graves.")
        observed = observation().model_copy(update={"objects": [
            *observation().objects,
            observation().objects[0].model_copy(update={"name": "grey couch"}),
        ]})
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock) as model:
            result = await review_scene_topic(picture_bytes(), {}, plan, ImageSettings(), observed)
        self.assertFalse(result.faithful)
        self.assertIn("sofa", result.issues[0])
        model.assert_not_awaited()

    async def test_same_sofa_is_allowed_for_a_living_room_lesson(self):
        plan = ScenePlan(scene="People sit on a sofa in their living room.", required_objects=["sofa", "lamp"],
                         excluded_objects=["grave"], source_evidence="People sit in their living room.")
        observed = observation().model_copy(update={"objects": [
            *observation().objects, observation().objects[0].model_copy(update={"name": "grey couch"}),
        ]})
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   return_value=topic_approval().model_dump_json()) as model, \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock) as unload:
            result = await review_scene_topic(picture_bytes(), {}, plan, ImageSettings(), observed)
        self.assertTrue(result.faithful)
        model.assert_awaited_once()
        self.assertEqual([call.args[0] for call in unload.await_args_list], [ImageSettings().vision_model])

    def test_duplicate_sightings_do_not_duplicate_the_word_bank(self):
        data = observation().model_dump()
        data["objects"].append(data["objects"][0])
        self.assertEqual(len(PictureObservation.model_validate(data).objects), 3)

    async def test_inspection_is_blind_and_fifth_image_can_succeed(self):
        plan = ScenePlan(scene="A simple cafe table beside a window.", required_objects=["cup", "plate", "menu"], excluded_objects=[], source_evidence="Cafe")
        wrong = observation().model_copy(update={"objects": observation().objects[:2]})
        progress = AsyncMock()
        with patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
             patch("app.picture_generation.plan_scene", new_callable=AsyncMock, return_value=plan) as planner, \
             patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
             patch("app.picture_generation.observe_picture", new_callable=AsyncMock, side_effect=[wrong] * 4 + [observation()]) as inspect, \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, side_effect=[
                 TopicRelevanceReview(faithful=False, missing_objects=["menu"], issues=["Menu is missing."], repair_instructions="Show the menu on the cafe table.")
             ] * 4 + [topic_approval()]), \
             patch("app.picture_generation.prepare_picture_questions", new_callable=AsyncMock, return_value=observation()) as questions:
            _, metadata = await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), progress)
        self.assertEqual(render.await_count, 5)
        self.assertEqual(metadata["seed"], 46)
        self.assertEqual(metadata["attempts"], 5)
        self.assertEqual([call.args[1] for call in render.await_args_list], [42, 43, 44, 45, 46])
        self.assertEqual(planner.await_count, 5)
        self.assertIn("Menu is missing", planner.await_args.kwargs["feedback"])
        self.assertEqual(planner.await_args.kwargs["max_attempts"], 1)
        for call in inspect.await_args_list:
            self.assertEqual(len(call.args), 2)  # Pixels and server settings; no requested objects or lesson context.
        self.assertEqual(questions.await_count, 1)

    async def test_five_unsuitable_images_fail_instead_of_inventing_answer_key(self):
        plan = ScenePlan(scene="A simple cafe table beside a window.", required_objects=["cup", "plate", "menu"], excluded_objects=[], source_evidence="Cafe")
        unsuitable = observation().model_copy(update={"objects": []})
        with patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
             patch("app.picture_generation.plan_scene", new_callable=AsyncMock, return_value=plan), \
             patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
             patch("app.picture_generation.observe_picture", new_callable=AsyncMock, return_value=unsuitable), \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, return_value=TopicRelevanceReview(
                 faithful=False, missing_objects=["cup", "plate", "menu"], issues=["All cafe objects are absent."],
                 repair_instructions="Show the cafe objects on the cafe table.")):
            with self.assertRaisesRegex(PictureError, "5 attempts"):
                await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), AsyncMock())
        self.assertEqual(render.await_count, 5)

    async def test_off_topic_picture_is_retried_even_when_all_required_objects_are_visible(self):
        plan = ScenePlan(scene="A simple cafe table beside a window.", required_objects=["cup", "plate", "menu"], excluded_objects=[], source_evidence="Cafe")
        rejected = TopicRelevanceReview(faithful=False, missing_objects=[], issues=["Objects are in an unrelated bedroom."],
                                        repair_instructions="Place the table in the cafe described by the lesson.")
        for reviews in ([rejected, topic_approval()], [rejected] * 5):
            with self.subTest(approved=reviews[-1].faithful), \
                 patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
                 patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
                 patch("app.picture_generation.plan_scene", new_callable=AsyncMock, return_value=plan) as planner, \
                 patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
                 patch("app.picture_generation.observe_picture", new_callable=AsyncMock, return_value=observation()), \
                 patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, side_effect=reviews), \
                 patch("app.picture_generation.prepare_picture_questions", new_callable=AsyncMock, return_value=observation()) as questions:
                if reviews[-1].faithful:
                    _, metadata = await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), AsyncMock())
                    self.assertTrue(metadata["topic_review"]["faithful"])
                    self.assertEqual(questions.await_count, 1)
                else:
                    with self.assertRaisesRegex(PictureError, "did not match the lesson topic after 5 attempts"):
                        await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), AsyncMock())
                    questions.assert_not_awaited()
                self.assertEqual(render.await_count, len(reviews))
                self.assertIn("table in the cafe", planner.await_args.kwargs["feedback"])

    async def test_failed_render_uses_repaired_plan_and_publishes_its_metadata(self):
        original = ScenePlan(scene="A cup and plate sit on a cluttered cafe table.", required_objects=["cup", "plate"],
                             excluded_objects=[], source_evidence="People order tea at a cafe.")
        repaired = original.model_copy(update={"scene": "A cup and menu sit on an uncluttered cafe table.",
                                               "required_objects": ["cup", "menu"]})
        rejected = TopicRelevanceReview(faithful=False, missing_objects=["plate"], issues=["The plate is obscured."],
                                        repair_instructions="Choose a simpler cafe composition.")
        with patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
             patch("app.picture_generation.plan_scene", new_callable=AsyncMock, side_effect=[original, repaired]) as planner, \
             patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
             patch("app.picture_generation.observe_picture", new_callable=AsyncMock, return_value=observation()), \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, side_effect=[rejected, topic_approval()]), \
             patch("app.picture_generation.prepare_picture_questions", new_callable=AsyncMock, return_value=observation()):
            _, metadata = await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), AsyncMock())
        self.assertEqual(planner.await_args.kwargs["previous_plan"], original)
        self.assertIn("plate is obscured", planner.await_args.kwargs["feedback"])
        self.assertEqual(metadata["required_objects"], ["cup", "menu"])
        self.assertIn("uncluttered", render.await_args.args[0])
        self.assertNotIn("plate", render.await_args.args[0])
        self.assertEqual(metadata["seed"], 43)

    async def test_invalid_repair_plan_keeps_valid_plan_within_five_render_budget(self):
        plan = ScenePlan(scene="A cup and plate sit on a cafe table.", required_objects=["cup", "plate"],
                         excluded_objects=[], source_evidence="People order tea at a cafe.")
        rejected = TopicRelevanceReview(faithful=False, missing_objects=["plate"], issues=["The plate is obscured."],
                                        repair_instructions="Choose a simpler cafe composition.")
        with patch("app.picture_generation.free_comfy", new_callable=AsyncMock), \
             patch("app.picture_generation.unload_ollama", new_callable=AsyncMock), \
             patch("app.picture_generation.plan_scene", new_callable=AsyncMock,
                   side_effect=[plan] + [PictureError("Invalid repair plan")] * 4) as planner, \
             patch("app.picture_generation.render_picture", new_callable=AsyncMock, return_value=picture_bytes()) as render, \
             patch("app.picture_generation.observe_picture", new_callable=AsyncMock, return_value=observation()), \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, return_value=rejected), \
             patch("app.picture_generation.prepare_picture_questions", new_callable=AsyncMock) as questions:
            with self.assertRaisesRegex(PictureError, "after 5 attempts"):
                await prepare_picture({}, "A2", "Cafe", "photo", 42, ImageSettings(), AsyncMock())
        self.assertEqual(render.await_count, 5)
        self.assertEqual(planner.await_count, 5)
        questions.assert_not_awaited()

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
        metadata = {"model": self.job["model"], "vision_model": "test", "seed": 42, "style": "photo", "observation": observation().model_dump(),
                    "practice_words": ["menu", "tea", "bill", "waiter"]}
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
            self.assertEqual(lesson.lesson["visual_activity"]["target_words"], ["menu", "tea", "bill", "waiter"])
            self.assertIn("cup", lesson.lesson["visual_activity"]["visible_objects"])
            self.assertEqual(session.get(LessonImageRecord, job.asset_id).metadata_json["seed"], 42)

    def test_restart_requeues_interrupted_jobs(self):
        with Session(self.engine) as session:
            session.get(PictureJobRecord, self.job["id"]).status = "checking"
            session.commit()
        recover_jobs(self.engine)
        with Session(self.engine) as session:
            self.assertEqual(session.get(PictureJobRecord, self.job["id"]).status, "queued")
