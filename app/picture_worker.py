import asyncio
from datetime import UTC, datetime
import logging
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.comfy_client import PictureError, free_comfy
from app.config import ImageSettings
from app.models import ACTIVE_PICTURE_STATES, LessonImageRecord, PictureJobRecord, SavedLessonRecord
from app.ollama_client import OllamaError
from app.picture_generation import prepare_picture, unload_ollama
from app.picture_storage import write_image

logger = logging.getLogger(__name__)


def next_job(engine):
    with Session(engine) as session:
        job = session.scalar(select(PictureJobRecord).where(PictureJobRecord.status == "queued")
                             .order_by(PictureJobRecord.created_at, PictureJobRecord.id).limit(1))
        if job is None:
            return None
        lesson = session.get(SavedLessonRecord, job.lesson_id)
        if lesson is None:
            session.delete(job)
            session.commit()
            return None
        return {"id": job.id, "lesson_id": job.lesson_id, "lesson": lesson.lesson, "level": lesson.level,
                "theme": lesson.theme, "style": job.style, "seed": job.seed, "model": job.model}


def set_status(engine, job_id: str, status: str, error: str | None = None):
    with Session(engine) as session:
        job = session.get(PictureJobRecord, job_id)
        if job is None:
            raise PictureError("The saved lesson was removed.")
        job.status, job.error, job.updated_at = status, error, datetime.now(UTC)
        session.commit()


def attach_picture(engine, job: dict, content: bytes, metadata: dict, settings: ImageSettings):
    asset_id = str(uuid4())
    write_image(asset_id, content, settings)
    with Session(engine) as session:
        # Replace only the picture and its activity; favorite/text changes survive.
        lesson = session.scalar(select(SavedLessonRecord).where(SavedLessonRecord.id == job["lesson_id"]).with_for_update())
        record = session.get(PictureJobRecord, job["id"])
        if lesson is None or record is None:
            return
        observation = metadata["observation"]
        visible_words = [obj["name"] for obj in observation["objects"]]
        words = metadata.get("practice_words") or visible_words
        image = {"asset_id": asset_id, "url": f"/api/lesson-images/{asset_id}", "description": observation["caption"],
                 "score": 1.0, "source": "local-generation", "model": metadata["model"], "seed": metadata["seed"], "style": metadata["style"]}
        visual = {"activity_type": "describe_picture", "title": "Describe, connect and discuss", "scene_prompt": "",
                  "instruction": "Use the lesson words to describe the scene, connect it to the lesson and explain your ideas.",
                  "target_words": words, "visible_objects": visible_words, "answer_key": [],
                  "sentence_frames": observation.get("sentence_frames", []), "questions": observation["questions"],
                  "asset_id": asset_id, "verified": True}
        metadata = {**metadata, "visual_activity": visual}
        session.add(LessonImageRecord(id=asset_id, lesson_id=lesson.id, created_at=datetime.now(UTC), metadata_json=metadata))
        content_json = dict(lesson.lesson)
        content_json.update(images=[image], primary_image=image, visual_activity=visual, image_requested=True,
                            picture_context_stale=False)
        content_json["worksheet_assets"] = [asset for asset in content_json.get("worksheet_assets", [])
                                            if asset.get("asset_type") not in ("reading_illustration", "find_in_picture", "label_the_picture")]
        lesson.lesson = content_json
        record.status, record.asset_id, record.error, record.updated_at = "ready", asset_id, None, datetime.now(UTC)
        session.commit()


def recover_jobs(engine):
    with Session(engine) as session:
        session.execute(update(PictureJobRecord).where(PictureJobRecord.status.in_(ACTIVE_PICTURE_STATES))
                        .values(status="queued", error=None, updated_at=datetime.now(UTC)))
        session.commit()


async def process_job(app, engine, job: dict, settings: ImageSettings):
    async def progress(status):
        await run_in_threadpool(set_status, engine, job["id"], status)

    async with app.state.generation_slots:
        try:
            # A queued job keeps the model preset it was created with.
            from dataclasses import replace
            job_settings = replace(settings, model=job["model"])
            content, metadata = await prepare_picture(job["lesson"], job["level"], job["theme"], job["style"],
                                                       job["seed"], job_settings, progress)
            await run_in_threadpool(attach_picture, engine, job, content, metadata, settings)
        except asyncio.CancelledError:
            try:
                await run_in_threadpool(set_status, engine, job["id"], "queued")
            except PictureError:
                pass
            raise
        except Exception as error:
            message = str(error) if isinstance(error, (PictureError, OllamaError)) else "The picture could not be prepared. Your existing lesson has been kept. Try again."
            logger.warning("Picture job failed (%s)", type(error).__name__)
            try:
                await run_in_threadpool(set_status, engine, job["id"], "failed", message)
            except PictureError:
                pass  # Deleted lesson; never recreate it from a stale snapshot.
        finally:
            try:
                if not settings.managed_runtime:
                    await free_comfy(settings)
                await unload_ollama(settings.planner_model)
                await unload_ollama(settings.vision_model)
            except PictureError:
                logger.warning("Local picture model cleanup failed")


async def picture_worker(app, engine, settings: ImageSettings):
    await run_in_threadpool(recover_jobs, engine)
    while True:
        app.state.picture_wakeup.clear()
        try:
            job = await run_in_threadpool(next_job, engine)
            if job:
                await process_job(app, engine, job, settings)
                continue
            await asyncio.wait_for(app.state.picture_wakeup.wait(), timeout=10)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("Picture queue is unavailable (%s)", type(error).__name__)
            await asyncio.sleep(5)
