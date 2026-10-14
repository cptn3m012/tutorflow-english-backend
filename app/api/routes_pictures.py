import asyncio
from datetime import UTC, datetime
import secrets
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.routes_saved_lessons import DatabaseSession, find_lesson, to_lesson
from app.config import ImageSettings
from app.models import ACTIVE_PICTURE_STATES, LessonImageRecord, PictureJobRecord, SavedLessonRecord
from app.picture_schemas import PictureJob, PictureRequest
from app.picture_storage import image_path
from app.saved_lesson_schemas import SavedLesson, SavedLessonWrite

router = APIRouter(tags=["Lesson pictures"])


@router.put("/lessons/picture-drafts/{lesson_id}", response_model=SavedLesson)
def prepare_picture_draft(lesson_id: UUID, payload: SavedLessonWrite, session: DatabaseSession):
    if lesson_id != payload.id:
        raise HTTPException(422, "Path ID must match the working lesson ID.")
    if not ImageSettings.from_environment().enabled:
        raise HTTPException(503, "Local picture generation is not configured.")
    existing = session.get(SavedLessonRecord, str(lesson_id))
    if existing:
        return to_lesson(existing)
    record = SavedLessonRecord(
        id=str(lesson_id), saved_at=payload.saved_at, favorite=False, is_draft=True,
        level=payload.level, duration=payload.duration, theme=payload.theme,
        mode=payload.mode, lesson=payload.lesson.model_dump(mode="json", exclude_unset=True),
    )
    session.add(record)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return to_lesson(find_lesson(session, lesson_id, allow_draft=True))
    session.refresh(record)
    return to_lesson(record)


@router.get("/lessons/picture-drafts/{lesson_id}", response_model=SavedLesson)
def get_picture_draft(lesson_id: UUID, session: DatabaseSession):
    return to_lesson(find_lesson(session, lesson_id, allow_draft=True))


@router.get("/lessons/pictures/settings")
def picture_settings():
    settings = ImageSettings.from_environment()
    return {"enabled": settings.enabled, "model": settings.model, "styles": ["photo", "illustration"]}


@router.get("/lessons/saved/{lesson_id}/picture-jobs", response_model=PictureJob | None)
def latest_picture_job(lesson_id: UUID, session: DatabaseSession):
    find_lesson(session, lesson_id, allow_draft=True)
    return session.scalar(select(PictureJobRecord).where(PictureJobRecord.lesson_id == str(lesson_id))
                          .order_by(PictureJobRecord.created_at.desc(), PictureJobRecord.id).limit(1))


@router.post("/lessons/saved/{lesson_id}/picture-jobs", response_model=PictureJob, status_code=202)
async def create_picture_job(lesson_id: UUID, payload: PictureRequest, request: Request, session: DatabaseSession):
    settings = ImageSettings.from_environment()
    if not settings.enabled:
        raise HTTPException(503, "Local picture generation is not configured.")
    async with request.app.state.picture_job_lock:
        find_lesson(session, lesson_id, allow_draft=True)
        active = session.scalar(select(PictureJobRecord).where(
            PictureJobRecord.lesson_id == str(lesson_id), PictureJobRecord.status.in_(ACTIVE_PICTURE_STATES)))
        if active:
            return active
        if session.scalar(select(func.count()).select_from(PictureJobRecord).where(
                PictureJobRecord.status.in_(ACTIVE_PICTURE_STATES))) >= 4:
            raise HTTPException(429, "The picture queue is full. Please try again later.", headers={"Retry-After": "30"})
        now = datetime.now(UTC)
        job = PictureJobRecord(id=str(uuid4()), lesson_id=str(lesson_id), status="queued", style=payload.style,
                               model=settings.model, seed=secrets.randbelow(2**31), created_at=now, updated_at=now)
        session.add(job)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(PictureJobRecord).where(
                PictureJobRecord.lesson_id == str(lesson_id), PictureJobRecord.status.in_(ACTIVE_PICTURE_STATES)))
            if existing is None:
                raise
            return existing
        session.refresh(job)
        request.app.state.picture_wakeup.set()
        return job


def job_snapshot(engine, lesson_id: str, job_id: str) -> PictureJob | None:
    with Session(engine) as session:
        job = session.get(PictureJobRecord, job_id)
        if job is None or job.lesson_id != lesson_id or session.get(SavedLessonRecord, lesson_id) is None:
            return None
        return PictureJob.model_validate(job)


@router.get("/lessons/saved/{lesson_id}/picture-jobs/{job_id}/events")
async def picture_events(lesson_id: UUID, job_id: UUID, request: Request, session: DatabaseSession):
    engine = session.get_bind()
    snapshot = job_snapshot(engine, str(lesson_id), str(job_id))
    if snapshot is None:
        raise HTTPException(404, "Picture job not found.")

    async def events():
        previous = None
        heartbeat = 0
        while not await request.is_disconnected():
            job = await run_in_threadpool(job_snapshot, engine, str(lesson_id), str(job_id))
            if job is None:
                yield 'event: removed\ndata: {}\n\n'
                return
            serialized = job.model_dump_json()
            if serialized != previous:
                yield f"event: progress\ndata: {serialized}\n\n"
                previous = serialized
            if job.status in ("ready", "failed"):
                return
            heartbeat += 1
            if heartbeat % 12 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


@router.get("/lesson-images/{asset_id}")
def lesson_image(asset_id: UUID, session: DatabaseSession):
    record = session.get(LessonImageRecord, str(asset_id))
    if record is None or session.get(SavedLessonRecord, record.lesson_id) is None:
        raise HTTPException(404, "Lesson picture not found.")
    path = image_path(asset_id)
    if not path.is_file():
        raise HTTPException(404, "Lesson picture not found.")
    return FileResponse(path, media_type="image/jpeg")
