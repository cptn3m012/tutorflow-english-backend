from datetime import UTC
from typing import Annotated
from uuid import UUID

from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_session
from starlette.concurrency import run_in_threadpool

from app.lesson_editing import content_token, edit_content, regenerate_content
from app.models import ACTIVE_PICTURE_STATES, PictureJobRecord, SavedLessonRecord
from app.ollama_client import OllamaError
from app.quiz_schemas import QuizGenerateRequest
from app.quiz_service import generate_exit_quiz, quiz_is_stale
from app.saved_lesson_schemas import (
    FavoriteUpdate, LessonContentUpdate, LessonSectionRegenerate, SavedLesson, SavedLessonPage, SavedLessonWrite,
)

router = APIRouter(prefix="/lessons/saved", tags=["Saved lessons"])
DatabaseSession = Annotated[Session, Depends(get_session)]


def to_lesson(record: SavedLessonRecord) -> SavedLesson:
    timestamp = record.saved_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    content = deepcopy(record.lesson)
    if content.get("exit_quiz"):
        content["exit_quiz_stale"] = quiz_is_stale(content, record.level)
    return SavedLesson(
        id=record.id, saved_at=timestamp, favorite=record.favorite, content_token=content_token(record),
        level=record.level, duration=record.duration, theme=record.theme,
        mode=record.mode, lesson=content,
    )


def find_lesson(session: Session, lesson_id: UUID, *, allow_draft: bool = False) -> SavedLessonRecord:
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is None or (record.is_draft and not allow_draft):
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    return record


def editable_record(session: Session, lesson_id: UUID, expected_token: str):
    record = find_lesson(session, lesson_id)
    if content_token(record) != expected_token:
        raise HTTPException(409, "This lesson changed in another tab. Reload it before saving your changes.")
    if session.scalar(select(PictureJobRecord.id).where(
            PictureJobRecord.lesson_id == str(lesson_id), PictureJobRecord.status.in_(ACTIVE_PICTURE_STATES)).limit(1)):
        raise HTTPException(409, "Wait for the current picture job to finish before editing this lesson.")
    return record


def read_edit_snapshot(engine, lesson_id: UUID, expected_token: str):
    with Session(engine) as session:
        record = editable_record(session, lesson_id, expected_token)
        return {"lesson": deepcopy(record.lesson), "level": record.level,
                "duration": record.duration, "theme": record.theme}


def publish_edit(engine, lesson_id: UUID, expected_token: str, snapshot: dict, content: dict):
    with Session(engine) as session:
        editable_record(session, lesson_id, expected_token)
        # Compare-and-swap also protects SQLite and simultaneous backend processes.
        result = session.execute(update(SavedLessonRecord).where(
            SavedLessonRecord.id == str(lesson_id), SavedLessonRecord.is_draft.is_(False),
            SavedLessonRecord.lesson == snapshot["lesson"], SavedLessonRecord.level == snapshot["level"],
            SavedLessonRecord.duration == snapshot["duration"], SavedLessonRecord.theme == snapshot["theme"],
        ).values(lesson=content).execution_options(synchronize_session=False))
        if result.rowcount != 1:
            session.rollback()
            raise HTTPException(409, "This lesson changed while the section was being prepared. Reload it and try again.")
        session.commit()
        session.expire_all()
        return to_lesson(find_lesson(session, lesson_id))


@router.get("", response_model=SavedLessonPage)
def list_saved_lessons(
    session: DatabaseSession,
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    records = session.scalars(
        select(SavedLessonRecord)
        .where(SavedLessonRecord.is_draft.is_(False))
        .order_by(SavedLessonRecord.saved_at.desc(), SavedLessonRecord.id)
        .offset(offset).limit(limit)
    ).all()
    total = session.scalar(select(func.count()).select_from(SavedLessonRecord).where(SavedLessonRecord.is_draft.is_(False)))
    return SavedLessonPage(items=[to_lesson(record) for record in records], total=total)


@router.get("/{lesson_id}", response_model=SavedLesson)
def get_saved_lesson(lesson_id: UUID, session: DatabaseSession):
    return to_lesson(find_lesson(session, lesson_id))


@router.put("/{lesson_id}", response_model=SavedLesson)
def save_lesson(lesson_id: UUID, payload: SavedLessonWrite, session: DatabaseSession):
    """Idempotent create for saves, browser imports and undo. Never overwrite a save."""
    if lesson_id != payload.id:
        raise HTTPException(status_code=422, detail="Path ID must match the saved lesson ID.")
    record = session.scalar(select(SavedLessonRecord).where(SavedLessonRecord.id == str(lesson_id)).with_for_update())
    if record is not None:
        if record.is_draft:
            # Preserve the canonical picture, including a worker finishing while
            # the browser saves. Promoting does not replace the stored content.
            record.is_draft = False
            record.favorite = payload.favorite
            record.saved_at = payload.saved_at
            session.commit()
            session.refresh(record)
        return to_lesson(record)
    timestamp = payload.saved_at
    timestamp = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)
    record = SavedLessonRecord(
        id=str(payload.id), saved_at=timestamp, favorite=payload.favorite,
        level=payload.level, duration=payload.duration, theme=payload.theme,
        mode=payload.mode, lesson=payload.lesson.model_dump(mode="json", exclude_unset=True),
    )
    session.add(record)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        # Another request can complete the same retry/import concurrently.
        existing = session.get(SavedLessonRecord, str(lesson_id))
        if existing is None:
            raise
        return to_lesson(existing)
    session.refresh(record)
    return to_lesson(record)


@router.patch("/{lesson_id}", response_model=SavedLesson)
def update_favorite(lesson_id: UUID, payload: FavoriteUpdate, session: DatabaseSession):
    record = find_lesson(session, lesson_id)
    record.favorite = payload.favorite
    session.commit()
    session.refresh(record)
    return to_lesson(record)


@router.patch("/{lesson_id}/content", response_model=SavedLesson)
async def edit_saved_lesson(lesson_id: UUID, payload: LessonContentUpdate, request: Request, session: DatabaseSession):
    engine = session.get_bind()
    async with request.app.state.picture_job_lock:
        snapshot = await run_in_threadpool(read_edit_snapshot, engine, lesson_id, payload.expected_token)
        try:
            content = edit_content(snapshot["lesson"], payload.changes.model_dump(mode="json", exclude_unset=True))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return await run_in_threadpool(publish_edit, engine, lesson_id, payload.expected_token, snapshot, content)


@router.post("/{lesson_id}/regenerate", response_model=SavedLesson)
async def regenerate_saved_section(lesson_id: UUID, payload: LessonSectionRegenerate, request: Request, session: DatabaseSession):
    engine = session.get_bind()
    snapshot = await run_in_threadpool(read_edit_snapshot, engine, lesson_id, payload.expected_token)
    slots = request.app.state.generation_slots
    if slots.locked():
        raise HTTPException(429, "The local model is busy. Please wait and try again.", headers={"Retry-After": "30"})
    try:
        async with slots:
            content = await regenerate_content(snapshot["lesson"], snapshot["level"], snapshot["theme"],
                                               snapshot["duration"], payload.section, payload.instructions)
            async with request.app.state.picture_job_lock:
                return await run_in_threadpool(publish_edit, engine, lesson_id, payload.expected_token, snapshot, content)
    except OllamaError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/{lesson_id}/quiz", response_model=SavedLesson)
async def prepare_exit_quiz(lesson_id: UUID, payload: QuizGenerateRequest, request: Request, session: DatabaseSession):
    engine = session.get_bind()
    snapshot = await run_in_threadpool(read_edit_snapshot, engine, lesson_id, payload.expected_token)
    slots = request.app.state.generation_slots
    if slots.locked():
        raise HTTPException(429, "The local model is busy. Please wait and try again.", headers={"Retry-After": "30"})
    try:
        async with slots:
            quiz = await generate_exit_quiz(snapshot["lesson"], snapshot["level"], payload.question_count)
            content = {**snapshot["lesson"], "exit_quiz": quiz.model_dump(mode="json"), "exit_quiz_stale": False}
            async with request.app.state.picture_job_lock:
                return await run_in_threadpool(publish_edit, engine, lesson_id, payload.expected_token, snapshot, content)
    except OllamaError as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.delete("/{lesson_id}", status_code=204)
def delete_saved_lesson(lesson_id: UUID, session: DatabaseSession):
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is not None:
        session.execute(delete(PictureJobRecord).where(PictureJobRecord.lesson_id == str(lesson_id)))
        session.delete(record)
        session.commit()
    return Response(status_code=204)
