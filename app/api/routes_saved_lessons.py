from datetime import UTC
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import SavedLessonRecord
from app.saved_lesson_schemas import FavoriteUpdate, SavedLesson, SavedLessonPage

router = APIRouter(prefix="/lessons/saved", tags=["Saved lessons"])
DatabaseSession = Annotated[Session, Depends(get_session)]


def to_lesson(record: SavedLessonRecord) -> SavedLesson:
    timestamp = record.saved_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return SavedLesson(
        id=record.id, saved_at=timestamp, favorite=record.favorite,
        level=record.level, duration=record.duration, theme=record.theme,
        mode=record.mode, lesson=record.lesson,
    )


def find_lesson(session: Session, lesson_id: UUID) -> SavedLessonRecord:
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is None:
        raise HTTPException(status_code=404, detail="Saved lesson not found.")
    return record


@router.get("", response_model=SavedLessonPage)
def list_saved_lessons(
    session: DatabaseSession,
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    records = session.scalars(
        select(SavedLessonRecord)
        .order_by(SavedLessonRecord.saved_at.desc(), SavedLessonRecord.id)
        .offset(offset).limit(limit)
    ).all()
    total = session.scalar(select(func.count()).select_from(SavedLessonRecord))
    return SavedLessonPage(items=[to_lesson(record) for record in records], total=total)


@router.get("/{lesson_id}", response_model=SavedLesson)
def get_saved_lesson(lesson_id: UUID, session: DatabaseSession):
    return to_lesson(find_lesson(session, lesson_id))


@router.put("/{lesson_id}", response_model=SavedLesson)
def save_lesson(lesson_id: UUID, payload: SavedLesson, session: DatabaseSession):
    """Idempotent create for saves, browser imports and undo. Never overwrite a save."""
    if lesson_id != payload.id:
        raise HTTPException(status_code=422, detail="Path ID must match the saved lesson ID.")
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is not None:
        return to_lesson(record)
    timestamp = payload.saved_at
    timestamp = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)
    record = SavedLessonRecord(
        id=str(payload.id), saved_at=timestamp, favorite=payload.favorite,
        level=payload.level, duration=payload.duration, theme=payload.theme,
        mode=payload.mode, lesson=payload.lesson.model_dump(mode="json"),
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


@router.delete("/{lesson_id}", status_code=204)
def delete_saved_lesson(lesson_id: UUID, session: DatabaseSession):
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is not None:
        session.delete(record)
        session.commit()
    return Response(status_code=204)
