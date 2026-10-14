from datetime import UTC
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import PictureJobRecord, SavedLessonRecord
from app.saved_lesson_schemas import FavoriteUpdate, SavedLesson, SavedLessonPage, SavedLessonWrite

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


def find_lesson(session: Session, lesson_id: UUID, *, allow_draft: bool = False) -> SavedLessonRecord:
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is None or (record.is_draft and not allow_draft):
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


@router.delete("/{lesson_id}", status_code=204)
def delete_saved_lesson(lesson_id: UUID, session: DatabaseSession):
    record = session.get(SavedLessonRecord, str(lesson_id))
    if record is not None:
        session.execute(delete(PictureJobRecord).where(PictureJobRecord.lesson_id == str(lesson_id)))
        session.delete(record)
        session.commit()
    return Response(status_code=204)
