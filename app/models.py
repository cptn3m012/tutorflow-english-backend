from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, JSON, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SavedLessonRecord(Base):
    __tablename__ = "saved_lessons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    favorite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    level: Mapped[str] = mapped_column(String(2), index=True)
    duration: Mapped[int] = mapped_column(Integer)
    theme: Mapped[str] = mapped_column(String(500))
    mode: Mapped[str] = mapped_column(String(4))
    lesson: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))


ACTIVE_PICTURE_STATES = ("queued", "planning", "rendering", "checking")


class LessonImageRecord(Base):
    __tablename__ = "lesson_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Retain immutable versions when a lesson is deleted so Undo can restore it.
    # The asset endpoint also requires the associated saved lesson to exist.
    lesson_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))


class PictureJobRecord(Base):
    __tablename__ = "picture_jobs"
    __table_args__ = (
        Index(
            "uq_picture_job_active_lesson", "lesson_id", unique=True,
            postgresql_where=text("status IN ('queued', 'planning', 'rendering', 'checking')"),
            sqlite_where=text("status IN ('queued', 'planning', 'rendering', 'checking')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    lesson_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    style: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(80))
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    asset_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
