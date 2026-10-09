from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, JSON, String
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
