from datetime import datetime, UTC
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class SavedLessonContent(BaseModel):
    # Keep optional/future frontend fields, including demo CEFR descriptions.
    model_config = ConfigDict(extra="allow")
    title: str = Field(min_length=1, max_length=300)
    lesson_goal: str
    reading_text: str
    target_vocabulary: list[str] = Field(default_factory=list)
    reading_questions: list[str] = Field(default_factory=list)
    speaking_questions: list[str] = Field(default_factory=list)
    pair_work_task: str = ""
    role_play_scenario: str = ""
    sentence_frames: list[str] = Field(default_factory=list)


class SavedLesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    saved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    favorite: bool = False
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
    duration: int = Field(ge=1, le=180)
    theme: str = Field(min_length=1, max_length=500)
    mode: Literal["demo", "live"]
    lesson: SavedLessonContent


class SavedLessonPage(BaseModel):
    items: list[SavedLesson]
    total: int


class FavoriteUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    favorite: bool
