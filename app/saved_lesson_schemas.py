from datetime import datetime, UTC
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.reading_service import validate_reading_comprehension
from app.schemas import (
    LessonImage, LessonVariant, LessonVisualActivity, LessonWorksheetAsset, NonEmptyText, ReadingQuestionAnswer,
)


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
    content_token: str | None = None
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
    duration: int = Field(ge=1, le=180)
    theme: str = Field(min_length=1, max_length=500)
    mode: Literal["demo", "live"]
    lesson: SavedLessonContent


class SavedLessonImage(LessonImage):
    model_config = ConfigDict(extra="allow")
    url: NonEmptyText
    description: NonEmptyText
    score: float = Field(allow_inf_nan=False)

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str | None) -> str | None:
        if value is not None:
            UUID(value)
        return value


class SavedPictureQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")
    question: NonEmptyText
    answer: NonEmptyText
    evidence: NonEmptyText
    # Older picture questions did not include object or vocabulary tags.
    object_names: list[NonEmptyText] = Field(default_factory=list)
    kind: Literal["observation", "application", "discussion"] = "observation"
    target_words: list[NonEmptyText] = Field(default_factory=list)


class SavedVisualActivity(LessonVisualActivity):
    model_config = ConfigDict(extra="allow")
    activity_type: NonEmptyText = "find_in_picture"
    title: NonEmptyText
    instruction: NonEmptyText
    target_words: list[NonEmptyText] = Field(default_factory=list)
    visible_objects: list[NonEmptyText] = Field(default_factory=list)
    sentence_frames: list[NonEmptyText] = Field(default_factory=list)
    answer_key: list[NonEmptyText] = Field(default_factory=list)
    questions: list[SavedPictureQuestion] = Field(default_factory=list)

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str | None) -> str | None:
        if value is not None:
            UUID(value)
        return value


class SavedWorksheetAsset(LessonWorksheetAsset):
    model_config = ConfigDict(extra="allow")
    asset_type: NonEmptyText
    title: NonEmptyText
    instruction: NonEmptyText
    lines: list[NonEmptyText] = Field(default_factory=list)
    word_bank: list[NonEmptyText] = Field(default_factory=list)
    answer_key: list[NonEmptyText] = Field(default_factory=list)


class SavedLessonWriteContent(LessonVariant):
    """Validate known sections while retaining legacy and future browser fields."""

    model_config = ConfigDict(extra="allow")
    title: NonEmptyText = Field(max_length=300)
    lesson_goal: NonEmptyText
    reading_text: NonEmptyText
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=180)
    theme: NonEmptyText | None = Field(default=None, max_length=500)
    target_vocabulary: list[NonEmptyText] = Field(default_factory=list)
    reading_questions: list[NonEmptyText] = Field(default_factory=list)
    speaking_questions: list[NonEmptyText] = Field(default_factory=list)
    pair_work_task: str = ""
    role_play_scenario: str = ""
    sentence_frames: list[NonEmptyText] = Field(default_factory=list)
    # Demo lessons historically used a description instead of an array.
    cefr_focus: list[str] | str = Field(default_factory=list)
    images: list[SavedLessonImage] = Field(default_factory=list)
    primary_image: SavedLessonImage | None = None
    visual_activity: SavedVisualActivity | None = None
    worksheet_assets: list[SavedWorksheetAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_linked_sections(self):
        # Missing/empty answer keys remain supported for older lessons.
        if self.reading_answers:
            validate_reading_comprehension(self, len(self.reading_questions))
        image = self.primary_image or (self.images[0] if self.images else None)
        if self.visual_activity and self.visual_activity.asset_id:
            if image is None or image.asset_id is None or UUID(image.asset_id) != UUID(self.visual_activity.asset_id):
                raise ValueError("Picture and questions must refer to the same image version.")
        if self.primary_image and self.primary_image.asset_id and self.images:
            if not any(image.asset_id and UUID(image.asset_id) == UUID(self.primary_image.asset_id)
                       for image in self.images):
                raise ValueError("The primary picture must belong to the lesson images.")
        return self


class SavedLessonWrite(SavedLesson):
    # Keep reads permissive so existing records need no migration to stay readable.
    theme: NonEmptyText = Field(max_length=500)
    lesson: SavedLessonWriteContent

    @model_validator(mode="after")
    def validate_lesson_metadata(self):
        if self.lesson.level is not None and self.lesson.level != self.level:
            raise ValueError("Saved level must match the lesson level.")
        if self.lesson.duration_minutes is not None and self.lesson.duration_minutes != self.duration:
            raise ValueError("Saved duration must match the lesson duration.")
        if self.lesson.theme is not None and self.lesson.theme != self.theme:
            raise ValueError("Saved theme must match the lesson theme.")
        if self.lesson.exit_quiz and self.lesson.exit_quiz.level != self.level:
            raise ValueError("Exit quiz level must match the saved lesson level.")
        return self


class SavedLessonPage(BaseModel):
    items: list[SavedLesson]
    total: int


class FavoriteUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    favorite: bool


EditableSection = Literal["overview", "vocabulary", "reading", "comprehension", "speaking", "activities", "sentence_frames"]


class LessonContentChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: NonEmptyText | None = Field(default=None, max_length=300)
    lesson_goal: NonEmptyText | None = None
    target_vocabulary: list[NonEmptyText] | None = Field(default=None, min_length=1, max_length=16)
    reading_text: NonEmptyText | None = None
    reading_questions: list[NonEmptyText] | None = Field(default=None, min_length=1, max_length=8)
    reading_answers: list[ReadingQuestionAnswer] | None = None
    speaking_questions: list[NonEmptyText] | None = Field(default=None, min_length=1, max_length=12)
    pair_work_task: NonEmptyText | None = None
    role_play_scenario: NonEmptyText | None = None
    sentence_frames: list[NonEmptyText] | None = Field(default=None, min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_changes(self):
        if not self.model_fields_set or any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("Supply at least one non-null lesson field to edit.")
        return self


class LessonContentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_token: str = Field(pattern=r"^[0-9a-f]{64}$")
    changes: LessonContentChanges


class LessonSectionRegenerate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_token: str = Field(pattern=r"^[0-9a-f]{64}$")
    section: EditableSection
    instructions: str = Field(default="", max_length=600)
