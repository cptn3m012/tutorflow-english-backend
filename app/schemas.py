from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing import Annotated, List, Optional


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ReadingQuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: NonEmptyText
    answer: NonEmptyText
    evidence: NonEmptyText


class ReadingComprehension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[ReadingQuestionAnswer] = Field(min_length=1, max_length=8)


class LessonAdvancedOptions(BaseModel):
    reading_min_words: Optional[int] = Field(default=None, ge=40, le=400, example=90)
    reading_max_words: Optional[int] = Field(default=None, ge=50, le=500, example=140)
    target_vocabulary_count: int = Field(default=8, ge=4, le=16, example=8)
    reading_question_count: int = Field(default=3, ge=1, le=8, example=3)
    speaking_question_count: int = Field(default=5, ge=2, le=12, example=5)
    sentence_frame_count: int = Field(default=3, ge=1, le=8, example=3)
    visual_word_count: int = Field(default=6, ge=3, le=10, example=6)
    include_visual_activity: bool = Field(default=True, example=True)
    include_worksheet_assets: bool = Field(default=True, example=True)
    include_images: bool = Field(default=True, example=True)
    skill_focus: str = Field(default="speaking", pattern="^(speaking|reading|vocabulary|grammar|balanced)$")
    lesson_style: str = Field(default="practical", pattern="^(practical|exam|conversation|story|business|travel)$")
    extra_instructions: Optional[str] = Field(default=None, max_length=600, example="Use cafe role-play situations.")

    @model_validator(mode="after")
    def validate_reading_range(self):
        if (
            self.reading_min_words is not None
            and self.reading_max_words is not None
            and self.reading_min_words > self.reading_max_words
        ):
            raise ValueError("reading_min_words must be lower than or equal to reading_max_words.")
        return self


class PdfExportOptions(BaseModel):
    template: str = Field(default="clean", pattern="^(clean|compact|worksheet)$")
    font_size: str = Field(default="medium", pattern="^(small|medium|large)$")
    font_family: str = Field(default="helvetica", pattern="^(helvetica|times|courier)$")
    title_font_size: Optional[int] = Field(default=None, ge=14, le=36)
    heading_font_size: Optional[int] = Field(default=None, ge=10, le=24)
    body_font_size: Optional[int] = Field(default=None, ge=8, le=16)
    include_images: bool = Field(default=True)
    include_answers: bool = Field(default=False)
    include_visual_activity: bool = Field(default=True)
    include_worksheet_assets: bool = Field(default=True)
    page_size: str = Field(default="A4", pattern="^A4$")
    spacing: str = Field(default="normal", pattern="^(compact|normal|relaxed)$")
    accent_color: str = Field(default="0.10 0.34 0.55")
    custom_images: List[dict] = Field(default_factory=list)


class LessonGenerateRequest(BaseModel):
    level: str = Field(..., example="A2")
    duration: int = Field(..., example=30)
    topic: Optional[str] = Field(default=None, example="food")
    lesson_date: Optional[str] = Field(default=None, example="2026-12-20")
    variant_count: int = Field(default=3, example=3)
    advanced_options: LessonAdvancedOptions = Field(default_factory=LessonAdvancedOptions)

    @model_validator(mode="after")
    def validate_topic_or_date(self):
        if not self.topic and not self.lesson_date:
            raise ValueError("You must provide either 'topic' or 'lesson_date'.")
        return self


class LessonImage(BaseModel):
    url: str
    description: str
    score: float
    source: Optional[str] = None
    source_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    creator: Optional[str] = None
    license: Optional[str] = None
    license_version: Optional[str] = None


class LessonVisualActivity(BaseModel):
    activity_type: str = Field(default="find_in_picture")
    title: str
    instruction: str
    scene_prompt: str
    target_words: List[str] = Field(default_factory=list)
    answer_key: List[str] = Field(default_factory=list)


class LessonWorksheetAsset(BaseModel):
    asset_type: str
    title: str
    instruction: str
    prompt: Optional[str] = None
    lines: List[str] = Field(default_factory=list)
    word_bank: List[str] = Field(default_factory=list)
    answer_key: List[str] = Field(default_factory=list)


class CefrProfile(BaseModel):
    level: str
    label: str
    study_hours: str
    source_name: str
    source_url: str
    can_do: List[str]
    communication_targets: List[str]
    vocabulary_targets: List[str]
    grammar_targets: List[str]


class LessonVariant(BaseModel):
    id: Optional[str] = None
    level: Optional[str] = None
    duration_minutes: Optional[int] = None
    theme: Optional[str] = None
    title: str
    lesson_goal: str
    target_vocabulary: List[str]

    reading_text: str
    reading_questions: List[str] = Field(default_factory=list)
    reading_answers: List[ReadingQuestionAnswer] = Field(default_factory=list)

    speaking_questions: List[str]
    pair_work_task: str
    role_play_scenario: str
    sentence_frames: List[str]
    communication_focus: Optional[str] = None
    vocabulary_focus: Optional[str] = None
    grammar_focus: Optional[str] = None
    cefr_focus: List[str] = Field(default_factory=list)

    image_search_query: Optional[str] = None
    image_keywords: List[str] = Field(default_factory=list)
    images: List[LessonImage] = Field(default_factory=list)
    primary_image: Optional[LessonImage] = None
    visual_activity: Optional[LessonVisualActivity] = None
    worksheet_assets: List[LessonWorksheetAsset] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class LessonGenerateResponse(BaseModel):
    requested_level: str
    lesson_theme: str
    duration_minutes: int
    source_type: str
    total_lessons: int
    generated_at: str
    cefr_profile: CefrProfile
    lessons: List[LessonVariant]
