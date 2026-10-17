from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

QuizText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
QuizLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


class QuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["multiple_choice", "short_answer"]
    focus: Literal["vocabulary", "grammar", "communication"]
    target: QuizText
    question: QuizText = Field(max_length=400)
    options: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=180)]] = Field(max_length=4)
    answer: QuizText = Field(max_length=500)
    explanation: QuizText = Field(max_length=600)

    @model_validator(mode="after")
    def validate_answer(self):
        if self.kind == "multiple_choice":
            if len(self.options) not in (3, 4) or len({item.casefold() for item in self.options}) != len(self.options):
                raise ValueError("Multiple-choice questions need 3 or 4 distinct options.")
            if self.answer not in self.options:
                raise ValueError("The answer must match exactly one option.")
        elif self.options:
            raise ValueError("Short-answer questions must have an empty options list.")
        return self


class ExitQuizDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: QuizText = Field(max_length=160)
    instruction: QuizText = Field(max_length=400)
    questions: list[QuizQuestion] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def validate_coverage(self):
        if {item.focus for item in self.questions} != {"vocabulary", "grammar", "communication"}:
            raise ValueError("Cover vocabulary, grammar and the communication goal.")
        if not any(item.kind == "short_answer" and item.focus == "communication" for item in self.questions):
            raise ValueError("Include a short communication task with an example answer.")
        if not any(item.kind == "multiple_choice" for item in self.questions):
            raise ValueError("Include at least one multiple-choice question.")
        if len({item.question.casefold() for item in self.questions}) != len(self.questions):
            raise ValueError("Use distinct questions.")
        return self


class LessonExitQuiz(ExitQuizDraft):
    level: QuizLevel
    source_token: str = Field(pattern=r"^[0-9a-f]{64}$")


class QuizGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_token: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_count: int = Field(default=3, ge=3, le=5)


class QuizReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issues: list[QuizText]
