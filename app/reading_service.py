import re

from pydantic import BaseModel, ConfigDict, Field

from app.errors import LessonQualityError
from app.generation_policy import LESSON_MAX_ATTEMPTS
from app.ollama_client import generate_from_ollama
from app.prompts import build_reading_comprehension_prompt
from app.schemas import LessonVariant, NonEmptyText, ReadingQuestionAnswer


class ReadingEvidenceSelection(BaseModel):
    """Internal model output; public answer keys retain literal passage evidence."""

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyText
    answer: NonEmptyText
    evidence_start: int = Field(ge=1, strict=True)
    evidence_end: int = Field(ge=1, strict=True)


class ReadingComprehensionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ReadingEvidenceSelection] = Field(min_length=1, max_length=8)


def reading_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Keep offsets into the original passage so punctuation and spacing survive."""
    spans = []
    start = 0
    for boundary in re.finditer(r'(?<=[.!?])\s+(?=[A-Z"“])', text):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(text)))
    return [span for span in spans if text[span[0]:span[1]].strip()]


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def validate_reading_comprehension(lesson: LessonVariant, question_count: int) -> None:
    if len(lesson.reading_questions) != question_count or len(lesson.reading_answers) != question_count:
        raise LessonQualityError(f"Expected {question_count} reading questions with answer evidence.")

    seen = set()
    passage = normalize_whitespace(lesson.reading_text)
    for question, item in zip(lesson.reading_questions, lesson.reading_answers):
        normalized_question = normalize_whitespace(question).casefold()
        if not normalized_question or normalized_question in seen:
            raise LessonQualityError("Reading questions must be non-empty and distinct.")
        seen.add(normalized_question)
        if question != item.question:
            raise LessonQualityError("The answer key must follow the reading question order.")
        evidence = normalize_whitespace(item.evidence)
        if not re.search(r"(?<!\w)" + re.escape(evidence) + r"(?!\w)", passage):
            raise LessonQualityError(f"Evidence for question '{question}' is not quoted from the final reading.")


async def generate_reading_comprehension(
    lesson: LessonVariant, level: str, question_count: int, feedback: str = "",
) -> LessonVariant:
    spans = reading_sentence_spans(lesson.reading_text)
    if not spans:
        raise LessonQualityError("Cannot generate reading questions from an empty passage.")
    sentences = [lesson.reading_text[start:end] for start, end in spans]
    schema = ReadingComprehensionSelection.model_json_schema()
    schema["properties"]["items"].update(minItems=question_count, maxItems=question_count)
    for field in ("evidence_start", "evidence_end"):
        schema["$defs"]["ReadingEvidenceSelection"]["properties"][field]["maximum"] = len(spans)
    last_error = feedback

    for _ in range(LESSON_MAX_ATTEMPTS):
        prompt = build_reading_comprehension_prompt(
            level, lesson.title, lesson.lesson_goal, lesson.reading_text, question_count, last_error,
            sentences=sentences,
        )
        raw_response = await generate_from_ollama(prompt, json_schema=schema, task="comprehension")
        try:
            result = ReadingComprehensionSelection.model_validate_json(raw_response)
            answers = []
            for item in result.items:
                if not 1 <= item.evidence_start <= item.evidence_end <= len(spans):
                    raise LessonQualityError(
                        f"Evidence sentence range for '{item.question}' must satisfy "
                        f"1 <= evidence_start <= evidence_end <= {len(spans)}."
                    )
                start = spans[item.evidence_start - 1][0]
                end = spans[item.evidence_end - 1][1]
                answers.append(ReadingQuestionAnswer(
                    question=item.question, answer=item.answer,
                    evidence=lesson.reading_text[start:end],
                ))
            candidate = lesson.model_copy(update={
                "reading_questions": [item.question for item in answers],
                "reading_answers": answers,
            })
            validate_reading_comprehension(candidate, question_count)
        except ValueError as error:
            last_error = f"{feedback}\nStructural validation also failed: {error}".strip()
            continue

        lesson.reading_questions = candidate.reading_questions
        lesson.reading_answers = candidate.reading_answers
        return lesson

    raise LessonQualityError(
        f"Reading comprehension repair failed for '{lesson.title}' after {LESSON_MAX_ATTEMPTS} attempts: {last_error}"
    )
