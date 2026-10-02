import re

from app.errors import LessonQualityError
from app.ollama_client import generate_from_ollama
from app.prompts import build_reading_comprehension_prompt
from app.schemas import LessonVariant, ReadingComprehension


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
    schema = ReadingComprehension.model_json_schema()
    schema["properties"]["items"].update(minItems=question_count, maxItems=question_count)
    last_error = feedback

    for _ in range(3):
        prompt = build_reading_comprehension_prompt(
            level, lesson.title, lesson.lesson_goal, lesson.reading_text, question_count, last_error,
        )
        raw_response = await generate_from_ollama(prompt, json_schema=schema)
        try:
            result = ReadingComprehension.model_validate_json(raw_response)
            candidate = lesson.model_copy(update={
                "reading_questions": [item.question for item in result.items],
                "reading_answers": result.items,
            })
            validate_reading_comprehension(candidate, question_count)
        except ValueError as error:
            last_error = str(error)
            continue

        lesson.reading_questions = candidate.reading_questions
        lesson.reading_answers = candidate.reading_answers
        return lesson

    raise LessonQualityError(
        f"Reading comprehension repair failed for '{lesson.title}' after 3 attempts: {last_error}"
    )
