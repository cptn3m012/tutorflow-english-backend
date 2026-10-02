from app.errors import LessonQualityError
from app.ollama_client import generate_from_ollama
from app.prompts import build_lesson_quality_prompt
from app.schemas import LessonQualityIssue, LessonQualityReview, LessonVariant


async def review_lesson_quality(lesson: LessonVariant, level: str) -> list[LessonQualityIssue]:
    cloze_assets = [asset for asset in lesson.worksheet_assets if asset.asset_type == "fill_in_the_blanks"]
    context = {
        "level": level,
        "title": lesson.title,
        "theme": lesson.theme,
        "lesson_goal": lesson.lesson_goal,
        "grammar_focus": lesson.grammar_focus,
        "target_vocabulary": lesson.target_vocabulary,
        "reading_text": lesson.reading_text,
        "reading_answers": [item.model_dump() for item in lesson.reading_answers],
        "cloze_exercises": [asset.model_dump() for asset in cloze_assets],
    }
    feedback = ""
    for _ in range(3):
        raw_response = await generate_from_ollama(
            build_lesson_quality_prompt(context, feedback),
            json_schema=LessonQualityReview.model_json_schema(),
        )
        try:
            review = LessonQualityReview.model_validate_json(raw_response)
            if not cloze_assets and any(issue.section == "cloze" for issue in review.issues):
                raise ValueError("No cloze exercise is supplied; review only reading and comprehension.")
            return review.issues
        except ValueError as error:
            feedback = str(error)

    raise LessonQualityError(
        f"Could not obtain a valid quality review for '{lesson.title}' after 3 attempts: {feedback}"
    )
