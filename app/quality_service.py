from app.audience import audience_guidance
from app.errors import LessonQualityError
from app.cefr_service import get_cefr_profile
from app.generation_policy import LESSON_MAX_ATTEMPTS
from app.ollama_client import generate_from_ollama
from app.prompts import build_lesson_quality_prompt
from app.schemas import LessonQualityIssue, LessonQualityReview, LessonVariant


async def review_lesson_quality(
    lesson: LessonVariant, level: str, *, sections: set[str] | None = None,
) -> list[LessonQualityIssue]:
    cloze_assets = [asset for asset in lesson.worksheet_assets if asset.asset_type == "fill_in_the_blanks"]
    available = {"overview", "reading"}
    for section, supplied in {
        "vocabulary": lesson.target_vocabulary, "comprehension": lesson.reading_questions,
        "speaking": lesson.speaking_questions, "activities": lesson.pair_work_task or lesson.role_play_scenario,
        "sentence_frames": lesson.sentence_frames, "cloze": cloze_assets,
    }.items():
        if supplied:
            available.add(section)
    reviewed_sections = available if sections is None else available & sections
    context = {
        "level": level,
        "teaching_context": audience_guidance(lesson.english_type, lesson.age_group),
        "cefr_profile": get_cefr_profile(level),
        "review_sections": sorted(reviewed_sections),
        "title": lesson.title,
        "duration_minutes": lesson.duration_minutes,
        "theme": lesson.theme,
        "lesson_goal": lesson.lesson_goal,
        "grammar_focus": lesson.grammar_focus,
        "communication_focus": lesson.communication_focus,
        "vocabulary_focus": lesson.vocabulary_focus,
        "cefr_focus": lesson.cefr_focus,
        "target_vocabulary": lesson.target_vocabulary,
        "reading_text": lesson.reading_text,
        "reading_questions": lesson.reading_questions,
        "reading_answers": [item.model_dump() for item in lesson.reading_answers],
        "speaking_questions": lesson.speaking_questions,
        "pair_work_task": lesson.pair_work_task,
        "role_play_scenario": lesson.role_play_scenario,
        "sentence_frames": lesson.sentence_frames,
        "cloze_exercises": [asset.model_dump() for asset in cloze_assets],
    }
    feedback = ""
    for _ in range(LESSON_MAX_ATTEMPTS):
        raw_response = await generate_from_ollama(
            build_lesson_quality_prompt(context, feedback),
            json_schema=LessonQualityReview.model_json_schema(),
            task="review",
        )
        try:
            review = LessonQualityReview.model_validate_json(raw_response)
            if any(issue.section not in reviewed_sections for issue in review.issues):
                raise ValueError("Report issues only in review_sections: " + ", ".join(sorted(reviewed_sections)))
            return review.issues
        except ValueError as error:
            feedback = str(error)

    raise LessonQualityError(
        f"Could not obtain a valid quality review for '{lesson.title}' after {LESSON_MAX_ATTEMPTS} attempts: {feedback}"
    )
