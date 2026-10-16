"""Section edits keep unrelated content and publish only complete, validated results."""
from copy import deepcopy
import hashlib
import json

from app.errors import LessonQualityError
from app.generation_policy import LESSON_MAX_ATTEMPTS
from app.lesson_service import build_mini_dialogue_asset, count_words, rewrite_reading_text_if_needed
from app.lesson_sections import SECTION_FIELDS, replace_text_section
from app.quality_service import review_lesson_quality
from app.reading_service import generate_reading_comprehension
from app.saved_lesson_schemas import SavedLessonWriteContent
from app.schemas import LessonAdvancedOptions, LessonVariant
from app.worksheet_service import build_fill_in_the_blanks_asset, repair_cloze_asset


def content_token(record) -> str:
    content = {"id": record.id, "level": record.level, "duration": record.duration,
               "theme": record.theme, "lesson": record.lesson}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def invalidate_picture_activity(content: dict):
    if content.get("visual_activity") or content.get("images") or content.get("primary_image"):
        content["visual_activity"] = None
        content["picture_context_stale"] = True


def edit_content(original: dict, changes: dict) -> dict:
    content = deepcopy(original)
    changed = {field for field, value in changes.items() if content.get(field) != value}
    content.update(changes)
    if "reading_text" in changed:
        content.update(reading_questions=[], reading_answers=[])
    if changed & {"reading_text", "target_vocabulary"}:
        content["worksheet_assets"] = [asset for asset in content.get("worksheet_assets", [])
                                       if asset.get("asset_type") not in ("fill_in_the_blanks", "reading_illustration")]
    if changed & {"reading_text", "target_vocabulary", "lesson_goal"}:
        invalidate_picture_activity(content)
    if changed & {"sentence_frames", "target_vocabulary"}:
        content["worksheet_assets"] = [asset for asset in content.get("worksheet_assets", [])
                                       if asset.get("asset_type") != "mini_dialogue"]
    if "reading_questions" in changed and "reading_answers" not in changes:
        content["reading_answers"] = []
    SavedLessonWriteContent.model_validate(content)
    return content


def as_variant(content: dict) -> LessonVariant:
    data = {**content}
    if isinstance(data.get("cefr_focus"), str):
        data["cefr_focus"] = []
    return LessonVariant.model_validate(data)


async def rebuild_cloze(lesson: LessonVariant, original: dict):
    if not any(asset.get("asset_type") == "fill_in_the_blanks" for asset in original.get("worksheet_assets", [])):
        return
    try:
        asset = build_fill_in_the_blanks_asset(lesson)
    except LessonQualityError as error:
        asset = await repair_cloze_asset(lesson, str(error))
    lesson.worksheet_assets = [item for item in lesson.worksheet_assets if item.asset_type != "fill_in_the_blanks"] + [asset]


async def regenerate_content(original: dict, level: str, theme: str, duration: int, section: str, instructions: str) -> dict:
    lesson = as_variant(original)
    question_count = max(1, min(8, len(lesson.reading_questions) or 3))
    async def generate_section(feedback: str):
        if section == "reading":
            words = count_words(as_variant(original).reading_text)
            options = LessonAdvancedOptions(reading_min_words=max(40, min(400, int(words * .8))),
                                            reading_max_words=max(50, min(500, int(words * 1.2))))
            await rewrite_reading_text_if_needed(lesson, level, theme, duration, options, force=True,
                quality_feedback=feedback or "Write a different passage on the same topic.")
            lesson.worksheet_assets = as_variant(original).worksheet_assets
            await generate_reading_comprehension(lesson, level, question_count)
            await rebuild_cloze(lesson, original)
        elif section == "comprehension":
            await generate_reading_comprehension(lesson, level, question_count, feedback)
        else:
            await replace_text_section(lesson, level, theme, section, feedback)
            if section == "vocabulary":
                await rebuild_cloze(lesson, original)
            if section in ("vocabulary", "sentence_frames"):
                lesson.worksheet_assets = [build_mini_dialogue_asset(lesson) if asset.asset_type == "mini_dialogue"
                                           else asset for asset in lesson.worksheet_assets]

    await generate_section(instructions)
    review_sections = ({"reading", "comprehension", "cloze"} if section == "reading" else
                       {"reading", "comprehension"} if section == "comprehension" else
                       {"vocabulary", "cloze"} if section == "vocabulary" else {section})
    for attempt in range(LESSON_MAX_ATTEMPTS):
        issues = await review_lesson_quality(lesson, level, sections=review_sections)
        if not issues:
            break
        feedback = "; ".join(f"{issue.section}: {issue.message}" for issue in issues)
        if section == "comprehension" and any(issue.section == "reading" for issue in issues):
            raise LessonQualityError("The unchanged reading needs revision: " + feedback)
        if attempt + 1 == LESSON_MAX_ATTEMPTS:
            raise LessonQualityError("The regenerated section did not pass the quality review: " + feedback)
        if any(issue.section == section for issue in issues):
            await generate_section(instructions + "\nRepair feedback: " + feedback)
            continue
        if any(issue.section == "comprehension" for issue in issues):
            await generate_reading_comprehension(lesson, level, question_count, feedback)
        if any(issue.section == "cloze" for issue in issues):
            asset = await repair_cloze_asset(lesson, feedback)
            lesson.worksheet_assets = [asset if item.asset_type == "fill_in_the_blanks" else item
                                       for item in lesson.worksheet_assets]

    fields = set(SECTION_FIELDS[section])
    if section == "reading":
        fields.update(("reading_questions", "reading_answers", "worksheet_assets"))
    if section in ("vocabulary", "sentence_frames"):
        fields.add("worksheet_assets")
    content = deepcopy(original)
    generated = lesson.model_dump(mode="json")
    content.update({field: generated[field] for field in fields})
    if "worksheet_assets" in fields:
        replaced_types = ({"fill_in_the_blanks"} if section == "reading" else
                          {"fill_in_the_blanks", "mini_dialogue"} if section == "vocabulary" else {"mini_dialogue"})
        replacements = {asset["asset_type"]: asset for asset in generated["worksheet_assets"]}
        content["worksheet_assets"] = [
            {**asset, **replacements[asset["asset_type"]]} if asset.get("asset_type") in replaced_types
            and asset["asset_type"] in replacements else deepcopy(asset)
            for asset in original.get("worksheet_assets", [])
        ]
    if section in ("overview", "reading", "vocabulary"):
        invalidate_picture_activity(content)
    SavedLessonWriteContent.model_validate(content)
    return content
