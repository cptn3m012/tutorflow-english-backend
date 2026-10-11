"""Structured model output contracts, separate from the public lesson response."""

from app.schemas import LessonAdvancedOptions


def _text_schema() -> dict:
    return {"type": "string", "minLength": 1}


def _text_array_schema(count: int) -> dict:
    return {
        "type": "array",
        "items": _text_schema(),
        "minItems": count,
        "maxItems": count,
    }


def _object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_lesson_draft_schema(
    variant_count: int, options: LessonAdvancedOptions, cefr_profile: dict,
) -> dict:
    """Constrain authored fields; later stages add answers, assets and metadata."""
    properties = {
        "title": _text_schema(),
        "lesson_goal": _text_schema(),
        "target_vocabulary": _text_array_schema(options.target_vocabulary_count),
        "reading_text": _text_schema(),
        "speaking_questions": _text_array_schema(options.speaking_question_count),
        "pair_work_task": _text_schema(),
        "role_play_scenario": _text_schema(),
        "sentence_frames": _text_array_schema(options.sentence_frame_count),
    }
    focus_schemas = []
    for category in ("communication", "vocabulary", "grammar"):
        focus_schema = {
            **_text_schema(),
            "enum": list(cefr_profile[f"{category}_targets"]),
        }
        properties[f"{category}_focus"] = focus_schema
        focus_schemas.append(focus_schema)
    properties["cefr_focus"] = {
        "type": "array",
        # Ollama's installed schema converter requires an object for array items.
        # Category-specific fields and the lesson validator retain the CEFR order.
        "items": {**_text_schema(), "enum": [target for schema in focus_schemas for target in schema["enum"]]},
        "minItems": 3,
        "maxItems": 3,
    }
    if options.include_visual_activity:
        properties["visual_activity"] = _object_schema({
            "activity_type": {"type": "string", "enum": ["find_in_picture"]},
            "title": _text_schema(),
            "instruction": _text_schema(),
            "scene_prompt": _text_schema(),
            "target_words": _text_array_schema(options.visual_word_count),
            "answer_key": _text_array_schema(options.visual_word_count),
        })
    return _object_schema({
        "lessons": {
            "type": "array",
            "items": _object_schema(properties),
            "minItems": variant_count,
            "maxItems": variant_count,
        },
    })


IMAGE_METADATA_SCHEMA = _object_schema({
    "image_search_query": _text_schema(),
    "image_keywords": _text_array_schema(4),
})
