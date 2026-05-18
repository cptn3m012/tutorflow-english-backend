import json
from datetime import datetime, UTC

from app.prompts import build_lesson_prompt, build_reading_rewrite_prompt
from app.ollama_client import generate_from_ollama
from app.schemas import (
    CefrProfile,
    LessonAdvancedOptions,
    LessonGenerateRequest,
    LessonGenerateResponse,
    LessonVariant,
    LessonVisualActivity,
    LessonWorksheetAsset,
)
from app.theme_service import resolve_theme
from app.image_service import attach_images_to_lessons
from app.cefr_service import (
    choose_best_grammar_target,
    choose_best_target_from_text,
    format_cefr_profile_for_prompt,
    get_cefr_profile,
    get_cefr_targets_by_category,
    normalize_cefr_focus_item,
)


def extract_json_from_text(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find valid JSON object in model response:\n\n{raw_text}")

    return raw_text[start:end + 1]


def count_words(text: str) -> int:
    return len([word for word in text.replace("\n", " ").split() if word.strip(" ,.!?;:()[]{}\"'")])


def get_reading_word_targets(
    level: str,
    duration: int,
    options: LessonAdvancedOptions | None = None,
) -> tuple[int, int]:
    if options and (options.reading_min_words is not None or options.reading_max_words is not None):
        default_minimum, default_maximum = get_default_reading_word_targets(level, duration)
        return (
            options.reading_min_words if options.reading_min_words is not None else default_minimum,
            options.reading_max_words if options.reading_max_words is not None else default_maximum,
        )

    return get_default_reading_word_targets(level, duration)


def get_default_reading_word_targets(level: str, duration: int) -> tuple[int, int]:
    normalized_level = level.upper()
    base_targets = {
        "A1": (55, 90),
        "A2": (75, 115),
        "B1": (95, 145),
        "B2": (120, 180),
        "C1": (150, 220),
        "C2": (180, 260),
    }
    minimum, maximum = base_targets.get(normalized_level, (80, 120))
    duration_bonus_steps = max(duration - 30, 0) // 15
    minimum += duration_bonus_steps * 15
    maximum += duration_bonus_steps * 20
    return minimum, maximum


def build_reading_wording_guidance(
    level: str,
    duration: int,
    options: LessonAdvancedOptions | None = None,
) -> str:
    minimum, maximum = get_reading_word_targets(level, duration, options)
    return (
        f"reading_text must be between {minimum} and {maximum} words for this "
        f"{duration}-minute {level} lesson"
    )


def build_reading_length_fallback_paragraph(lesson: LessonVariant, level: str, theme: str) -> str:
    vocabulary = lesson.target_vocabulary[:6]
    if len(vocabulary) < 6:
        vocabulary = vocabulary + ["place", "people", "food", "drink", "table", "menu"][: 6 - len(vocabulary)]

    return " ".join([
        f"This {theme} lesson stays at {level} level and uses everyday language.",
        f"In the story, people talk about {vocabulary[0]}, {vocabulary[1]}, and {vocabulary[2]} in a natural way.",
        f"They also use words like {vocabulary[3]}, {vocabulary[4]}, and {vocabulary[5]} when they describe the situation and their choices.",
        "The reader can understand the main situation, follow simple actions, and notice useful expressions for daily communication.",
        "The text gives a few extra details about people, place, food, and choices, so it feels more complete and easier to discuss.",
    ])


def trim_text_to_word_limit(text: str, maximum_words: int) -> str:
    words = text.replace("\n", " ").split()
    if len(words) <= maximum_words:
        return text.strip()

    trimmed = " ".join(words[:maximum_words]).rstrip(" ,;:")
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


async def rewrite_reading_text_if_needed(
    lesson: LessonVariant,
    level: str,
    theme: str,
    duration: int,
    options: LessonAdvancedOptions | None = None,
) -> LessonVariant:
    minimum_words, maximum_words = get_reading_word_targets(level, duration, options)
    current_text = lesson.reading_text

    for _ in range(3):
        current_word_count = count_words(current_text)
        if minimum_words <= current_word_count <= maximum_words:
            lesson.reading_text = current_text
            return lesson

        if current_word_count < minimum_words:
            rewrite_feedback = (
                f"The current version is too short at {current_word_count} words. "
                f"Return a fuller version with at least {minimum_words} words and no more than {maximum_words} words."
            )
        else:
            rewrite_feedback = (
                f"The current version is too long at {current_word_count} words. "
                f"Return a tighter version with at least {minimum_words} words and no more than {maximum_words} words."
            )

        prompt = build_reading_rewrite_prompt(
            level=level,
            theme=theme,
            lesson_title=lesson.title,
            lesson_goal=lesson.lesson_goal,
            vocabulary_focus=lesson.vocabulary_focus or "",
            grammar_focus=lesson.grammar_focus or "",
            target_vocabulary=lesson.target_vocabulary,
            original_reading_text=current_text,
            minimum_words=minimum_words,
            maximum_words=maximum_words,
            rewrite_feedback=rewrite_feedback,
        )
        rewritten_text = (await generate_from_ollama(prompt)).strip()
        cleaned_text = rewritten_text.strip().strip('"').strip()

        if cleaned_text:
            current_text = cleaned_text

    current_word_count = count_words(current_text)
    if current_word_count < minimum_words:
        fallback_paragraph = build_reading_length_fallback_paragraph(lesson, level, theme)
        current_text = f"{current_text.strip()}\n\n{fallback_paragraph}".strip()
        while count_words(current_text) < minimum_words:
            current_text = (
                f"{current_text} "
                "The passage adds more simple details about the setting, the people, and the choices they make during the lesson topic."
            ).strip()

    current_text = trim_text_to_word_limit(current_text, maximum_words)
    lesson.reading_text = current_text

    return lesson


async def repair_lesson_readings(
    lessons: list[LessonVariant],
    level: str,
    theme: str,
    duration: int,
    options: LessonAdvancedOptions | None = None,
) -> list[LessonVariant]:
    repaired_lessons = []

    for lesson in lessons:
        repaired_lessons.append(
            await rewrite_reading_text_if_needed(
                lesson=lesson,
                level=level,
                theme=theme,
                duration=duration,
                options=options,
            )
        )

    return repaired_lessons


def normalize_lessons_content(lessons: list[LessonVariant], level: str) -> list[LessonVariant]:
    category_targets = get_cefr_targets_by_category(level)

    for lesson in lessons:
        normalized_items = [normalize_cefr_focus_item(item, level) for item in lesson.cefr_focus]

        communication_focus = next(
            (item for item in normalized_items if item in category_targets["communication"]),
            None,
        )
        vocabulary_focus = next(
            (item for item in normalized_items if item in category_targets["vocabulary"]),
            None,
        )
        grammar_focus = next(
            (item for item in normalized_items if item in category_targets["grammar"]),
            None,
        )

        communication_text = " ".join([
            lesson.title,
            lesson.lesson_goal,
            lesson.pair_work_task,
            lesson.role_play_scenario,
            *lesson.speaking_questions,
        ])
        vocabulary_text = " ".join([
            lesson.title,
            lesson.lesson_goal,
            lesson.theme or "",
            *lesson.target_vocabulary,
        ])
        grammar_text = " ".join([
            lesson.reading_text,
            lesson.pair_work_task,
            lesson.role_play_scenario,
            *lesson.reading_questions,
            *lesson.speaking_questions,
            *lesson.sentence_frames,
        ])

        lesson.communication_focus = communication_focus or choose_best_target_from_text(
            category_targets["communication"],
            communication_text,
            category_targets["communication"][0],
        )
        lesson.vocabulary_focus = vocabulary_focus or choose_best_target_from_text(
            category_targets["vocabulary"],
            vocabulary_text,
            category_targets["vocabulary"][0],
        )
        lesson.grammar_focus = grammar_focus or choose_best_grammar_target(
            category_targets["grammar"],
            grammar_text,
        )
        lesson.cefr_focus = [
            lesson.communication_focus,
            lesson.vocabulary_focus,
            lesson.grammar_focus,
        ]

    return lessons


def build_visual_activity_fallback(
    lesson: LessonVariant,
    visual_word_count: int = 6,
) -> LessonVisualActivity:
    target_words = []
    for word in lesson.target_vocabulary:
        normalized = word.strip()
        if normalized and normalized not in target_words:
            target_words.append(normalized)
        if len(target_words) == visual_word_count:
            break

    if len(target_words) < visual_word_count:
        for word in ["table", "menu", "coffee", "cup", "window", "bag"]:
            if word not in target_words:
                target_words.append(word)
            if len(target_words) == visual_word_count:
                break

    return LessonVisualActivity(
        activity_type="find_in_picture",
        title=f"Find and say: {lesson.title}",
        instruction="Look at the scene, find the objects, and say each word in English.",
        scene_prompt=(
            f"Create one clear educational scene about {lesson.theme or lesson.title}. "
            f"Show these items clearly: {', '.join(target_words)}. "
            "Use a realistic, learner-friendly style with no text labels in the image."
        ),
        target_words=target_words,
        answer_key=target_words.copy(),
    )


def replace_first_case_insensitive(text: str, target: str, replacement: str) -> str:
    lower_text = text.lower()
    lower_target = target.lower()
    index = lower_text.find(lower_target)

    if index == -1:
        return text

    return f"{text[:index]}{replacement}{text[index + len(target):]}"


def split_into_sentences(text: str) -> list[str]:
    normalized_text = text.replace("\n", " ").strip()
    if not normalized_text:
        return []

    sentences = []
    current = []

    for character in normalized_text:
        current.append(character)
        if character in ".!?":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []

    trailing = "".join(current).strip()
    if trailing:
        sentences.append(trailing)

    return sentences


def normalize_frame_text(frame: str, fallback_word: str) -> str:
    text = frame.strip()
    if not text:
        return fallback_word.capitalize() + "."

    text = text.replace("...", f" {fallback_word}")
    text = text.replace("…", f" {fallback_word}")

    if text[-1] not in ".!?":
        text += "."

    return text[0].upper() + text[1:]


def build_cloze_line(word: str, theme: str) -> str:
    return f"In this lesson about {theme}, students use the word ____ in a real-life context."


def build_reading_illustration_asset(
    lesson: LessonVariant,
    visual_word_count: int = 6,
) -> LessonWorksheetAsset:
    visual_activity = lesson.visual_activity or build_visual_activity_fallback(lesson, visual_word_count)
    word_bank = visual_activity.target_words[:visual_word_count]

    return LessonWorksheetAsset(
        asset_type="reading_illustration",
        title=f"Illustration brief: {lesson.title}",
        instruction="Use this brief to create or sketch one illustration that matches the reading.",
        prompt=visual_activity.scene_prompt,
        lines=[
            lesson.lesson_goal,
            f"Focus on the theme: {lesson.theme or lesson.title}.",
            "Keep the scene clear, realistic, and useful for language practice.",
        ],
        word_bank=word_bank,
        answer_key=[],
    )


def build_find_in_picture_asset(
    lesson: LessonVariant,
    visual_word_count: int = 6,
) -> LessonWorksheetAsset:
    visual_activity = lesson.visual_activity or build_visual_activity_fallback(lesson, visual_word_count)

    return LessonWorksheetAsset(
        asset_type="find_in_picture",
        title=visual_activity.title,
        instruction=visual_activity.instruction,
        prompt=visual_activity.scene_prompt,
        lines=[
            "Ask the learner to point to each object and say the word aloud.",
            "After finding the items, ask one simple sentence about the scene.",
        ],
        word_bank=visual_activity.target_words,
        answer_key=visual_activity.answer_key,
    )


def build_label_the_picture_asset(lesson: LessonVariant) -> LessonWorksheetAsset:
    visual_activity = lesson.visual_activity or build_visual_activity_fallback(lesson)
    labels = visual_activity.target_words

    return LessonWorksheetAsset(
        asset_type="label_the_picture",
        title=f"Label the picture: {lesson.title}",
        instruction="Write the correct English labels on the illustration or generated scene.",
        prompt=visual_activity.scene_prompt,
        lines=[
            "Place the labels next to the matching objects in the picture.",
            "Then read all labels aloud in English.",
        ],
        word_bank=labels,
        answer_key=labels.copy(),
    )


def build_fill_in_the_blanks_asset(lesson: LessonVariant) -> LessonWorksheetAsset:
    sentences = split_into_sentences(lesson.reading_text)
    selected_words = []

    for word in lesson.target_vocabulary:
        normalized_word = word.strip()
        if normalized_word and normalized_word.lower() in lesson.reading_text.lower():
            selected_words.append(normalized_word)
        if len(selected_words) == 4:
            break

    if len(selected_words) < 4:
        for word in lesson.target_vocabulary:
            normalized_word = word.strip()
            if normalized_word and normalized_word not in selected_words:
                selected_words.append(normalized_word)
            if len(selected_words) == 4:
                break

    lines = []
    for index, word in enumerate(selected_words):
        matching_sentence = next(
            (sentence for sentence in sentences if word.lower() in sentence.lower() and sentence not in lines),
            None,
        )
        if matching_sentence:
            lines.append(replace_first_case_insensitive(matching_sentence, word, "____"))
        else:
            lines.append(build_cloze_line(word, lesson.theme or lesson.title))

    return LessonWorksheetAsset(
        asset_type="fill_in_the_blanks",
        title=f"Fill in the blanks: {lesson.title}",
        instruction="Complete the sentences with words from the word bank.",
        prompt=None,
        lines=lines,
        word_bank=selected_words,
        answer_key=selected_words.copy(),
    )


def build_mini_dialogue_asset(lesson: LessonVariant) -> LessonWorksheetAsset:
    fallback_words = lesson.target_vocabulary[:3] or ["coffee", "table", "menu"]
    while len(fallback_words) < 3:
        fallback_words.append("lesson")

    frames = lesson.sentence_frames[:3]
    while len(frames) < 3:
        frames.append(f"I would like {fallback_words[len(frames)]}")

    dialogue_lines = [
        "A: Hello. Can I help you?",
        f"B: {normalize_frame_text(frames[0], fallback_words[0])}",
        f"A: {normalize_frame_text(frames[1], fallback_words[1])}",
        f"B: {normalize_frame_text(frames[2], fallback_words[2])}",
    ]

    return LessonWorksheetAsset(
        asset_type="mini_dialogue",
        title=f"Mini dialogue: {lesson.title}",
        instruction="Read the dialogue, practise it in pairs, and then change two details.",
        prompt=None,
        lines=dialogue_lines,
        word_bank=fallback_words,
        answer_key=dialogue_lines.copy(),
    )


def build_worksheet_assets(
    lesson: LessonVariant,
    visual_word_count: int = 6,
) -> list[LessonWorksheetAsset]:
    return [
        build_reading_illustration_asset(lesson, visual_word_count),
        build_find_in_picture_asset(lesson, visual_word_count),
        build_label_the_picture_asset(lesson),
        build_fill_in_the_blanks_asset(lesson),
        build_mini_dialogue_asset(lesson),
    ]


def ensure_worksheet_assets(
    lessons: list[LessonVariant],
    options: LessonAdvancedOptions,
) -> list[LessonVariant]:
    for lesson in lessons:
        if not options.include_worksheet_assets:
            lesson.worksheet_assets = []
            continue

        if lesson.worksheet_assets:
            continue

        lesson.worksheet_assets = build_worksheet_assets(lesson, options.visual_word_count)

    return lessons


def ensure_visual_activities(
    lessons: list[LessonVariant],
    options: LessonAdvancedOptions,
) -> list[LessonVariant]:
    for lesson in lessons:
        if not options.include_visual_activity:
            lesson.visual_activity = None
            continue

        if lesson.visual_activity and len(lesson.visual_activity.target_words) == options.visual_word_count:
            lesson.visual_activity.answer_key = lesson.visual_activity.target_words.copy()
            continue

        lesson.visual_activity = build_visual_activity_fallback(lesson, options.visual_word_count)

    return lessons


def validate_lessons_content(
    lessons: list[LessonVariant],
    expected_count: int,
    level: str,
    duration: int,
    options: LessonAdvancedOptions,
) -> None:
    if len(lessons) != expected_count:
        raise ValueError(f"Model returned {len(lessons)} lessons, but expected {expected_count}.")

    minimum_reading_words, maximum_reading_words = get_reading_word_targets(level, duration, options)

    for lesson in lessons:
        if len(lesson.target_vocabulary) != options.target_vocabulary_count:
            raise ValueError(
                f"Lesson '{lesson.title}' does not have exactly "
                f"{options.target_vocabulary_count} vocabulary items."
            )
        if len(lesson.reading_questions) != options.reading_question_count:
            raise ValueError(
                f"Lesson '{lesson.title}' does not have exactly "
                f"{options.reading_question_count} reading questions."
            )
        if len(lesson.speaking_questions) != options.speaking_question_count:
            raise ValueError(
                f"Lesson '{lesson.title}' does not have exactly "
                f"{options.speaking_question_count} speaking questions."
            )
        if len(lesson.sentence_frames) != options.sentence_frame_count:
            raise ValueError(
                f"Lesson '{lesson.title}' does not have exactly "
                f"{options.sentence_frame_count} sentence frames."
            )
        if len(lesson.cefr_focus) != 3:
            raise ValueError(f"Lesson '{lesson.title}' does not have exactly 3 CEFR focus items.")
        if not lesson.reading_text or not lesson.reading_text.strip():
            raise ValueError(f"Lesson '{lesson.title}' has an empty reading_text.")
        reading_word_count = count_words(lesson.reading_text)
        if reading_word_count < minimum_reading_words:
            raise ValueError(
                f"Lesson '{lesson.title}' reading_text is too short: "
                f"{reading_word_count} words, expected at least {minimum_reading_words}."
            )
        if reading_word_count > maximum_reading_words:
            raise ValueError(
                f"Lesson '{lesson.title}' reading_text is too long: "
                f"{reading_word_count} words, expected at most {maximum_reading_words}."
            )
        if not options.include_visual_activity:
            continue
        if not lesson.visual_activity:
            raise ValueError(f"Lesson '{lesson.title}' does not include visual_activity.")
        if lesson.visual_activity.activity_type != "find_in_picture":
            raise ValueError(f"Lesson '{lesson.title}' has unsupported visual activity type.")
        if len(lesson.visual_activity.target_words) != options.visual_word_count:
            raise ValueError(
                f"Lesson '{lesson.title}' does not have exactly "
                f"{options.visual_word_count} visual target words."
            )
        if options.include_worksheet_assets and len(lesson.worksheet_assets) < 5:
            raise ValueError(f"Lesson '{lesson.title}' does not include enough worksheet assets.")


def prepare_lessons_for_frontend(
    lessons: list[LessonVariant],
    level: str,
    theme: str,
    duration: int,
) -> list[LessonVariant]:
    prepared_lessons = []

    for index, lesson in enumerate(lessons, start=1):
        lesson.id = f"lesson-{index}"
        lesson.level = level
        lesson.duration_minutes = duration
        lesson.theme = theme
        lesson.primary_image = lesson.images[0] if lesson.images else None

        merged_tags = []
        for item in [*lesson.cefr_focus, *lesson.image_keywords]:
            normalized = item.strip()
            if normalized and normalized not in merged_tags:
                merged_tags.append(normalized)

        lesson.tags = merged_tags
        prepared_lessons.append(lesson)

    return prepared_lessons


def attach_lesson_context(
    lessons: list[LessonVariant],
    level: str,
    theme: str,
    duration: int,
) -> list[LessonVariant]:
    for lesson in lessons:
        lesson.level = level
        lesson.duration_minutes = duration
        lesson.theme = theme

    return lessons

async def try_generate_once(request: LessonGenerateRequest) -> LessonGenerateResponse:
    options = request.advanced_options
    final_theme, source_type = resolve_theme(
        topic=request.topic,
        lesson_date=request.lesson_date
    )
    cefr_profile = get_cefr_profile(request.level)

    prompt = build_lesson_prompt(
        level=cefr_profile["level"],
        topic=final_theme,
        duration=request.duration,
        variant_count=request.variant_count,
        cefr_profile_prompt=format_cefr_profile_for_prompt(request.level),
        reading_wording_guidance=build_reading_wording_guidance(
            cefr_profile["level"],
            request.duration,
            options,
        ),
        target_vocabulary_count=options.target_vocabulary_count,
        reading_question_count=options.reading_question_count,
        speaking_question_count=options.speaking_question_count,
        sentence_frame_count=options.sentence_frame_count,
        visual_word_count=options.visual_word_count,
        include_visual_activity=options.include_visual_activity,
        skill_focus=options.skill_focus,
        lesson_style=options.lesson_style,
        extra_instructions=options.extra_instructions,
    )

    raw_response = await generate_from_ollama(prompt)

    try:
        cleaned_json = extract_json_from_text(raw_response)
        parsed_json = json.loads(cleaned_json)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON.\n\nRaw response:\n{raw_response}")

    if "lessons" not in parsed_json:
        raise ValueError("Model response does not contain 'lessons' field.")

    lessons = [LessonVariant.model_validate(item) for item in parsed_json["lessons"]]

    lessons = normalize_lessons_content(lessons, cefr_profile["level"])
    lessons = await repair_lesson_readings(
        lessons=lessons,
        level=cefr_profile["level"],
        theme=final_theme,
        duration=request.duration,
        options=options,
    )
    lessons = ensure_visual_activities(lessons, options)
    lessons = ensure_worksheet_assets(lessons, options)
    validate_lessons_content(
        lessons,
        request.variant_count,
        cefr_profile["level"],
        request.duration,
        options,
    )
    lessons = attach_lesson_context(
        lessons=lessons,
        level=cefr_profile["level"],
        theme=final_theme,
        duration=request.duration,
    )

    if options.include_images:
        lessons = await attach_images_to_lessons(lessons)
    else:
        for lesson in lessons:
            lesson.images = []
            lesson.primary_image = None
    lessons = prepare_lessons_for_frontend(
        lessons=lessons,
        level=cefr_profile["level"],
        theme=final_theme,
        duration=request.duration,
    )

    return LessonGenerateResponse(
        requested_level=cefr_profile["level"],
        lesson_theme=final_theme,
        duration_minutes=request.duration,
        source_type=source_type,
        total_lessons=len(lessons),
        generated_at=datetime.now(UTC).isoformat(),
        cefr_profile=CefrProfile.model_validate(cefr_profile),
        lessons=lessons
    )


async def generate_lesson_variants(request: LessonGenerateRequest) -> LessonGenerateResponse:
    last_error = None

    for attempt in range(3):
        try:
            return await try_generate_once(request)
        except Exception as e:
            last_error = e

    raise ValueError(f"Generation failed after 3 attempts. Last error: {str(last_error)}")
