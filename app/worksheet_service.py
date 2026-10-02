import re

from app.errors import LessonQualityError
from app.ollama_client import generate_from_ollama
from app.prompts import build_cloze_repair_prompt
from app.reading_service import normalize_whitespace
from app.schemas import ClozeExercise, LessonVariant, LessonWorksheetAsset


def word_pattern(word: str) -> re.Pattern:
    phrase = r"\s+".join(re.escape(part) for part in word.strip().split())
    return re.compile(r"(?<![\w'’\-])" + phrase + r"(?![\w'’\-])", re.IGNORECASE)


def replace_first_case_insensitive(text: str, target: str, replacement: str) -> str:
    if not target.strip():
        return text
    return word_pattern(target).sub(lambda _: replacement, text, count=1)


def split_into_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", normalized)


def validate_cloze_asset(lesson: LessonVariant, asset: LessonWorksheetAsset) -> None:
    if not 1 <= len(asset.lines) <= 4:
        raise LessonQualityError("A cloze exercise must have 1-4 grounded sentences.")
    if len(asset.lines) != len(asset.answer_key) or len(asset.lines) != len(asset.word_bank):
        raise LessonQualityError("Cloze sentences, word bank and answer key must have matching lengths.")

    bank = [normalize_whitespace(word).casefold() for word in asset.word_bank]
    answers = [normalize_whitespace(word).casefold() for word in asset.answer_key]
    vocabulary = {normalize_whitespace(word).casefold() for word in lesson.target_vocabulary}
    if any(not word for word in bank) or len(set(bank)) != len(bank):
        raise LessonQualityError("The cloze word bank must contain distinct, non-empty words.")
    if set(answers) != set(bank) or not set(bank) <= vocabulary:
        raise LessonQualityError("Cloze answers must use the word bank and target vocabulary.")

    source_sentences = {sentence.casefold() for sentence in split_into_sentences(lesson.reading_text)}
    used_sentences = set()
    for line, answer in zip(asset.lines, asset.answer_key):
        if line.count("____") != 1 or "_" in line.replace("____", ""):
            raise LessonQualityError("Every cloze sentence must contain exactly one blank.")
        restored = normalize_whitespace(line.replace("____", answer, 1)).casefold()
        if restored not in source_sentences or restored in used_sentences:
            raise LessonQualityError("Cloze answers must restore distinct sentences from the final reading.")
        # Check the actual blank location, not another occurrence of the answer.
        prefix, suffix = line.split("____")
        match = word_pattern(answer).match(prefix + answer + suffix, len(prefix))
        if match is None or match.end() != len(prefix) + len(answer):
            raise LessonQualityError("Cloze blanks must replace whole words or phrases.")
        used_sentences.add(restored)


def build_fill_in_the_blanks_asset(lesson: LessonVariant) -> LessonWorksheetAsset:
    sentences = split_into_sentences(lesson.reading_text)
    lines, answers = [], []
    used_sentences, used_words = set(), set()
    for word in lesson.target_vocabulary:
        normalized = normalize_whitespace(word)
        if not normalized or normalized.casefold() in used_words:
            continue
        for sentence in sentences:
            if sentence.casefold() in used_sentences:
                continue
            match = word_pattern(normalized).search(sentence)
            if match is None:
                continue
            lines.append(sentence[:match.start()] + "____" + sentence[match.end():])
            answers.append(match.group())
            used_sentences.add(sentence.casefold())
            used_words.add(normalized.casefold())
            break
        if len(lines) == 4:
            break

    asset = LessonWorksheetAsset(
        asset_type="fill_in_the_blanks",
        title=f"Fill in the blanks: {lesson.title}",
        instruction="Complete the sentences from the reading with words from the word bank.",
        lines=lines,
        word_bank=answers.copy(),
        answer_key=answers,
    )
    validate_cloze_asset(lesson, asset)
    return asset


async def repair_cloze_asset(lesson: LessonVariant, feedback: str) -> LessonWorksheetAsset:
    context = {"reading_text": lesson.reading_text, "target_vocabulary": lesson.target_vocabulary}
    last_error = feedback
    for _ in range(3):
        raw_response = await generate_from_ollama(
            build_cloze_repair_prompt(context, last_error), json_schema=ClozeExercise.model_json_schema(),
        )
        try:
            exercise = ClozeExercise.model_validate_json(raw_response)
            asset = LessonWorksheetAsset(
                asset_type="fill_in_the_blanks", title=f"Fill in the blanks: {lesson.title}",
                instruction="Complete the sentences from the reading with words from the word bank.",
                **exercise.model_dump(),
            )
            validate_cloze_asset(lesson, asset)
            return asset
        except ValueError as error:
            last_error = f"{feedback}\nStructural validation also failed: {error}"

    raise LessonQualityError(f"Cloze repair failed for '{lesson.title}' after 3 attempts: {last_error}")
