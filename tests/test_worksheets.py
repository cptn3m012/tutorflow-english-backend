import unittest

from app.errors import LessonQualityError
from app.lesson_service import ensure_worksheet_assets, validate_lessons_content
from app.schemas import LessonAdvancedOptions, ReadingQuestionAnswer
from app.worksheet_service import (
    build_fill_in_the_blanks_asset, replace_first_case_insensitive, validate_cloze_asset,
)
from tests.helpers import COMPREHENSION, make_lesson


class WorksheetTests(unittest.TestCase):
    def test_replacement_matches_whole_words_and_phrases(self):
        self.assertEqual(replace_first_case_insensitive("The teacher drinks tea.", "tea", "____"),
                         "The teacher drinks ____.")
        self.assertEqual(replace_first_case_insensitive("I can't go.", "can", "____"), "I can't go.")
        self.assertEqual(replace_first_case_insensitive("I order ICE CREAM.", "ice cream", "____"),
                         "I order ____.")

    def test_cloze_restores_distinct_reading_sentences(self):
        lesson = make_lesson()
        asset = build_fill_in_the_blanks_asset(lesson)
        self.assertEqual(len(asset.lines), 4)
        restored = [line.replace("____", answer) for line, answer in zip(asset.lines, asset.answer_key)]
        self.assertEqual(len(set(restored)), 4)
        for sentence in restored:
            self.assertIn(sentence, lesson.reading_text)

    def test_missing_vocabulary_never_produces_generic_filler(self):
        lesson = make_lesson(reading_text="The teacher is kind.", target_vocabulary=["tea", "menu", "coffee"])
        with self.assertRaises(LessonQualityError):
            build_fill_in_the_blanks_asset(lesson)

    def test_cloze_can_use_fewer_than_four_real_contexts(self):
        lesson = make_lesson(reading_text="Anna orders tea.", target_vocabulary=["tea", "menu", "coffee"])
        asset = build_fill_in_the_blanks_asset(lesson)
        self.assertEqual(asset.lines, ["Anna orders ____."])
        self.assertEqual(asset.answer_key, ["tea"])

    def test_wrong_answer_stale_sentence_and_partial_word_are_rejected(self):
        lesson = make_lesson()
        asset = build_fill_in_the_blanks_asset(lesson)
        invalid = asset.model_copy(deep=True)
        invalid.answer_key[0] = "coffee"
        with self.assertRaises(LessonQualityError):
            validate_cloze_asset(lesson, invalid)
        invalid = asset.model_copy(deep=True)
        invalid.lines[0] = "Ben eats ____ in the park."
        with self.assertRaises(LessonQualityError):
            validate_cloze_asset(lesson, invalid)
        lesson = make_lesson(reading_text="The teacher is kind.", target_vocabulary=["tea"])
        invalid = asset.model_copy(update={"lines": ["The ____cher is kind."],
                                         "word_bank": ["tea"], "answer_key": ["tea"]})
        with self.assertRaisesRegex(LessonQualityError, "whole words"):
            validate_cloze_asset(lesson, invalid)

    def test_incoming_worksheets_are_rebuilt_from_final_reading(self):
        lesson = make_lesson()
        stale = build_fill_in_the_blanks_asset(lesson)
        stale.lines = ["Stale worksheet"]
        lesson.worksheet_assets = [stale]
        ensure_worksheet_assets([lesson], LessonAdvancedOptions())
        cloze = next(asset for asset in lesson.worksheet_assets if asset.asset_type == "fill_in_the_blanks")
        validate_cloze_asset(lesson, cloze)
        self.assertNotIn("Stale worksheet", cloze.lines)

    def test_worksheets_are_validated_when_visual_activity_is_disabled(self):
        lesson = make_lesson(reading_questions=[item["question"] for item in COMPREHENSION["items"]],
                             reading_answers=[ReadingQuestionAnswer(**item) for item in COMPREHENSION["items"]])
        options = LessonAdvancedOptions(include_visual_activity=False)
        ensure_worksheet_assets([lesson], options)
        validate_lessons_content([lesson], 1, "A2", 30, options)
        lesson.worksheet_assets = lesson.worksheet_assets[:1]
        with self.assertRaisesRegex(ValueError, "worksheet assets"):
            validate_lessons_content([lesson], 1, "A2", 30, options)

    def test_empty_items_and_duplicate_variant_passages_are_rejected(self):
        lesson = make_lesson(reading_questions=[item["question"] for item in COMPREHENSION["items"]],
                             reading_answers=[ReadingQuestionAnswer(**item) for item in COMPREHENSION["items"]])
        options = LessonAdvancedOptions(include_visual_activity=False, include_worksheet_assets=False)
        with self.assertRaisesRegex(ValueError, "different reading passages"):
            validate_lessons_content([lesson, lesson.model_copy(deep=True)], 2, "A2", 30, options)
        lesson.target_vocabulary[0] = " "
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_lessons_content([lesson], 1, "A2", 30, options)


if __name__ == "__main__":
    unittest.main()
