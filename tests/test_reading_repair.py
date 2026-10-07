import unittest
from unittest.mock import AsyncMock, patch

from app.errors import LessonQualityError
from app.lesson_service import (
    generate_lesson_variants, get_reading_word_targets, rewrite_reading_text_if_needed,
)
from app.schemas import LessonAdvancedOptions, LessonGenerateRequest, LessonWorksheetAsset
from tests.helpers import READING, make_lesson


class ReadingRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_reading_does_not_call_model(self):
        lesson = make_lesson()
        with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock) as generate:
            result = await rewrite_reading_text_if_needed(lesson, "A2", "cafe", 30)
        self.assertEqual(result.reading_text, READING)
        generate.assert_not_awaited()

    async def test_third_rewrite_is_checked_and_dependent_assets_are_invalidated(self):
        lesson = make_lesson(reading_text="Anna drinks tea.", worksheet_assets=[
            LessonWorksheetAsset(asset_type="fill_in_the_blanks", title="Old", instruction="Old")
        ])
        with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=["Too short.", "Still short.", READING]) as generate:
            result = await rewrite_reading_text_if_needed(lesson, "A2", "cafe", 30)
        self.assertEqual(result.reading_text, READING)
        self.assertEqual(result.worksheet_assets, [])
        self.assertEqual(generate.await_count, 3)

    async def test_failed_rewrites_do_not_pad_or_truncate_reading(self):
        for rewritten in ("Anna drinks tea.", READING * 4):
            with self.subTest(rewritten=rewritten[:20]):
                lesson = make_lesson(reading_text="Original short text.")
                with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                           return_value=rewritten) as generate:
                    with self.assertRaisesRegex(LessonQualityError, "after 3 attempts"):
                        await rewrite_reading_text_if_needed(lesson, "A2", "cafe", 30)
                self.assertEqual(lesson.reading_text, "Original short text.")
                self.assertEqual(generate.await_count, 3)

    async def test_exhausted_section_repair_does_not_restart_whole_generation(self):
        request = LessonGenerateRequest(level="A2", duration=30, topic="cafe")
        with patch("app.lesson_service.try_generate_once", new_callable=AsyncMock,
                   side_effect=LessonQualityError("Reading repair exhausted")) as generate:
            with self.assertRaises(LessonQualityError):
                await generate_lesson_variants(request)
        self.assertEqual(generate.await_count, 1)

    async def test_empty_responses_cannot_approve_a_forced_content_repair(self):
        lesson = make_lesson()
        with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=" ") as generate:
            with self.assertRaises(LessonQualityError):
                await rewrite_reading_text_if_needed(
                    lesson, "A2", "cafe", 30, force=True, quality_feedback="Fix the passage content.",
                )
        self.assertEqual(generate.await_count, 3)
        self.assertEqual(lesson.reading_text, READING)

    def test_one_sided_overrides_cannot_create_an_impossible_range(self):
        for options in (LessonAdvancedOptions(reading_min_words=200),
                        LessonAdvancedOptions(reading_max_words=50)):
            with self.assertRaisesRegex(LessonQualityError, "Invalid effective reading range"):
                get_reading_word_targets("A2", 30, options)


if __name__ == "__main__":
    unittest.main()
