import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.errors import LessonQualityError
from app.lesson_service import ensure_worksheet_assets, repair_lesson_consistency
from app.main import app
from app.quality_service import review_lesson_quality
from app.schemas import LessonAdvancedOptions, LessonQualityIssue, ReadingQuestionAnswer
from app.worksheet_service import repair_cloze_asset, validate_cloze_asset
from tests.helpers import COMPREHENSION, READING, make_lesson


def answered_lesson(**updates):
    return make_lesson(
        reading_questions=[item["question"] for item in COMPREHENSION["items"]],
        reading_answers=[ReadingQuestionAnswer(**item) for item in COMPREHENSION["items"]],
        **updates,
    )


class QualityReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_answer_with_real_quote_repairs_only_comprehension(self):
        lesson = answered_lesson()
        lesson.reading_answers[1].answer = "Coffee and cake."
        options = LessonAdvancedOptions(include_visual_activity=False)
        ensure_worksheet_assets([lesson], options)
        original_assets = [asset.model_dump() for asset in lesson.worksheet_assets]
        reviews = [json.dumps({"issues": [{"section": "comprehension",
                                            "message": "What does Anna order? Answer must be tea and a sandwich."}]}),
                   '{"issues": []}']
        with patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=reviews) as review_generate, \
             patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(COMPREHENSION)) as question_generate, \
             patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock) as reading_generate:
            await repair_lesson_consistency(lesson, "A2", "cafe", 30, options)
        reading_generate.assert_not_awaited()
        self.assertEqual(question_generate.await_count, 1)
        self.assertEqual(review_generate.await_count, 2)
        self.assertIn("Answer must be tea and a sandwich", question_generate.call_args.args[0])
        self.assertEqual(lesson.reading_answers[1].answer, "Tea and a sandwich.")
        self.assertEqual(lesson.reading_text, READING)
        self.assertEqual([asset.model_dump() for asset in lesson.worksheet_assets], original_assets)

    async def test_reading_content_repair_rebuilds_all_dependent_sections(self):
        lesson = answered_lesson()
        options = LessonAdvancedOptions(include_visual_activity=False)
        ensure_worksheet_assets([lesson], options)
        rewritten = READING.replace("Anna", "Maria")
        updated_answers = json.dumps(COMPREHENSION).replace("Anna", "Maria")
        with patch("app.lesson_service.review_lesson_quality", new_callable=AsyncMock,
                   side_effect=[[LessonQualityIssue(section="reading", message="Repair the passage content.")], []]), \
             patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=rewritten) as reading_generate, \
             patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=updated_answers) as question_generate:
            await repair_lesson_consistency(lesson, "A2", "cafe", 30, options)
        self.assertEqual(reading_generate.await_count, 1)
        self.assertIn("Repair the passage content", reading_generate.call_args.args[0])
        self.assertEqual(question_generate.await_count, 1)
        self.assertEqual(lesson.reading_text, rewritten)
        self.assertNotIn("Anna", " ".join(lesson.reading_questions))
        cloze = next(asset for asset in lesson.worksheet_assets if asset.asset_type == "fill_in_the_blanks")
        validate_cloze_asset(lesson, cloze)
        self.assertNotIn("Anna", " ".join(cloze.lines))

    async def test_ambiguous_cloze_repairs_only_exercise(self):
        lesson = answered_lesson()
        options = LessonAdvancedOptions(include_visual_activity=False)
        ensure_worksheet_assets([lesson], options)
        exercise = {
            "lines": ["After breakfast, she asks for the ____ and pays with her card."],
            "word_bank": ["bill"], "answer_key": ["bill"],
        }
        with patch("app.lesson_service.review_lesson_quality", new_callable=AsyncMock,
                   side_effect=[[LessonQualityIssue(section="cloze", message="Choose unambiguous contexts.")], []]), \
             patch("app.worksheet_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(exercise)) as exercise_generate, \
             patch("app.lesson_service.generate_reading_comprehension", new_callable=AsyncMock) as comprehension, \
             patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock) as reading_generate:
            await repair_lesson_consistency(lesson, "A2", "cafe", 30, options)
        comprehension.assert_not_awaited()
        reading_generate.assert_not_awaited()
        self.assertEqual(exercise_generate.await_count, 1)
        cloze = next(asset for asset in lesson.worksheet_assets if asset.asset_type == "fill_in_the_blanks")
        self.assertEqual(cloze.lines, exercise["lines"])
        self.assertEqual(lesson.reading_text, READING)

    async def test_missing_vocabulary_contexts_trigger_targeted_reading_repair(self):
        lesson = answered_lesson(reading_text="A dog runs outside.")
        options = LessonAdvancedOptions(include_visual_activity=False)
        with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=READING) as reading_generate, \
             patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(COMPREHENSION)), \
             patch("app.lesson_service.review_lesson_quality", new_callable=AsyncMock, return_value=[]) as review:
            await repair_lesson_consistency(lesson, "A2", "cafe", 30, options)
        self.assertEqual(reading_generate.await_count, 1)
        self.assertEqual(review.await_count, 1)
        self.assertIn("distinct complete sentences", reading_generate.call_args.args[0])

    async def test_persistent_content_errors_have_bounded_section_repairs(self):
        lesson = answered_lesson()
        issue = LessonQualityIssue(section="comprehension", message="Question duplicates another detail.")
        options = LessonAdvancedOptions(include_visual_activity=False, include_worksheet_assets=False)
        with patch("app.lesson_service.review_lesson_quality", new_callable=AsyncMock,
                   return_value=[issue]) as review, \
             patch("app.lesson_service.generate_reading_comprehension", new_callable=AsyncMock) as repair:
            with self.assertRaisesRegex(LessonQualityError, "consistency checks failed"):
                await repair_lesson_consistency(lesson, "A2", "cafe", 30, options)
        self.assertEqual(review.await_count, 3)
        self.assertEqual(repair.await_count, 2)

    async def test_missing_review_verdict_never_counts_as_approval(self):
        with patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value="{}") as generate:
            with self.assertRaisesRegex(LessonQualityError, "valid quality review"):
                await review_lesson_quality(answered_lesson(), "A2")
        self.assertEqual(generate.await_count, 3)

    async def test_disabled_worksheets_are_excluded_from_review_and_repair(self):
        lesson = answered_lesson()
        options = LessonAdvancedOptions(include_visual_activity=False, include_worksheet_assets=False)
        responses = [json.dumps({"issues": [{"section": "cloze", "message": "Missing cloze"}]}),
                     '{"issues": []}']
        with patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=responses) as generate, \
             patch("app.lesson_service.repair_cloze_asset", new_callable=AsyncMock) as repair:
            await repair_lesson_consistency(lesson, "A2", "cafe", 30, options)
        self.assertEqual(generate.await_count, 2)
        self.assertIn('"cloze_exercises": []', generate.call_args.args[0])
        repair.assert_not_awaited()
        self.assertEqual(lesson.worksheet_assets, [])

    async def test_invalid_cloze_repair_is_retried_with_validation_feedback(self):
        invalid = {"lines": ["Invented sentence ____."] , "word_bank": ["tea"], "answer_key": ["tea"]}
        valid = {"lines": ["After breakfast, she asks for the ____ and pays with her card."],
                 "word_bank": ["bill"], "answer_key": ["bill"]}
        with patch("app.worksheet_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(invalid), json.dumps(valid)]) as generate:
            result = await repair_cloze_asset(answered_lesson(), "Remove ambiguity.")
        self.assertEqual(generate.await_count, 2)
        self.assertIn("Remove ambiguity", generate.call_args.args[0])
        self.assertIn("distinct sentences from the final reading", generate.call_args.args[0])
        self.assertEqual(result.answer_key, ["bill"])


class GenerationIntegrationTests(unittest.TestCase):
    def test_api_generates_final_reading_then_answers_then_reviewed_worksheets(self):
        draft = make_lesson(reading_text="Old short reading.").model_dump()
        draft.pop("reading_questions")
        draft.pop("reading_answers")
        events = []
        async def content(prompt, **kwargs):
            if "rewrite one reading passage" in prompt:
                events.append("reading_repair")
                return READING
            events.append("draft")
            return json.dumps({"lessons": [draft]})
        async def questions(prompt, **kwargs):
            events.append("questions")
            self.assertIn(READING, prompt)
            return json.dumps(COMPREHENSION)
        async def review(prompt, **kwargs):
            events.append("review")
            self.assertIn('"answer": "Tea and a sandwich."', prompt)
            self.assertIn('"asset_type": "fill_in_the_blanks"', prompt)
            self.assertIn('"theme": "food and drinks"', prompt)
            return '{"issues": []}'
        with patch("app.lesson_service.generate_from_ollama", side_effect=content), \
             patch("app.reading_service.generate_from_ollama", side_effect=questions), \
             patch("app.quality_service.generate_from_ollama", side_effect=review), TestClient(app) as client:
            response = client.post("/lessons/generate", json={
                "level": "A2", "duration": 30, "topic": "food and drinks", "variant_count": 1,
                "advanced_options": {"include_images": False, "include_visual_activity": False},
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(events, ["draft", "reading_repair", "questions", "review"])
        lesson = response.json()["lessons"][0]
        self.assertEqual(lesson["reading_text"], READING)
        self.assertEqual(lesson["reading_questions"], [item["question"] for item in COMPREHENSION["items"]])
        self.assertEqual(lesson["reading_answers"], COMPREHENSION["items"])
        self.assertEqual(len(lesson["worksheet_assets"]), 5)

    def test_quality_failure_does_not_restart_api_draft_generation(self):
        draft = make_lesson().model_dump()
        with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps({"lessons": [draft]})) as generate, \
             patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(COMPREHENSION)), \
             patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value="{}"), TestClient(app) as client:
            response = client.post("/lessons/generate", json={
                "level": "A2", "duration": 30, "topic": "cafe", "variant_count": 1,
                "advanced_options": {"include_images": False},
            })
        self.assertEqual(response.status_code, 500)
        self.assertIn("valid quality review", response.json()["detail"])
        self.assertEqual(generate.await_count, 1)


if __name__ == "__main__":
    unittest.main()
