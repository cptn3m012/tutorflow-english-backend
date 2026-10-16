import json
import unittest
from unittest.mock import AsyncMock, patch

from app.cefr_service import choose_best_grammar_target, get_cefr_profile, normalize_cefr_focus_item
from app.errors import LessonQualityError
from app.lesson_editing import regenerate_content
from app.lesson_service import repair_lesson_consistency
from app.prompts import build_reading_comprehension_prompt, build_reading_rewrite_prompt
from app.quality_service import review_lesson_quality
from app.schemas import LessonAdvancedOptions, LessonQualityIssue
from app.worksheet_service import validate_cloze_asset
from tests.helpers import READING
from tests.test_lesson_quality import answered_lesson


class GrammarCueTests(unittest.TestCase):
    def test_copular_be_is_not_progressive_and_no_match_is_not_invented(self):
        targets = ["Present continuous", "Past continuous", "Modals: can, can't"]
        for text in ("She is a teacher.", "They were friends.", "The candy is on a shelf.", "Do you work?"):
            self.assertIsNone(choose_best_grammar_target(targets, text), text)
        self.assertEqual(choose_best_grammar_target(targets, "She is reading a book."), "Present continuous")
        self.assertEqual(choose_best_grammar_target(targets, "They were quietly waiting."), "Past continuous")
        self.assertEqual(choose_best_grammar_target(targets, "I can't swim."), "Modals: can, can't")
        self.assertEqual(choose_best_grammar_target(["Present continuous (now or in progress)"],
                                                   "She is not reading."), "Present continuous (now or in progress)")
        self.assertIsNone(choose_best_grammar_target(["Question words"], "Do you work?"))
        self.assertEqual(choose_best_grammar_target(["Question words"], "Where do you work?"), "Question words")

    def test_legacy_modal_label_still_normalizes_without_calling_do_a_modal(self):
        profile = get_cefr_profile("A1")
        self.assertIn("Modals: can, can't", profile["grammar_targets"])
        self.assertNotIn("Modals: can, can't, do, doesn't", profile["grammar_targets"])
        self.assertEqual(normalize_cefr_focus_item("Modals: can, can't, do, doesn't", "A1"), "Modals: can, can't")

    def test_reading_repair_and_comprehension_receive_full_profile_at_advanced_levels(self):
        prompts = [build_reading_rewrite_prompt("C1", "work", "Meetings", "Negotiate a solution", "Business",
                        "Mixed conditionals", ["meeting"], "A meeting.", 150, 200),
                   build_reading_comprehension_prompt("C1", "Meetings", "Negotiate a solution", "A meeting.",
                        1, sentences=["A meeting."])]
        for prompt in prompts:
            self.assertIn("Negotiating and conceding", prompt)
            self.assertIn("recognise implicit meaning", prompt)


class CefrReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_sees_full_profile_and_all_learner_tasks(self):
        lesson = answered_lesson(communication_focus="Exchanging greetings", grammar_focus="Present simple")
        lesson.speaking_questions = ["Critically evaluate the epistemological implications of surveillance."]
        lesson.sentence_frames = ["Had the safeguards been enacted earlier, __."]
        issues = [{"section": section, "message": "Quoted task exceeds familiar everyday A1 communication; simplify it."}
                  for section in ("speaking", "activities", "sentence_frames")]
        with patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps({"issues": issues})) as model:
            result = await review_lesson_quality(lesson, "A1")
        prompt = model.call_args.args[0]
        self.assertIn("introduce yourself and others", prompt)
        for value in (lesson.speaking_questions[0], lesson.pair_work_task, lesson.role_play_scenario,
                      lesson.sentence_frames[0], "Exchanging greetings", "Present simple"):
            self.assertIn(value, prompt)
        self.assertEqual([issue.section for issue in result], [item["section"] for item in issues])

    async def test_section_review_rejects_verdicts_outside_requested_scope(self):
        with patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock, side_effect=[
            '{"issues": [{"section": "reading", "message": "Rewrite it"}]}', '{"issues": []}',
        ]) as model:
            self.assertEqual(await review_lesson_quality(answered_lesson(), "A2", sections={"speaking"}), [])
        self.assertEqual(model.await_count, 2)
        self.assertIn('"review_sections": ["speaking"]', model.call_args.args[0])

    async def test_generation_repairs_each_new_section_without_changing_reading(self):
        cases = {
            "speaking": {"speaking_questions": [f"What do you order on day {i}?" for i in range(5)]},
            "activities": {"pair_work_task": "Ask your partner for a drink.", "role_play_scenario": "Order tea politely."},
            "sentence_frames": {"sentence_frames": ["Can I have __?", "I like __.", "Where is __?"]},
        }
        for section, replacement in cases.items():
            with self.subTest(section=section):
                lesson = answered_lesson()
                before = lesson.model_dump()
                issue = LessonQualityIssue(section=section, message="Practise the selected target using simpler learner language.")
                with patch("app.lesson_service.review_lesson_quality", new_callable=AsyncMock, side_effect=[[issue], []]), \
                     patch("app.lesson_sections.generate_from_ollama", new_callable=AsyncMock,
                           return_value=json.dumps(replacement)) as model:
                    await repair_lesson_consistency(lesson, "A2", "Cafe", 30,
                        LessonAdvancedOptions(include_worksheet_assets=False, include_visual_activity=False))
                self.assertEqual(lesson.model_dump(), {**before, **replacement})
                self.assertEqual(lesson.reading_text, READING)
                self.assertIn(issue.message, model.call_args.args[0])
                self.assertIn("Making requests (e.g. at a restaurant)", model.call_args.args[0])

    async def test_regeneration_retries_rejected_speaking_and_preserves_unrelated_content(self):
        original = answered_lesson().model_dump()
        hard = [f"Critically evaluate philosophical position {i}." for i in range(5)]
        simple = [f"What do you order on day {i}?" for i in range(5)]
        issue = LessonQualityIssue(section="speaking", message="Questions demand abstract argument; use familiar routine requests.")
        with patch("app.lesson_sections.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps({"speaking_questions": hard}), json.dumps({"speaking_questions": simple})]) as model, \
             patch("app.lesson_editing.review_lesson_quality", new_callable=AsyncMock, side_effect=[[issue], []]) as review:
            result = await regenerate_content(original, "A2", "Cafe", 30, "speaking", "Use a cafe.")
        self.assertEqual(result, {**original, "speaking_questions": simple})
        self.assertEqual(review.call_args.kwargs["sections"], {"speaking"})
        self.assertIn(issue.message, model.call_args.args[0])
        self.assertIn("Use a cafe.", model.call_args.args[0])

    async def test_vocabulary_repair_rebuilds_cloze_and_keeps_other_tasks(self):
        lesson = answered_lesson()
        original_questions = lesson.speaking_questions.copy()
        words = ["office", "menu", "water", "book", "bill", "card", "friend", "sandwich"]
        issue = LessonQualityIssue(section="vocabulary", message="Choose familiar words from the passage.")
        with patch("app.lesson_service.review_lesson_quality", new_callable=AsyncMock, side_effect=[[issue], []]), \
             patch("app.lesson_sections.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps({"target_vocabulary": words})):
            await repair_lesson_consistency(lesson, "A2", "Cafe", 30,
                LessonAdvancedOptions(include_visual_activity=False))
        self.assertEqual(lesson.target_vocabulary, words)
        self.assertEqual(lesson.reading_text, READING)
        self.assertEqual(lesson.speaking_questions, original_questions)
        cloze = next(asset for asset in lesson.worksheet_assets if asset.asset_type == "fill_in_the_blanks")
        validate_cloze_asset(lesson, cloze)
        self.assertTrue(set(cloze.answer_key) <= set(words))

    async def test_repeated_cefr_rejections_are_bounded_and_leave_original_intact(self):
        original = answered_lesson().model_dump()
        snapshot = json.dumps(original, sort_keys=True)
        issue = LessonQualityIssue(section="activities", message="Activity requires advanced abstract debate.")
        replacement = {"pair_work_task": "Debate epistemology.", "role_play_scenario": "Negotiate a treaty."}
        with patch("app.lesson_sections.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(replacement)) as model, \
             patch("app.lesson_editing.review_lesson_quality", new_callable=AsyncMock, return_value=[issue]) as review:
            with self.assertRaisesRegex(LessonQualityError, "advanced abstract debate"):
                await regenerate_content(original, "A1", "Cafe", 30, "activities", "")
        self.assertEqual(review.await_count, 5)
        self.assertEqual(model.await_count, 5)
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot)
