import copy
import unittest
from unittest.mock import AsyncMock, patch

from app.cefr_service import get_cefr_profile
from app.generation_schemas import IMAGE_METADATA_SCHEMA, build_lesson_draft_schema
from app.lesson_service import generate_lesson_variants, normalize_lessons_content
from app.ollama_client import OllamaError
from app.prompts import build_lesson_prompt, build_reading_rewrite_prompt
from app.schemas import LessonAdvancedOptions, LessonGenerateRequest
from tests.helpers import make_lesson


class GenerationSchemaTests(unittest.TestCase):
    def test_valid_declared_cefr_categories_take_precedence_over_malformed_legacy_array(self):
        profile = get_cefr_profile("A2")
        expected = [profile[f"{category}_targets"][-1] for category in ("communication", "vocabulary", "grammar")]
        lesson = make_lesson()
        lesson.communication_focus, lesson.vocabulary_focus, lesson.grammar_focus = expected
        lesson.cefr_focus = [profile["communication_targets"][0]] * 3
        normalize_lessons_content([lesson], "A2")
        self.assertEqual(lesson.cefr_focus, expected)
        self.assertEqual([lesson.communication_focus, lesson.vocabulary_focus, lesson.grammar_focus], expected)

    def test_legacy_category_array_remains_a_fallback_for_missing_declared_targets(self):
        profile = get_cefr_profile("A2")
        expected = [profile[f"{category}_targets"][-1] for category in ("communication", "vocabulary", "grammar")]
        lesson = make_lesson()
        lesson.communication_focus = None
        lesson.vocabulary_focus = "Not a syllabus target"
        lesson.grammar_focus = None
        lesson.cefr_focus = expected[::-1]
        normalize_lessons_content([lesson], "A2")
        self.assertEqual(lesson.cefr_focus, expected)

    def test_draft_requires_authored_fields_and_excludes_later_stage_fields(self):
        schema = build_lesson_draft_schema(
            1, LessonAdvancedOptions(include_visual_activity=False), get_cefr_profile("A2"),
        )
        self.assertEqual(schema["required"], ["lessons"])
        self.assertFalse(schema["additionalProperties"])
        lesson = schema["properties"]["lessons"]["items"]
        expected = {
            "title", "lesson_goal", "target_vocabulary", "reading_text", "speaking_questions",
            "pair_work_task", "role_play_scenario", "sentence_frames", "communication_focus",
            "vocabulary_focus", "grammar_focus", "cefr_focus",
        }
        self.assertEqual(set(lesson["properties"]), expected)
        self.assertEqual(set(lesson["required"]), expected)
        self.assertFalse(lesson["additionalProperties"])
        for field in ("title", "lesson_goal", "reading_text", "pair_work_task", "role_play_scenario"):
            self.assertEqual(lesson["properties"][field], {"type": "string", "minLength": 1})

    def test_variant_and_activity_counts_follow_request_boundaries(self):
        for variant_count in (1, 3):
            for options in (
                LessonAdvancedOptions(),
                LessonAdvancedOptions(
                    target_vocabulary_count=16, speaking_question_count=12,
                    sentence_frame_count=8, visual_word_count=10,
                ),
            ):
                with self.subTest(variants=variant_count, options=options.model_dump()):
                    lessons = build_lesson_draft_schema(
                        variant_count, options, get_cefr_profile("C2"),
                    )["properties"]["lessons"]
                    self.assertEqual((lessons["minItems"], lessons["maxItems"]),
                                     (variant_count, variant_count))
                    properties = lessons["items"]["properties"]
                    counts = {
                        "target_vocabulary": options.target_vocabulary_count,
                        "speaking_questions": options.speaking_question_count,
                        "sentence_frames": options.sentence_frame_count,
                    }
                    for field, count in counts.items():
                        self.assertEqual((properties[field]["minItems"], properties[field]["maxItems"]),
                                         (count, count))
                        self.assertEqual(properties[field]["items"]["minLength"], 1)
                    visual = properties["visual_activity"]
                    self.assertFalse(visual["additionalProperties"])
                    self.assertEqual(set(visual["required"]), set(visual["properties"]))
                    self.assertEqual(visual["properties"]["activity_type"]["enum"], ["find_in_picture"])
                    for field in ("target_words", "answer_key"):
                        self.assertEqual(
                            (visual["properties"][field]["minItems"], visual["properties"][field]["maxItems"]),
                            (options.visual_word_count, options.visual_word_count),
                        )

    def test_disabled_visual_activity_cannot_be_authored(self):
        options = LessonAdvancedOptions(include_visual_activity=False)
        lesson = build_lesson_draft_schema(1, options, get_cefr_profile("A1"))["properties"]["lessons"]["items"]
        self.assertNotIn("visual_activity", lesson["properties"])
        self.assertNotIn("visual_activity", lesson["required"])
        self.assertFalse(lesson["additionalProperties"])

    def test_cefr_choices_stay_in_their_category_with_ollama_compatible_array_items(self):
        for level in ("A1", "A2", "B1", "B2", "C1", "C2"):
            with self.subTest(level=level):
                profile = get_cefr_profile(level)
                original = copy.deepcopy(profile)
                properties = build_lesson_draft_schema(
                    1, LessonAdvancedOptions(), profile,
                )["properties"]["lessons"]["items"]["properties"]
                focus = properties["cefr_focus"]
                self.assertEqual((focus["minItems"], focus["maxItems"]), (3, 3))
                self.assertIsInstance(focus["items"], dict)
                self.assertEqual(focus["items"]["enum"], [
                    target for category in ("communication", "vocabulary", "grammar")
                    for target in profile[f"{category}_targets"]
                ])
                for category in ("communication", "vocabulary", "grammar"):
                    allowed = profile[f"{category}_targets"]
                    self.assertEqual(properties[f"{category}_focus"]["enum"], allowed)
                    properties[f"{category}_focus"]["enum"].append("Test-only mutation")
                self.assertEqual(profile, original)

    def test_image_metadata_requires_one_query_and_four_nonempty_keywords(self):
        self.assertEqual(set(IMAGE_METADATA_SCHEMA["required"]), {"image_search_query", "image_keywords"})
        self.assertFalse(IMAGE_METADATA_SCHEMA["additionalProperties"])
        self.assertEqual(IMAGE_METADATA_SCHEMA["properties"]["image_search_query"]["minLength"], 1)
        keywords = IMAGE_METADATA_SCHEMA["properties"]["image_keywords"]
        self.assertEqual((keywords["minItems"], keywords["maxItems"]), (4, 4))
        self.assertEqual(keywords["items"], {"type": "string", "minLength": 1})


class LessonPromptTests(unittest.TestCase):
    def test_long_reading_guidance_targets_the_middle_of_the_range_without_short_paragraph_limit(self):
        prompt = build_reading_rewrite_prompt(
            "B2", "travel", "City trip", "Discuss travel", "Travel", "Conditionals",
            ["train", "hotel", "station", "ticket"], "A short passage.", 400, 500,
        )
        self.assertIn("approximately 450 words", prompt)
        self.assertIn("about 5 connected paragraphs", prompt)
        self.assertIn("do not add repetitive filler", prompt)
        self.assertNotIn("two short paragraphs", prompt)

    def build_prompt(self, **options):
        return build_lesson_prompt(
            level="A2", topic="cafe", duration=30, variant_count=1,
            cefr_profile_prompt="Required CEFR targets",
            reading_wording_guidance="reading_text must be between 75 and 115 words",
            **options,
        )

    def test_primary_skills_are_not_overridden_by_speaking_priority(self):
        for skill in ("speaking", "reading", "vocabulary", "grammar", "balanced"):
            with self.subTest(skill=skill):
                prompt = self.build_prompt(skill_focus=skill)
                self.assertIn(f"Primary skill focus: {skill}", prompt)
                self.assertIn("prioritize the requested primary skill focus", prompt)
                self.assertNotIn("focus mainly on speaking", prompt)

    def test_initial_reading_uses_vocabulary_forms_in_distinct_complete_sentences(self):
        for vocabulary_count, minimum in ((2, 2), (4, 4), (16, 4)):
            with self.subTest(vocabulary_count=vocabulary_count):
                prompt = self.build_prompt(target_vocabulary_count=vocabulary_count)
                self.assertIn(
                    f"use at least {minimum} different target vocabulary items in their exact supplied forms "
                    "across distinct complete sentences in reading_text", prompt,
                )


class GenerationFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_infrastructure_failure_does_not_repeat_the_whole_lesson(self):
        request = LessonGenerateRequest(level="A2", duration=30, topic="Cafe", variant_count=1)
        with patch("app.lesson_service.try_generate_once", new_callable=AsyncMock,
                   side_effect=OllamaError("Start Ollama")) as generate:
            with self.assertRaisesRegex(OllamaError, "Start Ollama"):
                await generate_lesson_variants(request)
        self.assertEqual(generate.await_count, 1)


if __name__ == "__main__":
    unittest.main()
