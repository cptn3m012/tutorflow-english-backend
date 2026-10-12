import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.comfy_client import PictureError
from app.config import ImageSettings
from app.picture_generation import (
    ObservedScene, prepare_picture_questions, plan_scene, review_scene_topic, scene_prompt, ScenePlan,
)
from app.picture_learning import PictureLearningActivity, validate_learning_activity, uses_vocabulary
from tests.test_picture_generation import topic_approval, picture_bytes

VOCABULARY = ["surveillance", "privacy", "data collection", "algorithm", "monitoring", "security", "invasion", "evidence"]
CONTEXT = {"title": "Surveillance and privacy", "theme": "Surveillance and privacy", "level": "B2",
           "vocabulary": VOCABULARY, "reading": "Surveillance can improve security but raise concerns about privacy."}
OBSERVED = ObservedScene(caption="A camera points at people beside a desk with a monitor.", objects=[
    {"name": "camera", "evidence": "A camera above the doorway points at the people."},
    {"name": "monitor", "evidence": "A monitor on the desk displays simple graphical shapes."},
    {"name": "person", "evidence": "A person stands beside the desk."},
])
ACTIVITY = {"questions": [
    {"kind": "observation", "question": "Which visible objects could be used for surveillance and monitoring?",
     "answer": "The camera and monitor could support surveillance and monitoring.",
     "evidence": "The camera above the doorway and monitor on the desk are visible.",
     "object_names": ["camera", "monitor"], "target_words": ["surveillance", "monitoring"]},
    {"kind": "application", "question": "How might an algorithm use data collection from the visible camera?",
     "answer": "An algorithm might analyse images obtained through data collection.",
     "evidence": "The visible camera provides a starting point for applying the lesson's concepts; its actual processing is unknown.",
     "object_names": ["camera"], "target_words": ["algorithm", "data collection"]},
    {"kind": "discussion", "question": "Would this surveillance support security or feel like an invasion of privacy? Explain.",
     "answer": "It could support security, but excessive surveillance might feel like an invasion of privacy.",
     "evidence": "The camera is aimed towards people; the judgement depends on the student's view.",
     "object_names": ["camera", "person"], "target_words": ["security", "invasion", "privacy"]},
], "sentence_frames": ["Surveillance could help ____.", "This may affect privacy because ____."]}


class PictureLearningTests(unittest.IsolatedAsyncioTestCase):
    def test_lettering_guidance_does_not_add_screens_or_signs_to_a_street_scene(self):
        plan = ScenePlan(scene="A camera is monitoring people walking beside a plain building.",
                         required_objects=["camera", "person"], excluded_objects=[], source_evidence=CONTEXT["reading"])
        prompt = scene_prompt(plan, "photo").lower()
        for unrelated in ("screen", "monitor", "sign", "marker", "board"):
            self.assertNotRegex(prompt, r"\b" + unrelated + r"s?\b")
        screen_plan = plan.model_copy(update={"scene": "A person works at a computer monitor on a desk."})
        self.assertIn("showing backs or side edges", scene_prompt(screen_plan, "photo"))

    async def test_abstract_words_guide_scene_without_becoming_fake_physical_objects(self):
        planned = {"scene": "A security desk with a camera aimed at people and an unlabelled graphical monitor.",
                   "required_objects": ["camera", "monitor", "person"], "excluded_objects": ["sofa"],
                   "source_evidence": CONTEXT["reading"], "illustrative": True,
                   "vocabulary_links": [
                       {"word": "surveillance", "visual_cue": "The camera points at people."},
                       {"word": "privacy", "visual_cue": "Discuss the people being watched."},
                   ]}
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock, return_value=json.dumps(planned)) as model, \
             patch("app.picture_generation.review_scene_topic", new_callable=AsyncMock, return_value=topic_approval()):
            plan = await plan_scene({"title": CONTEXT["title"], "reading_text": CONTEXT["reading"], "target_vocabulary": VOCABULARY},
                                    "B2", CONTEXT["theme"], ImageSettings())
        self.assertTrue(plan.illustrative)
        self.assertEqual(plan.required_objects, ["camera", "monitor", "person"])
        schema = model.await_args.args[1]
        self.assertEqual(schema["$defs"]["VocabularyVisualLink"]["properties"]["word"]["enum"], VOCABULARY)

    async def test_three_lesson_linked_question_kinds_use_original_words_and_sentence_starters(self):
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   side_effect=[json.dumps(ACTIVITY), '{"approved": true, "issues": []}']) as model:
            activity = await prepare_picture_questions(picture_bytes(), "B2", ImageSettings(), OBSERVED, CONTEXT)
        self.assertEqual([item.kind for item in activity.questions], ["observation", "application", "discussion"])
        self.assertEqual(activity.sentence_frames, ACTIVITY["sentence_frames"])
        fields = model.await_args_list[0].args[1]["$defs"]["PictureQuestion"]["properties"]
        self.assertEqual(fields["target_words"]["items"]["enum"], VOCABULARY)
        self.assertEqual(fields["object_names"]["items"]["enum"], ["camera", "monitor", "person"])

    async def test_bad_questions_repair_only_activity_and_use_review_feedback(self):
        rejected = '{"approved": false, "issues": ["Do not claim the camera actually runs a specific algorithm."]}'
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   side_effect=[json.dumps(ACTIVITY), rejected, json.dumps(ACTIVITY), '{"approved": true, "issues": []}']) as model:
            await prepare_picture_questions(picture_bytes(), "B2", ImageSettings(), OBSERVED, CONTEXT)
        self.assertEqual(model.await_count, 4)
        self.assertIn("specific algorithm", model.await_args_list[2].args[0])

    async def test_surplus_vocabulary_tag_does_not_discard_supported_questions(self):
        data = copy.deepcopy(ACTIVITY)
        data["questions"][0]["target_words"].append("evidence")
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   side_effect=[json.dumps(data), '{"approved": true, "issues": []}']) as model:
            activity = await prepare_picture_questions(picture_bytes(), "B2", ImageSettings(), OBSERVED, CONTEXT)
        self.assertEqual(activity.questions[0].target_words, ["surveillance", "monitoring"])
        self.assertEqual(model.await_count, 2)

    async def test_words_are_identified_from_text_when_model_omits_tags(self):
        data = copy.deepcopy(ACTIVITY)
        for item in data["questions"]:
            item.pop("target_words")
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock,
                   side_effect=[json.dumps(data), '{"approved": true, "issues": []}']):
            activity = await prepare_picture_questions(picture_bytes(), "B2", ImageSettings(), OBSERVED, CONTEXT)
        self.assertEqual(activity.questions[0].target_words, ["surveillance", "monitoring"])
        self.assertEqual(activity.questions[1].target_words, ["data collection", "algorithm"])

    def test_abstract_vocabulary_is_not_forced_into_factual_observations(self):
        data = copy.deepcopy(ACTIVITY)
        data["questions"][0].update(question="What can you see by the doorway?",
                                    answer="A camera points towards a person.", target_words=[])
        activity = PictureLearningActivity.model_validate(data)
        validate_learning_activity(activity, VOCABULARY, ["camera", "monitor", "person"])
        activity.questions[1].target_words = []
        with self.assertRaisesRegex(ValueError, "Application and discussion"):
            validate_learning_activity(activity, VOCABULARY, ["camera", "monitor", "person"])

    def test_invalid_kinds_unseen_objects_unused_or_unknown_words_are_rejected(self):
        for mutation in (
            lambda items: items[1].update(kind="observation"),
            lambda items: items[0].update(object_names=["sofa"]),
            lambda items: items[0].update(target_words=["banana"]),
            lambda items: items[0].update(target_words=["evidence"]),
        ):
            data = copy.deepcopy(ACTIVITY)
            mutation(data["questions"])
            with self.assertRaises(ValueError):
                validate_learning_activity(PictureLearningActivity.model_validate(data), VOCABULARY, ["camera", "monitor", "person"])

    def test_natural_vocabulary_forms_and_hyphenated_phrases_are_allowed(self):
        for word, phrase in (("monitoring", "Cameras monitor the entrance."),
                             ("data collection", "Data is collected here."),
                             ("data collection", "A data-collection system."),
                             ("sandwich", "Would you order sandwiches?"),
                             ("invasion", "Some people may feel their privacy is invaded.")):
            with self.subTest(word=word, phrase=phrase):
                self.assertTrue(uses_vocabulary(word, phrase))
        self.assertFalse(uses_vocabulary("evidence", "The camera monitors the entrance."))

    async def test_structural_failure_is_bounded_and_never_counts_as_approval(self):
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock, return_value='{}') as model:
            with self.assertRaisesRegex(PictureError, "after two attempts"):
                await prepare_picture_questions(picture_bytes(), "B2", ImageSettings(), OBSERVED, CONTEXT)
        self.assertEqual(model.await_count, 2)

    async def test_garbled_prominent_lettering_requires_new_render(self):
        plan = ScenePlan(scene="A camera points at people beside an unlabelled monitoring desk.", required_objects=["camera", "monitor"],
                         excluded_objects=[], source_evidence=CONTEXT["reading"])
        with patch("app.picture_generation.planner_json", new_callable=AsyncMock) as model:
            review = await review_scene_topic(picture_bytes(), CONTEXT, plan, ImageSettings(), OBSERVED.model_copy(update={"unreadable_text": True}))
        self.assertFalse(review.faithful)
        self.assertIn("blank", review.repair_instructions)
        model.assert_not_awaited()
