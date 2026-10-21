import asyncio
import json
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import LessonGenerateRequest, TopicSuggestionsRequest, TopicSuggestionsResponse
from app.topic_service import suggest_topics
from app.lesson_service import try_generate_once
from app.ollama_client import OllamaError
from tests.helpers import make_lesson


def ideas(prefix="School shop"):
    return {"topics": [{"title": f"{prefix} {n}", "description": "Practise polite requests in a pretend shop."} for n in range(6)]}


class TopicSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_topics_use_course_age_and_level_and_do_not_generate_lessons(self):
        request = TopicSuggestionsRequest(english_type="business", age_group="children", level="B1", exclude_topics=["Old topic"])
        with patch("app.topic_service.generate_from_ollama", new_callable=AsyncMock, return_value=json.dumps(ideas())) as model:
            result = await suggest_topics(request)
        self.assertEqual(len(result.topics), 6)
        prompt = model.call_args.args[0]
        for text in ("Business English", "children", "pretend shops", "B1", "Old topic", "Do not generate lessons yet"):
            self.assertIn(text, prompt)
        self.assertEqual(model.call_args.kwargs["task"], "topics")

    async def test_teen_topics_use_age_appropriate_context_for_both_courses(self):
        for course in ("general", "business"):
            with self.subTest(course=course):
                request = TopicSuggestionsRequest(english_type=course, age_group="teens", level="B1")
                with patch("app.topic_service.generate_from_ollama", new_callable=AsyncMock, return_value=json.dumps(ideas())) as model:
                    await suggest_topics(request)
                prompt = model.call_args.args[0]
                self.assertIn("Learners: teenagers", prompt)
                self.assertNotIn("Learners: adults", prompt)
                if course == "business":
                    self.assertIn("internship", prompt)

    async def test_repeated_titles_are_rejected(self):
        request = TopicSuggestionsRequest(level="A2", exclude_topics=["School shop 0"])
        with patch("app.topic_service.generate_from_ollama", new_callable=AsyncMock, return_value=json.dumps(ideas())):
            with self.assertRaises(ValueError):
                await suggest_topics(request)

    async def test_generation_carries_audience_into_repairs_and_response(self):
        lesson = make_lesson()
        request = LessonGenerateRequest(level="A2", duration=45, topic="School shop", variant_count=1,
                                        english_type="business", age_group="teens")
        with ExitStack() as stack:
            model = stack.enter_context(patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock,
                                             return_value=json.dumps({"lessons": [lesson.model_dump()]})))
            stack.enter_context(patch("app.lesson_service.normalize_lessons_content", side_effect=lambda lessons, *_: lessons))
            stack.enter_context(patch("app.lesson_service.repair_lesson_readings", new_callable=AsyncMock,
                                     side_effect=lambda **kwargs: kwargs["lessons"]))
            stack.enter_context(patch("app.lesson_service.generate_reading_comprehension", new_callable=AsyncMock))
            stack.enter_context(patch("app.lesson_service.ensure_visual_activities", side_effect=lambda lessons, *_: lessons))
            repair = stack.enter_context(patch("app.lesson_service.repair_lesson_consistency", new_callable=AsyncMock))
            stack.enter_context(patch("app.lesson_service.validate_lessons_content"))
            response = await try_generate_once(request)
        self.assertIn("Business English", model.call_args.args[0])
        self.assertIn("Learners: teenagers", model.call_args.args[0])
        self.assertEqual(repair.call_args.args[0].age_group, "teens")
        self.assertEqual(response.age_group, "teens")
        self.assertEqual(response.lessons[0].english_type, "business")
        self.assertEqual(response.lessons[0].age_group, "teens")

    def test_validation_and_legacy_defaults(self):
        legacy = LessonGenerateRequest(level="A2", duration=45, topic="Travel")
        self.assertEqual((legacy.english_type, legacy.age_group), ("general", "adults"))
        for overrides in ({"age_group": "unsupported"}, {"english_type": "invalid"}, {"level": "B9"}):
            with self.assertRaises(ValidationError):
                TopicSuggestionsRequest(**{"level": "A2", **overrides})
        duplicate = ideas()
        duplicate["topics"][1] = duplicate["topics"][0]
        with self.assertRaises(ValidationError):
            TopicSuggestionsResponse.model_validate(duplicate)


class TopicApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_invalid_age_is_422_without_calling_model(self):
        with patch("app.api.routes_lessons.suggest_topics", new_callable=AsyncMock) as suggest:
            result = self.client.post("/lessons/topics", json={"level": "A2", "age_group": "unsupported"})
        self.assertEqual(result.status_code, 422)
        suggest.assert_not_called()

    def test_busy_model_is_429(self):
        self.app.state.generation_slots = asyncio.Semaphore(0)
        result = self.client.post("/lessons/topics", json={"level": "A2"})
        self.assertEqual(result.status_code, 429)
        self.assertIn("Retry-After", result.headers)

    def test_success_and_model_failure_release_queue(self):
        with patch("app.api.routes_lessons.suggest_topics", new_callable=AsyncMock,
                   return_value=TopicSuggestionsResponse.model_validate(ideas())):
            result = self.client.post("/lessons/topics", json={"level": "A2"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()["topics"]), 6)
        for error, status in ((OllamaError("Model unavailable"), 503), (ValueError("Invalid JSON"), 502)):
            with patch("app.api.routes_lessons.suggest_topics", new_callable=AsyncMock, side_effect=error):
                result = self.client.post("/lessons/topics", json={"level": "A2"})
            self.assertEqual(result.status_code, status)
            self.assertFalse(self.app.state.generation_slots.locked())
