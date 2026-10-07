import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.errors import LessonQualityError
from app.ollama_client import generate_from_ollama
from app.reading_service import generate_reading_comprehension, validate_reading_comprehension
from app.schemas import ReadingQuestionAnswer
from tests.helpers import COMPREHENSION, READING, make_lesson


class ComprehensionTests(unittest.IsolatedAsyncioTestCase):
    async def test_replaces_stale_questions_using_final_reading_and_schema(self):
        lesson = make_lesson(reading_questions=["Who is Ben?"])
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(COMPREHENSION)) as generate:
            await generate_reading_comprehension(lesson, "A2", 3)
        self.assertEqual(lesson.reading_questions, [item["question"] for item in COMPREHENSION["items"]])
        self.assertEqual(len(lesson.reading_answers), 3)
        self.assertIn(READING, generate.call_args.args[0])
        items_schema = generate.call_args.kwargs["json_schema"]["properties"]["items"]
        self.assertEqual((items_schema["minItems"], items_schema["maxItems"]), (3, 3))

    async def test_invented_evidence_is_repaired_without_changing_reading(self):
        invalid = copy.deepcopy(COMPREHENSION)
        invalid["items"][0]["evidence"] = "Ben goes to a library."
        lesson = make_lesson()
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(invalid), json.dumps(COMPREHENSION)]) as generate:
            await generate_reading_comprehension(lesson, "A2", 3)
        self.assertEqual(lesson.reading_text, READING)
        self.assertEqual(generate.await_count, 2)
        self.assertIn("not quoted from the final reading", generate.call_args.args[0])

    async def test_duplicate_questions_and_invalid_json_exhaust_only_section_attempts(self):
        duplicate = copy.deepcopy(COMPREHENSION)
        duplicate["items"][1] = duplicate["items"][0]
        for response in (json.dumps(duplicate), "not JSON", '{"items": []}'):
            with self.subTest(response=response[:30]):
                lesson = make_lesson()
                original = lesson.model_dump()
                with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                           return_value=response) as generate:
                    with self.assertRaises(LessonQualityError):
                        await generate_reading_comprehension(lesson, "A2", 3)
                self.assertEqual(generate.await_count, 3)
                self.assertEqual(lesson.model_dump(), original)

    def test_answer_key_order_and_evidence_word_boundaries(self):
        lesson = make_lesson(reading_questions=["Who is kind?"], reading_text="The teacher is kind.",
                             reading_answers=[ReadingQuestionAnswer(question="Who is kind?",
                                                                    answer="The teacher.", evidence="tea")])
        with self.assertRaisesRegex(LessonQualityError, "not quoted"):
            validate_reading_comprehension(lesson, 1)
        lesson.reading_answers[0].evidence = "The teacher is kind."
        lesson.reading_answers[0].question = "Different question?"
        with self.assertRaisesRegex(LessonQualityError, "question order"):
            validate_reading_comprehension(lesson, 1)

    async def test_ollama_receives_schema_and_legacy_calls_remain_supported(self):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"response": "result"})

        client_type = httpx.AsyncClient
        for schema in ({"type": "object"}, None):
            client = client_type(transport=httpx.MockTransport(respond))
            with patch("app.ollama_client.httpx.AsyncClient", return_value=client):
                self.assertEqual(await generate_from_ollama("prompt", json_schema=schema), "result")
        self.assertEqual(requests[0]["format"], {"type": "object"})
        self.assertNotIn("format", requests[1])


if __name__ == "__main__":
    unittest.main()
