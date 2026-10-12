import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.errors import LessonQualityError
from app.ollama_client import generate_from_ollama
from app.reading_service import generate_reading_comprehension, validate_reading_comprehension
from app.schemas import ReadingQuestionAnswer
from tests.helpers import COMPREHENSION, COMPREHENSION_SELECTION, READING, make_lesson


class ComprehensionTests(unittest.IsolatedAsyncioTestCase):
    async def test_replaces_stale_questions_using_final_reading_and_schema(self):
        lesson = make_lesson(reading_questions=["Who is Ben?"])
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(COMPREHENSION_SELECTION)) as generate:
            await generate_reading_comprehension(lesson, "A2", 3)
        self.assertEqual(lesson.reading_questions, [item["question"] for item in COMPREHENSION["items"]])
        self.assertEqual(len(lesson.reading_answers), 3)
        self.assertIn(READING, generate.call_args.args[0])
        items_schema = generate.call_args.kwargs["json_schema"]["properties"]["items"]
        self.assertEqual((items_schema["minItems"], items_schema["maxItems"]), (3, 3))
        fields = generate.call_args.kwargs["json_schema"]["$defs"]["ReadingEvidenceSelection"]["properties"]
        self.assertEqual(fields["evidence_start"]["maximum"], 7)
        self.assertEqual(fields["evidence_end"]["maximum"], 7)
        self.assertNotIn("evidence", fields)
        self.assertEqual([item.model_dump() for item in lesson.reading_answers], COMPREHENSION["items"])

    async def test_invalid_sentence_range_is_repaired_without_changing_reading(self):
        invalid = copy.deepcopy(COMPREHENSION_SELECTION)
        invalid["items"][0]["evidence_end"] = 99
        lesson = make_lesson()
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(invalid), json.dumps(COMPREHENSION_SELECTION)]) as generate:
            await generate_reading_comprehension(lesson, "A2", 3)
        self.assertEqual(lesson.reading_text, READING)
        self.assertEqual(generate.await_count, 2)
        self.assertIn("Evidence sentence range", generate.call_args.args[0])

    async def test_fifth_comprehension_attempt_uses_validation_feedback(self):
        invalid = copy.deepcopy(COMPREHENSION_SELECTION)
        invalid["items"][0]["evidence_end"] = 99
        lesson = make_lesson()
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(invalid)] * 4 + [json.dumps(COMPREHENSION_SELECTION)]) as generate:
            await generate_reading_comprehension(lesson, "A2", 3)
        self.assertEqual(generate.await_count, 5)
        self.assertEqual(lesson.reading_text, READING)
        self.assertIn("Evidence sentence range", generate.await_args.args[0])
        validate_reading_comprehension(lesson, 3)

    async def test_all_saints_activities_copy_multiple_sentences_exactly(self):
        reading = (
            "All Saints’ Day is on November 1.\n\n"
            "Families visit their relatives’ graves.  They light candles and leave flowers.\n"
            "It is a quiet time to remember loved ones."
        )
        question = "What are some common activities people do on All Saints’ Day according to the passage?"
        selection = {"items": [{
            "question": question, "answer": "They visit graves, light candles and leave flowers.",
            "evidence_start": 2, "evidence_end": 3,
        }]}
        lesson = make_lesson(title="Celebrating All Saints’ Day", reading_text=reading)
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(selection)) as generate:
            await generate_reading_comprehension(lesson, "A2", 1)
        self.assertEqual(generate.await_count, 1)
        self.assertEqual(lesson.reading_text, reading)
        self.assertEqual(lesson.reading_answers[0].evidence,
                         "Families visit their relatives’ graves.  They light candles and leave flowers.")
        self.assertIn(lesson.reading_answers[0].evidence, reading)
        validate_reading_comprehension(lesson, 1)

    async def test_invalid_ranges_and_free_form_evidence_do_not_publish_answers(self):
        for start, end in ((0, 1), (2, 1), (1, 8), (True, 1), ("1", 1)):
            with self.subTest(start=start, end=end):
                invalid = copy.deepcopy(COMPREHENSION_SELECTION)
                invalid["items"][0].update(evidence_start=start, evidence_end=end)
                lesson = make_lesson()
                original = lesson.model_dump()
                with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                           return_value=json.dumps(invalid)) as generate:
                    with self.assertRaises(LessonQualityError):
                        await generate_reading_comprehension(lesson, "A2", 3)
                self.assertEqual(generate.await_count, 5)
                self.assertEqual(lesson.model_dump(), original)
        lesson = make_lesson()
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                   return_value=json.dumps(COMPREHENSION)):
            with self.assertRaises(LessonQualityError):
                await generate_reading_comprehension(lesson, "A2", 3)

    async def test_empty_passage_fails_before_model_call(self):
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock) as generate:
            with self.assertRaisesRegex(LessonQualityError, "empty passage"):
                await generate_reading_comprehension(make_lesson(reading_text=" \n "), "A2", 3)
        generate.assert_not_awaited()

    async def test_duplicate_questions_and_invalid_json_exhaust_only_section_attempts(self):
        duplicate = copy.deepcopy(COMPREHENSION_SELECTION)
        duplicate["items"][1] = duplicate["items"][0]
        for response in (json.dumps(duplicate), "not JSON", '{"items": []}'):
            with self.subTest(response=response[:30]):
                lesson = make_lesson()
                original = lesson.model_dump()
                with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock,
                           return_value=response) as generate:
                    with self.assertRaises(LessonQualityError):
                        await generate_reading_comprehension(lesson, "A2", 3)
                self.assertEqual(generate.await_count, 5)
                self.assertEqual(lesson.model_dump(), original)

    def test_answer_key_order_and_evidence_word_boundaries(self):
        lesson = make_lesson(reading_questions=["Who is kind?"], reading_text="The teacher is kind.",
                             reading_answers=[ReadingQuestionAnswer(question="Who is kind?",
                                                                    answer="The teacher.", evidence="tea")])
        with self.assertRaisesRegex(LessonQualityError, "not quoted"):
            validate_reading_comprehension(lesson, 1)
        lesson.reading_answers[0].evidence = "The teacher is friendly."
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
