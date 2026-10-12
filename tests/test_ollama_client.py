import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.config import OllamaSettings
from app.ollama_client import OllamaError, generate_from_ollama


class OllamaSettingsTests(unittest.TestCase):
    def test_defaults_select_gpt_oss_and_bounded_local_context(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = OllamaSettings.from_environment()
        self.assertEqual(settings.model, "gpt-oss:20b")
        self.assertEqual(settings.context_tokens, 12288)
        self.assertEqual(settings.reasoning_effort, "low")
        self.assertEqual(settings.review_reasoning_effort, "medium")

    def test_configuration_rejects_invalid_resources_and_credentials_in_url(self):
        for changes in (
            {"model": ""}, {"context_tokens": 100}, {"timeout_seconds": 0},
            {"reasoning_effort": "false"}, {"base_url": "http://user:password@localhost:11434"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                OllamaSettings(**changes).validate()


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, answer, *, task="lesson", schema=None, model="gpt-oss:20b", calls=None):
        calls = calls if calls is not None else []
        sequence = iter(answer) if isinstance(answer, tuple) else None
        real_client = httpx.AsyncClient

        def handle(request):
            calls.append(json.loads(request.content))
            value = next(sequence) if sequence is not None else answer
            if isinstance(value, Exception):
                raise value
            if isinstance(value, httpx.Response):
                return value
            return httpx.Response(200, json=value)

        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(handle), **kwargs)

        with patch.dict(os.environ, {"OLLAMA_MODEL": model, "OLLAMA_NUM_CTX": "12288",
                                    "OLLAMA_REASONING_EFFORT": "low", "OLLAMA_REVIEW_REASONING_EFFORT": "medium"}), \
             patch("app.ollama_client.httpx.AsyncClient", side_effect=client):
            result = await generate_from_ollama("Prepare the requested materials.", task=task, json_schema=schema)
        return result, calls[0]

    async def test_structured_output_uses_schema_and_returns_only_final_answer(self):
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        result, payload = await self.request({"response": ' {"title":"Cafe"} ',
                                             "thinking": "Private reasoning with unrelated JSON.", "done": True}, schema=schema)
        self.assertEqual(json.loads(result), {"title": "Cafe"})
        self.assertEqual(payload["format"], schema)
        self.assertIn(json.dumps(schema), payload["prompt"])
        self.assertEqual(payload["think"], "low")
        self.assertEqual(payload["options"]["num_predict"], 8192)
        self.assertFalse(payload["stream"])

    async def test_plain_reading_has_no_json_constraint(self):
        result, payload = await self.request({"response": "A complete reading passage."}, task="reading")
        self.assertEqual(result, "A complete reading passage.")
        self.assertNotIn("format", payload)
        self.assertEqual(payload["options"]["num_predict"], 2048)

    async def test_review_and_metadata_have_distinct_reasoning_and_token_budgets(self):
        _, review = await self.request({"response": '{"issues": []}'}, task="review")
        _, metadata = await self.request({"response": '{"image_search_query": "cafe"}'}, task="image_metadata")
        self.assertEqual(review["think"], "medium")
        self.assertEqual(review["options"]["temperature"], 0)
        self.assertEqual(metadata["think"], "low")
        self.assertLess(metadata["options"]["num_predict"], review["options"]["num_predict"])

    async def test_other_models_do_not_receive_gpt_oss_reasoning_controls(self):
        _, payload = await self.request({"response": "Passage."}, model="another-model:latest")
        self.assertNotIn("think", payload)

    async def test_incomplete_empty_and_unexpected_output_is_rejected(self):
        for answer in (
            {"response": "partial", "done_reason": "length"},
            {"response": "partial", "done": False},
            {"response": "", "thinking": "Only reasoning."},
            {"response": None}, [], httpx.Response(200, text="invalid-json"),
        ):
            with self.subTest(answer=answer), self.assertRaises(OllamaError):
                await self.request(answer)

    async def test_transport_failures_are_actionable(self):
        for answer, expected in (
            (httpx.Response(404, json={"error": "model missing"}), "is unavailable"),
            (httpx.Response(400, json={"error": "bad settings"}), "HTTP 400"),
            (httpx.ConnectError("refused"), "Start Ollama"),
            (httpx.ReadTimeout("slow"), "timed out"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(OllamaError, expected):
                await self.request(answer)

    async def test_gpu_initialization_failure_retries_identical_request_once(self):
        failure = httpx.Response(500, json={
            "error": "llama-server process has terminated: CUDA error: shared object initialization failed",
        })
        calls = []
        with patch("app.ollama_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            answer, _ = await self.request((failure, {"response": "Recovered lesson."}), calls=calls)
        self.assertEqual(answer, "Recovered lesson.")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        sleep.assert_awaited_once_with(1)

    async def test_persistent_gpu_failure_stops_after_one_retry_with_safe_message(self):
        failure = httpx.Response(500, json={
            "error": "Private runtime path: CUDA error: shared object initialization failed",
        })
        calls = []
        with patch("app.ollama_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(OllamaError) as raised:
                await self.request((failure, failure), calls=calls)
        self.assertEqual(len(calls), 2)
        self.assertIn("Restart Ollama", str(raised.exception))
        self.assertNotIn("Private runtime path", str(raised.exception))
        sleep.assert_awaited_once_with(1)

    async def test_other_server_errors_are_not_retried(self):
        for failure in (
            httpx.Response(500, json={"error": "out of memory"}),
            httpx.Response(500, text="invalid-json"),
            httpx.Response(400, json={"error": "CUDA error: shared object initialization failed"}),
        ):
            with self.subTest(status=failure.status_code):
                calls = []
                with patch("app.ollama_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
                    with self.assertRaisesRegex(OllamaError, f"HTTP {failure.status_code}"):
                        await self.request(failure, calls=calls)
                self.assertEqual(len(calls), 1)
                sleep.assert_not_awaited()
