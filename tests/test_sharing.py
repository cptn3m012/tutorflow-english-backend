import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.config import AppSettings
from app.main import create_app

PASSWORD = "test-password-for-sharing-only"


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.application = create_app(AppSettings(
            public_share=True, share_username="tutorflow", share_password=PASSWORD,
        ))
        self.client = TestClient(self.application)

    def tearDown(self):
        self.client.close()

    def test_sharing_refuses_missing_or_short_credentials(self):
        for username, password in (("", PASSWORD), ("tutorflow", "short"), ("invalid:user", PASSWORD)):
            with self.subTest(username=username):
                with self.assertRaisesRegex(ValueError, "Public sharing requires"):
                    create_app(AppSettings(public_share=True, share_username=username, share_password=password))

    def test_every_entry_point_requires_authentication(self):
        for path in ("/", "/studio", "/library", "/health", "/api/health", "/docs", "/openapi.json", "/api/openapi.json", "/lessons/saved", "/api/lessons/saved"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 401)
                self.assertIn("Basic", response.headers["www-authenticate"])
        self.assertEqual(self.client.post("/api/lessons/generate", json={}).status_code, 401)

    def test_wrong_and_malformed_authorization_is_rejected(self):
        self.assertEqual(self.client.get("/api/health", auth=("tutorflow", "wrong")).status_code, 401)
        for header in ("Bearer abc", "Basic !!!", "Basic /w==", "Basic YWJj"):
            self.assertEqual(self.client.get("/api/health", headers={"Authorization": header}).status_code, 401)
        self.assertEqual(self.client.get("/api/health", auth=("tutorflow", PASSWORD)).status_code, 200)

    def test_shared_responses_disable_cache_and_framing(self):
        response = self.client.get("/api/health", auth=("tutorflow", PASSWORD))
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_authenticated_cross_origin_writes_are_blocked(self):
        response = self.client.post("/api/lessons/generate", auth=("tutorflow", PASSWORD), json={}, headers={"Origin": "https://other.example"})
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/api/lessons/generate", auth=("tutorflow", PASSWORD), json={}, headers={"Origin": "http://testserver"})
        self.assertEqual(response.status_code, 422)

    def test_sharing_rejects_cross_origin_preflight(self):
        response = self.client.options("/api/lessons/saved", headers={
            "Origin": "https://other.example", "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_real_body_limit_applies_to_chunked_requests(self):
        body = iter([b"a" * (3 * 1024 * 1024), b"b" * (3 * 1024 * 1024)])
        response = self.client.post("/api/lessons/export/pdf", auth=("tutorflow", PASSWORD), content=body)
        self.assertEqual(response.status_code, 413)

    def test_generation_bounds_and_capacity_are_enforced(self):
        payload = {"level": "A2", "duration": 30, "topic": "Cafe", "variant_count": 100}
        self.assertEqual(self.client.post("/api/lessons/generate", auth=("tutorflow", PASSWORD), json=payload).status_code, 422)
        payload["variant_count"] = 1
        self.application.state.generation_slots = asyncio.Semaphore(0)
        response = self.client.post("/api/lessons/generate", auth=("tutorflow", PASSWORD), json=payload)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "30")

    def test_frontend_routes_and_missing_api_or_assets(self):
        with TemporaryDirectory() as directory:
            dist = Path(directory)
            (dist / "index.html").write_text("<html>TutorFlow test</html>", encoding="utf-8")
            application = create_app(AppSettings(frontend_dist=dist))
            with TestClient(application) as client:
                for path in ("/", "/studio", "/library"):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIn("TutorFlow test", response.text)
                self.assertEqual(client.get("/assets/missing.js").status_code, 404)
                self.assertEqual(client.get("/api/not-an-endpoint").status_code, 404)
                self.assertEqual(client.get("/%2e%2e/.env").status_code, 404)
                self.assertIn("LessonGenerateRequest", client.get("/api/openapi.json").json()["components"]["schemas"])

    def test_origins_must_be_exact(self):
        for origin in ("*", "https://*.example", "https://example/path", "https://user:password@example"):
            with self.assertRaises(ValueError):
                create_app(AppSettings(cors_origins=(origin,)))


class GenerationConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_requests_do_not_call_model_and_failure_releases_capacity(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate(payload):
            started.set()
            await release.wait()
            raise ValueError("Controlled generation failure")

        application = create_app(AppSettings())
        payload = {"level": "A2", "duration": 30, "topic": "Cafe", "variant_count": 1}
        with patch("app.api.routes_lessons.generate_lesson_variants", new_callable=AsyncMock, side_effect=generate) as mocked:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
                first = asyncio.create_task(client.post("/lessons/generate", json=payload))
                await asyncio.wait_for(started.wait(), timeout=2)
                busy = await client.post("/lessons/generate", json=payload)
                self.assertEqual(busy.status_code, 429)
                self.assertEqual(mocked.await_count, 1)
                release.set()
                self.assertEqual((await first).status_code, 500)
                self.assertEqual((await client.post("/lessons/generate", json=payload)).status_code, 500)
                self.assertEqual(mocked.await_count, 2)


if __name__ == "__main__":
    unittest.main()
