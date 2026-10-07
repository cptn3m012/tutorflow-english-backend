import unittest

from fastapi.testclient import TestClient

from app.main import app


class LocalCorsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_local_frontend_can_send_json_posts_without_cross_origin_credentials(self):
        for origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
            with self.subTest(origin=origin):
                response = self.client.options("/lessons/generate", headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], origin)
                self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_unlisted_origin_is_rejected(self):
        response = self.client.options("/lessons/generate", headers={
            "Origin": "https://unlisted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_unneeded_method_and_custom_header_are_rejected(self):
        for method, headers in (("DELETE", "content-type"), ("POST", "x-private-key")):
            with self.subTest(method=method, headers=headers):
                response = self.client.options("/lessons/generate", headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": method,
                    "Access-Control-Request-Headers": headers,
                })
                self.assertEqual(response.status_code, 400)

    def test_get_exposes_download_filename_for_allowed_frontend(self):
        response = self.client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:5173")
        self.assertEqual(response.headers["access-control-expose-headers"], "Content-Disposition")


if __name__ == "__main__":
    unittest.main()
