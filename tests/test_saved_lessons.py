from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database import build_engine, get_session, migrate_database
from app.main import app
from tests.helpers import make_lesson


class SavedLessonTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.url = f"sqlite:///{Path(self.directory.name).as_posix()}/library.db"
        self.engine = build_engine(self.url)
        migrate_database(self.engine)

        def session_override():
            with Session(self.engine) as session:
                yield session

        app.dependency_overrides[get_session] = session_override
        self.client = TestClient(app)
        self.id = str(uuid4())
        self.payload = {
            "id": self.id, "saved_at": "2026-10-09T09:00:00Z", "favorite": False,
            "level": "A2", "duration": 30, "theme": "Cafe", "mode": "live",
            "lesson": make_lesson().model_dump(),
        }

    def tearDown(self):
        self.client.close()
        app.dependency_overrides.pop(get_session, None)
        self.engine.dispose()
        self.directory.cleanup()

    def save(self, payload=None):
        value = payload or self.payload
        return self.client.put(f"/lessons/saved/{value['id']}", json=value)

    def test_save_round_trips_content_and_survives_new_engine(self):
        self.payload["lesson"]["cefr_focus"] = "Demo guidance"
        self.payload["lesson"]["future_field"] = {"notes": ["Zażółć gęślą jaźń"]}
        response = self.save()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["lesson"], self.payload["lesson"])
        self.engine.dispose()
        self.engine = build_engine(self.url)
        migrate_database(self.engine)  # Repeated startup must preserve saves.
        retrieved = self.client.get(f"/lessons/saved/{self.id}")
        self.assertEqual(retrieved.json(), response.json())

    def test_idempotent_save_does_not_duplicate_or_overwrite_favorite(self):
        self.save()
        self.client.patch(f"/lessons/saved/{self.id}", json={"favorite": True})
        self.assertTrue(self.save().json()["favorite"])
        page = self.client.get("/lessons/saved").json()
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["id"], self.id)

    def test_delete_and_undo_with_same_id(self):
        self.save()
        for _ in range(2):
            self.assertEqual(self.client.delete(f"/lessons/saved/{self.id}").status_code, 204)
        self.assertEqual(self.client.get(f"/lessons/saved/{self.id}").status_code, 404)
        self.assertEqual(self.save().status_code, 200)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 1)

    def test_pagination_keeps_repeated_generator_ids_separate(self):
        self.save()
        other = dict(self.payload, id=str(uuid4()), saved_at="2026-10-09T10:00:00Z")
        self.save(other)
        first = self.client.get("/lessons/saved?limit=1").json()
        second = self.client.get("/lessons/saved?limit=1&offset=1").json()
        self.assertEqual(first["total"], 2)
        self.assertEqual(first["items"][0]["id"], other["id"])
        self.assertEqual(second["items"][0]["id"], self.id)

    def test_validation_and_missing_lesson(self):
        self.assertEqual(self.client.get("/lessons/saved?limit=101").status_code, 422)
        self.assertEqual(self.client.put(f"/lessons/saved/{uuid4()}", json=self.payload).status_code, 422)
        self.assertEqual(self.client.patch(f"/lessons/saved/{self.id}", json={"favorite": True}).status_code, 404)
        self.payload["lesson"] = {"title": "Incomplete"}
        self.assertEqual(self.save().status_code, 422)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)

    def test_database_errors_are_safe_and_do_not_claim_save_success(self):
        error = OperationalError("sensitive SQL", {}, Exception("password=secret"))
        with patch("sqlalchemy.orm.Session.get", side_effect=error):
            response = self.save()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)

    def test_migration_version_is_recorded(self):
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(text("select version_num from alembic_version")), "0001_saved_lessons")


if __name__ == "__main__":
    unittest.main()
