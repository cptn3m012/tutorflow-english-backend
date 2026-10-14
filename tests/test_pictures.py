from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from app.config import AppSettings, ImageSettings
from app.database import build_engine, get_session, migrate_database
from app.main import create_app
from app.models import LessonImageRecord, PictureJobRecord, SavedLessonRecord
from app.picture_storage import image_path, write_image
from tests.helpers import make_lesson


class PictureTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.settings = ImageSettings(enabled=True, storage_dir=Path(self.directory.name) / "images")
        self.engine = build_engine(f"sqlite:///{self.directory.name}/test.db")
        migrate_database(self.engine)
        self.app = create_app(AppSettings())

        def session_override():
            with Session(self.engine) as session:
                yield session

        self.app.dependency_overrides[get_session] = session_override
        self.client = TestClient(self.app)
        self.config = patch("app.config.ImageSettings.from_environment", return_value=self.settings)
        self.config.start()
        self.lesson_id = str(uuid4())
        self.payload = {"id": self.lesson_id, "saved_at": "2026-10-12T09:00:00Z", "favorite": False,
                        "level": "A2", "duration": 30, "theme": "Cafe", "mode": "live",
                        "lesson": make_lesson().model_dump()}
        self.client.put(f"/lessons/saved/{self.lesson_id}", json=self.payload)
        self.url = f"/lessons/saved/{self.lesson_id}/picture-jobs"

    def tearDown(self):
        self.client.close()
        self.config.stop()
        self.engine.dispose()
        self.directory.cleanup()

    def test_job_is_persistent_and_duplicate_requests_return_active_job(self):
        first = self.client.post(self.url, json={"style": "photo"})
        second = self.client.post(self.url, json={"style": "illustration"})
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.engine.dispose()
        self.assertEqual(self.client.get(self.url).json()["status"], "queued")
        self.assertEqual(self.client.get(f"/lessons/saved/{self.lesson_id}").json()["lesson"], self.payload["lesson"])

    def test_picture_draft_stays_outside_library_until_explicit_save(self):
        draft_id = str(uuid4())
        payload = dict(self.payload, id=draft_id)
        draft_url = f"/lessons/picture-drafts/{draft_id}"
        self.assertEqual(self.client.put(draft_url, json=payload).status_code, 200)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 1)
        self.assertEqual(self.client.get(f"/lessons/saved/{draft_id}").status_code, 404)
        self.assertEqual(self.client.patch(f"/lessons/saved/{draft_id}", json={"favorite": True}).status_code, 404)
        self.assertEqual(self.client.post(f"/lessons/saved/{draft_id}/picture-jobs", json={}).status_code, 202)
        # A completed worker update must survive stale browser save/retry bodies.
        with Session(self.engine) as session:
            record = session.get(SavedLessonRecord, draft_id)
            record.lesson = {**record.lesson, "picture_marker": "canonical worker update"}
            session.commit()
        self.assertIn("picture_marker", self.client.put(draft_url, json=payload).json()["lesson"])
        saved = self.client.put(f"/lessons/saved/{draft_id}", json=payload)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["lesson"]["picture_marker"], "canonical worker update")
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 2)
        self.assertEqual(self.client.get(f"/lessons/saved/{draft_id}").status_code, 200)
        self.assertEqual(self.client.get(draft_url).status_code, 200)
        self.assertEqual(self.client.put(f"/lessons/saved/{draft_id}", json=payload).json(), saved.json())

    def test_draft_creation_validates_content_and_path_id(self):
        self.assertEqual(self.client.put(f"/lessons/picture-drafts/{uuid4()}", json=self.payload).status_code, 422)
        self.assertEqual(self.client.put(f"/lessons/picture-drafts/{self.lesson_id}", json={}).status_code, 422)

    def test_draft_rejects_invalid_lesson_content_before_insertion(self):
        for updates in ({"reading_text": "   "}, {"reading_answers": "bad type"},
                        {"visual_activity": {"questions": [{}]}}, {"level": "C2"}):
            with self.subTest(updates=updates):
                payload = deepcopy(self.payload)
                payload["id"] = str(uuid4())
                payload["lesson"].update(updates)
                url = f"/lessons/picture-drafts/{payload['id']}"
                self.assertEqual(self.client.put(url, json=payload).status_code, 422)
                self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 1)

    def test_api_does_not_accept_workflows_paths_prompts_or_unknown_styles(self):
        for body in ({"workflow": {}}, {"path": "../../.env"}, {"prompt": "anything"}, {"style": "unknown"}):
            self.assertEqual(self.client.post(self.url, json=body).status_code, 422)
        self.assertEqual(self.client.post(f"/lessons/saved/{uuid4()}/picture-jobs", json={}).status_code, 404)

    def test_disabled_runtime_and_queue_limit(self):
        with patch("app.config.ImageSettings.from_environment", return_value=ImageSettings()):
            self.assertEqual(self.client.post(self.url, json={}).status_code, 503)
        with Session(self.engine) as session:
            for _ in range(4):
                session.add(PictureJobRecord(id=str(uuid4()), lesson_id=str(uuid4()), status="queued", style="photo",
                                             model="flux2-klein-4b", seed=1, created_at=datetime.now(UTC), updated_at=datetime.now(UTC)))
            session.commit()
        self.assertEqual(self.client.post(self.url, json={}).status_code, 429)

    def test_asset_is_private_and_delete_undo_preserves_immutable_image(self):
        asset_id = str(uuid4())
        raw = BytesIO()
        Image.new("RGB", (256, 256), "orange").save(raw, "PNG")
        write_image(asset_id, raw.getvalue(), self.settings)
        with Session(self.engine) as session:
            session.add(LessonImageRecord(id=asset_id, lesson_id=self.lesson_id, created_at=datetime.now(UTC), metadata_json={}))
            session.commit()
        url = f"/lesson-images/{asset_id}"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        with Image.open(BytesIO(response.content)) as image:
            self.assertEqual(image.format, "JPEG")
        self.client.delete(f"/lessons/saved/{self.lesson_id}")
        self.assertEqual(self.client.get(url).status_code, 404)
        self.client.put(f"/lessons/saved/{self.lesson_id}", json=self.payload)
        self.assertEqual(self.client.get(url).status_code, 200)

    def test_public_access_requires_auth_for_jobs_and_assets(self):
        protected = create_app(AppSettings(public_share=True, share_username="tester", share_password="test-password-long-enough"))
        protected.dependency_overrides[get_session] = self.app.dependency_overrides[get_session]
        with patch("app.main.migrate_database"), patch("app.main.get_engine", return_value=self.engine), TestClient(protected) as client:
            for url in (self.url, f"/lesson-images/{uuid4()}", "/lessons/pictures/settings",
                        f"/lessons/picture-drafts/{self.lesson_id}"):
                self.assertEqual(client.get(url).status_code, 401)
            self.assertEqual(client.put(f"/lessons/picture-drafts/{uuid4()}", json=self.payload).status_code, 401)
            self.assertEqual(client.get(self.url, auth=("tester", "test-password-long-enough")).status_code, 200)

    def test_completed_events_close_and_wrong_lesson_cannot_subscribe(self):
        job = self.client.post(self.url, json={}).json()
        with Session(self.engine) as session:
            record = session.get(PictureJobRecord, job["id"])
            record.status = "failed"
            record.error = "Please try again."
            session.commit()
        response = self.client.get(f'{self.url}/{job["id"]}/events')
        self.assertIn('event: progress', response.text)
        self.assertIn('"status":"failed"', response.text)
        self.assertEqual(self.client.get(f'/lessons/saved/{uuid4()}/picture-jobs/{job["id"]}/events').status_code, 404)

    def test_file_path_requires_uuid_and_invalid_image_does_not_write(self):
        with self.assertRaises(ValueError):
            image_path("../../.env", self.settings)
        with self.assertRaises(ValueError):
            write_image(str(uuid4()), b"not an image", self.settings)


if __name__ == "__main__":
    unittest.main()
