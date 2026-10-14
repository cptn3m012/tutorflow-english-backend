from copy import deepcopy
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
from app.models import SavedLessonRecord
from tests.helpers import COMPREHENSION, make_lesson


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

    def test_blank_required_text_and_list_items_are_rejected_before_save(self):
        for field in ("title", "lesson_goal", "reading_text", "target_vocabulary",
                      "reading_questions", "speaking_questions", "sentence_frames"):
            with self.subTest(field=field):
                payload = deepcopy(self.payload)
                payload["lesson"][field] = [" \n "] if isinstance(payload["lesson"][field], list) else " \n "
                self.assertEqual(self.save(payload).status_code, 422)
        payload = dict(self.payload, theme="   ")
        self.assertEqual(self.save(payload).status_code, 422)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)

    def test_malformed_optional_sections_are_rejected_before_save(self):
        cases = (
            {"reading_answers": "bad type"},
            {"reading_answers": [{}]},
            {"images": [{"url": "image.jpg"}]},
            {"primary_image": {"url": "image.jpg"}},
            {"visual_activity": {"questions": [{}]}},
            {"visual_activity": {"title": "Picture", "instruction": "Describe it.",
                                 "scene_prompt": "", "questions": [{"question": "What is visible?"}]}},
            {"worksheet_assets": [{"asset_type": "mini_dialogue", "title": "Dialogue",
                                   "instruction": "Read it.", "lines": [None]}]},
        )
        for updates in cases:
            with self.subTest(updates=updates):
                payload = deepcopy(self.payload)
                payload["lesson"].update(updates)
                self.assertEqual(self.save(payload).status_code, 422)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)

    def test_metadata_must_match_the_lesson_when_present(self):
        for updates in ({"level": "C2"}, {"duration_minutes": 180}, {"theme": "Travel"},
                        {"level": "A9"}, {"duration_minutes": -1}, {"theme": "   "}):
            with self.subTest(updates=updates):
                payload = deepcopy(self.payload)
                payload["lesson"].update(updates)
                self.assertEqual(self.save(payload).status_code, 422)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)

    def test_answer_keys_must_match_questions_and_the_final_reading(self):
        payload = deepcopy(self.payload)
        payload["lesson"].update(
            reading_questions=[item["question"] for item in COMPREHENSION["items"]],
            reading_answers=deepcopy(COMPREHENSION["items"]),
        )
        cases = []
        missing_answer = deepcopy(payload)
        missing_answer["lesson"]["reading_answers"].pop()
        cases.append(missing_answer)
        wrong_order = deepcopy(payload)
        wrong_order["lesson"]["reading_answers"].reverse()
        cases.append(wrong_order)
        stale_evidence = deepcopy(payload)
        stale_evidence["lesson"]["reading_answers"][0]["evidence"] = "Anna orders coffee at home."
        cases.append(stale_evidence)
        for invalid in cases:
            with self.subTest(lesson=invalid["lesson"]):
                self.assertEqual(self.save(invalid).status_code, 422)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)
        response = self.save(payload)
        self.assertEqual(response.status_code, 200, response.text)
        pdf = self.client.post("/lessons/export/pdf", json={"lesson": response.json()["lesson"]})
        self.assertEqual(pdf.status_code, 200, pdf.text[:100])

    def test_legacy_import_without_new_sections_remains_exportable(self):
        self.payload["lesson"] = {
            "title": "Legacy cafe", "lesson_goal": "Order tea.",
            "reading_text": make_lesson().reading_text, "cefr_focus": ["Making requests"],
            "future_field": {"notes": ["Zażółć gęślą jaźń"]},
        }
        response = self.save()
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["lesson"]
        self.assertEqual(content["cefr_focus"], ["Making requests"])
        self.assertEqual(content["future_field"], self.payload["lesson"]["future_field"])
        self.assertNotIn("reading_answers", content)
        self.assertEqual(self.client.get(f"/lessons/saved/{self.id}").json(), response.json())
        pdf = self.client.post("/lessons/export/pdf", json={"lesson": content})
        self.assertEqual(pdf.status_code, 200, pdf.text[:100])

    def test_picture_sections_round_trip_and_reject_mismatched_versions(self):
        asset_id = str(uuid4())
        image = {"url": f"/api/lesson-images/{asset_id}", "description": "A cup on a table.",
                 "score": 1.0, "asset_id": asset_id, "future_image_field": "kept"}
        visual = {"activity_type": "describe_picture", "title": "Describe the picture",
                  "instruction": "Describe the cup.", "scene_prompt": "", "asset_id": asset_id,
                  "questions": [{"question": "Where is the cup?", "answer": "On the table.",
                                 "evidence": "The cup is on the table."}]}
        payload = deepcopy(self.payload)
        payload["lesson"].update(level="A2", duration_minutes=30, theme="Cafe",
                                 images=[image], primary_image=image, visual_activity=visual)
        for replacement in (str(uuid4()), "../../.env"):
            with self.subTest(asset_id=replacement):
                invalid = deepcopy(payload)
                invalid["lesson"]["visual_activity"]["asset_id"] = replacement
                self.assertEqual(self.save(invalid).status_code, 422)
        response = self.save(payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["lesson"], payload["lesson"])

    def test_rejected_retry_keeps_saved_content_and_favorite(self):
        self.assertEqual(self.save().status_code, 200)
        previous = self.client.patch(f"/lessons/saved/{self.id}", json={"favorite": True}).json()
        invalid = deepcopy(self.payload)
        invalid["lesson"]["reading_answers"] = "bad type"
        self.assertEqual(self.save(invalid).status_code, 422)
        self.assertEqual(self.client.get(f"/lessons/saved/{self.id}").json(), previous)

    def test_existing_legacy_records_remain_readable(self):
        self.save()
        with Session(self.engine) as session:
            record = session.get(SavedLessonRecord, self.id)
            record.lesson = {**record.lesson, "reading_answers": "old unsupported data"}
            session.commit()
        response = self.client.get(f"/lessons/saved/{self.id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["lesson"]["reading_answers"], "old unsupported data")
        self.assertEqual(self.client.get("/lessons/saved").status_code, 200)

    def test_database_errors_are_safe_and_do_not_claim_save_success(self):
        error = OperationalError("sensitive SQL", {}, Exception("password=secret"))
        with patch("sqlalchemy.orm.Session.scalar", side_effect=error):
            response = self.save()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        self.assertEqual(self.client.get("/lessons/saved").json()["total"], 0)

    def test_migration_version_is_recorded(self):
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(text("select version_num from alembic_version")), "0003_picture_drafts")


if __name__ == "__main__":
    unittest.main()
