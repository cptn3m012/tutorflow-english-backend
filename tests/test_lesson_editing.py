from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import AppSettings
from app.database import build_engine, get_session, migrate_database
from app.lesson_editing import edit_content, regenerate_content
from app.main import create_app
from app.models import PictureJobRecord, SavedLessonRecord
from app.ollama_client import OllamaError
from app.schemas import ReadingQuestionAnswer
from app.worksheet_service import build_fill_in_the_blanks_asset
from tests.helpers import COMPREHENSION, COMPREHENSION_SELECTION, READING, make_lesson


def answered_lesson():
    lesson = make_lesson(reading_questions=[item["question"] for item in COMPREHENSION["items"]],
                         reading_answers=[ReadingQuestionAnswer(**item) for item in COMPREHENSION["items"]])
    lesson.worksheet_assets = [build_fill_in_the_blanks_asset(lesson)]
    return lesson


class LessonEditingApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.engine = build_engine(f"sqlite:///{Path(self.directory.name).as_posix()}/editing.db")
        migrate_database(self.engine)
        self.app = create_app(AppSettings())
        def sessions():
            with Session(self.engine) as session:
                yield session
        self.app.dependency_overrides[get_session] = sessions
        self.client = TestClient(self.app)
        self.id = str(uuid4())
        self.url = f"/lessons/saved/{self.id}"
        payload = {"id": self.id, "level": "A2", "duration": 30, "theme": "Cafe", "mode": "live",
                   "lesson": answered_lesson().model_dump()}
        saved = self.client.put(self.url, json=payload)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.saved = saved.json()

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.directory.cleanup()

    def edit(self, changes, token=None):
        return self.client.patch(self.url + "/content", json={
            "expected_token": token or self.saved["content_token"], "changes": changes})

    def regenerate(self, section="speaking", token=None):
        return self.client.post(self.url + "/regenerate", json={
            "expected_token": token or self.saved["content_token"], "section": section,
            "instructions": "Use practical classroom situations."})

    def test_edit_persists_and_preserves_other_material_and_favorite(self):
        self.client.patch(self.url, json={"favorite": True})
        response = self.edit({"title": "A different cafe lesson"})
        self.assertEqual(response.status_code, 200, response.text)
        updated = response.json()
        self.assertTrue(updated["favorite"])
        self.assertEqual(updated["saved_at"], self.saved["saved_at"])
        self.assertEqual(updated["lesson"], {**self.saved["lesson"], "title": "A different cafe lesson"})
        self.assertNotEqual(updated["content_token"], self.saved["content_token"])
        self.engine.dispose()
        self.assertEqual(self.client.get(self.url).json(), updated)

    def test_reading_edit_invalidates_only_linked_sections_without_model_calls(self):
        with patch("app.lesson_editing.generate_from_ollama", new_callable=AsyncMock) as model:
            response = self.edit({"reading_text": "Anna visits a cafe and orders tea."})
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["lesson"]
        self.assertEqual(content["reading_questions"], [])
        self.assertEqual(content["reading_answers"], [])
        self.assertEqual(content["worksheet_assets"], [])
        self.assertEqual(content["speaking_questions"], self.saved["lesson"]["speaking_questions"])
        model.assert_not_called()
        pdf = self.client.post("/lessons/export/pdf", json={"lesson": content})
        self.assertEqual(pdf.status_code, 200, pdf.text[:100])

    def test_stale_editor_cannot_overwrite_a_new_save(self):
        first = self.edit({"title": "First teacher"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(self.edit({"title": "Second teacher"}).status_code, 409)
        self.assertEqual(self.client.get(self.url).json()["lesson"]["title"], "First teacher")

    def test_invalid_edits_do_not_change_the_record(self):
        for changes in ({}, {"title": " "}, {"title": None}, {"images": []},
                        {"reading_answers": [{"question": "Where?", "answer": "Home", "evidence": "Not in the text"}]}):
            with self.subTest(changes=changes):
                self.assertEqual(self.edit(changes).status_code, 422)
        self.assertEqual(self.client.get(self.url).json(), self.saved)

    def test_active_picture_job_blocks_edits_and_regeneration(self):
        with Session(self.engine) as session:
            session.add(PictureJobRecord(id=str(uuid4()), lesson_id=self.id, status="queued", style="photo",
                        model="flux2-klein-4b", seed=1, created_at=datetime.now(UTC), updated_at=datetime.now(UTC)))
            session.commit()
        self.assertEqual(self.edit({"title": "Another title"}).status_code, 409)
        with patch("app.api.routes_saved_lessons.regenerate_content", new_callable=AsyncMock) as regenerate:
            self.assertEqual(self.regenerate().status_code, 409)
            regenerate.assert_not_called()

    def test_regeneration_changes_only_the_requested_section(self):
        prepared = {**self.saved["lesson"], "speaking_questions": ["Which cafe would you choose?"]}
        with patch("app.api.routes_saved_lessons.regenerate_content", new_callable=AsyncMock, return_value=prepared) as regenerate:
            response = self.regenerate()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["lesson"], prepared)
        self.assertEqual(regenerate.call_args.args[-2:], ("speaking", "Use practical classroom situations."))
        self.assertFalse(self.app.state.generation_slots.locked())

    def test_failed_or_busy_model_keeps_the_previous_record(self):
        with patch("app.api.routes_saved_lessons.regenerate_content", new_callable=AsyncMock, side_effect=OllamaError("Model unavailable")):
            self.assertEqual(self.regenerate().status_code, 503)
        self.assertFalse(self.app.state.generation_slots.locked())
        self.assertEqual(self.client.get(self.url).json(), self.saved)
        import asyncio
        self.app.state.generation_slots = asyncio.Semaphore(0)
        self.assertEqual(self.regenerate().status_code, 429)

    def test_edit_during_regeneration_is_not_overwritten(self):
        async def other_teacher(*args):
            with Session(self.engine) as session:
                record = session.get(SavedLessonRecord, self.id)
                record.lesson = {**record.lesson, "title": "Another teacher's edit"}
                session.commit()
            return {**args[0], "speaking_questions": ["A new question?"]}
        with patch("app.api.routes_saved_lessons.regenerate_content", side_effect=other_teacher):
            self.assertEqual(self.regenerate().status_code, 409)
        self.assertEqual(self.client.get(self.url).json()["lesson"]["title"], "Another teacher's edit")

    def test_favorite_change_during_regeneration_is_preserved(self):
        async def favorite(*args):
            with Session(self.engine) as session:
                session.get(SavedLessonRecord, self.id).favorite = True
                session.commit()
            return {**args[0], "speaking_questions": ["A new question?"]}
        with patch("app.api.routes_saved_lessons.regenerate_content", side_effect=favorite):
            response = self.regenerate()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["favorite"])


class SectionRegenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaking_uses_bounded_section_schema_and_keeps_reading(self):
        original = answered_lesson().model_dump()
        questions = [f"What would you order at cafe number {number}?" for number in range(5)]
        with patch("app.lesson_editing.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=["{}", json.dumps({"speaking_questions": questions})]) as generate:
            result = await regenerate_content(original, "A2", "Cafe", 30, "speaking", "Use cafe situations.")
        self.assertEqual(result, {**original, "speaking_questions": questions})
        self.assertEqual(generate.await_count, 2)
        self.assertEqual(generate.call_args.kwargs["json_schema"]["required"], ["speaking_questions"])

    async def test_reading_rebuilds_questions_answers_and_cloze_from_new_text(self):
        original = answered_lesson().model_dump()
        custom = {"asset_type": "custom_activity", "title": "Custom exercise", "instruction": "Discuss it.",
                  "lines": ["Your own activity."], "future_field": {"notes": ["Keep this"]}}
        original["worksheet_assets"].append(custom)
        new_reading = READING.replace("Monday morning", "Tuesday morning")
        with patch("app.lesson_service.generate_from_ollama", new_callable=AsyncMock, return_value=new_reading), \
             patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock, return_value=json.dumps(COMPREHENSION_SELECTION)), \
             patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock, return_value='{"issues": []}'):
            result = await regenerate_content(original, "A2", "Cafe", 30, "reading", "Change the day.")
        self.assertEqual(result["reading_text"], new_reading)
        self.assertIn("Tuesday morning", result["reading_answers"][0]["evidence"])
        self.assertEqual(result["speaking_questions"], original["speaking_questions"])
        self.assertTrue(result["worksheet_assets"])
        self.assertEqual(result["worksheet_assets"][-1], custom)

    async def test_comprehension_keeps_reading_and_other_sections(self):
        original = answered_lesson().model_dump()
        with patch("app.reading_service.generate_from_ollama", new_callable=AsyncMock, return_value=json.dumps(COMPREHENSION_SELECTION)), \
             patch("app.quality_service.generate_from_ollama", new_callable=AsyncMock, return_value='{"issues": []}'):
            result = await regenerate_content(original, "A2", "Cafe", 30, "comprehension", "")
        self.assertEqual(result["reading_text"], original["reading_text"])
        self.assertEqual(result["worksheet_assets"], original["worksheet_assets"])

    def test_linked_picture_becomes_stale_but_the_image_is_kept(self):
        original = answered_lesson().model_dump()
        image = {"asset_id": str(uuid4()), "url": "/picture.jpg", "description": "A cup", "score": 1}
        original.update(images=[image], primary_image=image, visual_activity=None)
        result = edit_content(original, {"reading_text": "Anna drinks tea."})
        self.assertEqual(result["primary_image"], image)
        self.assertTrue(result["picture_context_stale"])
        self.assertIsNone(result["visual_activity"])
