import asyncio
from copy import deepcopy
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
from app.errors import LessonQualityError
from app.main import create_app
from app.models import PictureJobRecord, SavedLessonRecord
from app.ollama_client import OllamaError
from app.pdf_service import PdfDocument, build_lesson_pdf, parse_lesson_pdf_payload, pdf_literal
from app.schemas import PdfExportOptions
from app.quiz_schemas import ExitQuizDraft
from app.quiz_service import generate_exit_quiz, quiz_is_stale, quiz_source_token
from tests.helpers import make_lesson
from tests.quiz_fixtures import quiz_draft, quiz_for


class ExitQuizGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_uses_full_cefr_profile_and_only_publishes_reviewed_quiz(self):
        content = make_lesson(grammar_focus="Question words").model_dump()
        with patch("app.quiz_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(quiz_draft()), '{"issues": []}']) as model:
            quiz = await generate_exit_quiz(content, "A2", 3)
        self.assertEqual(len(quiz.questions), 3)
        self.assertEqual(quiz.source_token, quiz_source_token(content, "A2"))
        self.assertIn("communicate in simple and routine tasks", model.call_args_list[0].args[0])
        self.assertIn(content["lesson_goal"], model.call_args_list[0].args[0])
        self.assertEqual([call.kwargs["task"] for call in model.call_args_list], ["quiz", "review"])

    async def test_wrong_key_bad_count_and_unrelated_target_are_retried_before_review(self):
        bad_key = quiz_draft()
        bad_key["questions"][0]["answer"] = "not an option"
        bad_count = quiz_draft()
        bad_count["questions"].append({**bad_count["questions"][0], "question": "Another question?"})
        bad_target = quiz_draft()
        bad_target["questions"][0]["target"] = "quantum mechanics"
        with patch("app.quiz_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(item) for item in (bad_key, bad_count, bad_target, quiz_draft())] + ['{"issues": []}']) as model:
            await generate_exit_quiz(make_lesson().model_dump(), "A2", 3)
        self.assertEqual(model.await_count, 5)
        self.assertIn("exactly 3", model.call_args_list[2].args[0])
        self.assertIn("exact item", model.call_args_list[3].args[0])

    async def test_four_and_five_question_choices_are_enforced_in_model_schema(self):
        for count in (4, 5):
            with self.subTest(count=count):
                draft = quiz_draft()
                for index in range(3, count):
                    draft["questions"].append({**draft["questions"][0], "question": f"Vocabulary check {index + 1}?"})
                with patch("app.quiz_service.generate_from_ollama", new_callable=AsyncMock,
                           side_effect=[json.dumps(draft), '{"issues": []}']) as model:
                    quiz = await generate_exit_quiz(make_lesson().model_dump(), "A2", count)
                self.assertEqual(len(quiz.questions), count)
                schema = model.call_args_list[0].kwargs["json_schema"]["properties"]["questions"]
                self.assertEqual((schema["minItems"], schema["maxItems"]), (count, count))

    async def test_semantic_review_feedback_repairs_draft_and_malformed_verdict_is_not_approval(self):
        with patch("app.quiz_service.generate_from_ollama", new_callable=AsyncMock, side_effect=[
            json.dumps(quiz_draft()), '{}', '{"issues": ["Question 2 has two valid options; replace the distractor."]}',
            json.dumps(quiz_draft()), '{"issues": []}',
        ]) as model:
            await generate_exit_quiz(make_lesson().model_dump(), "A2", 3)
        self.assertIn("two valid options", model.call_args_list[3].args[0])
        self.assertEqual(model.await_count, 5)

    async def test_persistent_issues_are_bounded_and_leave_lesson_unchanged(self):
        content = make_lesson().model_dump()
        snapshot = deepcopy(content)
        with patch("app.quiz_service.generate_from_ollama", new_callable=AsyncMock,
                   side_effect=[json.dumps(quiz_draft()), '{"issues": ["Wrong key."]}'] * 5) as model:
            with self.assertRaisesRegex(LessonQualityError, "after 5 attempts"):
                await generate_exit_quiz(content, "A2", 3)
        self.assertEqual(model.await_count, 10)
        self.assertEqual(content, snapshot)

    async def test_empty_vocabulary_requires_teacher_input_before_calling_model(self):
        with patch("app.quiz_service.generate_from_ollama", new_callable=AsyncMock) as model:
            with self.assertRaisesRegex(LessonQualityError, "Add vocabulary"):
                await generate_exit_quiz(make_lesson(target_vocabulary=[]).model_dump(), "A2", 3)
            model.assert_not_called()

    def test_choices_and_coverage_are_validated(self):
        for change in ("duplicate_options", "missing_focus", "options_in_short_answer"):
            draft = quiz_draft()
            if change == "duplicate_options":
                draft["questions"][0]["options"] = ["bill", "BILL", "cup"]
            elif change == "missing_focus":
                draft["questions"][1]["focus"] = "vocabulary"
            else:
                draft["questions"][2]["options"] = ["one"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                ExitQuizDraft.model_validate(draft)


class ExitQuizApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.engine = build_engine(f"sqlite:///{Path(self.directory.name).as_posix()}/quiz.db")
        migrate_database(self.engine)
        self.app = create_app(AppSettings())
        def sessions():
            with Session(self.engine) as session:
                yield session
        self.app.dependency_overrides[get_session] = sessions
        self.client = TestClient(self.app)
        self.url = f"/lessons/saved/{uuid4()}"
        self.content = make_lesson(grammar_focus="Question words", level="A2").model_dump()
        response = self.client.put(self.url, json={"id": self.url.rsplit('/', 1)[-1], "level": "A2",
            "duration": 30, "theme": "Cafe", "mode": "live", "lesson": self.content})
        self.assertEqual(response.status_code, 200, response.text)
        self.saved = response.json()

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.directory.cleanup()

    def prepare(self, **updates):
        return self.client.post(self.url + "/quiz", json={"expected_token": self.saved["content_token"], **updates})

    def save_quiz(self):
        with patch("app.api.routes_saved_lessons.generate_exit_quiz", new_callable=AsyncMock,
                   return_value=quiz_for(self.content)):
            response = self.prepare()
        self.assertEqual(response.status_code, 200, response.text)
        self.saved = response.json()
        return self.saved

    def test_quiz_persists_with_same_identity_favorite_and_existing_material(self):
        self.client.patch(self.url, json={"favorite": True})
        updated = self.save_quiz()
        self.assertTrue(updated["favorite"])
        self.assertFalse(updated["lesson"]["exit_quiz_stale"])
        self.assertEqual(updated["lesson"]["reading_text"], self.content["reading_text"])
        self.engine.dispose()
        self.assertEqual(self.client.get(self.url).json(), updated)

    def test_failures_and_busy_model_preserve_previous_quiz_and_release_slot(self):
        self.save_quiz()
        for error, status in ((LessonQualityError("Ambiguous quiz"), 422), (OllamaError("Offline"), 503)):
            with patch("app.api.routes_saved_lessons.generate_exit_quiz", new_callable=AsyncMock, side_effect=error):
                self.assertEqual(self.prepare().status_code, status)
            self.assertEqual(self.client.get(self.url).json(), self.saved)
            self.assertFalse(self.app.state.generation_slots.locked())
        self.app.state.generation_slots = asyncio.Semaphore(0)
        self.assertEqual(self.prepare().status_code, 429)

    def test_stale_token_and_invalid_counts_do_not_generate(self):
        with patch("app.api.routes_saved_lessons.generate_exit_quiz", new_callable=AsyncMock) as model:
            for count in (2, 6):
                self.assertEqual(self.prepare(question_count=count).status_code, 422)
            self.assertEqual(self.prepare(expected_token="0" * 64).status_code, 409)
            model.assert_not_called()

    def test_edit_during_generation_cannot_be_overwritten(self):
        async def concurrent(*args):
            with Session(self.engine) as session:
                record = session.get(SavedLessonRecord, self.saved["id"])
                record.lesson = {**record.lesson, "lesson_goal": "Another teacher's goal."}
                session.commit()
            return quiz_for(self.content)
        with patch("app.api.routes_saved_lessons.generate_exit_quiz", side_effect=concurrent):
            self.assertEqual(self.prepare().status_code, 409)
        self.assertIsNone(self.client.get(self.url).json()["lesson"]["exit_quiz"])

    def test_active_picture_job_blocks_generation(self):
        with Session(self.engine) as session:
            session.add(PictureJobRecord(id=str(uuid4()), lesson_id=self.saved["id"], status="queued", style="photo",
                model="flux2-klein-4b", seed=1, created_at=datetime.now(UTC), updated_at=datetime.now(UTC)))
            session.commit()
        with patch("app.api.routes_saved_lessons.generate_exit_quiz", new_callable=AsyncMock) as model:
            self.assertEqual(self.prepare().status_code, 409)
            model.assert_not_called()

    def test_edit_marks_quiz_stale_and_export_refuses_outdated_handout(self):
        self.save_quiz()
        edited = self.client.patch(self.url + "/content", json={"expected_token": self.saved["content_token"],
            "changes": {"lesson_goal": "Ask for directions."}})
        self.assertEqual(edited.status_code, 200, edited.text)
        lesson = edited.json()["lesson"]
        self.assertTrue(lesson["exit_quiz_stale"])
        exported = self.client.post("/lessons/export/pdf", json={"lesson": lesson, "pdf_options": {"content": "quiz"}})
        self.assertEqual(exported.status_code, 400)
        response, options = parse_lesson_pdf_payload({"lesson": lesson})
        self.assertNotIn(pdf_literal("Cafe quick check"), build_lesson_pdf(response, options))


class ExitQuizPdfTests(unittest.TestCase):
    def test_large_courier_text_wraps_instead_of_running_past_the_page_edge(self):
        document = PdfDocument(PdfExportOptions(font_family="courier", font_size="large"))
        document.text("Where " * 14, size=10)
        self.assertEqual(len(document.page.commands), 2)
        self.assertIn(b"(Where Where", document.page.commands[1])
    def content(self):
        content = make_lesson(level="A2").model_dump()
        content["exit_quiz"] = quiz_for(content).model_dump()
        return content

    def test_quiz_only_student_has_questions_and_no_keys_or_lesson_passage(self):
        content = self.content()
        response, options = parse_lesson_pdf_payload({"lesson": content,
            "pdf_options": {"content": "quiz", "include_answers": False}})
        pdf = build_lesson_pdf(response, options)
        self.assertIn(pdf_literal("Cafe quick check"), pdf)
        self.assertIn(pdf_literal("3. " + content["exit_quiz"]["questions"][2]["question"]), pdf)
        self.assertNotIn(pdf_literal(content["exit_quiz"]["questions"][2]["answer"]), pdf)
        self.assertNotIn(pdf_literal("Exit quiz answer key"), pdf)
        self.assertNotIn(pdf_literal("Reading"), pdf)

    def test_teacher_key_is_on_separate_page_and_full_lesson_quiz_can_be_omitted(self):
        content = self.content()
        response, options = parse_lesson_pdf_payload({"lesson": content,
            "pdf_options": {"content": "quiz", "include_answers": True}})
        pdf = build_lesson_pdf(response, options)
        self.assertIn(pdf_literal("Exit quiz answer key"), pdf)
        self.assertIn(b"/Count 2", pdf)
        self.assertIn(b"Example answer:", pdf)
        response, options = parse_lesson_pdf_payload({"lesson": content, "pdf_options": {"include_exit_quiz": False}})
        self.assertNotIn(pdf_literal("Cafe quick check"), build_lesson_pdf(response, options))

    def test_fingerprint_ignores_picture_changes_and_handles_legacy_optional_metadata(self):
        content = self.content()
        content.update(images=[], picture_context_stale=True, tags=["new tag"])
        self.assertFalse(quiz_is_stale(content, "A2"))
        content.pop("communication_focus")
        self.assertFalse(quiz_is_stale(content, "A2"))
        content["target_vocabulary"] = ["new word"]
        self.assertTrue(quiz_is_stale(content, "A2"))
