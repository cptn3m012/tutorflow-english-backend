from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from PIL import Image
from sqlalchemy.orm import Session

from app.config import ImageSettings
from app.database import build_engine, migrate_database
from app.models import LessonImageRecord, SavedLessonRecord
from app.pdf_service import build_lesson_pdf, parse_lesson_pdf_payload, pdf_literal
from app.picture_storage import write_image
from tests.helpers import make_lesson


class PicturePdfTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.engine = build_engine(f"sqlite:///{self.directory.name}/pdf.db")
        migrate_database(self.engine)
        self.settings = ImageSettings(storage_dir=Path(self.directory.name) / "images")
        self.asset_id, self.lesson_id = str(uuid4()), str(uuid4())
        raw = BytesIO()
        Image.new("RGB", (256, 256), "orange").save(raw, "PNG")
        self.path = write_image(self.asset_id, raw.getvalue(), self.settings)
        self.visual = {"activity_type": "describe_picture", "asset_id": self.asset_id, "verified": True,
                       "title": "Describe the picture", "instruction": "Answer the questions.", "scene_prompt": "",
                       "target_words": ["cup", "table"], "answer_key": [],
                       "questions": [{"question": "What color is the cup?", "answer": "It is blue.",
                                      "evidence": "The blue cup is visible on the table.", "object_names": ["cup"]}]}
        self.lesson = make_lesson().model_dump()
        image = {"asset_id": self.asset_id, "url": f"/api/lesson-images/{self.asset_id}", "description": "A blue cup.", "score": 1}
        self.lesson.update(images=[image], primary_image=image, visual_activity=self.visual)
        with Session(self.engine) as session:
            session.add(SavedLessonRecord(id=self.lesson_id, saved_at=datetime.now(UTC), favorite=False, level="A2",
                                          duration=30, theme="Cafe", mode="live", lesson=self.lesson))
            session.add(LessonImageRecord(id=self.asset_id, lesson_id=self.lesson_id, created_at=datetime.now(UTC),
                                          metadata_json={"visual_activity": self.visual}))
            session.commit()
        self.engine_patch = patch("app.pdf_service.get_engine", return_value=self.engine)
        self.settings_patch = patch("app.config.ImageSettings.from_environment", return_value=self.settings)
        self.engine_patch.start()
        self.settings_patch.start()

    def tearDown(self):
        self.settings_patch.stop()
        self.engine_patch.stop()
        self.engine.dispose()
        self.directory.cleanup()

    def build(self, teacher=False, include_images=True):
        response, options = parse_lesson_pdf_payload({"lesson": self.lesson,
            "pdf_options": {"include_answers": teacher, "include_images": include_images}})
        return build_lesson_pdf(response, options)

    def test_student_embeds_actual_image_and_hides_picture_answers(self):
        pdf = self.build()
        self.assertIn(self.path.read_bytes(), pdf)
        self.assertIn(pdf_literal("1. What color is the cup?"), pdf)
        self.assertNotIn(pdf_literal("Answer: It is blue."), pdf)
        self.assertNotIn(pdf_literal("Evidence: The blue cup is visible on the table."), pdf)

    def test_teacher_uses_answer_key_stored_with_image_version(self):
        self.lesson["visual_activity"] = dict(self.visual, questions=[{"question": "Wrong question", "answer": "Wrong answer"}])
        pdf = self.build(teacher=True)
        self.assertIn(pdf_literal("Answer: It is blue."), pdf)
        self.assertIn(pdf_literal("Evidence: The blue cup is visible on the table."), pdf)
        self.assertNotIn(b"Wrong answer", pdf)

    def test_discussion_keys_are_examples_and_student_receives_words_and_starters_only(self):
        visual = {**self.visual, "target_words": ["privacy", "surveillance"],
                  "sentence_frames": ["Surveillance may affect privacy because ____."],
                  "questions": [{"kind": "discussion", "question": "Could this camera affect privacy?",
                                 "answer": "It might affect privacy.", "evidence": "A camera is visible; opinions differ.",
                                 "object_names": ["camera"], "target_words": ["privacy"]}]}
        with Session(self.engine) as session:
            record = session.get(LessonImageRecord, self.asset_id)
            record.metadata_json = {"visual_activity": visual}
            session.commit()
        teacher = self.build(teacher=True)
        student = self.build()
        self.assertIn(pdf_literal("Example answer: It might affect privacy."), teacher)
        self.assertNotIn(pdf_literal("Example answer: It might affect privacy."), student)
        self.assertNotIn(pdf_literal("Discussion basis: A camera is visible; opinions differ."), student)
        self.assertIn(pdf_literal("Words to use: privacy, surveillance"), student)
        self.assertIn(pdf_literal("- Surveillance may affect privacy because ____."), student)

    def test_unicode_hyphens_remain_readable_in_pdf_text(self):
        self.assertEqual(pdf_literal("data\u2011privacy at the town\u2011hall"), pdf_literal("data-privacy at the town-hall"))

    def test_missing_file_unknown_uuid_and_mismatched_versions_are_rejected(self):
        self.lesson["visual_activity"]["asset_id"] = str(uuid4())
        with self.assertRaisesRegex(ValueError, "different image versions"):
            self.build()
        self.lesson["visual_activity"]["asset_id"] = self.asset_id
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.build()
        self.lesson["primary_image"]["asset_id"] = "../../.env"
        with self.assertRaises(ValueError):
            self.build()

    def test_omitting_images_does_not_read_image_file(self):
        self.path.unlink()
        pdf = self.build(include_images=False)
        self.assertNotIn(b"/DCTDecode", pdf)
        self.assertIn(pdf_literal("1. What color is the cup?"), pdf)
