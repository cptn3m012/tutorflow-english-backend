import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.pdf_service import build_lesson_pdf, parse_lesson_pdf_payload, pdf_literal
from app.schemas import PdfExportOptions, ReadingQuestionAnswer
from tests.helpers import COMPREHENSION, make_lesson


def answered_lesson():
    return make_lesson(
        reading_questions=[item["question"] for item in COMPREHENSION["items"]],
        reading_answers=[ReadingQuestionAnswer(**item) for item in COMPREHENSION["items"]],
    )


class PdfAnswerTests(unittest.TestCase):
    def test_teacher_pdf_includes_answers_and_evidence(self):
        response, options = parse_lesson_pdf_payload({
            "lesson": answered_lesson().model_dump(), "pdf_options": {"include_answers": True},
        })
        pdf = build_lesson_pdf(response, options)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn(pdf_literal("Reading answer key"), pdf)
        self.assertIn(pdf_literal("Answer: Tea and a sandwich."), pdf)
        self.assertIn(b"(Evidence:", pdf)

    def test_student_pdf_hides_reading_key_and_evidence(self):
        response, options = parse_lesson_pdf_payload({"lesson": answered_lesson().model_dump()})
        pdf = build_lesson_pdf(response, options)
        self.assertNotIn(pdf_literal("Reading answer key"), pdf)
        self.assertNotIn(b"(Answer:", pdf)
        self.assertNotIn(b"(Evidence:", pdf)

    def test_legacy_payload_without_new_answer_field_is_still_exportable(self):
        payload = make_lesson().model_dump()
        payload.pop("reading_answers")
        response, _ = parse_lesson_pdf_payload(payload)
        pdf = build_lesson_pdf(response, PdfExportOptions(include_answers=True))
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertNotIn(pdf_literal("Reading answer key"), pdf)

    def test_export_api_rejects_stale_answer_key_and_accepts_legacy_lesson(self):
        with TestClient(app) as client:
            lesson = answered_lesson()
            lesson.reading_answers[0].evidence = "A sentence absent from the reading."
            invalid = client.post("/lessons/export/pdf", json={"lesson": lesson.model_dump()})
            self.assertEqual(invalid.status_code, 400)
            self.assertIn("not quoted from the final reading", invalid.json()["detail"])
            legacy = make_lesson().model_dump()
            legacy.pop("reading_answers")
            valid = client.post("/lessons/export/pdf/preview", json={"lesson": legacy})
            self.assertEqual(valid.status_code, 200)
            self.assertEqual(valid.headers["content-type"], "application/pdf")
            self.assertIn("inline", valid.headers["content-disposition"])


if __name__ == "__main__":
    unittest.main()
