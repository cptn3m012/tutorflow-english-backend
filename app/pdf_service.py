from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any
from textwrap import wrap

from app.reading_service import validate_reading_comprehension
from app.schemas import (
    CefrProfile,
    LessonGenerateResponse,
    LessonVariant,
    LessonWorksheetAsset,
    PdfExportOptions,
)


A4_WIDTH = 595
A4_HEIGHT = 842
MARGIN_X = 54
MARGIN_TOP = 54
MARGIN_BOTTOM = 54
FONT_REGULAR = "F1"
FONT_BOLD = "F2"
TEXT_COLOR = "0.12 0.15 0.18"
MUTED_COLOR = "0.38 0.43 0.48"
RULE_COLOR = "0.78 0.82 0.86"
ACCENT_COLOR = "0.10 0.34 0.55"


FONT_SIZE_SCALE = {
    "small": 0.9,
    "medium": 1.0,
    "large": 1.15,
}

SPACING_SCALE = {
    "compact": 0.72,
    "normal": 1.0,
    "relaxed": 1.3,
}

TEMPLATE_SPACING = {
    "clean": 1.0,
    "compact": 0.75,
    "worksheet": 1.18,
}

FONT_FAMILIES = {
    "helvetica": ("Helvetica", "Helvetica-Bold"),
    "times": ("Times-Roman", "Times-Bold"),
    "courier": ("Courier", "Courier-Bold"),
}

DATA_URL_RE = re.compile(r"^data:image/(?P<kind>jpeg|jpg);base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)


def parse_lesson_pdf_payload(payload: Any) -> tuple[LessonGenerateResponse, PdfExportOptions]:
    payload = unwrap_payload(payload)
    options = PdfExportOptions()

    if isinstance(payload, dict) and "pdf_options" in payload:
        options = PdfExportOptions.model_validate(payload.get("pdf_options") or {})

    response = coerce_lesson_pdf_payload(payload)
    for lesson in response.lessons:
        if lesson.reading_answers:
            validate_reading_comprehension(lesson, len(lesson.reading_questions))
    return response, options


def coerce_lesson_pdf_payload(payload: Any) -> LessonGenerateResponse:
    payload = unwrap_payload(payload)

    if isinstance(payload, dict) and "lessons" in payload and "requested_level" in payload and "cefr_profile" in payload:
        return LessonGenerateResponse.model_validate(payload)

    if isinstance(payload, dict) and "lessons" in payload:
        lessons_payload = payload["lessons"]
        metadata = payload
    elif isinstance(payload, dict) and "lesson" in payload:
        lessons_payload = [payload["lesson"]]
        metadata = payload
    elif isinstance(payload, list):
        lessons_payload = payload
        metadata = {}
    elif isinstance(payload, dict):
        lessons_payload = [payload]
        metadata = payload
    else:
        raise ValueError("PDF export payload must be a lesson, a lessons array, or a generated lesson response.")

    lessons = [LessonVariant.model_validate(item) for item in lessons_payload]
    if not lessons:
        raise ValueError("PDF export payload does not contain any lessons.")

    first_lesson = lessons[0]
    level = (
        metadata.get("requested_level")
        or metadata.get("level")
        or first_lesson.level
        or "Unknown"
    )
    theme = (
        metadata.get("lesson_theme")
        or metadata.get("theme")
        or first_lesson.theme
        or first_lesson.title
    )
    duration = (
        metadata.get("duration_minutes")
        or metadata.get("duration")
        or first_lesson.duration_minutes
        or 0
    )

    cefr_profile_payload = metadata.get("cefr_profile")
    if cefr_profile_payload:
        cefr_profile = CefrProfile.model_validate(cefr_profile_payload)
    else:
        cefr_profile = CefrProfile(
            level=str(level),
            label=f"{level} lesson pack",
            study_hours="",
            source_name="Generated lesson data",
            source_url="",
            can_do=[],
            communication_targets=[],
            vocabulary_targets=[],
            grammar_targets=[],
        )

    return LessonGenerateResponse(
        requested_level=str(level),
        lesson_theme=str(theme),
        duration_minutes=int(duration),
        source_type=str(metadata.get("source_type") or "export"),
        total_lessons=len(lessons),
        generated_at=str(metadata.get("generated_at") or datetime.now(UTC).isoformat()),
        cefr_profile=cefr_profile,
        lessons=lessons,
    )


def unwrap_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    for key in ("data", "result", "response"):
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            return unwrap_payload(value)

    return payload


@dataclass
class PdfPage:
    commands: list[bytes] = field(default_factory=list)
    image_names: list[str] = field(default_factory=list)


@dataclass
class PdfEmbeddedImage:
    name: str
    data: bytes
    width: int
    height: int


class PdfDocument:
    def __init__(self, options: PdfExportOptions):
        self.options = options
        self.page_width = A4_WIDTH
        self.page_height = A4_HEIGHT
        self.margin_x = 42 if options.template == "compact" or options.spacing == "compact" else MARGIN_X
        self.margin_top = 42 if options.template == "compact" or options.spacing == "compact" else MARGIN_TOP
        self.margin_bottom = 42 if options.template == "compact" or options.spacing == "compact" else MARGIN_BOTTOM
        self.font_scale = FONT_SIZE_SCALE[options.font_size]
        self.spacing_scale = SPACING_SCALE[options.spacing] * TEMPLATE_SPACING[options.template]
        self.font_regular_base, self.font_bold_base = FONT_FAMILIES[options.font_family]
        self.accent_color = sanitize_pdf_color(options.accent_color, ACCENT_COLOR)
        self.embedded_images: list[PdfEmbeddedImage] = []
        self.pages: list[PdfPage] = [PdfPage()]
        self.y = self.page_height - self.margin_top

    @property
    def page(self) -> PdfPage:
        return self.pages[-1]

    def add_page(self):
        self.pages.append(PdfPage())
        self.y = self.page_height - self.margin_top

    def ensure_space(self, height: int):
        if self.y - height < self.margin_bottom:
            self.add_page()

    def text(
        self,
        value: str,
        size: int = 11,
        bold: bool = False,
        indent: int = 0,
        gap: int = 4,
        color: str = TEXT_COLOR,
        max_width: int | None = None,
    ):
        if not value:
            return

        size = self.scale_font(size)
        line_height = max(int((size + 4) * self.spacing_scale), size + 2, 10)
        gap = self.scale_gap(gap)
        width = max_width or (self.page_width - (self.margin_x * 2) - indent)
        max_chars = max(18, int(width / (size * 0.48)))
        lines = wrap(normalize_text(value), width=max_chars, break_long_words=False) or [""]

        self.ensure_space(line_height * len(lines) + gap)

        font = FONT_BOLD if bold else FONT_REGULAR
        x = self.margin_x + indent
        for line in lines:
            self.page.commands.append(
                b"BT /"
                + font.encode("ascii")
                + b" "
                + str(size).encode("ascii")
                + b" Tf "
                + color.encode("ascii")
                + b" rg 1 0 0 1 "
                + str(x).encode("ascii")
                + b" "
                + str(self.y).encode("ascii")
                + b" Tm "
                + pdf_literal(line)
                + b" Tj ET\n"
            )
            self.y -= line_height
        self.y -= gap

    def spacer(self, height: int = 8):
        height = self.scale_gap(height)
        self.ensure_space(height)
        self.y -= height

    def rule(self, gap_before: int = 4, gap_after: int = 10):
        self.spacer(gap_before + gap_after)

    def badge(self, label: str, value: str, x: int, y: int, width: int):
        write_text(self.page, label.upper(), x, y - 6, size=self.scale_font(7), color=MUTED_COLOR, bold=True)
        write_text(
            self.page,
            value,
            x,
            y - 24,
            size=self.scale_font(11),
            color=TEXT_COLOR,
            bold=True,
            max_width=width,
        )

    def column_list(self, items: list[str], columns: int = 2):
        if not items:
            return

        col_width = int((self.page_width - (self.margin_x * 2) - 18) / columns)
        row_height = self.scale_gap(18)
        rows = (len(items) + columns - 1) // columns
        self.ensure_space(rows * row_height + 8)

        start_y = self.y
        for index, item in enumerate(items):
            col = index // rows
            row = index % rows
            x = self.margin_x + col * (col_width + 18)
            y = start_y - row * row_height
            write_text(self.page, f"- {item}", x, y, size=self.scale_font(10), color=TEXT_COLOR, max_width=col_width)
        self.y -= rows * row_height + 8

    def answer_space(self, lines: int = 2):
        if self.options.template != "worksheet":
            return

        for _ in range(lines):
            self.spacer(18)

    def scale_font(self, size: int) -> int:
        if self.options.body_font_size and size <= 11:
            return self.options.body_font_size
        if self.options.heading_font_size and 12 <= size <= 16:
            return self.options.heading_font_size
        if self.options.title_font_size and size >= 20:
            return self.options.title_font_size
        return max(7, round(size * self.font_scale))

    def scale_gap(self, value: int) -> int:
        return max(1, round(value * self.spacing_scale))

    def add_custom_images(self):
        if not self.options.include_images or not self.options.custom_images:
            return

        add_section(self, 99, "Custom images")
        for index, image in enumerate(self.options.custom_images, start=1):
            title = str(image.get("title") or f"Image {index}")
            caption = str(image.get("caption") or "")
            data_url = image.get("data_url")

            self.text(title, size=11, bold=True, indent=8, gap=3)
            if isinstance(data_url, str):
                if self.draw_jpeg_data_url(data_url):
                    if caption:
                        self.text(caption, size=9, indent=8, color=MUTED_COLOR)
                    continue

            url = str(image.get("url") or "")
            if caption:
                self.text(caption, size=10, indent=8)
            if url:
                self.text(url, size=9, indent=8, color=MUTED_COLOR)

    def draw_jpeg_data_url(self, data_url: str, max_height: int = 220) -> bool:
        match = DATA_URL_RE.match(data_url.strip())
        if not match:
            return False

        try:
            image_data = base64.b64decode(match.group("data"), validate=True)
            image_width, image_height = parse_jpeg_size(image_data)
        except (ValueError, TypeError):
            return False

        available_width = self.page_width - (self.margin_x * 2) - 16
        scale = min(available_width / image_width, max_height / image_height, 1.0)
        draw_width = int(image_width * scale)
        draw_height = int(image_height * scale)

        self.ensure_space(draw_height + self.scale_gap(16))
        name = f"Im{len(self.embedded_images) + 1}"
        self.embedded_images.append(PdfEmbeddedImage(name, image_data, image_width, image_height))
        if name not in self.page.image_names:
            self.page.image_names.append(name)

        x = self.margin_x + 8
        y = self.y - draw_height
        self.page.commands.append(
            f"q {draw_width} 0 0 {draw_height} {x} {y} cm /{name} Do Q\n".encode("ascii")
        )
        self.y = y - self.scale_gap(12)
        return True


def write_text(
    page: PdfPage,
    value: str,
    x: int,
    y: int,
    size: int = 10,
    color: str = TEXT_COLOR,
    bold: bool = False,
    max_width: int | None = None,
):
    font = FONT_BOLD if bold else FONT_REGULAR
    text = normalize_text(value)
    if max_width:
        max_chars = max(8, int(max_width / (size * 0.48)))
        text = wrap(text, width=max_chars, break_long_words=False)[0] if text else ""
    page.commands.append(
        b"BT /"
        + font.encode("ascii")
        + b" "
        + str(size).encode("ascii")
        + b" Tf "
        + color.encode("ascii")
        + b" rg 1 0 0 1 "
        + str(x).encode("ascii")
        + b" "
        + str(y).encode("ascii")
        + b" Tm "
        + pdf_literal(text)
        + b" Tj ET\n"
    )


def sanitize_pdf_color(value: str, fallback: str) -> str:
    parts = str(value).split()
    if len(parts) != 3:
        return fallback

    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return fallback

    if any(number < 0 or number > 1 for number in numbers):
        return fallback

    return " ".join(f"{number:.2f}" for number in numbers)


def parse_jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("Only JPEG data URLs can be embedded in the PDF.")

    index = 2
    while index < len(data):
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break

        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9):
            continue
        if index + 2 > len(data):
            break

        segment_length = int.from_bytes(data[index:index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            break

        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            segment_start = index + 2
            height = int.from_bytes(data[segment_start + 1:segment_start + 3], "big")
            width = int.from_bytes(data[segment_start + 3:segment_start + 5], "big")
            if width > 0 and height > 0:
                return width, height
            break

        index += segment_length

    raise ValueError("Could not read JPEG dimensions.")


def normalize_text(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def pdf_literal(value: str) -> bytes:
    encoded = normalize_text(value).encode("cp1252", errors="replace")
    result = bytearray(b"(")
    for byte in encoded:
        if byte in (40, 41, 92):
            result.extend(b"\\")
            result.append(byte)
        elif byte < 32 or byte > 126:
            result.extend(f"\\{byte:03o}".encode("ascii"))
        else:
            result.append(byte)
    result.extend(b")")
    return bytes(result)


def add_bullets(doc: PdfDocument, items: list[str], size: int = 10, answer_lines: bool = False):
    for item in items:
        doc.text(f"- {item}", size=size, indent=12, gap=2)
        if answer_lines:
            doc.answer_space(1)


def add_section(doc: PdfDocument, number: int, title: str):
    doc.ensure_space(46)
    doc.spacer(12)
    section_title = title if number == 99 else title
    doc.text(section_title, size=13, bold=True, gap=8, color=doc.accent_color)


def add_worksheet_assets(doc: PdfDocument, assets: list[LessonWorksheetAsset], number: int):
    if not assets or not doc.options.include_worksheet_assets:
        return

    add_section(doc, number, "Printable activities")
    for asset in assets:
        doc.ensure_space(70)
        doc.text(asset.title, size=11, bold=True, indent=8, gap=2, color=TEXT_COLOR)
        doc.text(asset.instruction, size=10, indent=16, gap=2)
        if asset.word_bank:
            doc.text("Word bank: " + ", ".join(asset.word_bank), size=10, indent=16, gap=2, color=MUTED_COLOR)
        if asset.lines:
            add_bullets(doc, asset.lines, size=10, answer_lines=True)
        if asset.answer_key and doc.options.include_answers:
            doc.text("Answer key: " + ", ".join(asset.answer_key), size=10, indent=16, gap=8, color=MUTED_COLOR)


def add_reading_answer_key(doc: PdfDocument, lesson: LessonVariant):
    if not doc.options.include_answers or not lesson.reading_answers:
        return
    add_section(doc, 99, "Reading answer key")
    for index, item in enumerate(lesson.reading_answers, start=1):
        doc.text(f"{index}. {item.question}", size=10, bold=True, indent=8, gap=2)
        doc.text(f"Answer: {item.answer}", size=10, indent=16, gap=2)
        doc.text(f"Evidence: {item.evidence}", size=9, indent=16, gap=8, color=MUTED_COLOR)


def add_lesson(doc: PdfDocument, lesson: LessonVariant, index: int):
    doc.text(lesson.title, size=24, bold=True, gap=7)
    doc.text("English lesson", size=11, bold=True, gap=10, color=doc.accent_color)

    metadata = [
        ("Level", lesson.level or ""),
        ("Time", f"{lesson.duration_minutes} minutes" if lesson.duration_minutes else ""),
        ("Theme", lesson.theme or ""),
    ]
    badge_y = doc.y
    badge_width = 154
    for badge_index, (label, value) in enumerate(metadata):
        if value:
            doc.badge(label, value, doc.margin_x + badge_index * (badge_width + 14), badge_y, badge_width)
    doc.y -= doc.scale_gap(54)

    add_section(doc, 1, "Lesson goal")
    doc.text(lesson.lesson_goal, size=11, indent=8, gap=6)

    if lesson.cefr_focus:
        doc.text("Focus: " + " · ".join(lesson.cefr_focus), size=9, indent=8, color=MUTED_COLOR, gap=4)

    add_section(doc, 2, "Vocabulary")
    doc.column_list(lesson.target_vocabulary, columns=2)

    add_section(doc, 3, "Reading")
    doc.text(lesson.reading_text, size=10, indent=8)

    add_section(doc, 4, "Comprehension questions")
    add_bullets(doc, lesson.reading_questions, answer_lines=True)

    add_section(doc, 5, "Speaking practice")
    add_bullets(doc, lesson.speaking_questions, answer_lines=True)
    doc.spacer(4)

    add_section(doc, 6, "Pair work and role play")
    doc.text(f"Pair work: {lesson.pair_work_task}", size=10, indent=8, color=MUTED_COLOR)
    doc.answer_space(2)
    doc.text(f"Role play: {lesson.role_play_scenario}", size=10, indent=8, color=MUTED_COLOR)
    doc.answer_space(2)

    add_section(doc, 7, "Useful language")
    add_bullets(doc, lesson.sentence_frames)

    if lesson.visual_activity and doc.options.include_visual_activity:
        add_section(doc, 8, "Visual task")
        doc.text(lesson.visual_activity.title, size=11, bold=True, indent=8)
        doc.text(lesson.visual_activity.instruction, size=10, indent=8)
        doc.text("Target words: " + ", ".join(lesson.visual_activity.target_words), size=10, indent=8, color=MUTED_COLOR)
        if lesson.visual_activity.answer_key and doc.options.include_answers:
            doc.text(
                "Answer key: " + ", ".join(lesson.visual_activity.answer_key),
                size=10,
                indent=8,
                color=MUTED_COLOR,
            )
        doc.answer_space(3)
        worksheet_number = 9
    else:
        worksheet_number = 8

    add_worksheet_assets(doc, lesson.worksheet_assets, worksheet_number)
    add_reading_answer_key(doc, lesson)

    lesson_image = lesson.primary_image or (lesson.images[0] if lesson.images else None)
    if doc.options.include_images and lesson_image:
        add_section(doc, worksheet_number + 1 if doc.options.include_worksheet_assets else worksheet_number, "Image")
        doc.text(lesson_image.description, size=10, indent=8)
        doc.text(lesson_image.url, size=9, indent=8, color=MUTED_COLOR)


def build_lesson_pdf(response: LessonGenerateResponse, options: PdfExportOptions | None = None) -> bytes:
    options = options or PdfExportOptions()
    doc = PdfDocument(options)

    if len(response.lessons) > 1:
        title_size = 21 if options.template == "compact" else 24
        doc.text("English Lesson Pack", size=title_size, bold=True, gap=8)
        doc.text(response.lesson_theme, size=15, color=doc.accent_color, bold=True, gap=18)

        badge_y = doc.y
        badge_width = int((doc.page_width - (doc.margin_x * 2) - 36) / 3)
        doc.badge("Level", response.requested_level, doc.margin_x, badge_y, badge_width)
        doc.badge(
            "Duration",
            f"{response.duration_minutes} minutes",
            doc.margin_x + badge_width + 18,
            badge_y,
            badge_width,
        )
        doc.badge("Lessons", str(response.total_lessons), doc.margin_x + (badge_width + 18) * 2, badge_y, badge_width)
        doc.y -= doc.scale_gap(54)
        doc.text("Included lessons", size=14, bold=True, color=TEXT_COLOR, gap=6)
        for lesson in response.lessons:
            doc.text(f"- {lesson.title}", size=10, indent=8, gap=2)

    for index, lesson in enumerate(response.lessons, start=1):
        if index > 1 or len(response.lessons) > 1:
            doc.add_page()
        add_lesson(doc, lesson, index)

    doc.add_custom_images()

    return render_pdf(
        doc.pages,
        doc.page_width,
        doc.page_height,
        doc.embedded_images,
        doc.font_regular_base,
        doc.font_bold_base,
    )


def render_pdf(
    pages: list[PdfPage],
    page_width: int = A4_WIDTH,
    page_height: int = A4_HEIGHT,
    images: list[PdfEmbeddedImage] | None = None,
    font_regular_base: str = "Helvetica",
    font_bold_base: str = "Helvetica-Bold",
) -> bytes:
    images = images or []
    objects: list[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"")
    font_regular_id = add_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /"
        + font_regular_base.encode("ascii")
        + b" /Encoding /WinAnsiEncoding >>"
    )
    font_bold_id = add_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /"
        + font_bold_base.encode("ascii")
        + b" /Encoding /WinAnsiEncoding >>"
    )

    image_object_ids = {}
    for image in images:
        image_object_ids[image.name] = add_object(
            b"<< /Type /XObject /Subtype /Image /Width "
            + str(image.width).encode("ascii")
            + b" /Height "
            + str(image.height).encode("ascii")
            + b" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length "
            + str(len(image.data)).encode("ascii")
            + b" >>\nstream\n"
            + image.data
            + b"\nendstream"
        )

    page_ids: list[int] = []
    for page in pages:
        stream = b"".join(page.commands)
        content_id = add_object(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        )
        xobjects = b" ".join(
            b"/" + name.encode("ascii") + b" " + str(image_object_ids[name]).encode("ascii") + b" 0 R"
            for name in page.image_names
            if name in image_object_ids
        )
        xobject_resources = b" /XObject << " + xobjects + b" >>" if xobjects else b""

        page_id = add_object(
            b"<< /Type /Page /Parent "
            + str(pages_id).encode("ascii")
            + b" 0 R /MediaBox [0 0 "
            + str(page_width).encode("ascii")
            + b" "
            + str(page_height).encode("ascii")
            + b"] /Resources << /Font << /F1 "
            + str(font_regular_id).encode("ascii")
            + b" 0 R /F2 "
            + str(font_bold_id).encode("ascii")
            + b" 0 R >>"
            + xobject_resources
            + b" >> /Contents "
            + str(content_id).encode("ascii")
            + b" 0 R >>"
        )
        page_ids.append(page_id)

    kids = b" ".join(str(page_id).encode("ascii") + b" 0 R" for page_id in page_ids)
    objects[pages_id - 1] = (
        b"<< /Type /Pages /Kids ["
        + kids
        + b"] /Count "
        + str(len(page_ids)).encode("ascii")
        + b" >>"
    )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, content in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(str(object_id).encode("ascii") + b" 0 obj\n")
        pdf.extend(content)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n")
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root "
        + str(catalog_id).encode("ascii")
        + b" 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )

    return bytes(pdf)
