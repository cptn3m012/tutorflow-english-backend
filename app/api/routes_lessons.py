from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError
from app.schemas import CefrProfile, LessonGenerateRequest, LessonGenerateResponse
from app.lesson_service import generate_lesson_variants
from app.cefr_service import get_cefr_profile, list_cefr_profiles
from app.pdf_service import build_lesson_pdf, parse_lesson_pdf_payload

router = APIRouter(prefix="/lessons", tags=["Lessons"])


@router.get("/cefr-levels", response_model=list[CefrProfile])
async def list_cefr_levels_endpoint():
    return [CefrProfile.model_validate(profile) for profile in list_cefr_profiles()]


@router.get("/cefr-levels/{level}", response_model=CefrProfile)
async def get_cefr_level_endpoint(level: str):
    try:
        return CefrProfile.model_validate(get_cefr_profile(level))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/advanced-options")
async def get_advanced_options_endpoint():
    return {
        "reading_min_words": {"type": "number", "min": 40, "max": 400, "step": 5, "default": None},
        "reading_max_words": {"type": "number", "min": 50, "max": 500, "step": 5, "default": None},
        "target_vocabulary_count": {"type": "number", "min": 4, "max": 16, "step": 1, "default": 8},
        "reading_question_count": {"type": "number", "min": 1, "max": 8, "step": 1, "default": 3},
        "speaking_question_count": {"type": "number", "min": 2, "max": 12, "step": 1, "default": 5},
        "sentence_frame_count": {"type": "number", "min": 1, "max": 8, "step": 1, "default": 3},
        "visual_word_count": {"type": "number", "min": 3, "max": 10, "step": 1, "default": 6},
        "include_visual_activity": {"type": "boolean", "default": True},
        "include_worksheet_assets": {"type": "boolean", "default": True},
        "include_images": {"type": "boolean", "default": True},
        "skill_focus": {
            "type": "select",
            "default": "speaking",
            "options": ["speaking", "reading", "vocabulary", "grammar", "balanced"],
        },
        "lesson_style": {
            "type": "select",
            "default": "practical",
            "options": ["practical", "exam", "conversation", "story", "business", "travel"],
        },
        "extra_instructions": {"type": "text", "maxLength": 600, "default": None},
    }


@router.get("/pdf-options")
async def get_pdf_options_endpoint():
    return {
        "template": {
            "type": "select",
            "default": "clean",
            "options": ["clean", "compact", "worksheet"],
        },
        "font_size": {
            "type": "select",
            "default": "medium",
            "options": ["small", "medium", "large"],
        },
        "font_family": {
            "type": "select",
            "default": "helvetica",
            "options": ["helvetica", "times", "courier"],
        },
        "title_font_size": {"type": "number", "min": 14, "max": 36, "step": 1, "default": None},
        "heading_font_size": {"type": "number", "min": 10, "max": 24, "step": 1, "default": None},
        "body_font_size": {"type": "number", "min": 8, "max": 16, "step": 1, "default": None},
        "include_images": {"type": "boolean", "default": True},
        "include_answers": {"type": "boolean", "default": False},
        "include_visual_activity": {"type": "boolean", "default": True},
        "include_worksheet_assets": {"type": "boolean", "default": True},
        "page_size": {
            "type": "select",
            "default": "A4",
            "options": ["A4"],
        },
        "spacing": {
            "type": "select",
            "default": "normal",
            "options": ["compact", "normal", "relaxed"],
        },
        "accent_color": {"type": "rgbTriplet", "default": "0.10 0.34 0.55"},
        "custom_images": {
            "type": "array",
            "default": [],
            "item": {
                "title": "string",
                "caption": "string",
                "data_url": "JPEG data URL, e.g. data:image/jpeg;base64,...",
                "url": "optional fallback URL",
            },
        },
    }


@router.post("/generate", response_model=LessonGenerateResponse)
async def generate_lessons_endpoint(request: LessonGenerateRequest, http_request: Request):
    slots = http_request.app.state.generation_slots
    if slots.locked():
        raise HTTPException(status_code=429, detail="A lesson is already being generated. Please wait and try again.", headers={"Retry-After": "30"})
    try:
        async with slots:
            return await generate_lesson_variants(request)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def build_pdf_response(request: Request, disposition: str):
    try:
        payload = await request.json()
        response, pdf_options = parse_lesson_pdf_payload(payload)
        pdf_bytes = build_lesson_pdf(response, pdf_options)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not build PDF: {str(e)}")

    filename = f"english-lessons-{response.requested_level.lower()}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@router.post("/export/pdf")
async def export_lessons_pdf_endpoint(request: Request):
    return await build_pdf_response(request, "attachment")


@router.post("/export/pdf/preview")
async def preview_lessons_pdf_endpoint(request: Request):
    return await build_pdf_response(request, "inline")
