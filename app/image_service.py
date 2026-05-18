import json
import os

import httpx

from app.ollama_client import generate_from_ollama
from app.prompts import build_image_metadata_prompt
from app.schemas import LessonVariant, LessonImage

OPENVERSE_API_URL = os.getenv("OPENVERSE_API_URL", "https://api.openverse.org/v1/images/")
OPENVERSE_PAGE_SIZE = 3
PEXELS_API_URL = os.getenv("PEXELS_API_URL", "https://api.pexels.com/v1/search")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
IMAGE_PROVIDER_MODE = os.getenv("IMAGE_PROVIDER_MODE", "remote").lower()
IMAGE_QUERY_STOP_WORDS = {
    "classroom",
    "whiteboard",
    "worksheet",
    "students",
    "student",
    "lesson",
    "english",
    "school",
    "teaching",
}
IMAGE_RESULT_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "your",
    "have",
    "was",
    "were",
    "they",
    "their",
    "them",
}
IMAGE_NEGATIVE_HINTS = {
    "snake",
    "snakes",
    "giants",
    "monster",
    "halloween",
    "photoaday",
    "weekend",
    "candles",
    "palo",
    "santo",
}


def extract_json_from_text(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find valid JSON object in model response:\n\n{raw_text}")

    return raw_text[start:end + 1]


async def generate_image_metadata_for_lesson(lesson: LessonVariant) -> LessonVariant:
    prompt = build_image_metadata_prompt(
        lesson_title=lesson.title,
        lesson_goal=lesson.lesson_goal,
        role_play_scenario=lesson.role_play_scenario
    )

    raw_response = await generate_from_ollama(prompt)

    try:
        cleaned_json = extract_json_from_text(raw_response)
        parsed_json = json.loads(cleaned_json)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON for image metadata.\n\nRaw response:\n{raw_response}")

    image_search_query = parsed_json.get("image_search_query")
    image_keywords = parsed_json.get("image_keywords")

    if not image_search_query:
        raise ValueError(f"Missing image_search_query for lesson '{lesson.title}'.")

    if not isinstance(image_keywords, list) or len(image_keywords) != 4:
        raise ValueError(f"Lesson '{lesson.title}' does not have exactly 4 image keywords.")

    lesson.image_search_query = image_search_query
    lesson.image_keywords = image_keywords
    lesson.images = await search_images_for_lesson(lesson)

    return lesson


async def search_openverse_images(query: str) -> list[dict]:
    params = {
        "q": query,
        "page_size": OPENVERSE_PAGE_SIZE,
        "mature": "false",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(OPENVERSE_API_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    return payload.get("results", [])


async def search_pexels_images(query: str) -> list[dict]:
    if not PEXELS_API_KEY:
        return []

    params = {
        "query": query,
        "per_page": OPENVERSE_PAGE_SIZE,
        "orientation": "landscape",
    }
    headers = {
        "Authorization": PEXELS_API_KEY,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(PEXELS_API_URL, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()

    return payload.get("photos", [])


def map_openverse_image(image: dict, lesson_title: str) -> LessonImage:
    provider = image.get("source") or image.get("provider") or "openverse"
    source_url = image.get("foreign_landing_url") or image.get("url")
    title = image.get("title") or lesson_title
    image_url = image.get("thumbnail") or image.get("url")

    return LessonImage(
        url=image_url or "",
        description=f"{title} [{provider}]",
        score=1.0,
        source=provider,
        source_url=source_url,
        thumbnail_url=image.get("thumbnail"),
        creator=image.get("creator"),
        license=image.get("license"),
        license_version=image.get("license_version"),
    )


def map_pexels_image(image: dict, lesson_title: str) -> LessonImage:
    src = image.get("src", {})
    image_url = (
        src.get("large")
        or src.get("large2x")
        or src.get("landscape")
        or src.get("medium")
        or src.get("original")
    )
    thumbnail_url = src.get("medium") or src.get("small") or image_url
    title = image.get("alt") or lesson_title

    return LessonImage(
        url=image_url or "",
        description=f"{title} [pexels]",
        score=1.0,
        source="pexels",
        source_url=image.get("url"),
        thumbnail_url=thumbnail_url,
        creator=image.get("photographer"),
        license="Pexels License",
        license_version=None,
    )


def sanitize_image_query_text(text: str) -> str:
    cleaned_words = []

    for raw_word in text.split():
        normalized_word = raw_word.strip(" ,.!?;:()[]{}\"'").lower()
        if not normalized_word:
            continue
        if normalized_word in IMAGE_QUERY_STOP_WORDS:
            continue
        cleaned_words.append(raw_word.strip())

    return " ".join(cleaned_words).strip()


def tokenize_text(text: str) -> set[str]:
    tokens = set()

    for raw_word in text.lower().replace("/", " ").replace("-", " ").split():
        normalized_word = raw_word.strip(" ,.!?;:()[]{}\"'#")
        if len(normalized_word) < 3:
            continue
        if normalized_word in IMAGE_QUERY_STOP_WORDS or normalized_word in IMAGE_RESULT_STOP_WORDS:
            continue
        tokens.add(normalized_word)

    return tokens


def score_search_result_text(title: str, creator: str, source_url: str, lesson: LessonVariant, query: str) -> float:
    searchable_text = " ".join([title, creator, source_url])
    image_tokens = tokenize_text(searchable_text)
    lesson_tokens = tokenize_text(" ".join([
        lesson.title,
        lesson.theme or "",
        lesson.image_search_query or "",
        *lesson.image_keywords,
    ]))
    query_tokens = tokenize_text(query)

    overlap_with_lesson = len(image_tokens & lesson_tokens)
    overlap_with_query = len(image_tokens & query_tokens)
    hashtag_penalty = searchable_text.count("#") * 0.75
    long_title_penalty = 1.25 if len(title.split()) > 16 else 0.0
    negative_penalty = sum(1.0 for token in image_tokens if token in IMAGE_NEGATIVE_HINTS)

    return (overlap_with_lesson * 2.0) + overlap_with_query - hashtag_penalty - long_title_penalty - negative_penalty


def sanitize_image_keywords(keywords: list[str]) -> list[str]:
    sanitized = []

    for keyword in keywords:
        normalized = sanitize_image_query_text(keyword)
        if not normalized:
            continue
        sanitized.append(normalized)

    return sanitized


def build_image_search_queries(lesson: LessonVariant) -> list[str]:
    queries = []

    if lesson.image_search_query:
        queries.append(sanitize_image_query_text(lesson.image_search_query))

    sanitized_keywords = sanitize_image_keywords(lesson.image_keywords)
    if sanitized_keywords:
        queries.append(" ".join(sanitized_keywords))

    if lesson.title:
        queries.append(sanitize_image_query_text(lesson.title))

    if lesson.theme:
        queries.append(sanitize_image_query_text(lesson.theme))

    combined_query = " ".join(
        part for part in [lesson.theme or "", *sanitized_keywords[:3]] if part
    ).strip()
    if combined_query:
        queries.append(combined_query)

    unique_queries = []
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        if normalized.lower() not in [item.lower() for item in unique_queries]:
            unique_queries.append(normalized)

    return unique_queries


def score_openverse_image(image: dict, lesson: LessonVariant, query: str) -> float:
    return score_search_result_text(
        title=image.get("title", ""),
        creator=image.get("creator", ""),
        source_url=image.get("foreign_landing_url", ""),
        lesson=lesson,
        query=query,
    )


def score_pexels_image(image: dict, lesson: LessonVariant, query: str) -> float:
    base_score = score_search_result_text(
        title=image.get("alt", ""),
        creator=image.get("photographer", ""),
        source_url=image.get("url", ""),
        lesson=lesson,
        query=query,
    )
    # Prefer Pexels slightly as a visual fallback because it usually returns more directly usable stock photos.
    return base_score + 1.0


def select_best_images(results: list[dict], lesson: LessonVariant, query: str) -> list[LessonImage]:
    scored_results = []

    for image in results:
        if not image.get("url"):
            continue

        score = score_openverse_image(image, lesson, query)
        if score < 1.0:
            continue

        lesson_image = map_openverse_image(image, lesson.title)
        lesson_image.score = round(score, 2)
        scored_results.append(lesson_image)

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return scored_results[:OPENVERSE_PAGE_SIZE]


def select_best_pexels_images(results: list[dict], lesson: LessonVariant, query: str) -> list[LessonImage]:
    scored_results = []

    for image in results:
        src = image.get("src", {})
        if not src:
            continue

        score = score_pexels_image(image, lesson, query)
        if score < 1.0:
            continue

        lesson_image = map_pexels_image(image, lesson.title)
        lesson_image.score = round(score, 2)
        scored_results.append(lesson_image)

    scored_results.sort(key=lambda item: item.score, reverse=True)
    return scored_results[:OPENVERSE_PAGE_SIZE]


async def search_images_for_lesson(lesson: LessonVariant) -> list[LessonImage]:
    if IMAGE_PROVIDER_MODE in {"off", "offline", "local", "disabled"}:
        return []

    for query in build_image_search_queries(lesson):
        try:
            results = await search_openverse_images(query)
            mapped_results = select_best_images(results, lesson, query)

            if mapped_results:
                lesson.image_search_query = query
                return mapped_results
        except Exception:
            continue

    for query in build_image_search_queries(lesson):
        try:
            results = await search_pexels_images(query)
            mapped_results = select_best_pexels_images(results, lesson, query)

            if mapped_results:
                lesson.image_search_query = query
                return mapped_results
        except Exception:
            continue

    return []


async def attach_images_to_lessons(lessons: list[LessonVariant]) -> list[LessonVariant]:
    enriched_lessons = []

    for lesson in lessons:
        enriched_lesson = await generate_image_metadata_for_lesson(lesson)
        enriched_lessons.append(enriched_lesson)

    return enriched_lessons
