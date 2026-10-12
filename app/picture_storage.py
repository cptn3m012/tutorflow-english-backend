"""Private image files; callers supply UUIDs, never file paths or remote URLs."""
import base64
from io import BytesIO
from pathlib import Path
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from app.config import ImageSettings

MAX_IMAGE_BYTES = 12 * 1024 * 1024


def image_path(asset_id: str | UUID, settings: ImageSettings | None = None) -> Path:
    settings = settings or ImageSettings.from_environment()
    return settings.storage_dir / f"{UUID(str(asset_id))}.jpg"


def normalize_image(content: bytes) -> bytes:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Generated image is empty or too large.")
    try:
        with Image.open(BytesIO(content)) as source:
            if source.format not in ("PNG", "JPEG", "WEBP") or not 128 <= min(source.size) or max(source.size) > 2048:
                raise ValueError("Generated image has unsupported dimensions or format.")
            source.load()
            image = source.convert("RGB")
            image.thumbnail((1024, 1024))
            output = BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Generated image could not be decoded.") from error


def write_image(asset_id: str | UUID, content: bytes, settings: ImageSettings | None = None) -> Path:
    path = image_path(asset_id, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(normalize_image(content))
    temp.replace(path)
    return path


def image_data_url(asset_id: str | UUID, settings: ImageSettings | None = None) -> str:
    path = image_path(asset_id, settings)
    if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Lesson picture is unavailable. Generate it again before exporting.")
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
