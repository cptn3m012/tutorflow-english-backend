import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def database_url() -> str:
    # The file database makes local development usable without a running server.
    # An explicitly configured PostgreSQL URL never falls back to SQLite.
    return os.getenv("DATABASE_URL") or f"sqlite:///{(PROJECT_ROOT / 'data' / 'tutorflow.db').as_posix()}"


@dataclass(frozen=True)
class OllamaSettings:
    model: str = "gpt-oss:20b"
    base_url: str = "http://127.0.0.1:11434"
    context_tokens: int = 12288
    timeout_seconds: float = 180.0
    keep_alive: str = "10m"
    reasoning_effort: str = "low"
    review_reasoning_effort: str = "medium"

    @classmethod
    def from_environment(cls):
        settings = cls(
            model=os.getenv("OLLAMA_MODEL", cls.model).strip(),
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            context_tokens=int(os.getenv("OLLAMA_NUM_CTX", str(cls.context_tokens))),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(cls.timeout_seconds))),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", cls.keep_alive),
            reasoning_effort=os.getenv("OLLAMA_REASONING_EFFORT", cls.reasoning_effort),
            review_reasoning_effort=os.getenv("OLLAMA_REVIEW_REASONING_EFFORT", cls.review_reasoning_effort),
        )
        settings.validate()
        return settings

    def validate(self):
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment
        ):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) base URL without credentials or query parameters.")
        if not self.model:
            raise ValueError("OLLAMA_MODEL must not be empty.")
        if not 8192 <= self.context_tokens <= 131072:
            raise ValueError("OLLAMA_NUM_CTX must be between 8192 and 131072.")
        if not 1 <= self.timeout_seconds <= 600:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be between 1 and 600.")
        if any(value not in ("low", "medium", "high") for value in (
            self.reasoning_effort, self.review_reasoning_effort,
        )):
            raise ValueError("Ollama reasoning effort must be low, medium or high.")


@dataclass(frozen=True)
class ImageSettings:
    enabled: bool = False
    comfy_url: str = "http://127.0.0.1:8188"
    model: str = "flux2-klein-4b"
    vision_model: str = "qwen3-vl:4b-instruct"
    planner_model: str = "qwen3:14b"
    storage_dir: Path = PROJECT_ROOT / "data" / "lesson-images"
    timeout_seconds: float = 300.0
    managed_runtime: bool = False

    @classmethod
    def from_environment(cls):
        settings = cls(
            enabled=os.getenv("LOCAL_IMAGES_ENABLED", "false").lower() == "true",
            comfy_url=os.getenv("COMFYUI_BASE_URL", cls.comfy_url).rstrip("/"),
            model=os.getenv("IMAGE_MODEL", cls.model),
            vision_model=os.getenv("IMAGE_VISION_MODEL", cls.vision_model),
            planner_model=os.getenv("IMAGE_PLANNER_MODEL", cls.planner_model),
            storage_dir=Path(os.getenv("IMAGE_STORAGE_DIR", str(cls.storage_dir))).resolve(),
            timeout_seconds=float(os.getenv("IMAGE_TIMEOUT_SECONDS", str(cls.timeout_seconds))),
            managed_runtime=os.getenv("COMFYUI_MANAGED_RUNTIME", "false").lower() == "true",
        )
        settings.validate()
        return settings

    def validate(self):
        parsed = urlsplit(self.comfy_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("COMFYUI_BASE_URL must be a loopback HTTP origin.")
        if self.model not in ("flux2-klein-4b", "z-image-turbo"):
            raise ValueError("IMAGE_MODEL must be flux2-klein-4b or z-image-turbo.")
        if not self.vision_model.strip() or not self.planner_model.strip() or not 30 <= self.timeout_seconds <= 600:
            raise ValueError("Image generation requires a vision model and a timeout between 30 and 600 seconds.")
        if self.enabled:
            ollama_url = urlsplit(OllamaSettings.from_environment().base_url)
            if ollama_url.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError("Local image processing requires a loopback Ollama URL.")


@dataclass(frozen=True)
class AppSettings:
    public_share: bool = False
    share_username: str = ""
    share_password: str = ""
    frontend_dist: Path | None = None
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:4173", "http://localhost:4173",
    )

    @classmethod
    def from_environment(cls):
        dist = os.getenv("FRONTEND_DIST_DIR")
        origins = os.getenv("CORS_ORIGINS")
        return cls(
            public_share=os.getenv("PUBLIC_SHARE_MODE", "false").lower() == "true",
            share_username=os.getenv("SHARE_USERNAME", ""),
            share_password=os.getenv("SHARE_PASSWORD", ""),
            frontend_dist=Path(dist).resolve() if dist else None,
            cors_origins=tuple(value.strip() for value in origins.split(",") if value.strip()) if origins is not None else cls.cors_origins,
        )

    def validate(self):
        if self.public_share and (
            not self.share_username or ":" in self.share_username
            or len(self.share_password) < 16
        ):
            raise ValueError("Public sharing requires SHARE_USERNAME and SHARE_PASSWORD (at least 16 characters).")
        if self.frontend_dist and not (self.frontend_dist / "index.html").is_file():
            raise ValueError("FRONTEND_DIST_DIR must contain a built frontend index.html.")
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in ("http", "https") or not parsed.netloc
                or parsed.username or parsed.password or "*" in origin
                or parsed.path or parsed.query or parsed.fragment
            ):
                raise ValueError("CORS_ORIGINS must contain exact HTTP(S) origins, without paths or wildcards.")
