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
