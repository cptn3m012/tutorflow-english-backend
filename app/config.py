import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def database_url() -> str:
    # The file database makes local development usable without a running server.
    # An explicitly configured PostgreSQL URL never falls back to SQLite.
    return os.getenv("DATABASE_URL") or f"sqlite:///{(PROJECT_ROOT / 'data' / 'tutorflow.db').as_posix()}"
