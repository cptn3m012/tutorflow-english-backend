from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import database_url


class Base(DeclarativeBase):
    pass


def build_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.drivername in ("postgres", "postgresql"):
        parsed = parsed.set(drivername="postgresql+psycopg")
    options = {"pool_pre_ping": True}
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        options["connect_args"] = {"check_same_thread": False, "timeout": 15}
    elif parsed.get_backend_name() == "postgresql":
        options["connect_args"] = {"connect_timeout": 5}
    else:
        raise ValueError("DATABASE_URL must use PostgreSQL or SQLite.")
    return create_engine(parsed, **options)


@lru_cache
def get_engine() -> Engine:
    return build_engine(database_url())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session


def migrate_database(engine: Engine | None = None) -> None:
    from alembic import command
    from alembic.config import Config
    from app.config import PROJECT_ROOT

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    with (engine or get_engine()).begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
