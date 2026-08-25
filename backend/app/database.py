import logging
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

settings.ensure_dirs()

engine = create_engine(
    settings.database_url,
    # SQLite + FastAPI: connections are handed between threadpool workers.
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


INITIAL_REVISION = "0001_initial"


def _alembic_config():
    from alembic.config import Config

    backend_dir = Path(__file__).resolve().parent.parent
    config = Config(str(backend_dir / "alembic.ini"))
    config.set_main_option("script_location", str(backend_dir / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    return config


def init_db() -> None:
    """Bring the database to the latest revision.

    A database created by the pre-Alembic `create_all` path has the tables but no
    version table. Stamping it at the initial revision first means those installs
    upgrade in place instead of colliding with "table already exists".
    """
    from alembic import command

    from . import models  # noqa: F401  -- register mappers before autogenerate

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    config = _alembic_config()

    if "users" in tables and "alembic_version" not in tables:
        logger.info("Existing pre-Alembic database found; stamping %s", INITIAL_REVISION)
        command.stamp(config, INITIAL_REVISION)

    command.upgrade(config, "head")
