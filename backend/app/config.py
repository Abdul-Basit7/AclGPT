from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration, read from environment or a .env file."""

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Sourcery"

    # Auth
    secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    # Where the browser app lives, for post-OAuth redirects.
    frontend_url: str = "http://localhost:5173"

    # OAuth apps. Empty credentials mean the provider is simply not offered.
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    github_oauth_client_id: str = ""
    github_oauth_client_secret: str = ""
    oauth_state_ttl_seconds: int = 600

    # Provider keys (server side: the deployment owns them, not the end user)
    google_api_key: str = ""
    groq_api_key: str = ""

    # Storage
    data_dir: Path = BASE_DIR / "data"

    # CORS (the Vite dev server)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # RAG tuning
    embedding_model: str = "models/gemini-embedding-001"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    retrieval_fetch_k: int = 20
    history_turns: int = 6

    # Uploads
    max_upload_mb: int = 100

    # Embedding throughput. Google's free tier allows ~100 embed requests per
    # minute, and a large PDF is thousands of chunks, so indexing is paced and
    # retried rather than fired all at once.
    # Measured against the Google free tier: batches of 8 succeed in ~1s, while
    # 16 is rejected instantly with a 429. Batch size -- not throughput -- is the
    # binding constraint, so keep batches small and pace requests under the
    # documented ~100/minute quota.
    embed_batch_size: int = 8
    embed_requests_per_minute: int = 60
    embed_max_retries: int = 8

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'sourcery.db'}"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.upload_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
