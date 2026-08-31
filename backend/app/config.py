from functools import lru_cache
from pathlib import Path
from typing import List, Optional

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

    # Embeddings. "local" runs a small ONNX model on this machine: no API key, no
    # quota, and fast enough to index a large PDF in seconds. "google" uses
    # gemini-embedding-001, which scores slightly higher on retrieval benchmarks
    # but is capped at 1,000 embed requests per day on the free tier -- and since
    # every chunk is one request, a single large document can exhaust it.
    embedding_provider: str = "local"  # local | google
    # bge-small measured at ~11 chunks/sec here against ~4 for bge-base, for
    # about one point of retrieval quality. Indexing speed matters more on large
    # files, so small is the default; set LOCAL_EMBEDDING_MODEL to override.
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where the downloaded model lives. fastembed defaults to a temp directory,
    # which a container wipes on restart -- so every cold start would re-download
    # 67 MB, and a host without access to Hugging Face would fail outright. The
    # Docker image bakes the model in and points this at it.
    embedding_cache_dir: Optional[Path] = None
    # Only used when embedding_provider == "google".
    embedding_model: str = "models/gemini-embedding-001"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    retrieval_fetch_k: int = 20
    history_turns: int = 6

    # Uploads
    max_upload_mb: int = 100

    # Embedding throughput, used only for a remote provider. Google's free tier
    # caps embed requests per *day* (1,000) as well as per minute, and each chunk
    # is one request, so remote indexing is paced in small batches and retried.
    embed_batch_size: int = 8
    embed_requests_per_minute: int = 60
    embed_max_retries: int = 8

    # A local model has no quota, so batches are large and nothing is paced.
    local_embed_batch_size: int = 256

    @property
    def embeddings_are_local(self) -> bool:
        return self.embedding_provider.strip().lower() == "local"

    @property
    def effective_embed_batch_size(self) -> int:
        return (
            self.local_embed_batch_size
            if self.embeddings_are_local
            else self.embed_batch_size
        )

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
