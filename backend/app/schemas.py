from datetime import datetime, timezone
from typing import Annotated, List, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field

from .security import MAX_PASSWORD_BYTES


def _as_utc(value: datetime) -> datetime:
    """
    SQLite drops the offset even for DateTime(timezone=True), so timestamps come
    back naive and a browser would read them as local time -- an answer written
    at 09:00 UTC showing as 09:00 in a UTC+5 timezone. Everything is stored in
    UTC, so stamp that back on before serialising.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- auth ---


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)


class UserOut(ORMModel):
    id: int
    email: EmailStr
    created_at: UtcDatetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProviderOut(BaseModel):
    """An OAuth provider that has credentials configured on this server."""

    id: str
    label: str


# --- collections ---


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CollectionOut(ORMModel):
    id: int
    name: str
    created_at: UtcDatetime
    document_count: int = 0
    ready_count: int = 0


# --- documents ---


class DocumentOut(ORMModel):
    id: int
    collection_id: int
    filename: str
    content_type: str
    size_bytes: int
    pages: int
    chunk_count: int
    chunks_embedded: int = 0
    status: str
    error: Optional[str] = None
    created_at: UtcDatetime


# --- chats ---


class ChatCreate(BaseModel):
    collection_id: int
    title: Optional[str] = Field(default=None, max_length=200)
    model: Optional[str] = Field(default=None, max_length=80)
    web_search: bool = False


class ChatUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    model: Optional[str] = Field(default=None, max_length=80)
    web_search: Optional[bool] = None


class ChatOut(ORMModel):
    id: int
    collection_id: int
    title: str
    model: str
    web_search: bool = False
    created_at: UtcDatetime
    updated_at: UtcDatetime


class Source(BaseModel):
    document_id: Optional[int] = None
    filename: str
    page: Optional[int] = None
    snippet: str = ""
    # Set for web results; None for passages from an uploaded document.
    url: Optional[str] = None


class MessageOut(ORMModel):
    id: int
    role: str
    content: str
    sources: List[Source] = []
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    duration_ms: Optional[int] = None
    created_at: UtcDatetime


class SuggestionsOut(BaseModel):
    """Follow-up questions offered under the composer; empty when none fit."""

    suggestions: List[str] = []


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


# --- meta ---


class ModelInfo(BaseModel):
    id: str
    label: str
    # Whether the provider can search the web for this model. The UI disables
    # the web-search control for models where this is false.
    supports_web_search: bool = False


class HealthOut(BaseModel):
    status: str
    google_key_configured: bool
    groq_key_configured: bool
    models: List[ModelInfo]
