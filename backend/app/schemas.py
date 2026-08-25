from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .security import MAX_PASSWORD_BYTES


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- auth ---


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)


class UserOut(ORMModel):
    id: int
    email: EmailStr
    created_at: datetime


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
    created_at: datetime
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
    created_at: datetime


# --- chats ---


class ChatCreate(BaseModel):
    collection_id: int
    title: Optional[str] = Field(default=None, max_length=200)
    model: Optional[str] = Field(default=None, max_length=80)


class ChatUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    model: Optional[str] = Field(default=None, max_length=80)


class ChatOut(ORMModel):
    id: int
    collection_id: int
    title: str
    model: str
    created_at: datetime
    updated_at: datetime


class Source(BaseModel):
    document_id: Optional[int] = None
    filename: str
    page: Optional[int] = None
    snippet: str = ""


class MessageOut(ORMModel):
    id: int
    role: str
    content: str
    sources: List[Source] = []
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    created_at: datetime


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


# --- meta ---


class ModelInfo(BaseModel):
    id: str
    label: str


class HealthOut(BaseModel):
    status: str
    google_key_configured: bool
    groq_key_configured: bool
    models: List[ModelInfo]
