"""Groq chat models.

Model ids change frequently and differ per account, so the list is discovered from
the Groq API at runtime and cached. A static fallback keeps the app usable when
discovery is unavailable (no key, offline, or a transient API error).
"""

import logging
import re
import threading
import time
from typing import List, Optional

from langchain_groq import ChatGroq

from ..config import settings
from ..schemas import ModelInfo

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 600

# Used when the API cannot be reached. Kept short and general on purpose.
FALLBACK_MODELS: List[ModelInfo] = [
    ModelInfo(id="openai/gpt-oss-120b", label="GPT OSS 120B"),
    ModelInfo(id="openai/gpt-oss-20b", label="GPT OSS 20B"),
    ModelInfo(id="groq/compound", label="Compound"),
]

# Speech, moderation and embedding models cannot answer chat requests.
_EXCLUDE = ("whisper", "orpheus", "prompt-guard", "safeguard", "embed", "tts", "playai")

# Preferred first when the account exposes them.
_PREFERRED = (
    "openai/gpt-oss-120b",
    "groq/compound",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
)

_lock = threading.Lock()
_cache: Optional[List[ModelInfo]] = None
_cached_at = 0.0


def _label(model_id: str) -> str:
    """Turn 'openai/gpt-oss-120b' into 'GPT OSS 120B'."""
    words = []
    for part in model_id.split("/")[-1].replace("_", "-").split("-"):
        if re.fullmatch(r"\d+(\.\d+)?[bm]", part, re.I) or part.lower() in {
            "gpt",
            "oss",
            "ai",
        }:
            words.append(part.upper())
        elif re.fullmatch(r"[\d.]+", part):
            words.append(part)
        else:
            words.append(part.capitalize())
    return " ".join(words)


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(token in lowered for token in _EXCLUDE)


def _discover() -> Optional[List[ModelInfo]]:
    if not settings.groq_api_key:
        return None
    try:
        from groq import Groq

        ids = [m.id for m in Groq(api_key=settings.groq_api_key).models.list().data]
    except Exception as exc:
        logger.warning("Could not list Groq models (%s); using the fallback list.", exc)
        return None

    chat_ids = sorted(i for i in ids if _is_chat_model(i))
    if not chat_ids:
        return None

    ordered = [i for i in _PREFERRED if i in chat_ids]
    ordered += [i for i in chat_ids if i not in ordered]
    return [ModelInfo(id=i, label=_label(i)) for i in ordered]


def list_models(force_refresh: bool = False) -> List[ModelInfo]:
    global _cache, _cached_at
    with _lock:
        fresh = _cache is not None and (time.time() - _cached_at) < CACHE_TTL_SECONDS
        if fresh and not force_refresh:
            return _cache
        discovered = _discover()
        _cache = discovered if discovered is not None else list(FALLBACK_MODELS)
        _cached_at = time.time()
        return _cache


def default_model() -> str:
    return list_models()[0].id


def resolve_model(model: str) -> str:
    """Return `model` if this account can use it, otherwise the default."""
    available = {m.id for m in list_models()}
    return model if model in available else list_models()[0].id


def get_llm(model: str) -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured on the server. Add it to backend/.env."
        )
    return ChatGroq(
        model=resolve_model(model),
        temperature=0.1,
        api_key=settings.groq_api_key,
        streaming=True,
    )
