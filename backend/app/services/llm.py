"""Groq chat models.

Model ids change frequently and differ per account, so the list is discovered from
the Groq API at runtime and cached. A static fallback keeps the app usable when
discovery is unavailable (no key, offline, or a transient API error).
"""

import logging
import re
import threading
import time
from typing import Dict, Iterator, List, Optional, Sequence

from langchain_groq import ChatGroq

from ..config import settings
from ..schemas import ModelInfo

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 600

# Groq runs web search and page fetching server side, but only for its Compound
# systems; every other model can answer only from its training data. Verified
# against the live API -- a Compound response carries `executed_tools`, others
# never do. This is why the web-search control is disabled for other models.
WEB_SEARCH_MODELS = frozenset({"groq/compound", "groq/compound-mini"})


def _natively_searches(model_id: str) -> bool:
    return model_id in WEB_SEARCH_MODELS


# Used when the API cannot be reached. Kept short and general on purpose.
FALLBACK_MODELS: List[ModelInfo] = [
    ModelInfo(id="openai/gpt-oss-120b", label="GPT OSS 120B"),
    ModelInfo(id="openai/gpt-oss-20b", label="GPT OSS 20B"),
    ModelInfo(
        id="groq/compound",
        label="Compound",
        supports_web_search=_natively_searches("groq/compound"),
    ),
]

# Speech, moderation and embedding models cannot answer chat requests.
_EXCLUDE = ("whisper", "orpheus", "prompt-guard", "safeguard", "embed", "tts", "playai")

# Preferred first when the account exposes them. compound-mini is listed above
# compound because it makes fewer tool calls per answer, which keeps a web
# search inside the per-request token limit far more often on a small tier --
# measured 2/3 successful against 1/3 for compound on the same questions.
_PREFERRED = (
    "openai/gpt-oss-120b",
    "groq/compound-mini",
    "groq/compound",
    "openai/gpt-oss-20b",
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
    return [
        ModelInfo(id=i, label=_label(i), supports_web_search=_natively_searches(i))
        for i in ordered
    ]


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


def supports_web_search(model_id: str) -> bool:
    """Whether web search may be used with this model.

    Read from the advertised model list rather than the constant above, so the
    capability the UI is told about and the one the server enforces can never
    drift apart.
    """
    return any(m.id == model_id and m.supports_web_search for m in list_models())


def default_model() -> str:
    return list_models()[0].id


def resolve_model(model: str) -> str:
    """Return `model` if this account can use it, otherwise the default."""
    available = {m.id for m in list_models()}
    return model if model in available else list_models()[0].id


MAX_ERROR_CHARS = 400


def friendly_error(exc: Exception) -> str:
    """Turn a provider failure into something a user can act on.

    Groq answers 413 when a *single* request exceeds the account's per-minute
    token allowance, which is easy to trigger with web search: a Compound run
    pulls whole pages into the prompt, and one long page is enough. Reported
    verbatim it reads like an upload-size problem, which sends people looking in
    entirely the wrong place.
    """
    text = str(exc)
    lowered = text.lower()

    if "request entity too large" in lowered or "request_too_large" in lowered:
        return (
            "The web pages found for that question were too large for this Groq "
            "tier to process in one request. Try a narrower question, turn Web "
            "search off, or upgrade the Groq plan."
        )
    if "429" in text or "rate limit" in lowered:
        return (
            "Groq's rate limit was hit. Wait a moment and send the question "
            "again."
        )
    if "indexed with" in lowered and "re-index" in lowered:
        return text  # already a clear EmbeddingMismatch message
    return text[:MAX_ERROR_CHARS]


def get_llm(model: str, web_search: bool = False) -> ChatGroq:
    """Build a streaming client.

    `web_search` only reaches the API for models that support it. Groq decides
    per request whether a search is actually needed, so this enables the
    capability rather than forcing a search.
    """
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured on the server. Add it to backend/.env."
        )
    resolved = resolve_model(model)
    kwargs = {}
    if supports_web_search(resolved) and not web_search:
        # Best effort: Compound's built-in tools are server side and cannot be
        # switched off from the request in any documented way, so the prompt in
        # services.rag does the real work of telling it not to search.
        kwargs["compound_custom"] = {"tools": {"enabled_tools": []}}

    return ChatGroq(
        model=resolved,
        temperature=0.1,
        api_key=settings.groq_api_key,
        streaming=True,
        model_kwargs=kwargs,
    )


# LangChain message constructors take these names; the API uses the other set.
_LC_ROLES = {"system": "system", "user": "human", "assistant": "ai"}


def _lc_messages(messages: Sequence[Dict[str, str]]):
    return [(_LC_ROLES[m["role"]], m["content"]) for m in messages]


def _stream_via_sdk(model: str, messages: Sequence[Dict[str, str]]) -> Iterator[Dict]:
    """Stream straight from the Groq SDK.

    Needed only for web search: Compound reports the pages it consulted in
    `executed_tools`, and langchain_groq drops that field entirely -- verified by
    inspecting every chunk, where `additional_kwargs` is empty and
    `response_metadata` carries only `finish_reason`. Without this path the
    sources panel would silently never show a web result.
    """
    from groq import Groq

    stream = Groq(api_key=settings.groq_api_key).chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is not None:
            text = getattr(choice.delta, "content", None)
            if text:
                yield {"text": text}
            tools = getattr(choice.delta, "executed_tools", None)
            if tools:
                yield {"executed_tools": [dict(tool) for tool in tools]}

        # Groq reports usage on the final chunk, under its own namespace.
        usage = getattr(getattr(chunk, "x_groq", None), "usage", None)
        if usage:
            yield {
                "usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            }


def _stream_via_langchain(
    model: str, messages: Sequence[Dict[str, str]], web_search: bool
) -> Iterator[Dict]:
    client = get_llm(model, web_search=web_search)
    for chunk in client.stream(_lc_messages(messages)):
        text = getattr(chunk, "content", "")
        if text:
            yield {"text": text}
        usage = getattr(chunk, "usage_metadata", None)
        if usage:
            yield {
                "usage": {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            }


def stream_chat(
    model: str, messages: Sequence[Dict[str, str]], web_search: bool = False
) -> Iterator[Dict]:
    """Stream one answer as {"text"|"executed_tools"|"usage": ...} events.

    Two transports, because only one of them can report web search: the SDK
    directly when searching, LangChain otherwise.
    """
    resolved = resolve_model(model)
    if web_search and _natively_searches(resolved):
        for event in _stream_via_sdk(resolved, messages):
            yield event
        return
    for event in _stream_via_langchain(resolved, messages, web_search):
        yield event
