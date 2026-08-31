"""Turn an uploaded file into vectors. Runs as a background task after upload."""

import logging
import re
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..database import SessionLocal
from ..models import Document
from . import extract, vectorstore

logger = logging.getLogger(__name__)
MAX_ERROR_CHARS = 500
# Rate-limit windows can be a full minute, so allow a patient ceiling.
MAX_BACKOFF_SECONDS = 120.0


def stored_path(document_id: int, filename: str) -> Path:
    """On-disk name derived from the row id, so user filenames never touch the FS."""
    suffix = Path(filename).suffix.lower()
    return settings.upload_dir / f"{document_id}{suffix}"


class _RateLimiter:
    """Paces embedding work to stay under the provider's per-minute quota.

    The quota counts *documents*, not HTTP calls: a batch of 64 texts spends 64
    units. Pacing per call would therefore overshoot by the batch size, so each
    reservation consumes as many slots as the batch holds.
    """

    def __init__(self, per_minute: int) -> None:
        self._interval_per_unit = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def reserve(self, units: int = 1) -> None:
        cost = self._interval_per_unit * max(units, 1)
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + cost
        if delay > 0:
            time.sleep(delay)


# Shared across documents: the quota is per API key, not per upload.
_limiter = _RateLimiter(settings.embed_requests_per_minute)

_RETRY_AFTER = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


def _is_daily_quota(exc: Exception) -> bool:
    """A per-day cap will not clear by waiting, so retrying it is pointless.

    Google distinguishes the two in `quota_id`, e.g.
    `EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`.
    """
    return "perday" in str(exc).lower().replace("_", "")


def _is_rate_limit(exc: Exception) -> bool:
    if _is_daily_quota(exc):
        return False  # retrying a daily cap just wastes minutes
    text = str(exc).lower()
    return "429" in text or "quota" in text or "rate limit" in text


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honour the provider's suggested delay when it gives one."""
    match = _RETRY_AFTER.search(str(exc))
    if match:
        try:
            return min(float(match.group(1)) + 1.0, MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(2.0**attempt, MAX_BACKOFF_SECONDS)


def ingest_document(document_id: int) -> None:
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            return

        document.status = "processing"
        document.error = None
        document.chunks_embedded = 0
        db.commit()

        try:
            path = stored_path(document.id, document.filename)
            pages = extract.extract_pages(path, document.filename)
            chunks, ids = _split(document, pages)
            if not chunks:
                raise RuntimeError(
                    "No extractable text found. Scanned or image-only files need OCR first."
                )

            document.pages = len(pages)
            document.chunk_count = len(chunks)
            db.commit()

            # Track what actually landed, so a failure part-way through can be
            # rolled back instead of leaving orphaned vectors in the index.
            embedded_ids: List[str] = []

            def progress(done: int) -> None:
                embedded_ids[:] = ids[:done]
                document.chunks_embedded = done
                document.vector_ids = list(embedded_ids)
                db.commit()

            try:
                _embed_in_batches(document.collection_id, chunks, ids, progress)
            except Exception:
                # Without this, the index keeps chunks no document references,
                # so they can never be deleted and would pollute retrieval.
                if embedded_ids:
                    logger.warning(
                        "Rolling back %s partially embedded chunks for document %s",
                        len(embedded_ids),
                        document_id,
                    )
                    try:
                        vectorstore.delete_vectors(document.collection_id, embedded_ids)
                    except Exception:
                        logger.exception(
                            "Could not roll back partial vectors for document %s",
                            document_id,
                        )
                raise

            document.vector_ids = ids
            document.chunks_embedded = len(chunks)
            document.status = "ready"
            document.error = None
        except Exception as exc:  # surfaced to the UI via document.status/error
            logger.exception("Ingest failed for document %s", document_id)
            document.status = "failed"
            document.error = _friendly_error(exc)
            document.vector_ids = []
            document.chunks_embedded = 0
        db.commit()
    finally:
        db.close()


def _friendly_error(exc: Exception) -> str:
    if _is_daily_quota(exc):
        return (
            "The embedding provider's daily quota is used up. Google's free tier "
            "allows 1,000 embedding requests per day and each chunk counts as one, "
            "so roughly 1,000 chunks a day in total. Wait for the quota to reset, "
            "or use a paid API key."
        )
    if _is_rate_limit(exc):
        return (
            "The embedding provider's per-minute rate limit was exceeded and did not "
            "recover. Try again in a few minutes, split the file, or use a paid key."
        )
    return str(exc)[:MAX_ERROR_CHARS]


def _embed_in_batches(
    collection_id: int,
    chunks: List[LCDocument],
    ids: List[str],
    on_progress,
) -> None:
    """Index in batches, retrying rate-limit failures.

    Embedding every chunk in one call is what breaks large documents on a remote
    provider: the free tier caps requests per minute *and* per day, so a
    thousand-chunk PDF fails within seconds. Batching plus pacing plus backoff
    makes that slow but reliable. A local model has neither cap, so batches are
    large and pacing is skipped entirely. Progress is reported either way, so the
    UI shows movement instead of appearing hung.
    """
    batch_size = max(settings.effective_embed_batch_size, 1)
    # A local model has no quota to respect, so pacing would only slow it down.
    paced = not settings.embeddings_are_local
    done = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]

        for attempt in range(settings.embed_max_retries + 1):
            if paced:
                _limiter.reserve(len(batch))
            try:
                vectorstore.add_documents(collection_id, batch, batch_ids)
                break
            except Exception as exc:
                retryable = _is_rate_limit(exc)
                if not retryable or attempt == settings.embed_max_retries:
                    raise
                delay = _retry_delay(exc, attempt)
                logger.warning(
                    "Embedding batch %s-%s hit a rate limit; retrying in %.1fs (attempt %s)",
                    start,
                    start + len(batch),
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)

        done += len(batch)
        on_progress(done)


def _split(
    document: Document, pages: List[extract.Page]
) -> Tuple[List[LCDocument], List[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks: List[LCDocument] = []
    ids: List[str] = []
    total_pages = len(pages)

    for page_no, text in pages:
        for piece in splitter.split_text(text or ""):
            if not piece.strip():
                continue
            ids.append(f"{document.id}-{len(ids)}")
            chunks.append(
                LCDocument(
                    page_content=piece,
                    metadata={
                        "document_id": document.id,
                        "filename": document.filename,
                        "page": page_no,
                        "total_pages": total_pages,
                    },
                )
            )
    return chunks, ids
