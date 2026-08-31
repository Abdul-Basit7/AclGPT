"""Per-collection FAISS indexes stored on disk.

Each collection owns a directory under ``data/indexes/<collection_id>``. Writes are
serialised per collection and staged through a temp directory, so a crash mid-write
cannot leave a half-saved index behind.
"""

import json
import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from ..config import settings

_store_cache: Dict[int, FAISS] = {}
_locks: Dict[int, threading.Lock] = {}
_registry_lock = threading.Lock()

# Records which embedding model built an index, so a model change is caught
# instead of silently returning nonsense.
SIGNATURE_FILE = "embedding.json"


class EmbeddingMismatch(RuntimeError):
    """An index on disk was built by a different embedding model."""


def _lock_for(collection_id: int) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(collection_id, threading.Lock())


@lru_cache
def get_embeddings() -> Embeddings:
    """The embedding model, local by default.

    Local means an ONNX model running in this process: no API key, no quota and
    no network, which is what makes indexing a large document possible at all.
    The remote option scores marginally better on retrieval benchmarks but the
    free tier allows only 1,000 embed requests per day, and one chunk is one
    request.

    Changing the model changes the vector space, so every existing index has to
    be rebuilt; `_signature` exists to make that failure loud.
    """
    if settings.embeddings_are_local:
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=settings.local_embedding_model,
            # Documents are embedded in large batches during ingest; letting
            # onnxruntime use several threads is the difference between minutes
            # and tens of minutes on a big file.
            threads=None,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured on the server. Add it to backend/.env."
        )
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.google_api_key
    )


def _signature() -> str:
    """Identifies the current embedding model. Vectors are only comparable
    within one signature."""
    if settings.embeddings_are_local:
        return f"local:{settings.local_embedding_model}"
    return f"google:{settings.embedding_model}"


def index_path(collection_id: int) -> Path:
    return settings.index_dir / str(collection_id)


def _read_signature(path: Path) -> Optional[str]:
    try:
        return json.loads((path / SIGNATURE_FILE).read_text())["signature"]
    except Exception:
        return None


def _write_signature(path: Path) -> None:
    (path / SIGNATURE_FILE).write_text(json.dumps({"signature": _signature()}))


def _assert_compatible(path: Path) -> None:
    """Refuse to search an index built by another model.

    Without this the dimensions may happen to match and retrieval quietly
    returns unrelated passages, which is far worse than an error.
    """
    stored = _read_signature(path)
    if stored is not None and stored != _signature():
        raise EmbeddingMismatch(
            f"This collection was indexed with {stored} but the server is now "
            f"using {_signature()}. Re-index the documents to search them again."
        )


def _load_unlocked(collection_id: int) -> Optional[FAISS]:
    cached = _store_cache.get(collection_id)
    if cached is not None:
        return cached
    path = index_path(collection_id)
    if not (path / "index.faiss").exists():
        return None
    _assert_compatible(path)
    store = FAISS.load_local(
        str(path), get_embeddings(), allow_dangerous_deserialization=True
    )
    _store_cache[collection_id] = store
    return store


def _save_unlocked(collection_id: int, store: FAISS) -> None:
    path = index_path(collection_id)
    staging = path.parent / f"{path.name}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    store.save_local(str(staging))
    _write_signature(staging)
    shutil.rmtree(path, ignore_errors=True)
    staging.rename(path)
    _store_cache[collection_id] = store


def add_documents(
    collection_id: int, documents: List[LCDocument], ids: List[str]
) -> None:
    if not documents:
        return
    with _lock_for(collection_id):
        store = _load_unlocked(collection_id)
        if store is None:
            store = FAISS.from_documents(documents, get_embeddings(), ids=ids)
        else:
            store.add_documents(documents, ids=ids)
        _save_unlocked(collection_id, store)


def delete_vectors(collection_id: int, ids: List[str]) -> None:
    if not ids:
        return
    with _lock_for(collection_id):
        store = _load_unlocked(collection_id)
        if store is None:
            return
        present = set(store.index_to_docstore_id.values())
        target = [i for i in ids if i in present]
        if not target:
            return
        store.delete(target)
        if store.index.ntotal == 0:
            _drop_unlocked(collection_id)
        else:
            _save_unlocked(collection_id, store)


def retrieve(collection_id: int, query: str) -> List[LCDocument]:
    with _lock_for(collection_id):
        store = _load_unlocked(collection_id)
        if store is None or store.index.ntotal == 0:
            return []
        k = min(settings.retrieval_k, store.index.ntotal)
        fetch_k = min(settings.retrieval_fetch_k, store.index.ntotal)
        return store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def has_index(collection_id: int) -> bool:
    return (index_path(collection_id) / "index.faiss").exists()


def _drop_unlocked(collection_id: int) -> None:
    path = index_path(collection_id)
    shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(path.parent / f"{path.name}.tmp", ignore_errors=True)
    _store_cache.pop(collection_id, None)


def drop_collection(collection_id: int) -> None:
    with _lock_for(collection_id):
        _drop_unlocked(collection_id)
    with _registry_lock:
        _locks.pop(collection_id, None)
