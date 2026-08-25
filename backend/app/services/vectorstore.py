"""Per-collection FAISS indexes stored on disk.

Each collection owns a directory under ``data/indexes/<collection_id>``. Writes are
serialised per collection and staged through a temp directory, so a crash mid-write
cannot leave a half-saved index behind.
"""

import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config import settings

_store_cache: Dict[int, FAISS] = {}
_locks: Dict[int, threading.Lock] = {}
_registry_lock = threading.Lock()


def _lock_for(collection_id: int) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(collection_id, threading.Lock())


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured on the server. Add it to backend/.env."
        )
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.google_api_key
    )


def index_path(collection_id: int) -> Path:
    return settings.index_dir / str(collection_id)


def _load_unlocked(collection_id: int) -> Optional[FAISS]:
    cached = _store_cache.get(collection_id)
    if cached is not None:
        return cached
    path = index_path(collection_id)
    if not (path / "index.faiss").exists():
        return None
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
