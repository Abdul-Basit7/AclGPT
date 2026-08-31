"""Rebuild every vector index from the files already on disk.

Needed whenever the embedding model changes: vectors are only comparable within
one model's vector space, so an existing index becomes meaningless rather than
merely stale. The uploaded files are kept, so this needs no re-uploading.

    backend/.venv/bin/python scripts/reindex.py [--collection ID]
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import Document  # noqa: E402
from app.services import ingest, vectorstore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=int, help="Only this collection")
    args = parser.parse_args()

    print(f"Embedding model: {vectorstore._signature()}")

    db = SessionLocal()
    try:
        query = db.query(Document).order_by(Document.id)
        if args.collection:
            query = query.filter(Document.collection_id == args.collection)
        documents = [(d.id, d.collection_id, d.filename) for d in query.all()]
        collection_ids = sorted({c for _, c in ((d[0], d[1]) for d in documents)})
    finally:
        db.close()

    if not documents:
        print("No documents to re-index.")
        return 0

    # Drop the old indexes first: re-ingesting into a foreign index would be
    # refused by the signature guard, one document at a time.
    for collection_id in collection_ids:
        vectorstore.drop_collection(collection_id)
    if not args.collection:
        stale = [
            p
            for p in settings.index_dir.iterdir()
            if p.is_dir() and p.name.isdigit() and int(p.name) not in collection_ids
        ]
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
        if stale:
            print(f"Removed {len(stale)} index(es) with no documents left.")

    ok = failed = missing = 0
    started = time.time()
    for document_id, _collection_id, filename in documents:
        path = ingest.stored_path(document_id, filename)
        if not path.exists():
            print(f"  - {document_id:>3} {filename}: source file missing, skipped")
            missing += 1
            continue

        ingest.ingest_document(document_id)

        db = SessionLocal()
        try:
            stored = db.get(Document, document_id)
            status, error = stored.status, stored.error
            chunks = stored.chunks_embedded
        finally:
            db.close()

        if status == "ready":
            print(f"  - {document_id:>3} {filename}: {chunks} chunks")
            ok += 1
        else:
            print(f"  - {document_id:>3} {filename}: {status} -- {error}")
            failed += 1

    elapsed = time.time() - started
    print(f"\n{ok} re-indexed, {failed} failed, {missing} missing in {elapsed:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
