from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import owned_collection
from ..models import Collection, Document
from ..schemas import DocumentOut
from ..services import extract, ingest, vectorstore

router = APIRouter(prefix="/api/collections/{collection_id}/documents", tags=["documents"])


@router.get("", response_model=List[DocumentOut])
def list_documents(
    collection: Collection = Depends(owned_collection),
    db: Session = Depends(get_db),
) -> List[Document]:
    return list(
        db.scalars(
            select(Document)
            .where(Document.collection_id == collection.id)
            .order_by(Document.id.desc())
        ).all()
    )


@router.post("", response_model=List[DocumentOut], status_code=status.HTTP_201_CREATED)
async def upload_documents(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection: Collection = Depends(owned_collection),
    db: Session = Depends(get_db),
) -> List[Document]:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    created: List[Document] = []

    for upload in files:
        name = upload.filename or "untitled"
        if not extract.is_supported(name):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{name}: unsupported file type. "
                    f"Allowed: {', '.join(sorted(extract.SUPPORTED_SUFFIXES))}"
                ),
            )
        payload = await upload.read()
        if len(payload) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{name} is larger than the {settings.max_upload_mb} MB limit.",
            )
        if not payload:
            raise HTTPException(status_code=400, detail=f"{name} is empty.")

        document = Document(
            collection_id=collection.id,
            filename=name,
            content_type=upload.content_type or "",
            size_bytes=len(payload),
            status="pending",
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # Written under the row id, so user-supplied names never reach the filesystem.
        ingest.stored_path(document.id, name).write_bytes(payload)
        background.add_task(ingest.ingest_document, document.id)
        created.append(document)

    return created


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    collection: Collection = Depends(owned_collection),
    db: Session = Depends(get_db),
) -> None:
    document = db.get(Document, document_id)
    if document is None or document.collection_id != collection.id:
        raise HTTPException(status_code=404, detail="Document not found")

    vectorstore.delete_vectors(collection.id, list(document.vector_ids or []))
    ingest.stored_path(document.id, document.filename).unlink(missing_ok=True)
    db.delete(document)
    db.commit()
