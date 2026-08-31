from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user, owned_collection
from ..models import Collection, User
from ..schemas import CollectionCreate, CollectionOut, SuggestionsOut
from ..services import ingest, llm as llm_service, rag, vectorstore

router = APIRouter(prefix="/api/collections", tags=["collections"])


def _serialize(collection: Collection) -> CollectionOut:
    documents = collection.documents
    return CollectionOut(
        id=collection.id,
        name=collection.name,
        created_at=collection.created_at,
        document_count=len(documents),
        ready_count=sum(1 for d in documents if d.status == "ready"),
    )


@router.get("", response_model=List[CollectionOut])
def list_collections(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> List[CollectionOut]:
    rows = db.scalars(
        select(Collection).where(Collection.owner_id == user.id).order_by(Collection.id)
    ).all()
    return [_serialize(c) for c in rows]


@router.post("", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(
    payload: CollectionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> CollectionOut:
    collection = Collection(owner_id=user.id, name=payload.name.strip())
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return _serialize(collection)


@router.patch("/{collection_id}", response_model=CollectionOut)
def rename_collection(
    payload: CollectionCreate,
    collection: Collection = Depends(owned_collection),
    db: Session = Depends(get_db),
) -> CollectionOut:
    collection.name = payload.name.strip()
    db.commit()
    db.refresh(collection)
    return _serialize(collection)


@router.get("/{collection_id}/suggestions", response_model=SuggestionsOut)
def collection_suggestions(
    collection: Collection = Depends(owned_collection),
) -> SuggestionsOut:
    """Opening questions for a collection, before any chat exists to base them on."""
    return SuggestionsOut(
        suggestions=rag.suggest_questions(collection.id, [], llm_service.default_model())
    )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(
    collection: Collection = Depends(owned_collection),
    db: Session = Depends(get_db),
) -> None:
    for document in collection.documents:
        Path(ingest.stored_path(document.id, document.filename)).unlink(missing_ok=True)
    vectorstore.drop_collection(collection.id)
    db.delete(collection)  # cascades to documents and chats
    db.commit()
