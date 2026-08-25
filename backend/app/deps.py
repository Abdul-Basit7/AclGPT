from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import Chat, Collection, User
from .security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise CREDENTIALS_ERROR
    subject = decode_access_token(credentials.credentials)
    if subject is None or not subject.isdigit():
        raise CREDENTIALS_ERROR
    user = db.get(User, int(subject))
    if user is None:
        raise CREDENTIALS_ERROR
    return user


def owned_collection(
    collection_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Collection:
    collection = db.get(Collection, collection_id)
    if collection is None or collection.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


def owned_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Chat:
    chat = db.get(Chat, chat_id)
    if chat is None or chat.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
