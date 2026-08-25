import json
import logging
from typing import Iterator, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..deps import get_current_user, owned_chat
from ..models import Chat, Collection, Message, User, utcnow
from ..schemas import ChatCreate, ChatOut, ChatUpdate, MessageCreate, MessageOut
from ..services import llm as llm_service
from ..services import rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chats", tags=["chats"])

TITLE_MAX = 60
UNTITLED = "New chat"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.get("", response_model=List[ChatOut])
def list_chats(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> List[Chat]:
    return list(
        db.scalars(
            select(Chat).where(Chat.owner_id == user.id).order_by(Chat.updated_at.desc())
        ).all()
    )


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
def create_chat(
    payload: ChatCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Chat:
    collection = db.get(Collection, payload.collection_id)
    if collection is None or collection.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")

    chat = Chat(
        owner_id=user.id,
        collection_id=collection.id,
        title=(payload.title or UNTITLED).strip() or UNTITLED,
        model=llm_service.resolve_model(payload.model or ""),
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.patch("/{chat_id}", response_model=ChatOut)
def update_chat(
    payload: ChatUpdate,
    chat: Chat = Depends(owned_chat),
    db: Session = Depends(get_db),
) -> Chat:
    if payload.title is not None:
        chat.title = payload.title.strip()
    if payload.model is not None:
        chat.model = llm_service.resolve_model(payload.model)
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat: Chat = Depends(owned_chat), db: Session = Depends(get_db)
) -> None:
    db.delete(chat)
    db.commit()


@router.get("/{chat_id}/messages", response_model=List[MessageOut])
def list_messages(
    chat: Chat = Depends(owned_chat), db: Session = Depends(get_db)
) -> List[Message]:
    return list(
        db.scalars(
            select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
        ).all()
    )


@router.post("/{chat_id}/messages")
def send_message(
    payload: MessageCreate,
    chat: Chat = Depends(owned_chat),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Persist the question, then stream the answer back as server-sent events."""
    question = payload.content.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Read history *before* inserting the new turn; the question is passed separately.
    history: List[Tuple[str, str]] = [
        (m.role, m.content)
        for m in db.scalars(
            select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
        ).all()
    ]

    user_message = Message(chat_id=chat.id, role="user", content=question, sources=[])
    db.add(user_message)
    if chat.title.strip() in ("", UNTITLED):
        chat.title = question[:TITLE_MAX]
    chat.updated_at = utcnow()
    db.commit()
    db.refresh(user_message)

    chat_id = chat.id
    collection_id = chat.collection_id
    model = chat.model
    user_message_id = user_message.id

    def event_stream() -> Iterator[str]:
        yield _sse({"type": "user_message", "id": user_message_id})
        parts: List[str] = []
        sources: List[dict] = []
        usage: dict = {}
        failed = False

        try:
            for event in rag.stream_answer(collection_id, question, history, model):
                if event["type"] == "sources":
                    sources = event["sources"]
                elif event["type"] == "usage":
                    usage = event
                elif event["type"] == "token":
                    parts.append(event["text"])
                yield _sse(event)
        except Exception as exc:
            failed = True
            logger.exception("Streaming failed for chat %s", chat_id)
            yield _sse({"type": "error", "detail": str(exc)})

        answer = "".join(parts)
        message_id = None
        if answer.strip():
            # The request-scoped session is gone by the time the body streams.
            session = SessionLocal()
            try:
                message = Message(
                    chat_id=chat_id,
                    role="assistant",
                    content=answer,
                    sources=sources,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
                session.add(message)
                stored_chat = session.get(Chat, chat_id)
                if stored_chat is not None:
                    stored_chat.updated_at = utcnow()
                session.commit()
                session.refresh(message)
                message_id = message.id
            except Exception:
                logger.exception("Could not persist answer for chat %s", chat_id)
            finally:
                session.close()

        yield _sse(
            {
                "type": "done",
                "message_id": message_id,
                "sources": sources,
                "failed": failed,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
