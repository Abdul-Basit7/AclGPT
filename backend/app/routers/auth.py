import logging
from typing import List
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import Collection, OAuthAccount, User
from ..schemas import Credentials, ProviderOut, TokenOut, UserOut
from ..security import (
    create_access_token,
    create_state_token,
    hash_password,
    verify_password,
    verify_state_token,
)
from ..services import oauth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

STARTER_COLLECTION = "My documents"


def _token_response(user: User) -> TokenOut:
    return TokenOut(
        access_token=create_access_token(str(user.id)),
        user=UserOut.model_validate(user),
    )


def _seed_collection(db: Session, user: User) -> None:
    """Give a new account its first collection, so the UI is never empty."""
    db.add(Collection(owner_id=user.id, name=STARTER_COLLECTION))
    db.commit()


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(payload: Credentials, db: Session = Depends(get_db)) -> TokenOut:
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    user = User(email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    _seed_collection(db, user)
    return _token_response(user)


@router.post("/login", response_model=TokenOut)
def login(payload: Credentials, db: Session = Depends(get_db)) -> TokenOut:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    return _token_response(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


# --- OAuth ---


@router.get("/providers", response_model=List[ProviderOut])
def providers() -> List[ProviderOut]:
    """Only providers with credentials, so the UI never shows a button that fails."""
    return [
        ProviderOut(id=name, label=oauth.DISPLAY_NAMES.get(name, name.title()))
        for name in oauth.configured_providers()
    ]


def _frontend_redirect(**params: str) -> RedirectResponse:
    """Hand the result back to the browser app in the URL fragment.

    A fragment is not sent to servers and does not land in access logs, which a
    query string would.
    """
    return RedirectResponse(
        url=f"{settings.frontend_url.rstrip('/')}/#{urlencode(params)}",
        status_code=status.HTTP_302_FOUND,
    )


@router.get("/oauth/{provider}/start")
def oauth_start(provider: str, request: Request) -> RedirectResponse:
    try:
        callback_uri = oauth.redirect_uri(provider, str(request.base_url))
        url = oauth.authorization_url(
            provider, create_state_token(provider), callback_uri
        )
    except oauth.OAuthError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


@router.get("/oauth/{provider}/callback")
def oauth_callback(
    provider: str,
    request: Request,
    db: Session = Depends(get_db),
    code: str = "",
    state: str = "",
    error: str = "",
) -> RedirectResponse:
    if error:
        # The user declined consent, or the provider rejected the request.
        return _frontend_redirect(error="Sign-in was cancelled.")
    if not code or not verify_state_token(state, provider):
        return _frontend_redirect(error="Sign-in request expired. Please try again.")

    try:
        callback_uri = oauth.redirect_uri(provider, str(request.base_url))
        profile = oauth.resolve_identity(provider, code, callback_uri)
    except oauth.OAuthError as exc:
        logger.warning("OAuth callback failed for %s: %s", provider, exc)
        return _frontend_redirect(error=str(exc))

    user = _upsert_oauth_user(db, provider, profile)
    return _frontend_redirect(token=create_access_token(str(user.id)))


def _upsert_oauth_user(
    db: Session, provider: str, profile: oauth.ProviderProfile
) -> User:
    """Find, link, or create the local account for a verified provider identity."""
    link = db.scalar(
        select(OAuthAccount).where(
            OAuthAccount.provider == provider,
            OAuthAccount.provider_account_id == profile.account_id,
        )
    )
    if link is not None:
        user = db.get(User, link.user_id)
        if user is not None:
            return user
        db.delete(link)  # orphaned link, fall through and rebuild it
        db.commit()

    # `resolve_identity` has already refused unverified emails, so matching on
    # address here cannot be used to hijack an existing account.
    user = db.scalar(select(User).where(User.email == profile.email))
    is_new = user is None
    if user is None:
        user = User(email=profile.email, password_hash=None)
        db.add(user)
        db.commit()
        db.refresh(user)

    db.add(
        OAuthAccount(
            user_id=user.id,
            provider=provider,
            provider_account_id=profile.account_id,
        )
    )
    db.commit()

    if is_new:
        _seed_collection(db, user)
    return user
