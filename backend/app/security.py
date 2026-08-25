import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

from .config import settings

# bcrypt silently ignores bytes past 72; reject instead of truncating.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    # OAuth-only accounts have no hash; password login must fail, not error.
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.PyJWTError:
        return None
    if payload.get("typ") == STATE_TOKEN_TYPE:
        return None  # a CSRF state token must never authenticate a request
    subject = payload.get("sub")
    return str(subject) if subject is not None else None


STATE_TOKEN_TYPE = "oauth_state"  # noqa: S105 - a discriminator, not a credential


def create_state_token(provider: str) -> str:
    """Short-lived signed token carrying the OAuth flow's CSRF state.

    Signing the state means no server-side session store is needed: the callback
    can prove it issued the value itself, and that it has not expired.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "typ": STATE_TOKEN_TYPE,
        "provider": provider,
        "nonce": secrets.token_urlsafe(16),
        "iat": now,
        "exp": now + timedelta(seconds=settings.oauth_state_ttl_seconds),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_state_token(token: str, provider: str) -> bool:
    """True only for a state token this server issued, for this provider, unexpired."""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.PyJWTError:
        return False
    return (
        payload.get("typ") == STATE_TOKEN_TYPE
        and payload.get("provider") == provider
    )
