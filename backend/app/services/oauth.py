"""OAuth 2.0 authorization-code flow, shared across providers.

Each provider is a `ProviderConfig` entry: endpoints, scopes, and a parser that
turns the provider's profile payload into a `ProviderProfile`. Adding a provider
is configuration, not new control flow.
"""

import logging
from typing import Callable, Dict, List, NamedTuple, Optional
from urllib.parse import urlencode

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15


class OAuthError(Exception):
    """Raised when a provider interaction fails or returns something unusable."""


class ProviderProfile(NamedTuple):
    account_id: str
    email: str
    email_verified: bool


class ProviderConfig(NamedTuple):
    name: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: str
    # Given the userinfo payload and an authorised client, produce a profile.
    parse_profile: Callable[[dict, httpx.Client], ProviderProfile]


def _google_profile(payload: dict, _client: httpx.Client) -> ProviderProfile:
    account_id = payload.get("sub") or ""
    email = (payload.get("email") or "").lower()
    # Google returns this as a bool or the string "true" depending on endpoint.
    verified_raw = payload.get("email_verified")
    verified = verified_raw is True or str(verified_raw).lower() == "true"
    if not account_id or not email:
        raise OAuthError("Google did not return an account id and email.")
    return ProviderProfile(account_id=str(account_id), email=email, email_verified=verified)


def _github_profile(payload: dict, client: httpx.Client) -> ProviderProfile:
    account_id = payload.get("id")
    if account_id is None:
        raise OAuthError("GitHub did not return an account id.")

    # The main profile hides the address when the user keeps it private, and it
    # carries no verification flag, so always resolve it from /user/emails.
    email, verified = "", False
    try:
        response = client.get("https://api.github.com/user/emails")
        response.raise_for_status()
        entries = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OAuthError(f"Could not read your GitHub email addresses: {exc}") from exc

    if isinstance(entries, list):
        primary = next((e for e in entries if e.get("primary")), None)
        chosen = primary or next((e for e in entries if e.get("verified")), None)
        if chosen:
            email = (chosen.get("email") or "").lower()
            verified = bool(chosen.get("verified"))

    if not email:
        raise OAuthError(
            "GitHub did not expose an email address. Add one to your GitHub account."
        )
    return ProviderProfile(account_id=str(account_id), email=email, email_verified=verified)


PROVIDERS: Dict[str, ProviderConfig] = {
    "google": ProviderConfig(
        name="google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        scopes="openid email profile",
        parse_profile=_google_profile,
    ),
    "github": ProviderConfig(
        name="github",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        userinfo_url="https://api.github.com/user",
        scopes="read:user user:email",
        parse_profile=_github_profile,
    ),
}

DISPLAY_NAMES = {"google": "Google", "github": "GitHub"}


def _credentials(provider: str) -> tuple:
    if provider == "google":
        return settings.google_oauth_client_id, settings.google_oauth_client_secret
    if provider == "github":
        return settings.github_oauth_client_id, settings.github_oauth_client_secret
    return "", ""


def is_configured(provider: str) -> bool:
    client_id, client_secret = _credentials(provider)
    return bool(client_id and client_secret)


def configured_providers() -> List[str]:
    return [name for name in PROVIDERS if is_configured(name)]


def get_config(provider: str) -> ProviderConfig:
    config = PROVIDERS.get(provider)
    if config is None:
        raise OAuthError(f"Unknown provider '{provider}'.")
    if not is_configured(provider):
        raise OAuthError(
            f"{DISPLAY_NAMES.get(provider, provider)} sign-in is not configured on this server."
        )
    return config


def redirect_uri(provider: str, base_url: str) -> str:
    """The callback URI, which must match what is registered with the provider."""
    return f"{base_url.rstrip('/')}/api/auth/oauth/{provider}/callback"


def authorization_url(provider: str, state: str, callback_uri: str) -> str:
    config = get_config(provider)
    client_id, _ = _credentials(provider)
    params = {
        "client_id": client_id,
        "redirect_uri": callback_uri,
        "response_type": "code",
        "scope": config.scopes,
        "state": state,
    }
    if provider == "google":
        # Ask for a fresh consent screen rather than silently reusing a session.
        params["access_type"] = "online"
        params["prompt"] = "select_account"
    return f"{config.authorize_url}?{urlencode(params)}"


def exchange_code(provider: str, code: str, callback_uri: str) -> str:
    """Swap an authorization code for an access token."""
    config = get_config(provider)
    client_id, client_secret = _credentials(provider)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": callback_uri,
        "grant_type": "authorization_code",
    }
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                config.token_url, data=data, headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise OAuthError(f"Could not reach {DISPLAY_NAMES.get(provider, provider)}.") from exc
    except ValueError as exc:
        raise OAuthError("The provider returned an unreadable token response.") from exc

    token = payload.get("access_token")
    if not token:
        # Never log or surface `payload`: it may echo the client secret.
        raise OAuthError("The provider did not issue an access token.")
    return str(token)


def fetch_profile(provider: str, access_token: str) -> ProviderProfile:
    config = get_config(provider)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, headers=headers) as client:
            response = client.get(config.userinfo_url)
            response.raise_for_status()
            payload = response.json()
            return config.parse_profile(payload, client)
    except httpx.HTTPError as exc:
        raise OAuthError(
            f"Could not read your {DISPLAY_NAMES.get(provider, provider)} profile."
        ) from exc
    except ValueError as exc:
        raise OAuthError("The provider returned an unreadable profile.") from exc


def resolve_identity(provider: str, code: str, callback_uri: str) -> ProviderProfile:
    """Full server-side leg: code -> token -> verified profile."""
    access_token = exchange_code(provider, code, callback_uri)
    profile = fetch_profile(provider, access_token)
    if not profile.email_verified:
        # Linking on an unverified address would let someone who registers an
        # address they do not own take over the matching local account.
        raise OAuthError(
            f"Your {DISPLAY_NAMES.get(provider, provider)} email address is not verified. "
            "Verify it with the provider and try again."
        )
    return profile


def optional_profile(provider: str) -> Optional[ProviderConfig]:
    return PROVIDERS.get(provider)
