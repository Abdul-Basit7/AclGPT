"""OAuth sign-in. The provider round trip is mocked, so these run offline."""

from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import select

from app.config import settings
from app.models import OAuthAccount, User
from app.security import create_state_token
from app.services import oauth


@pytest.fixture(autouse=True)
def no_ambient_oauth(monkeypatch):
    """Blank real credentials from .env so tests never depend on local setup."""
    for field in (
        "google_oauth_client_id",
        "google_oauth_client_secret",
        "github_oauth_client_id",
        "github_oauth_client_secret",
    ):
        monkeypatch.setattr(settings, field, "")


@pytest.fixture
def configured(monkeypatch):
    """Pretend both OAuth apps are registered."""
    monkeypatch.setattr(settings, "google_oauth_client_id", "google-client-id")
    monkeypatch.setattr(settings, "google_oauth_client_secret", "google-secret")
    monkeypatch.setattr(settings, "github_oauth_client_id", "github-client-id")
    monkeypatch.setattr(settings, "github_oauth_client_secret", "github-secret")


@pytest.fixture
def identity(monkeypatch):
    """Stand in for code -> token -> profile, so no network call happens."""

    def _set(email, account_id="provider-account-1", verified=True):
        def fake_resolve(provider, code, callback_uri):
            profile = oauth.ProviderProfile(
                account_id=account_id, email=email, email_verified=verified
            )
            if not profile.email_verified:
                raise oauth.OAuthError("Your email address is not verified.")
            return profile

        monkeypatch.setattr(oauth, "resolve_identity", fake_resolve)

    return _set


def db_session():
    from app.database import SessionLocal

    return SessionLocal()


# --- provider advertisement ---


def test_providers_empty_when_unconfigured(client):
    assert client.get("/api/auth/providers").json() == []


def test_providers_lists_configured_only(client, configured, monkeypatch):
    monkeypatch.setattr(settings, "github_oauth_client_secret", "")
    body = client.get("/api/auth/providers").json()
    assert [p["id"] for p in body] == ["google"]
    assert body[0]["label"] == "Google"


def test_start_rejects_unconfigured_provider(client):
    response = client.get("/api/auth/oauth/google/start", follow_redirects=False)
    assert response.status_code == 400
    assert "not configured" in response.json()["detail"]


def test_start_rejects_unknown_provider(client, configured):
    response = client.get("/api/auth/oauth/nitter/start", follow_redirects=False)
    assert response.status_code == 400


# --- the redirect out ---


@pytest.mark.parametrize(
    "provider,host",
    [("google", "accounts.google.com"), ("github", "github.com")],
)
def test_start_redirects_to_provider_with_state(client, configured, provider, host):
    response = client.get(
        f"/api/auth/oauth/{provider}/start", follow_redirects=False
    )
    assert response.status_code == 302
    target = urlparse(response.headers["location"])
    assert target.netloc == host

    params = parse_qs(target.query)
    assert params["response_type"] == ["code"]
    assert params["client_id"] == [f"{provider}-client-id"]
    assert params["redirect_uri"][0].endswith(
        f"/api/auth/oauth/{provider}/callback"
    )
    assert params["state"][0], "a CSRF state token must be present"
    # The secret must never travel to the browser.
    assert "client_secret" not in params


# --- the callback ---


def test_callback_creates_user_without_a_password(client, configured, identity):
    identity("newperson@example.com", account_id="google-1")
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "auth-code", "state": create_state_token("google")},
        follow_redirects=False,
    )
    assert response.status_code == 302
    fragment = urlparse(response.headers["location"]).fragment
    token = parse_qs(fragment).get("token", [None])[0]
    assert token, f"expected a token in the fragment, got {fragment!r}"

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "newperson@example.com"

    session = db_session()
    try:
        user = session.scalar(select(User).where(User.email == "newperson@example.com"))
        assert user is not None
        assert user.password_hash is None, "OAuth signup must not invent a password"
        link = session.scalar(
            select(OAuthAccount).where(OAuthAccount.user_id == user.id)
        )
        assert link is not None
        assert (link.provider, link.provider_account_id) == ("google", "google-1")
    finally:
        session.close()


def test_callback_seeds_a_starter_collection(client, configured, identity):
    identity("seeded@example.com", account_id="google-seed")
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "c", "state": create_state_token("google")},
        follow_redirects=False,
    )
    token = parse_qs(urlparse(response.headers["location"]).fragment)["token"][0]
    collections = client.get(
        "/api/collections", headers={"Authorization": f"Bearer {token}"}
    ).json()
    assert len(collections) == 1
    assert collections[0]["name"] == "My documents"


def test_repeat_sign_in_reuses_the_same_account(client, configured, identity):
    identity("returning@example.com", account_id="google-repeat")
    args = {"code": "c", "state": create_state_token("google")}

    first = client.get(
        "/api/auth/oauth/google/callback", params=args, follow_redirects=False
    )
    second = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "c2", "state": create_state_token("google")},
        follow_redirects=False,
    )
    assert first.status_code == second.status_code == 302

    session = db_session()
    try:
        users = session.scalars(
            select(User).where(User.email == "returning@example.com")
        ).all()
        assert len(users) == 1, "a second sign-in must not create a second user"
        links = session.scalars(
            select(OAuthAccount).where(OAuthAccount.user_id == users[0].id)
        ).all()
        assert len(links) == 1, "the provider link must not be duplicated"
    finally:
        session.close()


def test_oauth_links_to_an_existing_password_account(client, configured, identity):
    email = "linkme@example.com"
    assert (
        client.post(
            "/api/auth/register", json={"email": email, "password": "supersecret123"}
        ).status_code
        == 201
    )

    identity(email, account_id="google-link")
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "c", "state": create_state_token("google")},
        follow_redirects=False,
    )
    assert response.status_code == 302

    session = db_session()
    try:
        users = session.scalars(select(User).where(User.email == email)).all()
        assert len(users) == 1, "linking must not create a duplicate account"
        assert users[0].password_hash is not None, "existing password must survive"
        assert session.scalar(
            select(OAuthAccount).where(OAuthAccount.user_id == users[0].id)
        )
    finally:
        session.close()

    # The original password still works.
    assert (
        client.post(
            "/api/auth/login", json={"email": email, "password": "supersecret123"}
        ).status_code
        == 200
    )


def test_unverified_provider_email_is_refused(client, configured, identity):
    identity("spoofed@example.com", account_id="google-bad", verified=False)
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "c", "state": create_state_token("google")},
        follow_redirects=False,
    )
    assert response.status_code == 302
    params = parse_qs(urlparse(response.headers["location"]).fragment)
    assert "token" not in params, "an unverified email must not yield a session"
    assert "not verified" in params["error"][0]

    session = db_session()
    try:
        assert (
            session.scalar(select(User).where(User.email == "spoofed@example.com"))
            is None
        )
    finally:
        session.close()


# --- state / CSRF ---


@pytest.mark.parametrize(
    "state",
    [
        "",
        "not-a-jwt",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.bogussignature",
    ],
)
def test_tampered_state_is_rejected(client, configured, identity, state):
    identity("attacker@example.com")
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "auth-code", "state": state},
        follow_redirects=False,
    )
    assert response.status_code == 302
    params = parse_qs(urlparse(response.headers["location"]).fragment)
    assert "token" not in params
    assert "expired" in params["error"][0]


def test_state_is_bound_to_its_provider(client, configured, identity):
    """A state issued for GitHub must not authorise a Google callback."""
    identity("crossed@example.com")
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "c", "state": create_state_token("github")},
        follow_redirects=False,
    )
    params = parse_qs(urlparse(response.headers["location"]).fragment)
    assert "token" not in params


def test_expired_state_is_rejected(client, configured, identity, monkeypatch):
    monkeypatch.setattr(settings, "oauth_state_ttl_seconds", -1)
    stale = create_state_token("google")
    monkeypatch.setattr(settings, "oauth_state_ttl_seconds", 600)

    identity("late@example.com")
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"code": "c", "state": stale},
        follow_redirects=False,
    )
    params = parse_qs(urlparse(response.headers["location"]).fragment)
    assert "token" not in params


def test_state_token_cannot_authenticate_a_request(client, configured):
    """A state token is signed with the same key: it must not pass as a session."""
    response = client.get(
        "/api/collections",
        headers={"Authorization": f"Bearer {create_state_token('google')}"},
    )
    assert response.status_code == 401


def test_provider_denial_is_reported_cleanly(client, configured):
    response = client.get(
        "/api/auth/oauth/google/callback",
        params={"error": "access_denied"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    params = parse_qs(urlparse(response.headers["location"]).fragment)
    assert "token" not in params
    assert "cancelled" in params["error"][0]


# --- provider profile parsing ---


def test_google_profile_requires_a_verified_email():
    profile = oauth._google_profile(
        {"sub": "42", "email": "A@Example.com", "email_verified": True}, None
    )
    assert profile == oauth.ProviderProfile("42", "a@example.com", True)

    unverified = oauth._google_profile(
        {"sub": "42", "email": "a@example.com", "email_verified": "false"}, None
    )
    assert unverified.email_verified is False

    with pytest.raises(oauth.OAuthError):
        oauth._google_profile({"sub": "42"}, None)


def test_github_profile_uses_the_primary_verified_email():
    class FakeClient:
        def get(self, url):
            assert url.endswith("/user/emails")

            class Response:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return [
                        {"email": "alt@example.com", "primary": False, "verified": True},
                        {"email": "Main@Example.com", "primary": True, "verified": True},
                    ]

            return Response()

    profile = oauth._github_profile({"id": 99}, FakeClient())
    assert profile == oauth.ProviderProfile("99", "main@example.com", True)
