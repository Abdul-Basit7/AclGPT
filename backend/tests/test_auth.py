def test_register_returns_token_and_starter_collection(client, auth):
    headers, collection_id, email = auth()

    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["email"] == email

    collections = client.get("/api/collections", headers=headers).json()
    assert len(collections) == 1
    assert collections[0]["name"] == "My documents"
    assert collections[0]["id"] == collection_id


def test_login_round_trip(client):
    payload = {"email": "login@example.com", "password": "supersecret123"}
    assert client.post("/api/auth/register", json=payload).status_code == 201

    ok = client.post("/api/auth/login", json=payload)
    assert ok.status_code == 200
    assert ok.json()["user"]["email"] == "login@example.com"

    bad = client.post(
        "/api/auth/login", json={**payload, "password": "wrongpassword1"}
    )
    assert bad.status_code == 401


def test_duplicate_email_rejected(client):
    payload = {"email": "dupe@example.com", "password": "supersecret123"}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_short_password_rejected(client):
    response = client.post(
        "/api/auth/register", json={"email": "short@example.com", "password": "abc"}
    )
    assert response.status_code == 422


def test_protected_routes_require_a_token(client):
    for method, path in [
        ("get", "/api/collections"),
        ("get", "/api/chats"),
        ("get", "/api/auth/me"),
    ]:
        assert getattr(client, method)(path).status_code == 401

    assert client.get(
        "/api/collections", headers={"Authorization": "Bearer not-a-jwt"}
    ).status_code == 401


def test_health_reports_configuration(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["google_key_configured"] is True
    assert body["groq_key_configured"] is True
    assert [m["id"] for m in body["models"]] == ["test-model-a", "test-model-b"]
