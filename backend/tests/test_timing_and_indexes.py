"""Response timing, and the guard against mixing embedding models."""

import json

import pytest

from app.config import settings
from app.services import vectorstore
from tests.conftest import SAMPLE_MD


def parse_sse(text):
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def seed(client, auth):
    headers, collection_id, _ = auth()
    client.post(
        f"/api/collections/{collection_id}/documents",
        headers=headers,
        files={"files": ("notes.md", SAMPLE_MD.encode(), "text/markdown")},
    )
    chat = client.post(
        "/api/chats", headers=headers, json={"collection_id": collection_id}
    ).json()
    return headers, chat


# --- response timing ---


def test_answer_records_how_long_it_took(client, auth):
    headers, chat = seed(client, auth)

    response = client.post(
        f"/api/chats/{chat['id']}/messages",
        headers=headers,
        json={"content": "How long does recovery take?"},
    )
    done = parse_sse(response.text)[-1]
    assert done["type"] == "done"
    assert isinstance(done["duration_ms"], int)
    assert done["duration_ms"] >= 0

    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    answer = messages[-1]
    assert answer["role"] == "assistant"
    assert isinstance(answer["duration_ms"], int), "timing must survive the round trip"
    assert answer["created_at"], "the UI shows when the answer arrived"


def test_user_messages_carry_no_generation_time(client, auth):
    headers, chat = seed(client, auth)
    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "hi"}
    )
    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert messages[0]["role"] == "user"
    assert messages[0]["duration_ms"] is None


# --- embedding signature guard ---


def test_signature_reflects_the_configured_provider(monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "local_embedding_model", "BAAI/bge-small-en-v1.5")
    assert vectorstore._signature() == "local:BAAI/bge-small-en-v1.5"

    monkeypatch.setattr(settings, "embedding_provider", "google")
    monkeypatch.setattr(settings, "embedding_model", "models/gemini-embedding-001")
    assert vectorstore._signature() == "google:models/gemini-embedding-001"


def test_index_records_which_model_built_it(client, auth):
    headers, collection_id, _ = auth()
    response = client.post(
        f"/api/collections/{collection_id}/documents",
        headers=headers,
        files={"files": ("notes.md", SAMPLE_MD.encode(), "text/markdown")},
    )
    assert response.status_code == 201, response.text

    path = vectorstore.index_path(collection_id)
    assert (path / vectorstore.SIGNATURE_FILE).exists()
    assert vectorstore._read_signature(path) == vectorstore._signature()


def test_searching_an_index_from_another_model_is_refused(tmp_path):
    """A dimension clash may not raise; vectors from another model can simply
    rank nonsense highly, which is worse than an error."""
    (tmp_path / vectorstore.SIGNATURE_FILE).write_text(
        json.dumps({"signature": "google:models/some-other-model"})
    )
    with pytest.raises(vectorstore.EmbeddingMismatch) as excinfo:
        vectorstore._assert_compatible(tmp_path)

    message = str(excinfo.value)
    assert "some-other-model" in message
    assert "Re-index" in message


def test_an_unstamped_index_is_accepted(tmp_path):
    """Indexes built before the stamp existed must keep working."""
    vectorstore._assert_compatible(tmp_path)


def test_a_matching_signature_is_accepted(tmp_path):
    vectorstore._write_signature(tmp_path)
    vectorstore._assert_compatible(tmp_path)
