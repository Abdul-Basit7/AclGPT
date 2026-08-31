import json

from tests.conftest import FAKE_ANSWER, SAMPLE_MD


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
    chat = client.post("/api/chats", headers=headers, json={"collection_id": collection_id})
    assert chat.status_code == 201, chat.text
    return headers, collection_id, chat.json()


def test_streaming_answer_with_sources(client, auth):
    headers, _, chat = seed(client, auth)

    response = client.post(
        f"/api/chats/{chat['id']}/messages",
        headers=headers,
        json={"content": "How long does ACL recovery take?"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "user_message"
    assert "sources" in kinds and "token" in kinds and kinds[-1] == "done"

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1, "answer should arrive as multiple chunks, not one blob"
    assert "".join(tokens) == FAKE_ANSWER

    sources = next(e["sources"] for e in events if e["type"] == "sources")
    assert sources, "expected retrieved sources"
    assert sources[0]["filename"] == "notes.md"
    assert sources[0]["page"] == 1
    assert sources[0]["snippet"]

    done = events[-1]
    assert done["failed"] is False
    assert done["message_id"] is not None


def test_turns_are_persisted_and_titled(client, auth):
    headers, _, chat = seed(client, auth)
    assert chat["title"] == "New chat"

    client.post(
        f"/api/chats/{chat['id']}/messages",
        headers=headers,
        json={"content": "How long does ACL recovery take?"},
    )

    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "How long does ACL recovery take?"
    assert messages[1]["content"] == FAKE_ANSWER
    assert messages[1]["sources"][0]["filename"] == "notes.md"

    chats = client.get("/api/chats", headers=headers).json()
    assert chats[0]["title"] == "How long does ACL recovery take?"


def test_history_accumulates_across_turns(client, auth):
    headers, _, chat = seed(client, auth)
    for question in ("First question?", "Second question?", "Third question?"):
        client.post(
            f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": question}
        )

    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert [m["role"] for m in messages] == ["user", "assistant"] * 3
    assert all(m["content"].strip() for m in messages), "no empty turns stored"


def test_empty_collection_still_answers(client, auth):
    """With no documents Sourcery is a general assistant, not a broken one."""
    headers, collection_id, _ = auth()
    chat = client.post(
        "/api/chats", headers=headers, json={"collection_id": collection_id}
    ).json()

    response = client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "Hello?"}
    )
    events = parse_sse(response.text)
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == FAKE_ANSWER, "the model should be asked, not short-circuited"
    assert next(e["sources"] for e in events if e["type"] == "sources") == []
    assert events[-1]["failed"] is False


def test_model_selection_is_validated(client, auth):
    headers, collection_id, _ = auth()

    default = client.post(
        "/api/chats", headers=headers, json={"collection_id": collection_id}
    ).json()
    assert default["model"] == "test-model-a", "default is the first available model"

    bogus = client.post(
        "/api/chats",
        headers=headers,
        json={"collection_id": collection_id, "model": "not-a-real-model"},
    ).json()
    assert bogus["model"] == "test-model-a", "unknown ids fall back to the default"

    updated = client.patch(
        f"/api/chats/{default['id']}", headers=headers, json={"model": "test-model-b"}
    ).json()
    assert updated["model"] == "test-model-b"


def test_llm_failure_streams_an_error_not_a_crash(client, auth, monkeypatch):
    from app.services import llm as llm_service

    headers, _, chat = seed(client, auth)

    class Exploding:
        def stream(self, messages):
            raise RuntimeError("groq is down")

    monkeypatch.setattr(
        llm_service, "get_llm", lambda model, web_search=False: Exploding()
    )

    response = client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "hi"}
    )
    assert response.status_code == 200
    events = parse_sse(response.text)
    error = next(e for e in events if e["type"] == "error")
    assert "groq is down" in error["detail"]
    assert events[-1]["type"] == "done" and events[-1]["failed"] is True

    # The failed turn leaves the question but stores no empty assistant message.
    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert [m["role"] for m in messages] == ["user"]


def test_chats_are_isolated_between_users(client, auth):
    headers_a, _, chat = seed(client, auth)
    headers_b, collection_b, _ = auth()

    assert client.get(f"/api/chats/{chat['id']}/messages", headers=headers_b).status_code == 404
    assert client.delete(f"/api/chats/{chat['id']}", headers=headers_b).status_code == 404
    assert client.get("/api/chats", headers=headers_b).json() == []
    # Cannot attach a chat to someone else's collection.
    assert client.post(
        "/api/chats", headers=headers_a, json={"collection_id": collection_b}
    ).status_code == 404


def test_delete_chat(client, auth):
    headers, _, chat = seed(client, auth)
    assert client.delete(f"/api/chats/{chat['id']}", headers=headers).status_code == 204
    assert client.get("/api/chats", headers=headers).json() == []


def test_editing_a_question_rewinds_the_chat(client, auth):
    headers, _, chat = seed(client, auth)
    for question in ("First question?", "Second question?"):
        client.post(
            f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": question}
        )

    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert len(messages) == 4
    second_question_id = messages[2]["id"]

    deleted = client.delete(
        f"/api/chats/{chat['id']}/messages/{second_question_id}", headers=headers
    )
    assert deleted.status_code == 204

    # The second turn is gone; the first survives, ready for a re-ask.
    remaining = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert [m["content"] for m in remaining] == ["First question?", FAKE_ANSWER]

    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "Rewritten?"}
    )
    after = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    assert [m["content"] for m in after][2] == "Rewritten?"


def test_rewinding_past_the_first_question_frees_the_title(client, auth):
    headers, _, chat = seed(client, auth)
    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "Original?"}
    )
    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()

    client.delete(f"/api/chats/{chat['id']}/messages/{messages[0]['id']}", headers=headers)
    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "Rewritten?"}
    )

    titles = {c["id"]: c["title"] for c in client.get("/api/chats", headers=headers).json()}
    assert titles[chat["id"]] == "Rewritten?"


def test_rewind_is_scoped_to_the_owning_chat(client, auth):
    headers_a, _, chat = seed(client, auth)
    headers_b, _, _ = auth()
    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers_a, json={"content": "Mine?"}
    )
    message_id = client.get(
        f"/api/chats/{chat['id']}/messages", headers=headers_a
    ).json()[0]["id"]

    assert client.delete(
        f"/api/chats/{chat['id']}/messages/{message_id}", headers=headers_b
    ).status_code == 404
    assert client.delete(
        f"/api/chats/{chat['id']}/messages/999999", headers=headers_a
    ).status_code == 404


def test_timestamps_are_serialised_as_utc(client, auth):
    """SQLite returns naive datetimes; without an offset a browser reads them as local."""
    headers, _, chat = seed(client, auth)
    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "When?"}
    )

    messages = client.get(f"/api/chats/{chat['id']}/messages", headers=headers).json()
    for message in messages:
        assert message["created_at"].endswith("Z") or "+00:00" in message["created_at"]

    assert chat["created_at"].endswith("Z") or "+00:00" in chat["created_at"]


def test_suggestions_follow_the_conversation(client, auth):
    headers, collection_id, chat = seed(client, auth)
    client.post(
        f"/api/chats/{chat['id']}/messages", headers=headers, json={"content": "Recovery?"}
    )

    response = client.get(f"/api/chats/{chat['id']}/suggestions", headers=headers)
    assert response.status_code == 200
    suggestions = response.json()["suggestions"]
    # The fake model replies with a single line, so at most one survives parsing.
    assert isinstance(suggestions, list) and len(suggestions) <= 3
    assert all(isinstance(s, str) and s for s in suggestions)

    starters = client.get(f"/api/collections/{collection_id}/suggestions", headers=headers)
    assert starters.status_code == 200
    assert isinstance(starters.json()["suggestions"], list)


def test_suggestions_are_scoped_to_the_owner(client, auth):
    _, _, chat = seed(client, auth)
    headers_b, _, _ = auth()
    assert client.get(
        f"/api/chats/{chat['id']}/suggestions", headers=headers_b
    ).status_code == 404


def test_suggestion_parsing_strips_list_markers():
    from app.services.rag import _first_lines

    reply = (
        "Here are some questions:\n"
        "1. What does the author say about deadlines?\n"
        "- How should key result areas be chosen?\n"
        "* How should key result areas be chosen?\n"
        '"What is zero-based thinking?"\n'
        "short\n"
        + "x" * 200
        + "\n"
        "Which chapter covers procrastination?\n"
    )
    assert _first_lines(reply, 3) == [
        "What does the author say about deadlines?",
        "How should key result areas be chosen?",
        "What is zero-based thinking?",
    ]


def test_suggestions_survive_a_model_failure(client, auth, monkeypatch):
    from app.services import llm as llm_service

    headers, _, chat = seed(client, auth)

    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_service, "stream_chat", boom)
    response = client.get(f"/api/chats/{chat['id']}/suggestions", headers=headers)
    assert response.status_code == 200
    assert response.json()["suggestions"] == []
