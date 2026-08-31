"""Web search is a model capability, not a wish -- these tests pin the gating."""

import json

import pytest

from app.services import llm as llm_service
from app.services import rag


def parse_sse(text):
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def make_chat(client, headers, collection_id, **body):
    response = client.post(
        "/api/chats", headers=headers, json={"collection_id": collection_id, **body}
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- capability reporting ---


def test_groq_compound_is_the_only_native_searcher():
    """Verified against the live API: only Compound responses carry
    `executed_tools`."""
    assert llm_service._natively_searches("groq/compound")
    assert llm_service._natively_searches("groq/compound-mini")
    assert not llm_service._natively_searches("openai/gpt-oss-120b")
    assert not llm_service._natively_searches("")


def test_capability_is_read_from_the_advertised_model_list():
    """The list the UI receives and the rule the server enforces must agree, or
    the UI offers a control the server silently ignores."""
    advertised = {m.id: m.supports_web_search for m in llm_service.list_models()}
    for model_id, expected in advertised.items():
        assert llm_service.supports_web_search(model_id) is expected

    assert not llm_service.supports_web_search("model-that-does-not-exist")


# --- persistence and clamping ---


def test_web_search_is_refused_for_incapable_models(client, auth):
    headers, collection_id, _ = auth()
    chat = make_chat(
        client, headers, collection_id, model="test-model-a", web_search=True
    )
    assert chat["web_search"] is False, "a model that cannot search must not claim to"


def test_web_search_is_kept_for_capable_models(client, auth):
    headers, collection_id, _ = auth()
    chat = make_chat(
        client, headers, collection_id, model="test-searcher", web_search=True
    )
    assert chat["web_search"] is True


def test_switching_to_an_incapable_model_clears_the_flag(client, auth):
    headers, collection_id, _ = auth()
    chat = make_chat(
        client, headers, collection_id, model="test-searcher", web_search=True
    )

    switched = client.patch(
        f"/api/chats/{chat['id']}", headers=headers, json={"model": "test-model-a"}
    ).json()
    assert switched["web_search"] is False, (
        "leaving the flag on would show web search enabled for a model that "
        "silently ignores it"
    )


def test_web_search_can_be_toggled_on_a_capable_model(client, auth):
    headers, collection_id, _ = auth()
    chat = make_chat(client, headers, collection_id, model="test-searcher")
    assert chat["web_search"] is False

    on = client.patch(
        f"/api/chats/{chat['id']}", headers=headers, json={"web_search": True}
    ).json()
    assert on["web_search"] is True

    off = client.patch(
        f"/api/chats/{chat['id']}", headers=headers, json={"web_search": False}
    ).json()
    assert off["web_search"] is False


# --- prompt selection ---


def fake_doc(text="rehab takes months"):
    from langchain_core.documents import Document as LCDocument

    return LCDocument(
        page_content=text, metadata={"filename": "notes.md", "page": 1, "document_id": 1}
    )


@pytest.mark.parametrize(
    "docs,web,expected,forbidden",
    [
        ([fake_doc()], False, "Do not use", "search the web"),
        ([fake_doc()], True, "You can search the web", "Do not use"),
        ([], True, "no indexed documents", "Do not use"),
        ([], False, "cannot search the web", "You can search the web"),
    ],
)
def test_prompt_matches_what_the_model_can_reach(docs, web, expected, forbidden):
    """Offering citations the model cannot obtain is how it starts inventing them."""
    prompt = rag.build_prompt(docs, web)
    assert expected in prompt
    assert forbidden not in prompt


def test_document_context_is_included_when_present():
    prompt = rag.build_prompt([fake_doc("quadriceps strength gates progression")], True)
    assert "quadriceps strength gates progression" in prompt


# --- web results become sources ---


# The shape Groq actually sends, confirmed against a live stream: search_results
# is an object wrapping a results list, not a list. An earlier version of this
# test asserted a bare list, passed, and the feature still failed in production.
def live_shape(results):
    return [{"index": 0, "type": "search", "search_results": {"results": results}}]


def test_executed_tools_become_linked_sources():
    executed = live_shape(
        [
            {
                "title": "Rust 1.99 released",
                "url": "https://blog.rust-lang.org/1.99",
                "content": "  Rust  1.99   is now available. ",
            },
            # A duplicate URL must not appear twice in the panel.
            {"title": "dup", "url": "https://blog.rust-lang.org/1.99", "content": "x"},
            {"title": "no url", "url": "", "content": "ignored"},
        ]
    )
    sources = rag.web_sources(executed)
    assert len(sources) == 1
    source = sources[0]
    assert source["url"] == "https://blog.rust-lang.org/1.99"
    assert source["filename"] == "Rust 1.99 released"
    assert source["snippet"] == "Rust 1.99 is now available."
    assert source["page"] is None
    assert source["document_id"] is None


def test_web_sources_survive_missing_fields():
    assert rag.web_sources([]) == []
    assert rag.web_sources([{"type": "search"}]) == []
    assert rag.web_sources([{"search_results": None}]) == []
    # A search that has started but returned nothing yet.
    assert rag.web_sources(live_shape([])) == []
    assert rag.web_sources([{"search_results": {}}]) == []


def test_a_bare_result_list_is_also_accepted():
    """Tolerate the other plausible shape rather than raising mid-answer."""
    sources = rag.web_sources(
        [{"search_results": [{"title": "T", "url": "https://x.test", "content": "c"}]}]
    )
    assert [s["url"] for s in sources] == ["https://x.test"]


def test_document_sources_are_not_marked_as_web():
    (source,) = rag.build_sources([fake_doc()])
    assert source["url"] is None, "a document passage has no URL to link to"


# --- error translation ---


def test_oversized_search_error_is_explained():
    message = llm_service.friendly_error(
        RuntimeError("Error code: 413 - Request Entity Too Large")
    )
    assert "web pages" in message.lower()
    assert "413" not in message, "the raw status code means nothing to a user"
    assert "search off" in message.lower() or "narrower" in message.lower()


def test_rate_limit_error_is_explained():
    assert "rate limit" in llm_service.friendly_error(
        RuntimeError("429 Too Many Requests")
    ).lower()


def test_other_errors_are_passed_through():
    assert "boom" in llm_service.friendly_error(RuntimeError("boom"))


# --- transport selection ---


def test_searching_models_bypass_langchain(monkeypatch):
    """langchain_groq drops `executed_tools`, so a search must use the SDK path.

    Verified against the live API: every LangChain chunk has an empty
    `additional_kwargs` and only `finish_reason` in `response_metadata`.
    """
    calls = []
    monkeypatch.setattr(
        llm_service, "_stream_via_sdk", lambda m, msgs: calls.append(("sdk", m)) or iter(())
    )
    monkeypatch.setattr(
        llm_service,
        "_stream_via_langchain",
        lambda m, msgs, web: calls.append(("langchain", m)) or iter(()),
    )
    monkeypatch.setattr(llm_service, "_natively_searches", lambda m: m == "searcher")
    monkeypatch.setattr(llm_service, "resolve_model", lambda m: m)

    messages = [{"role": "user", "content": "hi"}]

    list(llm_service.stream_chat("searcher", messages, web_search=True))
    list(llm_service.stream_chat("searcher", messages, web_search=False))
    list(llm_service.stream_chat("plain", messages, web_search=True))

    assert calls == [
        ("sdk", "searcher"),
        ("langchain", "searcher"),
        # Asking for web search on a model that cannot do it must not route to
        # the SDK path, which exists only to read a search record.
        ("langchain", "plain"),
    ]


def test_history_is_converted_for_langchain():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert llm_service._lc_messages(messages) == [
        ("system", "rules"),
        ("human", "q"),
        ("ai", "a"),
    ]


def test_history_roles_round_trip():
    history = [("user", "q1"), ("assistant", "a1"), ("assistant", "   ")]
    # A blank assistant turn is dropped rather than sent as an empty message.
    assert rag.to_chat_messages(history) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
