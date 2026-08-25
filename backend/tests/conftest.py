"""Test config. Environment is set before the app is imported, so the suite runs
against a throwaway data directory and never touches real provider APIs."""

import os
import tempfile

_TMP_DATA = tempfile.mkdtemp(prefix="aclbot-tests-")
os.environ["DATA_DIR"] = _TMP_DATA
os.environ["GOOGLE_API_KEY"] = "test-google-key"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["SECRET_KEY"] = "test-secret-key"

import itertools  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from app.schemas import ModelInfo  # noqa: E402

from app.main import app  # noqa: E402
from app.services import llm as llm_service  # noqa: E402
from app.services import vectorstore  # noqa: E402

FAKE_ANSWER = "ACL rehab takes months (notes.md, page 1)."
TEST_MODELS = [
    ModelInfo(id="test-model-a", label="Test A"),
    ModelInfo(id="test-model-b", label="Test B"),
]
_emails = itertools.count()


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch):
    """Swap Google embeddings and Groq for local fakes."""
    monkeypatch.setattr(
        vectorstore, "get_embeddings", lambda: DeterministicFakeEmbedding(size=64)
    )
    # Model discovery would otherwise hit the Groq API.
    monkeypatch.setattr(
        llm_service, "list_models", lambda force_refresh=False: TEST_MODELS
    )
    monkeypatch.setattr(
        llm_service,
        "get_llm",
        lambda model: GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content=FAKE_ANSWER))
        ),
    )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth(client):
    """Register a fresh user and return (headers, starter_collection_id)."""

    def _register():
        email = f"user{next(_emails)}@example.com"
        response = client.post(
            "/api/auth/register", json={"email": email, "password": "supersecret123"}
        )
        assert response.status_code == 201, response.text
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        collections = client.get("/api/collections", headers=headers).json()
        return headers, collections[0]["id"], email

    return _register


SAMPLE_MD = (
    "# ACL rehabilitation\n\n"
    "The anterior cruciate ligament stabilises the knee joint. "
    "Reconstruction surgery is followed by a structured rehabilitation programme. "
    "Early phases focus on restoring range of motion and reducing swelling. "
    "Later phases introduce strength work, balance training and sport-specific drills. "
    "Return to sport typically happens between nine and twelve months after surgery. "
    "Graft choice, age and adherence to physiotherapy all influence the timeline. "
) * 6
