import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from server import (  # noqa: E402
    STOP_PHRASES,
    create_app,
    strip_stop_phrases,
)


def make_client() -> TestClient:
    app = create_app(fake=True)
    return TestClient(app)


def test_health_endpoint_reports_fake_mode():
    client = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "fake": True}


def test_chat_completions_returns_openai_shaped_response():
    client = make_client()
    payload = {
        "model": "conversational-agent-qlora",
        "messages": [
            {"role": "system", "content": "You are a scheduling assistant."},
            {"role": "user", "content": "I need to book a repair."},
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "conversational-agent-qlora"
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["message"]["content"]
    assert choice["finish_reason"] == "stop"


def test_fake_mode_cycles_through_canned_responses_without_loading_a_model():
    client = make_client()
    payload = {
        "model": "conversational-agent-qlora",
        "messages": [{"role": "user", "content": "hi"}],
    }
    replies = []
    for _ in range(6):
        response = client.post("/v1/chat/completions", json=payload)
        replies.append(response.json()["choices"][0]["message"]["content"])
    # Cycles deterministically through the fixture list, so we see repeats.
    assert len(set(replies)) < len(replies)


def test_invalid_role_is_rejected():
    client = make_client()
    payload = {
        "model": "conversational-agent-qlora",
        "messages": [{"role": "narrator", "content": "hi"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_strip_stop_phrases_cuts_at_the_first_match():
    text = "You're all set for Tuesday.\nCaller: wait, one more question"
    cleaned = strip_stop_phrases(text, STOP_PHRASES)
    assert cleaned == "You're all set for Tuesday."
    assert "Caller:" not in cleaned


def test_strip_stop_phrases_no_match_returns_stripped_text():
    text = "You're all set for Tuesday.  "
    assert strip_stop_phrases(text, STOP_PHRASES) == "You're all set for Tuesday."
