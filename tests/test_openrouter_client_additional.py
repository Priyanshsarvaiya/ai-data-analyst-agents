from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient


@dataclass
class _FakeResponse:
    status_code: int
    payload: dict

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("error", request=req, response=resp)

    def json(self) -> dict:
        return self.payload


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse(200, {"choices": []})


def _mk_client(monkeypatch: pytest.MonkeyPatch, fake_http_client) -> OpenRouterClient:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "Client", lambda timeout=None: fake_http_client)
    return OpenRouterClient(timeout_s=1)


def test_openrouter_chat_parses_string_content(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient([_FakeResponse(200, {"choices": [{"message": {"content": "hello"}}]})])
    client = _mk_client(monkeypatch, fake)
    assert client.chat(model="x", messages=[{"role": "user", "content": "hi"}]) == "hello"


def test_openrouter_chat_parses_list_content(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": [{"text": "part1"}, {"text": "part2"}]}}]}
    fake = _FakeClient([_FakeResponse(200, payload)])
    client = _mk_client(monkeypatch, fake)
    assert client.chat(model="x", messages=[{"role": "user", "content": "hi"}]) == "part1\npart2"


def test_openrouter_chat_parses_dict_content(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": {"text": "dict-content"}}}]}
    fake = _FakeClient([_FakeResponse(200, payload)])
    client = _mk_client(monkeypatch, fake)
    assert client.chat(model="x", messages=[{"role": "user", "content": "hi"}]) == "dict-content"


def test_openrouter_chat_returns_empty_on_no_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient([_FakeResponse(200, {"choices": []})])
    client = _mk_client(monkeypatch, fake)
    assert client.chat(model="x", messages=[{"role": "user", "content": "hi"}]) == ""


def test_openrouter_chat_retries_and_returns_empty_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_FakeResponse(429, {}), _FakeResponse(429, {}), _FakeResponse(429, {}), _FakeResponse(429, {})]
    fake = _FakeClient(responses)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    client = _mk_client(monkeypatch, fake)
    out = client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
    assert out == ""
    assert fake.calls == 4
