"""LLM backend dispatch tests: no network, no API key needed.

The openai_compat backend is exercised against a fake requests.post (payload
shape, auth header, usage accounting, error surfacing); the anthropic default
is asserted structurally without touching the SDK.
"""
import json

import pytest

from quant_agent import config, llm


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _ok(content="hello", prompt_tokens=11, completion_tokens=7):
    return _FakeResp({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens},
    })


def test_default_backend_is_anthropic(monkeypatch):
    monkeypatch.delenv("QA_LLM_BACKEND", raising=False)
    assert config.llm_backend() == "anthropic"


def test_blank_backend_means_anthropic(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "")
    assert config.llm_backend() == "anthropic"


def test_openai_compat_request_shape_and_usage(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("QA_LLM_BASE_URL", "http://127.0.0.1:8645/v1")
    monkeypatch.delenv("QA_LLM_API_KEY", raising=False)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json, headers=headers,
                        timeout=timeout)
        return _ok("the answer", prompt_tokens=100, completion_tokens=25)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    llm.reset_usage()
    out = llm.complete("SYS", "USER", max_tokens=321)

    assert out == "the answer"
    assert captured["url"] == "http://127.0.0.1:8645/v1/chat/completions"
    # hermes proxy / local servers ignore the bearer; a placeholder is sent
    assert captured["headers"]["Authorization"] == "Bearer sk-unused"
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert captured["payload"]["max_tokens"] == 321
    assert captured["payload"]["model"] == config.MODEL
    u = llm.get_usage()
    assert (u["input_tokens"], u["output_tokens"], u["calls"]) == (100, 25, 1)
    assert u["backend"] == "openai_compat"
    assert u["base_url"] == "http://127.0.0.1:8645/v1"


def test_explicit_api_key_is_sent(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("QA_LLM_API_KEY", "my-token")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(headers=headers)
        return _ok()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    llm.complete("s", "u")
    assert captured["headers"]["Authorization"] == "Bearer my-token"


def test_typed_content_parts_are_joined(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    parts = [{"type": "text", "text": "foo "}, {"type": "text", "text": "bar"}]
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _ok(content=parts))
    assert llm.complete("s", "u") == "foo bar"


def test_complete_json_over_openai_backend(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    fenced = "```json\n{\"answer\": 42}\n```"
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _ok(content=fenced))
    assert llm.complete_json("s", "u") == {"answer": 42}


def test_connection_error_is_actionable(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")

    def boom(*a, **k):
        raise llm.requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(llm.requests, "post", boom)
    with pytest.raises(RuntimeError) as e:
        llm.complete("s", "u")
    msg = str(e.value)
    assert "hermes proxy" in msg and "QA_LLM_BASE_URL" in msg


def test_http_error_is_surfaced_with_status(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    monkeypatch.setattr(
        llm.requests, "post",
        lambda *a, **k: _FakeResp({"error": "no such model"}, status=404))
    with pytest.raises(RuntimeError) as e:
        llm.complete("s", "u")
    assert "404" in str(e.value)


def test_unexpected_shape_is_surfaced(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _FakeResp({"nonsense": True}))
    with pytest.raises(RuntimeError) as e:
        llm.complete("s", "u")
    assert "Unexpected response shape" in str(e.value)


def test_unknown_backend_fails_loudly(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "banana")
    with pytest.raises(RuntimeError) as e:
        llm.complete("s", "u")
    assert "banana" in str(e.value)


def test_unknown_model_pricing_is_flagged_not_guessed(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    monkeypatch.delenv("QA_PRICE_INPUT_PER_MTOK", raising=False)
    monkeypatch.delenv("QA_PRICE_OUTPUT_PER_MTOK", raising=False)
    p = config.pricing_for("Hermes-4-405B")
    assert p == {"input": 0.0, "output": 0.0, "unknown": True}


def test_anthropic_backend_keeps_default_pricing_fallback(monkeypatch):
    monkeypatch.delenv("QA_LLM_BACKEND", raising=False)
    monkeypatch.delenv("QA_PRICE_INPUT_PER_MTOK", raising=False)
    monkeypatch.delenv("QA_PRICE_OUTPUT_PER_MTOK", raising=False)
    assert config.pricing_for("some-unknown-model") == config.DEFAULT_PRICING


def test_pricing_env_override_wins(monkeypatch):
    monkeypatch.setenv("QA_PRICE_INPUT_PER_MTOK", "0.9")
    monkeypatch.setenv("QA_PRICE_OUTPUT_PER_MTOK", "1.1")
    assert config.pricing_for("anything") == {"input": 0.9, "output": 1.1}


def test_usage_marks_unknown_pricing(monkeypatch):
    monkeypatch.setenv("QA_LLM_BACKEND", "openai_compat")
    monkeypatch.delenv("QA_PRICE_INPUT_PER_MTOK", raising=False)
    monkeypatch.delenv("QA_PRICE_OUTPUT_PER_MTOK", raising=False)
    monkeypatch.setattr(config, "MODEL", "Hermes-4-405B")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _ok())
    llm.reset_usage()
    llm.complete("s", "u")
    u = llm.get_usage()
    assert u["pricing"] == "unknown" and u["cost_usd"] == 0.0
    assert u["input_tokens"] == 11  # counts stay exact even when unpriced
