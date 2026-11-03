"""Single wrapper around the LLM backend.

Every LLM layer goes through complete() or complete_json(); no layer talks to
an SDK directly. Two backends, selected by QA_LLM_BACKEND (read at call time):

- "anthropic" (default): the Anthropic SDK, exactly as before v11.8.
- "openai_compat": any OpenAI-compatible /chat/completions endpoint over plain
  HTTP. This is how the agent runs through Hermes Agent: `hermes proxy`
  exposes your Hermes-managed provider subscription as a credential-attaching
  pass-through at http://127.0.0.1:8645/v1 (the default base URL here), and
  the Hermes API server (default port 8642) speaks the same format with the
  full agent behind it. Ollama / vLLM / LiteLLM / LM Studio work identically.

Failure policy: an unreachable endpoint, non-200 status, or unexpected
response shape raises immediately with the URL and the remedy. There is
deliberately NO silent fallback to the other backend - a mid-run backend
switch would contaminate any eval comparison.

complete_json retries once with the parse error fed back, on both backends.
"""
import json
import os
import re

import requests

from . import config

_client = None

# Per-process token accumulator. Each LLM call adds its usage here; the eval
# runner reads and resets it per case to report tokens + cost. In-process and
# single-threaded per case (the harness runs each case in its own subprocess),
# so a module global is safe and simple.
_USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def reset_usage() -> None:
    _USAGE["input_tokens"] = 0
    _USAGE["output_tokens"] = 0
    _USAGE["calls"] = 0


def get_usage() -> dict:
    """Return a copy of accumulated usage plus the computed USD cost.

    Token counts are exact on both backends. On openai_compat, a model the
    pricing table does not know reports cost_usd=0 with "pricing": "unknown"
    (see config.pricing_for); the backend and base URL are included so eval
    reports state which stack produced the numbers.
    """
    price = config.pricing_for(config.MODEL)
    cost = (_USAGE["input_tokens"] / 1_000_000) * price["input"] \
        + (_USAGE["output_tokens"] / 1_000_000) * price["output"]
    out = {**_USAGE, "model": config.MODEL, "backend": config.llm_backend(),
           "input_price_per_mtok": price["input"],
           "output_price_per_mtok": price["output"],
           "cost_usd": round(cost, 6)}
    if price.get("unknown"):
        out["pricing"] = "unknown"
    if config.llm_backend() == "openai_compat":
        out["base_url"] = config.llm_base_url()
    return out


def client():
    """Anthropic SDK client (anthropic backend only; imported lazily so the
    package is not required when running openai_compat)."""
    global _client
    if _client is None:
        from anthropic import Anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Launch via run.py to configure it.")
        _client = Anthropic(api_key=key)
    return _client


def _complete_anthropic(system: str, user: str, max_tokens: int) -> str:
    resp = client().messages.create(
        model=config.MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Capture token usage. resp.usage carries the exact input/output token
    # counts billed for this call.
    try:
        _USAGE["input_tokens"] += int(resp.usage.input_tokens)
        _USAGE["output_tokens"] += int(resp.usage.output_tokens)
        _USAGE["calls"] += 1
    except Exception:
        pass
    return "".join(b.text for b in resp.content if b.type == "text")


def _complete_openai_compat(system: str, user: str, max_tokens: int) -> str:
    base = config.llm_base_url().rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.llm_api_key() or 'sk-unused'}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=config.llm_timeout_s())
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach the OpenAI-compatible endpoint at {url} "
            f"({e.__class__.__name__}: {e}). Is the server running? For "
            f"Hermes Agent, `hermes proxy` serves http://127.0.0.1:8645/v1; "
            f"set QA_LLM_BASE_URL in keys.local.py if yours differs."
        ) from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM endpoint {url} returned HTTP {resp.status_code} for model "
            f"'{config.MODEL}': {resp.text[:500]}"
        )
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected response shape from {url}: {json.dumps(data)[:500]}"
        ) from e
    if isinstance(content, list):  # some servers return typed content parts
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict))
    text = content or ""
    usage = data.get("usage") or {}
    try:
        _USAGE["input_tokens"] += int(usage.get("prompt_tokens", 0))
        _USAGE["output_tokens"] += int(usage.get("completion_tokens", 0))
        _USAGE["calls"] += 1
    except Exception:
        pass
    return text


def complete(system: str, user: str, max_tokens: int = config.MAX_TOKENS) -> str:
    backend = config.llm_backend()
    if backend == "openai_compat":
        return _complete_openai_compat(system, user, max_tokens)
    if backend == "anthropic":
        return _complete_anthropic(system, user, max_tokens)
    raise RuntimeError(
        f"Unknown QA_LLM_BACKEND '{backend}'. "
        f"Valid values: 'anthropic' (default), 'openai_compat'."
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def complete_json(system: str, user: str, max_tokens: int = config.MAX_TOKENS) -> dict:
    raw = complete(system, user, max_tokens)
    last_err = None
    for attempt in range(2):
        try:
            return json.loads(_FENCE.sub("", raw).strip())
        except json.JSONDecodeError as e:
            last_err = e
            if attempt == 0:
                raw = complete(
                    system,
                    user
                    + f"\n\nYour previous reply was not valid JSON ({e}). "
                    f"Reply again with valid JSON only.\nPrevious reply:\n{raw}",
                    max_tokens,
                )
    raise RuntimeError(f"LLM returned invalid JSON twice: {last_err}")
