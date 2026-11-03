"""Central configuration: budgets, precedence rules, paths, model names.

Data-root resolution (mirrors the plugin system):
- QUANT_DATA_DIR env var, else <project>/data  -> docs/, traces/, chroma/, outputs/
- QUANT_KNOWLEDGE_DIR env var, else <data root>/knowledge
Set the env vars for per-project workspaces with one shared knowledge library.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str):
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


DATA_DIR = _env_path("QUANT_DATA_DIR") or (ROOT / "data")
DOCS_DIR = DATA_DIR / "docs"
KNOWLEDGE_DIR = _env_path("QUANT_KNOWLEDGE_DIR") or (DATA_DIR / "knowledge")
OUTPUTS_DIR = DATA_DIR / "outputs"
API_CACHE_DIR = DATA_DIR / "api_cache"
TRACES_DIR = DATA_DIR / "traces"
CHROMA_DIR = DATA_DIR / "chroma"
for _p in (DOCS_DIR, KNOWLEDGE_DIR, OUTPUTS_DIR, TRACES_DIR, CHROMA_DIR, API_CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

_DOCS_NOTE = DOCS_DIR / "put-period-documents-here.txt"
if not _DOCS_NOTE.exists():
    _DOCS_NOTE.write_text(
        "FACTS about the world go here: filings, news, transcripts, notes.\n"
        "Prefix filenames with the publication date (2025-02-12_AAPL_10-K.txt);\n"
        "documents published after the as-of date are invisible to searches.\n"
        "This placeholder is never indexed.\n")
_KNOW_NOTE = KNOWLEDGE_DIR / "put-domain-knowledge-here.txt"
if not _KNOW_NOTE.exists():
    _KNOW_NOTE.write_text(
        "TIMELESS METHODOLOGY goes here: textbooks, journal articles, formula\n"
        "references. Not date-filtered - citable for methods and definitions\n"
        "only, never for facts about the as-of period. This placeholder is\n"
        "never indexed.\n")

APP_VERSION = "11.8"  # single source of truth for human-facing version labels
MODEL = os.environ.get("QA_MODEL", "claude-sonnet-4-6")
MAX_ITERS = int(os.environ.get("QA_MAX_ITERS", "6"))
MAX_TOKENS = 1500
EVIDENCE_BUDGET_CHARS = 12000

# Per-MTok (million-token) USD pricing for cost reporting, by model string.
# Used only to translate token usage into an approximate dollar figure in eval
# reports; update if Anthropic pricing changes. Keys are matched by prefix so
# dated model strings (e.g. claude-haiku-4-5-20251001) resolve to their family.
# Verify current prices at https://www.anthropic.com/pricing before relying on
# the dollar figures - token COUNTS are exact; the $ is only as right as this table.
PRICING_PER_MTOK = {
    "claude-opus-4":   {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4":  {"input": 1.00,  "output": 5.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}  # fall back to sonnet-class


# ---- LLM backend selection (v11.8) ----------------------------------------
# Read at CALL time (functions, not import-time constants) so values pushed
# into the environment by keys.apply() - and monkeypatched in tests - take
# effect without an import-order dance. MODEL stays an import-time constant
# because keys are applied before the app/eval subprocess imports this module.

def llm_backend() -> str:
    """"anthropic" (default) or "openai_compat" (any /chat/completions server:
    Hermes Agent's `hermes proxy` / API server, Ollama, vLLM, LiteLLM, ...)."""
    return os.environ.get("QA_LLM_BACKEND", "anthropic").strip().lower() or "anthropic"


def llm_base_url() -> str:
    """Base URL for the openai_compat backend. Default is Hermes Agent's
    subscription proxy (`hermes proxy` -> http://127.0.0.1:8645/v1)."""
    return os.environ.get("QA_LLM_BASE_URL", "http://127.0.0.1:8645/v1").strip()


def llm_api_key() -> str:
    """Bearer token for the openai_compat backend. Local proxies (hermes
    proxy, Ollama) ignore it; blank sends the placeholder "sk-unused"."""
    return os.environ.get("QA_LLM_API_KEY", "").strip()


def llm_timeout_s() -> float:
    return float(os.environ.get("QA_LLM_TIMEOUT_S", "120"))


def pricing_for(model: str) -> dict:
    """Return {input, output} $/MTok for a model string (prefix match).

    QA_PRICE_INPUT_PER_MTOK / QA_PRICE_OUTPUT_PER_MTOK (set together) override
    everything - the way to price non-Anthropic models served via
    openai_compat. Without an override, a model the table does not know is
    reported as $0 with unknown=True on the openai_compat backend rather than
    a silently wrong sonnet-class guess; token COUNTS stay exact either way.
    """
    pi = os.environ.get("QA_PRICE_INPUT_PER_MTOK")
    po = os.environ.get("QA_PRICE_OUTPUT_PER_MTOK")
    if pi and po:
        return {"input": float(pi), "output": float(po)}
    for prefix, price in PRICING_PER_MTOK.items():
        if model.startswith(prefix):
            return price
    if llm_backend() == "openai_compat":
        return {"input": 0.0, "output": 0.0, "unknown": True}
    return DEFAULT_PRICING

# Source precedence used by assembly: lower number = higher trust.
PRECEDENCE = {"edgar": 0, "doc_kb": 1, "prices": 2, "quant": 2, "model": 2, "chart": 3, "knowledge": 4}
DEFAULT_PRECEDENCE = 5
