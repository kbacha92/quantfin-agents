"""Tool interface. The as-of guardrail is enforced here, once, for every tool.

A tool that forgets the as-of date cannot be called: __call__ rejects any
invocation without a real date before store-specific code runs.
"""
import re
from abc import ABC, abstractmethod
from datetime import date

from ..schemas import Evidence

# Tokens that look like tickers but are filing-form / filler / finance-jargon
# noise, so they are never treated as a user-named ticker by the allowlist gate.
# Note the failure direction: a missing entry here only OVER-permits (a jargon
# token counts as "named", harmless); it never blocks a real ticker the user
# typed. Extend freely when new jargon shows up in refusal logs.
_TICKER_NOISE = {
    # filler / filing forms (v4 original)
    "K", "Q", "A", "S", "SEC", "AND", "THE", "FOR", "OF", "IS", "IN", "ON", "AS",
    # indicator / metric acronyms (v11.4): appeared as false "named tickers"
    "RSI", "SMA", "EMA", "MACD", "ATR", "VWAP", "IV", "PE", "PB", "PS",
    "EPS", "EBIT", "ROE", "ROA", "ROIC", "FCF", "CAGR", "YTD", "YOY", "QOQ",
    # instrument / filing / org jargon
    "ETF", "IPO", "CEO", "CFO", "GAAP", "USD", "EUR", "GBP", "OTC", "NYSE",
    "ATM", "OTM", "ITM", "CALL", "PUT", "VOL", "BS",
}


def query_tickers(query: str) -> set[str]:
    """Tickers the user actually named: 2-5 char all-caps tokens (optional
    leading $), the conventional way a ticker is written. Deliberately strict -
    a guessed or inferred ticker the user never typed is not in the set."""
    out = set()
    for m in re.findall(r"\$?\b([A-Za-z]{1,5})\b", query or ""):
        if m.isupper() and len(m) >= 2 and m not in _TICKER_NOISE:
            out.add(m.upper())
    return out


def assert_ticker_named(requested: str, query: str, tool_name: str) -> None:
    """Guardrail (v4 Option B): refuse a ticker the user did not name.

    Prevents the agent from mutating or inventing a ticker (e.g. SYN -> SYNB,
    ACME -> ACMR) and reaching live data behind the user's back. Multi-ticker
    args like 'AAPL,MSFT' are split and each part checked. If no query context
    was threaded in (query is empty), the gate is a no-op so the tool stays
    usable outside the orchestrator."""
    if not query:
        return
    allowed = query_tickers(query)
    parts = [p.strip().upper() for p in re.split(r"[,\s]+", str(requested)) if p.strip()]
    for part in parts:
        if part and part not in allowed:
            hint = (
                " If the user wrote a company NAME (e.g. 'Apple') rather than a ticker, do not "
                "guess the symbol - answer that the ticker must be stated explicitly, so the "
                "user can re-ask naming it (e.g. 'AAPL')."
                if not allowed else ""
            )
            raise RuntimeError(
                f"{tool_name} refused: ticker '{part}' was not named in the question "
                f"(named tickers: {sorted(allowed) or 'none'}). The agent must not fetch or "
                f"compute on a ticker the user did not explicitly request.{hint}"
            )


class Tool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, as_of: date, **kwargs) -> list[Evidence]: ...

    def __call__(self, as_of, **kwargs) -> list[Evidence]:
        if not isinstance(as_of, date):
            raise ValueError(f"{self.name}: as_of date is required (point-in-time guardrail).")
        return self.run(as_of=as_of, **kwargs)
