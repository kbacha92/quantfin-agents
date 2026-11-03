"""Deterministic-layer tests: no network, no API key needed.

The LLM layers get evaluated separately against the golden set (next milestone).
"""
from datetime import date

from quant_agent import assembly
from quant_agent.schemas import Evidence, State, ToolCall
from quant_agent.tools.base import Tool


def test_merge_dedupes_and_orders_by_precedence():
    s = State(query="q", as_of=date(2025, 1, 1))
    e1 = Evidence(source_id="a", store="prices", content="x", pub_date=date(2024, 1, 1))
    e2 = Evidence(source_id="a", store="prices", content="x")  # duplicate id
    e3 = Evidence(source_id="b", store="edgar", content="y", pub_date=date(2024, 6, 1))
    assert assembly.merge(s, [e1, e2, e3]) == 2
    assert s.evidence[0].source_id == "b"  # edgar (primary) outranks prices


def test_novelty_check():
    s = State(query="q", as_of=date(2025, 1, 1))
    s.action_history.append(ToolCall(tool="t", args={"x": 1}))
    assert s.seen(ToolCall(tool="t", args={"x": 1}))
    assert not s.seen(ToolCall(tool="t", args={"x": 2}))


def test_as_of_guardrail_is_unavoidable():
    class Dummy(Tool):
        name = "dummy"
        description = "dummy"
        def run(self, as_of, **kwargs):
            return []

    d = Dummy()
    try:
        d(as_of=None)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert d(as_of=date(2025, 1, 1)) == []


def test_evidence_digest_marks_non_pit():
    e = Evidence(source_id="api:x", store="apis", content="live", pit_safe=False)
    assert "NON-PIT" in e.digest()


def test_method_evidence_digest_markers():
    e = Evidence(source_id="knowledge:x", store="knowledge", content="c",
                 kind="method", pit_safe=False, meta={"postdates_as_of": True})
    d = e.digest()
    assert "METHOD" in d and "POSTDATES AS-OF" in d


def test_answer_artifacts_default_empty():
    from quant_agent.schemas import Answer
    assert Answer(status="ok", text="t").artifacts == []


# --- v3 hardening tests ---

def test_rsi_handles_no_down_days_without_inf():
    import numpy as np
    import pandas as pd
    from quant_agent.tools.quant.indicators import rsi

    up_only = pd.Series(np.arange(1, 40, dtype=float))  # strictly increasing
    r = rsi(up_only, 14).dropna()
    assert np.isfinite(r).all()
    assert abs(float(r.iloc[-1]) - 100.0) < 1e-9  # all gains -> 100, not inf

    flat = pd.Series([100.0] * 40)
    rf = rsi(flat, 14).dropna()
    assert np.isfinite(rf).all() and abs(float(rf.iloc[-1]) - 50.0) < 1e-9


def test_retroactive_caveat_digest_distinct_from_live():
    from quant_agent.schemas import Evidence

    live = Evidence(source_id="api:x", store="apis", content="c", pit_safe=False)
    retro = Evidence(source_id="yf:x", store="prices", content="c", pit_safe=False,
                     meta={"pit_caveat": "retroactive"})
    assert "LIVE DATA" in live.digest()
    assert "retroactively adjusted" in retro.digest() and "LIVE DATA" not in retro.digest()


def test_first_row_guards_empty_chroma_shapes():
    from quant_agent.tools.doc_kb import _first_row

    for res in ({}, {"documents": [], "metadatas": []},
                {"documents": [[]], "metadatas": [[]]},
                {"documents": [None], "metadatas": [None]}):
        d, m = _first_row(res)
        assert d == [] and m == []
    d, m = _first_row({"documents": [["a"]], "metadatas": [[{"k": 1}]]})
    assert d == ["a"] and m == [{"k": 1}]


def test_citation_detection_tolerates_reformatting():
    from datetime import date
    from quant_agent.schemas import Evidence, State
    from quant_agent.synthesize import _cited_ids

    s = State(query="q", as_of=date(2025, 1, 1))
    s.evidence = [Evidence(source_id="doc:2025-02-12_AAPL_10-K.txt#chunk3",
                           store="doc_kb", content="x")]
    # exact
    assert _cited_ids("Risk [doc:2025-02-12_AAPL_10-K.txt#chunk3].", s)
    # reformatted spacing/case inside brackets still recovered
    assert _cited_ids("See [ doc:2025-02-12_AAPL_10-K.txt#Chunk3 ].", s)
    # nothing cited
    assert _cited_ids("No sources.", s) == []


def test_edgar_refuses_unnamed_ticker():
    from datetime import date
    from quant_agent.tools.edgar import EdgarFetchTool

    t = EdgarFetchTool()
    # invented ticker for a query about a different name -> refused pre-network
    try:
        t.run(as_of=date(2025, 1, 1), ticker="ACMR",
              _query="What did ACME report in its annual report?")
        assert False, "should have refused unnamed ticker"
    except RuntimeError as e:
        assert "refused" in str(e).lower()

    # ticker extraction ignores form-name noise
    assert t._query_tickers("fetch AAPL latest 10-K") == {"AAPL"}
    assert "K" not in t._query_tickers("AAPL 10-K")


def test_price_tools_refuse_unnamed_ticker():
    from datetime import date
    from quant_agent.tools.prices import PriceHistoryTool
    from quant_agent.tools.quant.indicators import QuantIndicatorTool

    try:
        PriceHistoryTool().run(as_of=date(2025, 3, 14), ticker="SYNB",
                               _query="closing price for SYN", lookback_days=90)
        assert False, "should refuse SYNB"
    except RuntimeError as e:
        assert "refused" in str(e).lower()

    # correlation with an unnamed second ticker is refused
    try:
        QuantIndicatorTool().run(as_of=date(2025, 3, 14), ticker="CRA,CRZ",
                                 indicator="correlation",
                                 _query="correlation between CRA and CRB", window=252)
        assert False, "should refuse unnamed CRZ"
    except RuntimeError as e:
        assert "refused" in str(e).lower()


def test_offline_strict_refuses_live_fetch(monkeypatch, tmp_path):
    """v4 Option A: with QUANT_OFFLINE_ONLY set, load_prices never goes live;
    a missing snapshot raises instead of fetching."""
    import os
    from datetime import date
    monkeypatch.setenv("QUANT_OFFLINE_ONLY", "1")
    from quant_agent.tools.prices import load_prices
    try:
        load_prices("NOSNAPSHOT", date(2025, 3, 14), 90)
        assert False, "offline-strict should refuse a missing snapshot"
    except RuntimeError as e:
        assert "offline-strict" in str(e).lower()


def test_query_retry_on_empty_but_nonempty_collection(monkeypatch):
    import quant_agent.tools.doc_kb as dk
    from quant_agent.tools.doc_kb import _first_row

    state = {"n": 0}

    class FakeCol:
        def __init__(self, mode):
            self.mode = mode
        def query(self, **kw):
            state["n"] += 1
            if self.mode == "hiccup" and state["n"] == 1:
                return {"documents": [[]], "metadatas": [[]]}
            if self.mode == "empty":
                return {"documents": [[]], "metadatas": [[]]}
            return {"documents": [["doc"]], "metadatas": [[{"file": "x", "chunk": 0}]]}
        def count(self):
            return 0 if self.mode == "empty" else 5

    # hiccup: non-empty collection, first query empty -> retry recovers
    monkeypatch.setattr(dk, "_collection", lambda name: FakeCol("hiccup"))
    state["n"] = 0
    res = dk._query_with_retry("docs", ["q"], 4)
    assert _first_row(res)[0] and state["n"] == 2

    # genuine empty: count==0 -> no retry, stays empty
    monkeypatch.setattr(dk, "_collection", lambda name: FakeCol("empty"))
    state["n"] = 0
    res = dk._query_with_retry("docs", ["q"], 4)
    assert not _first_row(res)[0] and state["n"] == 1
