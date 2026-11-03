
import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from quant_agent.tools.quant.attribution import brinson_fachler, cooper_li

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval" / "attribution" / "attribution.jsonl"
BOOK = ROOT / "eval" / "attribution" / "2025-03-31_eval_book.csv"
BOOK_FUTURE = ROOT / "eval" / "attribution" / "2025-06-30_eval_book_revised.csv"
NOTIONAL = 250_000.0


def _cases():
    if not DS.exists():
        subprocess.run([sys.executable, str(ROOT / "eval" / "build_attribution_eval.py")],
                       cwd=str(ROOT), check=True)
    return [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]


def _score():
    spec = importlib.util.spec_from_file_location("run_harness",
                                                  ROOT / "eval" / "run_harness.py")
    rh = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["run_harness"]
    try:
        spec.loader.exec_module(rh)
    finally:
        sys.argv = argv
    return rh.score


def test_all_cases_have_ground_truth_or_are_refusals():
    for c in _cases():
        if c["expected_status"] == "ok":
            assert isinstance(c["expected_value"], (int, float)), c["id"]
        else:
            # refusal case pins forbidden leaks instead of a value
            assert c.get("forbid_citation_source_ids") and c.get("must_not_contain"), c["id"]


def test_cases_are_self_consistent():
    _cases()  # ensure built (also writes the CSVs)
    df = pd.read_csv(BOOK)
    bf_grp, bf = brinson_fachler(df, "Sector", "Portfolio Weight", "Cap Weight", "Return")
    cl_grp, cl = cooper_li(df, "Sector", "Portfolio Weight", "Cap Weight", "Return",
                           notional=NOTIONAL)
    expected = {
        "attr_bf_active": bf["active_return"],
        "attr_bf_allocation": bf["total_allocation"],
        "attr_bf_selection_beta": float(bf_grp.set_index("bucket").loc["Beta", "Selection"]),
        "attr_cl_total_pnl": cl["total_pnl"],
        "attr_cl_sse": cl["SSE_total"],
    }
    for c in _cases():
        if c["id"] in expected:
            assert abs(c["expected_value"] - expected[c["id"]]) <= c["value_tolerance"], c["id"]


def test_lookahead_values_genuinely_differ():
    # The forbidden future P&L must be far outside the visible book's tolerance,
    # otherwise the leak case can't discriminate.
    _cases()
    fut = pd.read_csv(BOOK_FUTURE)
    vis = pd.read_csv(BOOK)
    _, cl_f = cooper_li(fut, "Sector", "Portfolio Weight", "Cap Weight", "Return",
                        notional=NOTIONAL)
    _, cl_v = cooper_li(vis, "Sector", "Portfolio Weight", "Cap Weight", "Return",
                        notional=NOTIONAL)
    assert abs(cl_f["total_pnl"] - cl_v["total_pnl"]) > 1000


def test_scorer_passes_correct_and_fails_wrong():
    score = _score()
    cases = {c["id"]: c for c in _cases()}
    c = cases["attr_cl_sse"]
    sid = c["expect_citation_source_ids"][0]
    good = {"answer": {"status": "ok",
                       "text": f"The short selection effect is {c['expected_value']:.2f} dollars [{sid}].",
                       "citations": [sid], "artifacts": []},
            "tools_fired": ["equity_attribution"]}
    ok, fails = score(c, good)
    assert ok, fails
    # wrong sign (the classic by-hand error) must fail
    bad = {"answer": {"status": "ok",
                      "text": f"The short selection effect is {-c['expected_value']:.2f} dollars [{sid}].",
                      "citations": [sid], "artifacts": []},
           "tools_fired": ["equity_attribution"]}
    ok, fails = score(c, bad)
    assert not ok and any("expected_value" in f for f in fails)
    # right number without the tool citation must fail (vanilla arm signature)
    uncited = {"answer": {"status": "ok",
                          "text": f"I estimate SSE at {c['expected_value']:.2f} dollars.",
                          "citations": [], "artifacts": []},
               "tools_fired": []}
    ok, fails = score(c, uncited)
    assert not ok and any("required citation" in f for f in fails)


def test_scorer_lookahead_case():
    score = _score()
    cases = {c["id"]: c for c in _cases()}
    c = cases["attr_pit_lookahead"]
    # correct refusal passes
    refusal = {"answer": {"status": "insufficient_data",
                          "text": "The revised book is dated after the as-of date and is not visible.",
                          "citations": [], "artifacts": []},
               "tools_fired": ["equity_attribution"]}
    ok, fails = score(c, refusal)
    assert ok, fails
    # a leak (confident answer citing the future file) fails on status AND citation
    fut_sid = c["forbid_citation_source_ids"][0]
    leak = {"answer": {"status": "ok",
                       "text": "The revised book's total P&L is 5,254.68 dollars.",
                       "citations": [fut_sid], "artifacts": []},
            "tools_fired": ["equity_attribution"]}
    ok, fails = score(c, leak)
    assert not ok
    assert any("status" in f for f in fails)
    assert any("forbidden citation" in f for f in fails)
    assert any("must_not_contain" in f for f in fails)


def test_agent_tool_end_to_end_on_eval_book(tmp_path, monkeypatch):
    # The staged eval book, run through the actual tool at the eval as-of date,
    # must produce the case's expected value and the exact required source_id -
    # i.e. the assertion chain (builder -> tool -> scorer) is closed.
    from quant_agent import config
    from quant_agent.tools.quant.attribution_tool import AttributionTool
    _cases()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    d = tmp_path / "portfolios"
    d.mkdir()
    import shutil
    shutil.copy(BOOK, d / BOOK.name)
    shutil.copy(BOOK_FUTURE, d / BOOK_FUTURE.name)
    t = AttributionTool()
    cases = {c["id"]: c for c in _cases()}
    c = cases["attr_cl_total_pnl"]
    ev = t.run(as_of=date(2025, 4, 15), op="cooper_li", file=BOOK.name,
               notional=NOTIONAL)
    assert ev[0].source_id == c["expect_citation_source_ids"][0]
    assert f"{c['expected_value']:.1f}".rstrip("0").rstrip(".") in ev[0].content or \
           str(round(c["expected_value"], 2)) in ev[0].content
    # and the lookahead file is refused at that date
    import pytest
    with pytest.raises(RuntimeError, match="lookahead"):
        t.run(as_of=date(2025, 4, 15), op="cooper_li", file=BOOK_FUTURE.name,
              notional=NOTIONAL)
