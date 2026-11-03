import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval" / "attribution" / "attribution.jsonl"


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


def _cases():
    if not DS.exists():
        subprocess.run([sys.executable, str(ROOT / "eval" / "build_attribution_eval.py")],
                       cwd=str(ROOT), check=True)
    return {c["id"]: c for c in
            (json.loads(l) for l in DS.read_text().splitlines() if l.strip())}


CL_SID = "attribution:cl:2025-03-31_eval_book.csv:Sector:250000"

# Verbatim answer texts from the failing live run (agent arm, 2026-07-02).
LIVE_PNL_ANSWER = (
    "## Cooper-Li Attribution \u2014 Total Dollar P&L\n\nUsing the Cooper-Li "
    "attribution framework applied to the `2025-03-31_eval_book.csv` portfolio "
    "(Sector grouping, Cap Weight as the universe, $250,000 notional), the "
    "**total dollar P&L is \u2013$793.70**, decomposed as follows "
    "[attribution:cl:2025-03-31_eval_book.csv:Sector:250000]:\n\n"
    "| Component | USD |\n|---|---|\n"
    "| Market / Notional Exposure (NE) | +$2,569.08 |\n"
    "| Bucket / Allocation Effect (BE) | +$220.73 |\n"
    "| Long Selection Effect (LSE) | \u2013$1,900.12 |\n"
    "| Short Selection Effect (SSE) | \u2013$1,683.40 |\n"
    "| **Total P&L** | **\u2013$793.70** |\n"
)

LIVE_SSE_ANSWER = (
    "## Total Short Selection Effect (SSE) \u2014 Cooper-Li Attribution\n\n"
    "Under Cooper-Li attribution by Sector, with a notional of $250,000, the "
    "**total Short Selection Effect (SSE) is **\u2212$1,683.40****.\n\n"
    "[attribution:cl:2025-03-31_eval_book.csv:Sector:250000]\n\n"
    "### Breakdown by Sector:\n| Sector | SSE ($) |\n|--------|---------|\n"
    "| Alpha | \u221287.47 |\n| Beta | \u2212597.64 |\n| Chi | \u2212998.28 |\n"
    "| **Total** | **\u22121,683.40** |\n"
)


def _result(text, citations=(CL_SID,)):
    return {"answer": {"status": "ok", "text": text,
                       "citations": list(citations), "artifacts": []},
            "tools_fired": ["equity_attribution"]}


def test_live_pnl_answer_now_passes():
    score = _score()
    ok, fails = score(_cases()["attr_cl_total_pnl"], _result(LIVE_PNL_ANSWER))
    assert ok, fails


def test_live_sse_answer_now_passes():
    score = _score()
    ok, fails = score(_cases()["attr_cl_sse"], _result(LIVE_SSE_ANSWER))
    assert ok, fails


def test_format_matrix():
    score = _score()
    case = _cases()["attr_cl_total_pnl"]  # expected -793.7 (+/-5)
    forms = [
        "total P&L is -$793.70",        # ASCII minus + $
        "total P&L is \u2013$793.70",   # en dash + $
        "total P&L is \u2212$793.70",   # minus sign + $
        "total P&L is $-793.70",        # $ then minus
        "total P&L was ($793.70)",      # accounting parens with $
        "total P&L is -793.70 dollars", # plain
        "total P&L is \u22120.0031748 of notional, i.e. \u2212$793.70",
    ]
    for f in forms:
        ok, fails = score(case, _result(f))
        assert ok, (f, fails)


def test_sign_flip_still_fails():
    # The whole point of attr_cl_sse: a positive SSE (classic by-hand sign
    # error) must FAIL even though the magnitude is exact - in every format.
    score = _score()
    case = _cases()["attr_cl_sse"]  # expected -1683.4
    for f in ("SSE is $1,683.40", "SSE is +$1,683.40", "SSE is 1683.40 dollars"):
        ok, fails = score(case, _result(f))
        assert not ok, f
        assert any("expected_value" in x for x in fails), (f, fails)


def test_bf_typographic_minus_percent():
    # The FLAKY BF mechanism: unicode minus on a bare decimal / percent form.
    score = _score()
    case = _cases()["attr_bf_active"]  # expected -0.024019 (+/-0.0005)
    for f in ("active return of \u22120.024019",
              "active return of \u20132.4019%",
              "active return of -0.0240"):
        ok, fails = score(case, _result(f, citations=("attribution:bf:2025-03-31_eval_book.csv:Sector",)))
        assert ok, (f, fails)


def test_hyphen_range_no_false_negative_number():
    # Dash normalization turns '2024\u20132025' into '2024-2025', which the
    # extractor reads as [2024, -2025]. Guard the realistic hazard: that
    # spurious negative must not rescue a case whose expected value it happens
    # to be far from anyway - i.e. extraction still finds the REAL value when
    # present, and a range alone does not produce a match.
    score = _score()
    case = _cases()["attr_cl_total_pnl"]  # expected -793.7
    ok, fails = score(case, _result("Over 2024\u20132025 the book returned nothing specific."))
    assert not ok  # no matching number extracted from the range text
