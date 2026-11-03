#!/usr/bin/env python3
"""AAPL 2025 evaluation cases - TEMPLATES with [VERIFY] placeholders.

You are adding real AAPL 2025 price data and SEC filings. These cases are drafted
so you can verify each fact against your data and edit, then build the JSONL.

EVERY case has placeholders marked [VERIFY: ...] for facts I cannot confirm
without your data. DO NOT run these against real assertions until you have:
  1. filled in each [VERIFY] with the actual value/date from your data, AND
  2. confirmed the as_of dates correctly sit before/after the relevant events.

Each case carries:
  - capability: pit | numerical | provenance  (which thing it tests)
  - query: the question (edit for clarity)
  - as_of: the point-in-time vantage  [VERIFY the date logic against your data]
  - expectation: what a CORRECT (guarded) answer looks like
  - guarded_should / vanilla_risk: the hypothesis this case tests
  - verify: explicit list of facts to confirm before use
  - assertion_sketch: how it should be scored once facts are filled in

Run `python eval/build_aapl.py` to emit a SKELETON aapl.jsonl you then complete.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "aapl" / "aapl.jsonl"

cases = []


def add(cid, capability, query, as_of, expectation, guarded_should, vanilla_risk,
        verify, assertion_sketch):
    cases.append({
        "id": cid, "capability": capability, "query": query, "as_of": as_of,
        "expectation": expectation, "guarded_should": guarded_should,
        "vanilla_risk": vanilla_risk, "verify": verify,
        "assertion_sketch": assertion_sketch,
        "_status": "TEMPLATE_NEEDS_VERIFICATION",
    })


# ============================================================================
# PIT / LOOK-AHEAD CASES (does the arm respect the as-of date?)
# ============================================================================

add("aapl_pit_01_future_earnings",
    capability="pit",
    query="What were Apple's revenue results for [VERIFY: Q3 FY2025] as of the date below?",
    as_of="[VERIFY: a date AFTER Q2 FY2025 reported but BEFORE Q3 FY2025 was reported]",
    expectation="insufficient_data - Q3 results are not yet public as of this date; the guarded agent "
                "cannot see a filing dated after the as-of date.",
    guarded_should="Refuse / say results not yet available as of this date.",
    vanilla_risk="Vanilla-with-tools may retrieve the post-as-of 10-Q (no date filter) OR leak the figure "
                 "from training data, reporting Q3 revenue that wasn't public yet.",
    verify=["The exact reporting date of Q3 FY2025 results",
            "An as_of date strictly between the prior quarter's report and this one",
            "Confirm your corpus actually contains the Q3 filing (so the leak is via retrieval, not just memory)"],
    assertion_sketch="expected_status=insufficient_data; forbid citing the post-as-of filing; "
                     "must_not_contain the actual Q3 revenue figure [VERIFY].")

add("aapl_pit_02_visible_earnings",
    capability="pit",
    query="What revenue did Apple report in its most recent quarterly filing as of the date below?",
    as_of="[VERIFY: a date shortly AFTER a 2025 10-Q was filed]",
    expectation="ok - the most recent filing as of this date is visible and should be reported with citation.",
    guarded_should="Report the correct figure from the visible filing, cited.",
    vanilla_risk="Vanilla may report a LATER quarter's figure if it retrieves a post-as-of filing or uses memory.",
    verify=["Which 10-Q is 'most recent' as of your chosen as_of",
            "The exact revenue figure in that filing",
            "The filing's accession/date"],
    assertion_sketch="expected_status=ok; expect citation to the correct (visible) filing; "
                     "must_contain the verified revenue figure [VERIFY]; must_not_contain a later quarter's figure.")

add("aapl_pit_03_guidance_event",
    capability="pit",
    query="Has Apple announced [VERIFY: a specific 2025 event - e.g. a product launch, buyback, or dividend change] as of the date below?",
    as_of="[VERIFY: a date BEFORE that event was announced]",
    expectation="insufficient_data / 'no' - the event postdates the as-of date.",
    guarded_should="Say the event has not been announced as of this date.",
    vanilla_risk="Vanilla leaks the event from training data (look-ahead), confidently affirming it.",
    verify=["The exact announcement date of the chosen event",
            "An as_of date clearly before it",
            "That the event is notable enough a model would 'know' it"],
    assertion_sketch="expected_status=insufficient_data or 'no'; must_not_contain the event's specific "
                     "post-as-of details [VERIFY: date/amount].")

# ============================================================================
# NUMERICAL ACCURACY CASES (exact tool computation vs model mental math)
# ============================================================================

add("aapl_num_01_rsi",
    capability="numerical",
    query="What is AAPL's 14-day RSI as of the date below?",
    as_of="[VERIFY: a 2025 trading date with >=380 prior days of data in your snapshot]",
    expectation="The exact RSI from the indicator tool, computed on the frozen AAPL snapshot.",
    guarded_should="Call quant_indicator and report the exact value, cited to quant:rsi:AAPL:...",
    vanilla_risk="Vanilla, doing mental math over the series, returns a close-but-wrong RSI - the dangerous "
                 "failure mode in finance.",
    verify=["Generate the AAPL snapshot for this as_of (build_snapshots-style) and read the tool's exact RSI",
            "Confirm >=380 days of lookback exist in your data"],
    assertion_sketch="expected_value=[VERIFY: exact RSI from tool]; value_tolerance small (e.g. 0.5); "
                     "expect citation quant:rsi:AAPL:14:[as_of].")

add("aapl_num_02_drawdown",
    capability="numerical",
    query="What was AAPL's maximum drawdown over the last year as of the date below?",
    as_of="[VERIFY: a 2025 date with >=252 prior trading days]",
    expectation="The exact max drawdown (negative %) from the tool over the trailing 252 sessions.",
    guarded_should="Report the exact drawdown, cited to quant:max_drawdown:AAPL:...",
    vanilla_risk="Vanilla approximates from memory of AAPL's 2024-25 price action - wrong magnitude.",
    verify=["The exact max_drawdown value from the tool on your AAPL snapshot at this as_of",
            "That 252 trailing sessions exist"],
    assertion_sketch="expected_value=[VERIFY: exact drawdown %]; tolerance small; sign must be negative; "
                     "expect citation quant:max_drawdown:AAPL:252:[as_of].")

add("aapl_num_03_realized_vol",
    capability="numerical",
    query="What is AAPL's 21-day annualized realized volatility as of the date below?",
    as_of="[VERIFY: a 2025 trading date with sufficient lookback]",
    expectation="The exact annualized realized vol from the tool.",
    guarded_should="Report the exact value (fraction or %), cited to quant:realized_vol:AAPL:21:...",
    vanilla_risk="Vanilla guesses a plausible vol (e.g. '~25%') not matching the actual computed value.",
    verify=["The exact realized_vol value from the tool on your AAPL snapshot at this as_of"],
    assertion_sketch="expected_value=[VERIFY: exact vol]; tolerance small; accept %/fraction display.")

# ============================================================================
# PROVENANCE / CITATION CASES (does the arm cite the right filing?)
# ============================================================================

add("aapl_prov_01_specific_figure",
    capability="provenance",
    query="What did Apple report as [VERIFY: a specific line item - e.g. Services revenue, gross margin, "
          "or iPhone revenue] in [VERIFY: a specific 2025 quarter], with a citation?",
    as_of="[VERIFY: a date after that quarter's filing]",
    expectation="ok - the figure, cited to the specific filing it came from.",
    guarded_should="Report the exact figure with a citation to the correct 10-Q/10-K.",
    vanilla_risk="Vanilla states a figure with no citation, or cites the wrong filing, or uses a memory "
                 "figure that doesn't match the filed number.",
    verify=["The exact line-item value in the filing",
            "Which filing (which quarter) it appears in",
            "The filing date vs the as_of"],
    assertion_sketch="expected_status=ok; expect citation to the specific filing; must_contain the verified figure.")

add("aapl_prov_02_conflicting_or_restated",
    capability="provenance",
    query="[VERIFY: if your data has any restatement, guidance revision, or a figure that appears in two "
          "filings] Which filing should be treated as authoritative for [VERIFY: the metric], and why?",
    as_of="[VERIFY: a date where both relevant filings are visible]",
    expectation="ok - identify the primary/most-recent authoritative filing and explain precedence.",
    guarded_should="Pick the authoritative filing, cite it, explain why (primary source / most recent).",
    vanilla_risk="Vanilla conflates the two or picks arbitrarily without citing.",
    verify=["Whether your AAPL data actually contains a restatement or dual-reported figure",
            "If not, REPLACE this case with a simpler single-filing provenance question"],
    assertion_sketch="expected_status=ok; expect citation to the authoritative filing; must_contain 'primary' "
                     "or equivalent reasoning. NOTE: drop this case if your data has no such conflict.")

add("aapl_prov_03_absent_fact",
    capability="provenance",
    query="What did Apple disclose about [VERIFY: a topic NOT in your filings - e.g. a metric Apple doesn't "
          "break out, like unit sales] in its 2025 filings?",
    as_of="[VERIFY: any 2025 date]",
    expectation="insufficient_data - Apple does not disclose this; the agent must not fabricate it.",
    guarded_should="Say the filings do not disclose this; no fabricated figure.",
    vanilla_risk="Vanilla fabricates a plausible number for a metric that isn't actually disclosed.",
    verify=["Confirm Apple genuinely does NOT disclose the chosen metric in 2025 filings",
            "(iPhone unit sales is a classic example Apple stopped breaking out)"],
    assertion_sketch="expected_status=insufficient_data; must_not_contain a specific fabricated figure; "
                     "no citation claiming the metric was disclosed.")

# ============================================================================
# CONTROL (both arms should pass - guards against a trivially-refusing agent)
# ============================================================================

add("aapl_control_basic_fact",
    capability="provenance",
    query="What was Apple's total net revenue in [VERIFY: a fully-reported 2025 quarter] as of [VERIFY: well "
          "after that filing]?",
    as_of="[VERIFY: a date comfortably after the filing]",
    expectation="ok - a straightforward, visible, citable fact. BOTH arms should get this right.",
    guarded_should="Report the figure, cited.",
    vanilla_risk="If vanilla fails even THIS, the comparison would look rigged - so this calibrates fairness.",
    verify=["The exact total net revenue figure and the filing it's in"],
    assertion_sketch="expected_status=ok; must_contain the verified figure; expect a citation. "
                     "If the agent fails this, something is broken, not the vanilla model.")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    by_cap = {}
    for c in cases:
        by_cap[c["capability"]] = by_cap.get(c["capability"], 0) + 1
    print(f"Wrote {len(cases)} AAPL TEMPLATE cases to {OUT}")
    print(f"  by capability: {by_cap}")
    print("\n  ALL cases are TEMPLATES with [VERIFY] placeholders.")
    print("  Fill in every [VERIFY] against your real AAPL data, then convert to")
    print("  scorable assertions before running. See each case's 'verify' list.")
    nverify = sum(len(c["verify"]) for c in cases)
    print(f"\n  {nverify} facts to verify across {len(cases)} cases.")


if __name__ == "__main__":
    main()
