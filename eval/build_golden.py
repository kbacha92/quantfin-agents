#!/usr/bin/env python3
"""Assemble the golden evaluation set (~50 cases) as JSONL.

Each record is a typed contract describing one query, the corpus state it runs
against, and assertions over the agent's Answer. Quant cases pull their exact
expected values from expected_quant.json (produced by build_snapshots.py) so the
golden numbers are guaranteed to equal what the real indicator functions emit.

Record schema (see eval/README.md for the full spec):
  id              unique case id
  area            one of: pit | provenance | quant | methodology
  query           the user question
  as_of           ISO date for the PIT vantage point
  seed_docs       list of doc filenames that must be in the FACTS corpus
  seed_knowledge  list of knowledge filenames that must be in the METHOD corpus
  snapshots       list of snapshot CSV filenames to stage into api_cache
  expected_status ok | insufficient_data
  expect_tools_any/expect_tools_all   tool names that should fire
  expect_citation_source_ids          exact source_ids that MUST be cited
  forbid_citation_source_ids          exact source_ids that must NOT be cited (PIT leak)
  expect_citation_store_any           at least one citation from these stores
  forbid_citation_store               no citation may come from these stores
  must_contain / must_not_contain     case-insensitive substring asserts on text
  expected_value / value_tolerance    numeric ground truth (quant cases)
  rationale       what guardrail/behavior this case checks
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE / "golden_dataset"
EXP = json.loads((DS / "expected_quant.json").read_text())
QAS_OF = EXP["as_of"]  # quant as-of (2025-03-14)
QC = EXP["cases"]

# Corpus filenames (must match seed_docs / seed_knowledge folders)
DOC_10K = "2024-11-01_ACME_10-K.txt"
DOC_8K = "2025-02-10_ACME_8-K.txt"
DOC_PRESS = "2025-05-20_ACME_press.txt"
DOC_NOTE = "2025-01-15_ACME_analyst_note.txt"
KN_RSI = "rsi_methodology.txt"
KN_DD = "drawdown_definition.txt"
KN_VOL = "2025-09-01_vol_forecasting_paper.txt"
KN_EMA = "ema_definition.txt"

ALL_DOCS = [DOC_10K, DOC_8K, DOC_PRESS, DOC_NOTE]
ALL_KN = [KN_RSI, KN_DD, KN_VOL, KN_EMA]

# Exact source_ids of documents that must never be cited when they postdate the
# as-of date. Citing these is a point-in-time leak; citing a *visible* doc is
# not. These are the precise counterpart to a blanket store-level forbid.
SID_8K = f"doc:{DOC_8K}#chunk0"
SID_PRESS = f"doc:{DOC_PRESS}#chunk0"
SID_NOTE = f"doc:{DOC_NOTE}#chunk0"

cases: list[dict] = []


def add(**rec):
    rec.setdefault("seed_docs", [])
    rec.setdefault("seed_knowledge", [])
    rec.setdefault("snapshots", [])
    rec.setdefault("expect_tools_any", [])
    rec.setdefault("expect_tools_all", [])
    rec.setdefault("expect_citation_source_ids", [])
    rec.setdefault("forbid_citation_source_ids", [])
    rec.setdefault("expect_citation_store_any", [])
    rec.setdefault("forbid_citation_store", [])
    rec.setdefault("must_contain", [])
    rec.setdefault("must_not_contain", [])
    rec.setdefault("expect_artifact", False)
    cases.append(rec)


# ----------------------------------------------------------------------------
# AREA 1: PIT GUARDRAIL  (as-of filtering; post-as-of doc must be invisible)
# ----------------------------------------------------------------------------

add(id="pit_01_visible_10k",
    area="pit",
    query="What customer-concentration risk does ACME disclose in its annual report?",
    as_of="2024-12-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_tools_any=["doc_search"],
    expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0"],
    must_contain=["38", "customer"],
    rationale="10-K filed 2024-11-01 is visible at as-of 2024-12-01; concentration risk must surface.")

add(id="pit_02_future_8k_invisible",
    area="pit",
    query="Did ACME report a revenue decline in its Q4 fiscal 2025 results?",
    as_of="2024-12-01",
    seed_docs=ALL_DOCS,
    expected_status="insufficient_data",
    forbid_citation_source_ids=[SID_8K],
    must_not_contain=["1.15 billion", "4 percent decline"],
    rationale="The 8-K is dated 2025-02-10; at as-of 2024-12-01 it must be invisible, so the answer cannot cite it and should be insufficient. Citing the visible 10-K it searched is fine.")

add(id="pit_03_boundary_same_day",
    area="pit",
    query="Summarize ACME's preliminary Q4 fiscal 2025 revenue figure.",
    as_of="2025-02-10",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_8K}#chunk0"],
    must_contain=["1.15 billion"],
    rationale="pub_date <= as_of is inclusive; a doc filed exactly on the as-of date is visible.")

add(id="pit_04_boundary_day_before",
    area="pit",
    query="What were ACME's preliminary Q4 fiscal 2025 revenue results?",
    as_of="2025-02-09",
    seed_docs=ALL_DOCS,
    expected_status="insufficient_data",
    forbid_citation_source_ids=[SID_8K],
    rationale="One day before the 8-K filing date, it must not be retrievable; citing the visible 10-K is fine.")

add(id="pit_05_acquisition_future",
    area="pit",
    query="Has ACME acquired Beacon Materials?",
    as_of="2025-03-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    must_not_contain=["acquisition of Beacon", "800 million", "$800 million"],
    forbid_citation_source_ids=[SID_PRESS],
    rationale="Existential 'Has X happened?' before the deal is visible. Answering 'no such acquisition "
              "appears in the record as of this date' is a COMPLETE answer (ok), like pit_13. The PIT "
              "guardrail is enforced by forbidding the future press-release citation and the deal details: "
              "the agent must answer 'none found' WITHOUT seeing or naming the future acquisition. "
              "(Contrast combo_02, which asks about the price REACTION to the deal - content of an unseen "
              "event - and is therefore insufficient_data.)")

add(id="pit_06_acquisition_visible_later",
    area="pit",
    query="Describe ACME's acquisition of Beacon Materials and what supply-chain issue it addresses.",
    as_of="2025-06-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_PRESS}#chunk0"],
    must_contain=["Beacon", "supplier"],
    rationale="At a later as-of the press release is visible and the supply-chain link should be drawn.")

add(id="pit_07_empty_corpus_no_hallucination",
    area="pit",
    query="What did ACME report in its 2025 annual report?",
    as_of="2025-03-01",
    seed_docs=[],
    expected_status="insufficient_data",
    forbid_citation_store=["doc_kb", "edgar"],
    rationale="With no documents seeded, the agent must not fabricate filing content.")

add(id="pit_08_price_asof_truncation",
    area="pit",
    query="What is the latest available closing price summary for SYN?",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_90d.csv"],
    expected_status="ok",
    expect_tools_any=["price_history", "quant_indicator"],
    expect_citation_source_ids=[f"yfinance:SYN:{QAS_OF}:90d"],
    must_not_contain=["SYNB", "SYNA"],
    rationale="Price data is truncated at the as-of snapshot; replay is offline and deterministic. Pins the exact SYN citation so a mutated ticker (e.g. SYNB hitting live data) cannot pass on a substring match.")

# ----------------------------------------------------------------------------
# AREA 2: PROVENANCE & PRECEDENCE  (citations required; primary > derived)
# ----------------------------------------------------------------------------

add(id="prov_01_citation_required",
    area="provenance",
    query="What was ACME's reported fiscal 2024 revenue, with a citation?",
    as_of="2025-01-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0"],
    must_contain=["4.20 billion"],
    rationale="Every period-fact claim must carry a source_id citation.")

add(id="prov_02_primary_outranks_derived",
    area="provenance",
    query="ACME's fiscal 2024 revenue: the company filing says one number and an analyst note says another. Which figure should be reported and why?",
    as_of="2025-02-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0"],
    must_contain=["4.20 billion", "primary"],
    must_not_contain=[],
    rationale="On conflict, the primary 10-K (4.20B) must outrank the derived analyst estimate (4.50B); the answer should say so.")

add(id="prov_03_flag_conflict",
    area="provenance",
    query="Is there any disagreement between sources about ACME's fiscal 2024 revenue?",
    as_of="2025-02-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    must_contain=["4.20 billion", "4.50 billion"],
    rationale="The evaluator/synthesis should surface the conflict between the filing and the note, not silently pick one.")

add(id="prov_04_no_citation_for_absent_fact",
    area="provenance",
    query="What is ACME's CEO's name?",
    as_of="2025-02-01",
    seed_docs=ALL_DOCS,
    expected_status="insufficient_data",
    must_not_contain=["CEO is", "chief executive officer is"],
    rationale="No seeded document names the CEO; the agent must report insufficient data and must not state a fabricated name. It may still cite docs it searched.")

add(id="prov_05_derived_price_noncite_as_fact",
    area="provenance",
    query="Report SYN's RSI(14) and cite where the number comes from.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="ok",
    expect_tools_all=["quant_indicator"],
    expect_citation_source_ids=[QC["rsi_SYN_14"]["source_id"]],
    expected_value=QC["rsi_SYN_14"]["expected_value"], value_tolerance=0.1,
    rationale="Derived quant evidence carries its own source_id and must be cited. Value checked within display-rounding tolerance, not verbatim.")

# ----------------------------------------------------------------------------
# AREA 3: QUANT CORRECTNESS  (exact values via frozen snapshots)
# ----------------------------------------------------------------------------

def quant_case(cid, q, ckey, tol):
    c = QC[ckey]
    snaps = []
    if c["indicator"] == "correlation":
        snaps = [f"yfinance_CRA_{QAS_OF}_504d.csv", f"yfinance_CRB_{QAS_OF}_504d.csv"]
    elif c["indicator"] == "max_drawdown":
        snaps = [f"yfinance_DDN_{QAS_OF}_433d.csv"]
    else:
        snaps = [f"yfinance_SYN_{QAS_OF}_380d.csv"]
    # realized_vol is emitted as a fraction (0.1307) but the agent naturally
    # reports it as a percentage (13.07%). Skipping the verbatim-string assert
    # for it avoids a brittle false-failure; the percentage-tolerant
    # expected_value check (in the scorer) still verifies the number exactly.
    mc = [] if c["indicator"] == "realized_vol" else [c["content_value_str"]]
    add(id=cid, area="quant", query=q, as_of=QAS_OF,
        snapshots=snaps,
        expected_status="ok",
        expect_tools_all=["quant_indicator"],
        expect_citation_source_ids=[c["source_id"]],
        must_contain=mc,
        expected_value=c["expected_value"],
        value_tolerance=tol,
        rationale=f"Exact {c['indicator']}({c['window']}) value replayed offline; must match ground truth within tolerance (accepting %/fraction display).")


quant_case("quant_01_rsi", "Compute the 14-day RSI for SYN.", "rsi_SYN_14", 0.01)
quant_case("quant_02_sma", "What is the 50-day simple moving average of SYN?", "sma_SYN_50", 0.01)
quant_case("quant_03_ema", "Compute the 21-day EMA of SYN.", "ema_SYN_21", 0.01)
quant_case("quant_04_realized_vol", "What is SYN's 21-day annualized realized volatility?", "realized_vol_SYN_21", 0.001)
quant_case("quant_05_max_drawdown", "What is the maximum drawdown of DDN over the last year?", "max_drawdown_DDN_252", 0.01)
quant_case("quant_06_correlation", "What is the daily-return correlation between CRA and CRB over the last year?", "correlation_CRA_CRB_252", 0.005)

# Quant robustness / error-path cases
add(id="quant_07_correlation_one_ticker",
    area="quant",
    query="What is the correlation of CRA?",
    as_of=QAS_OF,
    snapshots=[f"yfinance_CRA_{QAS_OF}_504d.csv"],
    expected_status="insufficient_data",
    must_not_contain=["correlation of CRA and CRA"],
    rationale="Correlation needs two tickers; a single-ticker request should fail cleanly, not invent a self-correlation of 1.0.")

add(id="quant_08_unknown_indicator",
    area="quant",
    query="Compute the 'sharpe_supreme' indicator for SYN.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="insufficient_data",
    rationale="An unsupported indicator name must surface as a gap, not a fabricated value.")

add(id="quant_09_numbers_not_recomputed",
    area="quant",
    query="Give SYN's RSI(14) and do not round or recompute it.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="ok",
    expect_citation_source_ids=[QC["rsi_SYN_14"]["source_id"]],
    must_contain=[QC["rsi_SYN_14"]["content_value_str"]],
    expected_value=QC["rsi_SYN_14"]["expected_value"], value_tolerance=0.01,
    rationale="Synthesis must report the tool's number verbatim, never re-derive it.")

# ----------------------------------------------------------------------------
# AREA 4: METHODOLOGY VS FACTS  (knowledge corpus usage rules)
# ----------------------------------------------------------------------------

add(id="meth_01_definition_from_knowledge",
    area="methodology",
    query="How is the Relative Strength Index defined and what do readings above 70 mean?",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="ok",
    expect_tools_any=["knowledge_search"],
    expect_citation_store_any=["knowledge"],
    must_contain=["overbought"],
    rationale="A pure definitional question should be answered from the methodology corpus.")

add(id="meth_02_method_not_period_fact",
    area="methodology",
    query="What was ACME's realized volatility in the as-of period? Only use period facts for the number.",
    as_of="2025-03-14",
    seed_docs=ALL_DOCS,
    seed_knowledge=ALL_KN,
    expected_status="insufficient_data",
    must_not_contain=["14.2 percent", "14.2%"],
    rationale="No period realized-vol fact for ACME exists in the corpus, so the answer must be insufficient and must not state the vol paper's 14.2% finding as a period fact. Retrieving the methodology definition is acceptable; only asserting a period number from a METHOD source is the violation.")

add(id="meth_03_postdates_hindsight_flag",
    area="methodology",
    query="Define the realized-volatility estimator, citing a reference.",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="ok",
    expect_citation_store_any=["knowledge"],
    must_contain=["252"],
    rationale="The estimator definition is usable even from a post-as-of paper, but it should be flagged as methods-only/hindsight background, not a period fact.")

add(id="meth_04_drawdown_definition",
    area="methodology",
    query="What does maximum drawdown mean as a risk measure?",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="ok",
    expect_citation_store_any=["knowledge"],
    must_contain=["peak", "trough"],
    rationale="Definitional question answered from methodology, not from any price series.")

add(id="meth_05_method_plus_fact_blend",
    area="methodology",
    query="Define RSI, then compute SYN's RSI(14) using price data.",
    as_of=QAS_OF,
    seed_knowledge=ALL_KN,
    snapshots=[f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="ok",
    expect_tools_any=["knowledge_search", "quant_indicator"],
    expect_citation_source_ids=[QC["rsi_SYN_14"]["source_id"]],
    expect_citation_store_any=["knowledge", "quant"],
    must_contain=["overbought"],
    expected_value=QC["rsi_SYN_14"]["expected_value"], value_tolerance=0.1,
    rationale="A blended query should pull the definition from knowledge AND the value from the quant tool, citing each in its proper role. Value checked within display tolerance.")

# ----------------------------------------------------------------------------
# Extra PIT/EDGAR + chart behavior cases to reach a thorough ~50
# ----------------------------------------------------------------------------

add(id="pit_09_guidance_withdrawn",
    area="pit",
    query="Has ACME withdrawn its full-year revenue guidance, and when?",
    as_of="2025-03-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_8K}#chunk0"],
    must_contain=["withdrew", "guidance"],
    rationale="8-K (2025-02-10) is visible at 2025-03-01; the guidance withdrawal should be reported.")

add(id="pit_10_no_future_leakage_in_summary",
    area="pit",
    query="Give a complete summary of everything known about ACME's supply chain as of this date.",
    as_of="2025-03-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    must_contain=["single supplier"],
    must_not_contain=["Beacon", "second specialty-resin supplier"],
    rationale="A broad summary must include the 10-K supply risk but must NOT leak the 2025-05-20 Beacon acquisition that resolves it.")

for n, (cid, q, mc) in enumerate([
    ("prov_06_revenue_yoy", "By how much did ACME's fiscal 2024 revenue change year over year, with citation?", ["6 percent"]),
    ("prov_07_buyback", "Did ACME's board authorize a share buyback, and for how much?", ["300 million"]),
    ("prov_08_net_income", "What was ACME's fiscal 2024 net income, cited?", ["512 million"]),
], start=1):
    add(id=cid, area="provenance", query=q, as_of="2025-01-01",
        seed_docs=ALL_DOCS, expected_status="ok",
        expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0"],
        must_contain=mc,
        rationale="Specific period fact must be reported from the primary filing with a citation.")

add(id="meth_06_no_fabricated_knowledge",
    area="methodology",
    query="What does the Sortino ratio measure, per our methodology library?",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="insufficient_data",
    must_not_contain=["downside deviation", "sortino ratio is", "sortino ratio measures"],
    rationale="No Sortino reference is seeded; the agent must report insufficient data and must not state a fabricated Sortino definition. Retrieving nearby knowledge chunks is acceptable; asserting a period/definitional fact from them is not.")

add(id="quant_10_rsi_bounds_sanity",
    area="quant",
    query="Confirm SYN's RSI(14) is a valid value between 0 and 100.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="ok",
    expect_citation_source_ids=[QC["rsi_SYN_14"]["source_id"]],
    expected_value=QC["rsi_SYN_14"]["expected_value"], value_tolerance=0.01,
    rationale="Post-fix RSI must be finite and within [0,100]; guards the div-by-zero regression.")

# ----------------------------------------------------------------------------
# BATCH 2: additional cases to reach a thorough ~50
# ----------------------------------------------------------------------------

# More PIT boundary + multi-doc reasoning
add(id="pit_11_two_docs_visible",
    area="pit",
    query="As of this date, what do we know about ACME's revenue trajectory across its filings?",
    as_of="2025-03-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0", f"doc:{DOC_8K}#chunk0"],
    must_contain=["4.20 billion", "1.15 billion"],
    must_not_contain=["Beacon"],
    rationale="Both the 10-K and 8-K are visible and should be synthesized; the later press release must not leak.")

add(id="pit_12_analyst_note_visible",
    area="pit",
    query="What is the third-party analyst estimate of ACME fiscal 2024 revenue?",
    as_of="2025-02-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_NOTE}#chunk0"],
    must_contain=["4.50 billion"],
    rationale="The analyst note (2025-01-15) is itself a visible document and can be reported as an estimate when explicitly asked for it.")

add(id="pit_13_press_release_boundary",
    area="pit",
    query="Has ACME announced any acquisition?",
    as_of="2025-05-19",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    forbid_citation_source_ids=[SID_PRESS],
    must_not_contain=["Beacon", "$800 million"],
    rationale="Existential question: 'Has X happened?' at an as-of one day before the press release. "
              "Answering 'no acquisition is in the visible record' is a COMPLETE answer (ok), not a data gap. "
              "The point-in-time guardrail is enforced by forbidding the future press-release citation and "
              "the Beacon details - the agent must answer 'none found' WITHOUT seeing or naming the future deal. "
              "Contrast pit_02, which asks about the CONTENT of a specific invisible report (insufficient).")

add(id="pit_14_price_history_old_asof",
    area="pit",
    query="Summarize SYN's price action and period return up to the as-of date.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_90d.csv", f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="ok",
    expect_tools_any=["price_history"],
    must_contain=["SYN"],
    rationale="price_history default lookback (90) plus indicator lookback both need snapshots; tests multi-lookback staging.")

# Chart generation cases (artifact behavior)
add(id="chart_01_price_png",
    area="quant",
    query="Make a price chart for SYN.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_282d.csv"],
    expected_status="ok",
    expect_tools_any=["make_chart"],
    expect_artifact=True,
    must_contain=["SYN"],
    rationale="make_chart should produce a saved PNG artifact referenced in the answer.")

add(id="chart_02_rsi_panel",
    area="quant",
    query="Plot SYN's RSI panel.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_282d.csv"],
    expected_status="ok",
    expect_tools_any=["make_chart"],
    expect_artifact=True,
    rationale="RSI chart uses the canonical rsi() and must render without div-by-zero.")

# More provenance / precedence
add(id="prov_09_no_method_as_primary",
    area="provenance",
    query="State ACME's audited fiscal 2024 revenue. Do not use third-party estimates.",
    as_of="2025-02-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0"],
    must_contain=["4.20 billion"],
    rationale="When the user excludes estimates, the answer must state the primary 10-K figure (4.20B) as authoritative. The agent may reference the 4.50B estimate to explain why it is excluded - that is correct provenance reasoning, not a violation.")

add(id="prov_10_citation_for_each_claim",
    area="provenance",
    query="List ACME's two largest disclosed risk factors, each with a citation.",
    as_of="2025-01-01",
    seed_docs=ALL_DOCS,
    expected_status="ok",
    expect_citation_source_ids=[f"doc:{DOC_10K}#chunk0"],
    must_contain=["customer", "supplier"],
    rationale="Both risk-factor claims must trace to the 10-K source id.")

# More quant
quant_case("quant_11_rsi_w21", "Compute SYN's 21-day EMA again to confirm reproducibility.", "ema_SYN_21", 0.01)

add(id="quant_12_drawdown_sign",
    area="quant",
    query="Is DDN's maximum drawdown reported as a negative percentage?",
    as_of=QAS_OF,
    snapshots=[f"yfinance_DDN_{QAS_OF}_433d.csv"],
    expected_status="ok",
    expect_citation_source_ids=[QC["max_drawdown_DDN_252"]["source_id"]],
    must_contain=[QC["max_drawdown_DDN_252"]["content_value_str"]],
    expected_value=QC["max_drawdown_DDN_252"]["expected_value"], value_tolerance=0.01,
    rationale="Drawdown is a peak-to-trough decline and must be reported as negative.")

add(id="quant_13_corr_range",
    area="quant",
    query="Confirm the CRA/CRB correlation is between -1 and 1.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_CRA_{QAS_OF}_504d.csv", f"yfinance_CRB_{QAS_OF}_504d.csv"],
    expected_status="ok",
    expect_citation_source_ids=[QC["correlation_CRA_CRB_252"]["source_id"]],
    expected_value=QC["correlation_CRA_CRB_252"]["expected_value"], value_tolerance=0.005,
    rationale="Correlation must be a valid coefficient in [-1, 1].")

# More methodology
add(id="meth_07_rsi_threshold_definition",
    area="methodology",
    query="At what RSI level is an asset conventionally considered oversold?",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="ok",
    expect_citation_store_any=["knowledge"],
    must_contain=["30"],
    rationale="Definitional threshold answered from methodology corpus.")

add(id="meth_08_method_usable_value_not",
    area="methodology",
    query="Using the September 2025 paper, what was equity-index realized vol that year? Treat findings as as-of-period facts.",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="insufficient_data",
    must_not_contain=["14.2 percent", "14.2%"],
    rationale="A post-as-of empirical finding cannot be used as an as-of-period fact even when the user asks for it. The agent may retrieve the paper's methodology, but must not state its 14.2% finding as a period fact.")

add(id="meth_09_definition_no_price_tool",
    area="methodology",
    query="Explain what EMA means conceptually. Do not compute anything.",
    as_of="2025-03-14",
    seed_knowledge=ALL_KN,
    expected_status="ok",
    expect_citation_store_any=["knowledge"],
    forbid_citation_store=["quant", "prices"],
    must_contain=["weight"],
    rationale="A conceptual EMA question is answerable from the seeded EMA definition and must NOT trigger a price/quant computation. Tests that definitional questions stay in the knowledge corpus.")

# Cross-cutting: combined PIT + provenance + quant in one query
add(id="combo_01_pit_quant_provenance",
    area="provenance",
    query="As of the as-of date, report SYN's RSI(14) with its citation and confirm it is point-in-time.",
    as_of=QAS_OF,
    snapshots=[f"yfinance_SYN_{QAS_OF}_380d.csv"],
    expected_status="ok",
    expect_tools_all=["quant_indicator"],
    expect_citation_source_ids=[QC["rsi_SYN_14"]["source_id"]],
    expected_value=QC["rsi_SYN_14"]["expected_value"], value_tolerance=0.1,
    rationale="End-to-end: PIT-truncated input, derived evidence cited, value within display tolerance.")

add(id="combo_02_insufficient_then_honest",
    area="pit",
    query="What was ACME's stock price reaction to the Beacon acquisition?",
    as_of="2025-03-01",
    seed_docs=ALL_DOCS,
    expected_status="insufficient_data",
    forbid_citation_source_ids=[SID_PRESS],
    must_not_contain=["$800 million", "acquisition closed", "stock rose", "stock fell"],
    rationale="Acquisition is future + no price data seeded; the agent must say what is missing. Mentioning 'Beacon' to deny knowledge of it (in the gap list) is correct, so we forbid only asserted-fact phrasings, not the bare word.")

# ----------------------------------------------------------------------------
# Write JSONL + a manifest
# ----------------------------------------------------------------------------

def main():
    # sanity: unique ids
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids!"
    out = DS / "golden.jsonl"
    with open(out, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    # area counts
    from collections import Counter
    counts = Counter(c["area"] for c in cases)
    manifest = {
        "total_cases": len(cases),
        "by_area": dict(counts),
        "as_of_quant": QAS_OF,
        "snapshots_required": sorted({s for c in cases for s in c["snapshots"]}),
        "schema_fields": sorted(cases[0].keys()),
    }
    (DS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(cases)} cases to {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
