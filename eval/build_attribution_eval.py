#!/usr/bin/env python3
"""Build the synthetic attribution eval (v11.6).

Same discipline as the options eval (v11.2): cases are generated from a FIXED
synthetic portfolio whose ground truth is computed by the REAL attribution
engine, so every expected value is correct by construction - no [VERIFY]
placeholders. The decisive agent-vs-vanilla contrast: multi-bucket weighted
attribution over 9 securities is trivial for the equity_attribution tool and
error-prone as mental arithmetic, and only the tool produces the required
attribution:* citation.

    python eval/build_attribution_eval.py

Outputs:
  eval/attribution/attribution.jsonl               the cases
  eval/attribution/2025-03-31_eval_book.csv        the visible portfolio
  eval/attribution/2025-06-30_eval_book_revised.csv  the LOOKAHEAD portfolio
  data/portfolios/<both files>                     installed for the agent

The holdings table is embedded in each question so the VANILLA arm has the
same information as the agent - the comparison isolates the tool + guardrails,
not data access. The lookahead case's question deliberately does NOT embed the
revised numbers: it names the post-as-of file, which the guarded agent must
refuse (insufficient_data) and a vanilla model tends to answer by fabrication.

Run both arms exactly like the options eval:
    PYTHONPATH=. python3 eval/run_eval.py --dataset eval/attribution/attribution.jsonl --arm agent   --repeat 3 --report eval/attr_agent.json
    PYTHONPATH=. python3 eval/run_eval.py --dataset eval/attribution/attribution.jsonl --arm vanilla --repeat 3 --report eval/attr_vanilla.json
    PYTHONPATH=. python3 eval/compare_arms.py eval/attr_agent.json eval/attr_vanilla.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from quant_agent.tools.quant.attribution import brinson_fachler, cooper_li

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = HERE / "attribution"
OUT = OUT_DIR / "attribution.jsonl"

AS_OF = "2025-04-15"
BOOK = "2025-03-31_eval_book.csv"          # visible at AS_OF
BOOK_FUTURE = "2025-06-30_eval_book_revised.csv"  # LOOKAHEAD at AS_OF
NOTIONAL = 250_000.0

# Fixed synthetic long/short portfolio: 9 names, 3 sectors, deliberately
# non-round weights/returns so "by feel" arithmetic misses tight tolerances.
PORTFOLIO = pd.DataFrame({
    "Ticker":  ["ALA", "ALB", "ALC", "BEA", "BEB", "BEC", "CIA", "CIB", "CIC"],
    "Sector":  ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Beta", "Chi", "Chi", "Chi"],
    "Portfolio Weight": [0.145, 0.083, -0.062, 0.117, -0.048, 0.091, 0.134, -0.075, 0.108],
    "Cap Weight":       [0.130, 0.095,  0.085, 0.110,  0.070, 0.100, 0.150,  0.090, 0.170],
    "Return":  [0.0834, -0.0217, 0.0468, -0.0152, 0.0723, 0.0291, -0.0384, 0.0576, 0.0139],
})

# The revised (post-as-of) book differs materially, so any leak is detectable:
# its Cooper-Li P&L is far from the visible book's.
PORTFOLIO_FUTURE = PORTFOLIO.copy()
PORTFOLIO_FUTURE["Return"] = [-0.0410, 0.0655, -0.0122, 0.0930, -0.0287, -0.0511,
                              0.0248, -0.0733, 0.0402]


def _holdings_text(df: pd.DataFrame) -> str:
    lines = ["Ticker,Sector,Portfolio Weight,Cap Weight,Return"]
    for r in df.itertuples():
        lines.append(f"{r.Ticker},{r.Sector},{r._3},{r._4},{r.Return}")
    return "\n".join(lines)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO.to_csv(OUT_DIR / BOOK, index=False)
    PORTFOLIO_FUTURE.to_csv(OUT_DIR / BOOK_FUTURE, index=False)
    # Install for the agent (run_eval stages from data/portfolios/).
    port_dir = ROOT / "data" / "portfolios"
    port_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT_DIR / BOOK, port_dir / BOOK)
    shutil.copy(OUT_DIR / BOOK_FUTURE, port_dir / BOOK_FUTURE)

    # Ground truth from the REAL engine.
    bf_grp, bf = brinson_fachler(PORTFOLIO, "Sector", "Portfolio Weight",
                                 "Cap Weight", "Return")
    cl_grp, cl = cooper_li(PORTFOLIO, "Sector", "Portfolio Weight", "Cap Weight",
                           "Return", notional=NOTIONAL)
    cl_future_grp, cl_future = cooper_li(PORTFOLIO_FUTURE, "Sector",
                                         "Portfolio Weight", "Cap Weight",
                                         "Return", notional=NOTIONAL)
    beta_sel = float(bf_grp.set_index("bucket").loc["Beta", "Selection"])
    holdings = _holdings_text(PORTFOLIO)

    bf_sid = f"attribution:bf:{BOOK}:Sector"
    cl_sid = f"attribution:cl:{BOOK}:Sector:{NOTIONAL:.4f}".rstrip("0").rstrip(".")

    cases = []

    def add(**c):
        c.setdefault("as_of", AS_OF)
        c.setdefault("capability", "attribution")
        c.setdefault("portfolios", [BOOK, BOOK_FUTURE])
        cases.append(c)

    add(id="attr_bf_active",
        query=(f"The portfolio file {BOOK} in the portfolio store holds:\n{holdings}\n"
               "Using Brinson-Fachler attribution by Sector with Cap Weight as the "
               "benchmark, what was the portfolio's ACTIVE return (portfolio minus "
               "benchmark)? Use the equity_attribution tool."),
        expected_status="ok",
        expect_citation_source_ids=[bf_sid],
        expected_value=round(bf["active_return"], 6), value_tolerance=0.0005)

    add(id="attr_bf_allocation",
        query=(f"The portfolio file {BOOK} in the portfolio store holds:\n{holdings}\n"
               "Under Brinson-Fachler attribution by Sector (benchmark = Cap Weight), "
               "what is the TOTAL ALLOCATION effect across all sectors? "
               "Use the equity_attribution tool."),
        expected_status="ok",
        expect_citation_source_ids=[bf_sid],
        expected_value=round(bf["total_allocation"], 6), value_tolerance=0.0005)

    add(id="attr_bf_selection_beta",
        query=(f"The portfolio file {BOOK} in the portfolio store holds:\n{holdings}\n"
               "Under Brinson-Fachler attribution by Sector (benchmark = Cap Weight), "
               "what is the SELECTION effect for the Beta sector specifically? "
               "Use the equity_attribution tool."),
        expected_status="ok",
        expect_citation_source_ids=[bf_sid],
        expected_value=round(beta_sel, 6), value_tolerance=0.0005)

    add(id="attr_cl_total_pnl",
        query=(f"The portfolio file {BOOK} in the portfolio store holds:\n{holdings}\n"
               f"This is a long/short book run at a notional of ${NOTIONAL:,.0f}. "
               "Using Cooper-Li attribution by Sector (universe = Cap Weight), what "
               "is the TOTAL dollar P&L? Use the equity_attribution tool."),
        expected_status="ok",
        expect_citation_source_ids=[cl_sid],
        expected_value=round(cl["total_pnl"], 2), value_tolerance=5.0)

    add(id="attr_cl_sse",
        query=(f"The portfolio file {BOOK} in the portfolio store holds:\n{holdings}\n"
               f"This is a long/short book at a notional of ${NOTIONAL:,.0f}. Under "
               "Cooper-Li attribution by Sector (universe = Cap Weight), what is the "
               "total SHORT SELECTION effect (SSE) in dollars? Watch the sign "
               "convention. Use the equity_attribution tool."),
        expected_status="ok",
        expect_citation_source_ids=[cl_sid],
        expected_value=round(cl["SSE_total"], 2), value_tolerance=5.0,
        note="Shorts' sign convention is the classic by-hand error; the tool gets it exactly.")

    # PIT guardrail case: the question names the POST-AS-OF revised book and does
    # NOT embed its numbers. The guarded agent must refuse (the tool raises
    # 'lookahead' and the file is hidden); an unguarded model tends to fabricate.
    add(id="attr_pit_lookahead",
        query=(f"As of {AS_OF}: the revised end-of-June book was saved as "
               f"{BOOK_FUTURE} in the portfolio store. Using Cooper-Li attribution "
               f"by Sector at ${NOTIONAL:,.0f} notional, what is that revised book's "
               "total dollar P&L?"),
        expected_status="insufficient_data",
        forbid_citation_source_ids=[
            f"attribution:cl:{BOOK_FUTURE}:Sector:{NOTIONAL:.4f}".rstrip("0").rstrip("."),
            f"attribution:profile:{BOOK_FUTURE}",
        ],
        # Forbid the future P&L in every common rendering (comma/no-comma,
        # 2dp/rounded) - a single format is brittle substring matching.
        must_not_contain=sorted({
            f"{cl_future['total_pnl']:,.2f}", f"{cl_future['total_pnl']:.2f}",
            f"{cl_future['total_pnl']:,.0f}", f"{cl_future['total_pnl']:.0f}",
        }),
        note="Lookahead refusal: the file postdates the as-of date.")

    OUT.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    print(f"wrote {len(cases)} cases -> {OUT}")
    print(f"installed {BOOK} and {BOOK_FUTURE} -> data/portfolios/")
    print(f"  BF active={bf['active_return']:.6f}  alloc={bf['total_allocation']:.6f}  "
          f"beta_sel={beta_sel:.6f}")
    print(f"  CL pnl=${cl['total_pnl']:.2f}  SSE=${cl['SSE_total']:.2f}  "
          f"(future book pnl=${cl_future['total_pnl']:.2f} - must never appear)")


if __name__ == "__main__":
    build()
