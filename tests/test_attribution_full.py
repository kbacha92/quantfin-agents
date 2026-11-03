
import json
import math
import subprocess
import sys
from pathlib import Path

import importlib.util
import pandas as pd
import pytest

from quant_agent.tools.quant.attribution import (
    cooper_li, cooper_li_full, etf_decompose, market_making,
)

ROOT = Path(__file__).resolve().parent.parent


def _paper_holdings():
    return pd.DataFrame([
        ("AAAA", 10, 1, 0.025, 0.02, 0.05),  ("BBBB", 10, 1, 0.025, 0.03, 0.0),
        ("CCCC", 10, 1, 0.025, -0.02, -0.064), ("DDDD", 10, 1, 0.05, 0.03, 0.0),
        ("EEEE", 10, 2, 0.075, -0.02, 0.0),  ("FFFF", 10, 2, 0.10, 0.01, 0.19),
        ("GGGG", 10, 2, 0.10, -0.03, 0.0),   ("HHHH", 10, 2, 0.125, -0.01, -0.088),
        ("IIII", 20, 1, 0.05, 0.00, 0.0),    ("JJJJ", 20, 1, 0.05, -0.02, 0.08),
        ("KKKK", 20, 1, 0.075, 0.015, 0.0),  ("LLLL", 20, 2, 0.10, 0.01, 0.0),
        ("MMMM", 20, 2, 0.10, 0.011, -0.024), ("NNNN", 20, 2, 0.10, 0.045, 0.0),
    ], columns=["Ticker", "Sector", "Cap", "Cap Weight", "Return", "Portfolio Weight"])


def _paper_etf():
    etfs = pd.DataFrame([{"ETF": "XETF", "Beta": -2, "SharesHeld": 1,
                          "StartPrice": 20.33, "EndPrice": 19.77}])
    cons = pd.DataFrame([
        {"ETF": "XETF", "Ticker": "AAAA", "SharesPerETF": 1, "StartPrice": 20.0, "EndPrice": 20.40},
        {"ETF": "XETF", "Ticker": "BBBB", "SharesPerETF": 2, "StartPrice": 21.0, "EndPrice": 21.63},
        {"ETF": "XETF", "Ticker": "CCCC", "SharesPerETF": 1, "StartPrice": 20.0, "EndPrice": 19.60},
    ])
    return etfs, cons


V = 500_000


# ------------------------------------------------------- layer 1: paper ---

def test_chart2_base_attribution():
    grp, s = cooper_li_full(_paper_holdings(), ["Sector", "Cap"],
                            "Portfolio Weight", "Cap Weight", "Return", notional=V)
    assert abs(s["universe_return"] - 0.004225) < 1e-9
    assert abs(s["NE_total"] - 304.20) < 0.01
    assert abs(s["BE_total"] - (-1271.25)) < 0.01
    assert abs((s["LSE_total"] + s["SSE_total"]) - 2565.05) < 0.02
    assert abs(s["PDR"] - 1598.00) < 0.01
    assert abs(s["reconciliation_residual"]) < 1e-8
    g = grp.set_index("Sector | Cap")
    paper = {"10 | 1": (-96.43, 50.00, 1216.00), "10 | 2": (-821.10, 2078.13, -82.50),
             "20 | 1": (-140.43, -828.57, 0.00), "20 | 2": (-213.30, 0.00, 132.00)}
    for b, (be, lse, sse) in paper.items():
        assert abs(g.loc[b, "BE"] - be) < 0.02, b
        assert abs(g.loc[b, "LSE"] - lse) < 0.02, b
        assert abs(g.loc[b, "SSE"] - sse) < 0.02, b


def test_chart4_etf_implied_exposures():
    etfs, cons = _paper_etf()
    adj, ae, det = etf_decompose(etfs, cons, V)
    assert abs(adj["AAAA"] * V - (-9.91707)) < 1e-4
    assert abs(adj["BBBB"] * V - (-20.82585)) < 1e-4
    assert abs(adj["CCCC"] * V - (-9.91707)) < 1e-4
    assert abs(sum(adj) * V - (-40.66)) < 0.01  # beta x ETF value


def test_chart5_arbitrage_effect():
    etfs, cons = _paper_etf()
    _, ae, _ = etf_decompose(etfs, cons, V)
    assert abs(ae - 0.06478) < 1e-5


def test_market_making_paper_example():
    trades = pd.DataFrame([{"Ticker": "DDDD", "Dollars": -1500.0,
                            "Shares": 100, "EndPrice": 15.02}])
    assert abs(market_making(trades) - 2.00) < 1e-9


def test_chart7_full_pdr():
    etfs, cons = _paper_etf()
    trades = pd.DataFrame([{"Ticker": "DDDD", "Dollars": -1500.0,
                            "Shares": 100, "EndPrice": 15.02}])
    grp, s = cooper_li_full(_paper_holdings(), ["Sector", "Cap"],
                            "Portfolio Weight", "Cap Weight", "Return",
                            notional=V, trades=trades, etfs=etfs, constituents=cons)
    assert abs(s["base_pnl"] - 1597.38) < 0.01
    assert abs(s["NE_total"] - 304.03) < 0.01
    assert abs(s["MM"] - 2.00) < 1e-9
    assert abs(s["AE"] - 0.06478) < 1e-5
    assert abs(s["PDR"] - 1599.44) < 0.01
    assert abs(s["reconciliation_residual"]) < 1e-8


def test_full_reduces_to_base_without_companions():
    h = _paper_holdings()
    g1, s1 = cooper_li(h.assign(B=h["Sector"].astype(str) + "|" + h["Cap"].astype(str)),
                       "B", "Portfolio Weight", "Cap Weight", "Return", notional=V)
    g2, s2 = cooper_li_full(h, ["Sector", "Cap"], "Portfolio Weight",
                            "Cap Weight", "Return", notional=V)
    assert abs(s1["total_pnl"] - s2["PDR"]) < 1e-9
    assert s2["MM"] == 0.0 and s2["AE"] == 0.0


def test_unknown_constituent_raises():
    etfs, cons = _paper_etf()
    cons.loc[0, "Ticker"] = "ZZZZ"  # not in the universe
    with pytest.raises(ValueError, match="not present in holdings"):
        cooper_li_full(_paper_holdings(), ["Sector", "Cap"], "Portfolio Weight",
                       "Cap Weight", "Return", notional=V, etfs=etfs, constituents=cons)


# --------------------------------------------------- layer 2: generator ---

def _generate(tmp, seed=7, n=8):
    out = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "generate_attribution_cases.py"),
         "--n", str(n), "--seed", str(seed)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    p = ROOT / "eval" / "attribution" / f"generated_{seed}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()], p


def test_generator_deterministic():
    cases1, p = _generate(None, seed=7)
    txt1 = p.read_text()
    cases2, p = _generate(None, seed=7)
    assert p.read_text() == txt1


def test_generated_cases_recompute():
    cases, _ = _generate(None, seed=7)
    port = ROOT / "data" / "portfolios"
    kinds_seen = set()
    for c in cases:
        kinds_seen.add(c["kind"])
        files = c["portfolios"]
        h = pd.read_csv(port / files[0])
        trades = etfs = cons = None
        for f in files[1:]:
            df = pd.read_csv(port / f)
            if f.endswith("_trades.csv"):
                trades = df
            elif f.endswith("_etf_constituents.csv"):
                cons = df
            elif f.endswith("_etfs.csv"):
                etfs = df
        notional = float(c["expect_citation_source_ids"][0].split(":")[4])
        grp, s = cooper_li_full(h, "Sector", "Portfolio Weight", "Cap Weight",
                                "Return", notional=notional,
                                trades=trades, etfs=etfs, constituents=cons)
        vals = {round(s[k], 2) for k in ("PDR", "SSE_total", "NE_total", "MM", "AE")}
        assert c["expected_value"] in vals, (c["id"], c["expected_value"], vals)
        assert abs(s["reconciliation_residual"]) < 1e-6
    assert {"plain", "mm", "etf", "both"} <= kinds_seen


# ------------------------------------------------------- layer 3: stats ---

def _stats():
    spec = importlib.util.spec_from_file_location("eval_stats", ROOT / "eval" / "stats.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    return st


def test_mcnemar_hand_values():
    st = _stats()
    # b=6, c=0: p = 2 * 0.5^6 = 0.03125 (the 6-case sweep scenario)
    assert abs(st.mcnemar_exact(6, 0) - 0.03125) < 1e-12
    # b=5, c=0: 0.0625 - total separation on 5 discordant cases is NOT < .05
    assert abs(st.mcnemar_exact(5, 0) - 0.0625) < 1e-12
    # symmetric and capped
    assert st.mcnemar_exact(3, 3) == 1.0
    assert st.mcnemar_exact(0, 0) == 1.0


def test_bootstrap_ci_sanity():
    st = _stats()
    # total separation on 20 cases: delta = 1, CI degenerate at 1
    d, lo, hi = st.bootstrap_delta_ci([1.0] * 20, [0.0] * 20, n_boot=2000, seed=1)
    assert d == 1.0 and lo == 1.0 and hi == 1.0
    # no difference: CI straddles 0
    a = [1, 0] * 10
    d, lo, hi = st.bootstrap_delta_ci([float(x) for x in a], [float(x) for x in a],
                                      n_boot=2000, seed=1)
    assert d == 0.0 and lo <= 0.0 <= hi
    # deterministic given seed
    r1 = st.bootstrap_delta_ci([1.0, 0.5, 0.0], [0.0, 0.5, 0.0], seed=3)
    r2 = st.bootstrap_delta_ci([1.0, 0.5, 0.0], [0.0, 0.5, 0.0], seed=3)
    assert r1 == r2


def test_case_level_aggregation():
    st = _stats()
    # 1 case x 3 repeats must count as ONE observation, not three
    rep = {"cases": [{"id": "x", "passes": 2, "runs": 3}]}
    assert st.per_case_rates(rep) == {"x": 2 / 3}
