"""

Three layers of verification:
1) HAND-COMPUTED ground truth on a minimal portfolio (every number derivable
   on paper).
2) RECONCILIATION IDENTITIES on NLPHFEA's own example data, checked against
   the reference values in its shipped characteristics files ($1,598 and
   $6,377 P&L at $500k notional).
3) TOOL-LEVEL behavior: PIT file-date gating, path-traversal rejection,
   deterministic source_ids, percent/dollar-string coercion.
"""
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_agent import config
from quant_agent.tools.quant.attribution import (
    brinson_fachler, coerce_numeric, cooper_li, profile,
)
from quant_agent.tools.quant.attribution_tool import AttributionTool

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------ 1) hand math ---

def _one_per_bucket():
    # One security per bucket => rP == rB per bucket => Selection = Interaction
    # = 0 and Allocation alone must equal active return. Hand numbers:
    #   A: wP=0.6, wB=0.5, r=0.08   B: wP=0.4, wB=0.5, r=0.04
    #   rB_total = 0.06; active = (0.6*0.08+0.4*0.04) - 0.06 = 0.004
    #   Alloc_A = (0.1)(0.08-0.06) = 0.002; Alloc_B = (-0.1)(0.04-0.06) = 0.002
    return pd.DataFrame({
        "Sector": ["A", "B"],
        "Portfolio Weight": [0.6, 0.4],
        "Cap Weight": [0.5, 0.5],
        "Return": [0.08, 0.04],
    })


def test_bf_hand_computed():
    grp, s = brinson_fachler(_one_per_bucket(), "Sector",
                             "Portfolio Weight", "Cap Weight", "Return")
    assert abs(s["active_return"] - 0.004) < 1e-12
    a = grp.set_index("bucket")
    assert abs(a.loc["A", "Allocation"] - 0.002) < 1e-12
    assert abs(a.loc["B", "Allocation"] - 0.002) < 1e-12
    assert abs(a["Selection"]).max() < 1e-12
    assert abs(a["Interaction"]).max() < 1e-12
    assert abs(s["reconciliation_residual"]) < 1e-12


def test_bf_identity_multisecurity():
    # Two securities per bucket, unequal weights: identity must still hold.
    df = pd.DataFrame({
        "Sector": ["A", "A", "B", "B"],
        "Portfolio Weight": [0.3, 0.3, 0.1, 0.3],
        "Cap Weight": [0.2, 0.3, 0.25, 0.25],
        "Return": [0.10, 0.02, -0.04, 0.06],
    })
    grp, s = brinson_fachler(df, "Sector", "Portfolio Weight", "Cap Weight", "Return")
    assert abs(s["reconciliation_residual"]) < 1e-12
    assert abs((s["total_allocation"] + s["total_selection"] + s["total_interaction"])
               - s["active_return"]) < 1e-12


def test_cooper_li_hand_computed():
    # Single bucket, one long one short, universe = the two names equally.
    #   long  wP=+0.5 r=0.10 ; short wP=-0.3 r=0.02 ; wU=0.5 each
    #   rU = 0.06 ; NE = (0.2)(0.06)=0.012 ; BE = 0
    #   LSE = 0.5*(0.10-0.06)=0.02 ; SSE = -0.3*(0.02-0.06)=0.012
    #   total = 0.5*0.10 - 0.3*0.02 = 0.044 = 0.012+0+0.02+0.012
    df = pd.DataFrame({
        "Sector": ["X", "X"],
        "Portfolio Weight": [0.5, -0.3],
        "Cap Weight": [0.5, 0.5],
        "Return": [0.10, 0.02],
    })
    grp, s = cooper_li(df, "Sector", "Portfolio Weight", "Cap Weight", "Return")
    assert abs(s["NE_total"] - 0.012) < 1e-12
    assert abs(s["BE_total"]) < 1e-12
    assert abs(s["LSE_total"] - 0.020) < 1e-12
    assert abs(s["SSE_total"] - 0.012) < 1e-12
    assert abs(s["total_pnl"] - 0.044) < 1e-12
    assert abs(s["reconciliation_residual"]) < 1e-12


# ------------------------------------------- 2) NLPHFEA reference datasets ---

def test_nlphfea_example_reconciles_to_reference():
    # Shipped characteristics file states: Dollar P&L $1,598, net exposure 0.144
    # at $500,000 notional.
    df = pd.read_csv(FIXTURES / "nlphfea_example_portfolio.csv", encoding="utf-8-sig")
    grp, s = cooper_li(df, "Sector", "Portfolio Weight", "Cap Weight", "Return",
                       notional=500_000)
    assert abs(s["total_pnl"] - 1598) < 1.0
    assert abs(s["net_exposure"] - 0.144) < 1e-9
    assert abs(s["reconciliation_residual"]) < 1e-6


def test_nlphfea_test1_percent_strings_reconcile():
    # This file carries '1.62%'-style returns and short positions; the shipped
    # characteristics_test1 states Dollar P&L $6,377 at $500,000.
    df = pd.read_csv(FIXTURES / "nlphfea_sample_test1.csv")
    import pandas.api.types as ptypes
    assert not ptypes.is_numeric_dtype(df["Return"])  # strings before coercion
    grp, s = cooper_li(df, "Sector", "Portfolio Weight", "Cap Weight", "Return",
                       notional=500_000)
    assert abs(s["total_pnl"] - 6377) < 1.0
    assert abs(s["reconciliation_residual"]) < 1e-6
    p = profile(df)
    assert p["has_short_positions"] and p["suggested_method"] == "cooper_li"


def test_coerce_numeric_formats():
    s = pd.Series(["1.62%", "$1,598", "(0.05)", "0.03", "", "junk"])
    out = coerce_numeric(s)
    assert abs(out[0] - 0.0162) < 1e-12
    assert abs(out[1] - 1598.0) < 1e-9
    assert abs(out[2] + 0.05) < 1e-12
    assert abs(out[3] - 0.03) < 1e-12
    assert pd.isna(out[4]) and pd.isna(out[5])


def test_bad_rows_raise_not_silently_dropped():
    df = pd.DataFrame({
        "Sector": ["A", "B"],
        "Portfolio Weight": [0.6, 0.4],
        "Cap Weight": [0.5, 0.5],
        "Return": ["n/a", "0.04"],  # one unparseable value
    })
    with pytest.raises(ValueError, match="non-numeric"):
        brinson_fachler(df, "Sector", "Portfolio Weight", "Cap Weight", "Return")


# --------------------------------------------------------- 3) tool + PIT ---

@pytest.fixture
def portfolios_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    d = tmp_path / "portfolios"
    d.mkdir()
    shutil.copy(FIXTURES / "nlphfea_example_portfolio.csv",
                d / "2025-03-31_book.csv")
    shutil.copy(FIXTURES / "nlphfea_example_portfolio.csv",
                d / "2025-06-30_book.csv")
    return d


def test_tool_pit_gate(portfolios_dir):
    t = AttributionTool()
    # visible file works
    ev = t.run(as_of=date(2025, 4, 15), op="cooper_li", file="2025-03-31_book.csv",
               notional=500_000)
    assert "1598" in ev[0].content
    assert ev[0].source_id == "attribution:cl:2025-03-31_book.csv:Sector:500000"
    assert ev[0].pit_safe is True and ev[0].pub_date == date(2025, 3, 31)
    # post-as-of file refused as lookahead
    with pytest.raises(RuntimeError, match="lookahead"):
        t.run(as_of=date(2025, 4, 15), op="profile", file="2025-06-30_book.csv")
    # list shows one visible, one hidden
    lst = t.run(as_of=date(2025, 4, 15), op="list")[0].content
    assert "2025-03-31_book.csv" in lst and "2025-06-30_book.csv" not in lst
    assert "1 more hidden" in lst


def test_tool_rejects_path_traversal(portfolios_dir):
    t = AttributionTool()
    for bad in ("../../etc/passwd", "sub/dir.csv", ".hidden.csv"):
        with pytest.raises(ValueError, match="bare filename"):
            t.run(as_of=date(2025, 4, 15), op="profile", file=bad)


def test_tool_bf_deterministic_sid_and_identity(portfolios_dir):
    t = AttributionTool()
    ev1 = t.run(as_of=date(2025, 4, 15), op="brinson_fachler", file="2025-03-31_book.csv")
    ev2 = t.run(as_of=date(2025, 4, 15), op="brinson_fachler", file="2025-03-31_book.csv")
    assert ev1[0].source_id == ev2[0].source_id == "attribution:bf:2025-03-31_book.csv:Sector"
    assert ev1[0].content == ev2[0].content
    assert "ACTIVE" in ev1[0].content
