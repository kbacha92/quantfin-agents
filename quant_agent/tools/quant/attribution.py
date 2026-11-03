"""Equity performance attribution engine (ported from the NLPHFEA framework).

Pure functions with deterministic output, mirroring the indicators/options
modules. Two methodologies:

- Brinson-Fachler (long-only, RELATIVE return vs a benchmark):
      Allocation_i  = (wP_i - wB_i) * (rB_i - rB_total)
      Selection_i   = wB_i * (rP_i - rB_i)
      Interaction_i = (wP_i - wB_i) * (rP_i - rB_i)
  Identity: sum of all effects == active return (rP_total - rB_total).

- Cooper-Li 2012 (long/short, ABSOLUTE P&L; SSRN 2557128). Per bucket b with
  universe return rU_b (wU-weighted) and total universe return rU:
      NE  = notional * w_side * rU              (market drift on exposure;
                                                 shorts carry negative w -> pay it)
      BE  = notional * w_side * (rU_b - rU)     (bucket tilt)
      LSE = notional * sum_{wP>0} wP_i*(r_i - rU_b)   (long selection)
      SSE = notional * sum_{wP<0} wP_i*(r_i - rU_b)   (short selection)
  Identity: NE + BE + LSE + SSE == notional * sum(wP_i * r_i) == total P&L.

Plus the paper's market-making and ETF-arbitrage effects. What was NOT ported
from NLPHFEA: its planner/analyst/reviewer multi-agent loop - the host quant
agent already provides planning, evaluation, and synthesis; nesting a second
orchestrator inside a tool would duplicate (and fight) the host architecture.

Data hygiene: real-world portfolio files carry '1.62%', '$1,598', '(0.05)'
style values; coerce_numeric normalizes them so the math never silently runs
on strings. Both identities above are asserted in tests against the NLPHFEA
example data ($1,598 P&L on the shipped example portfolio at $500k notional).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- loading ---

_PCT = re.compile(r"^\s*\(?\s*-?[\d,]*\.?\d+\s*%\s*\)?\s*$")
_NUMISH = re.compile(r"^\s*\(?\s*-?\$?\s*-?[\d,]*\.?\d+\s*\)?\s*$")


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Normalize finance-formatted strings to floats.

    Handles '%' (divided by 100), '$', thousands commas, and accounting-style
    parentheses negatives. Non-string numeric series pass through unchanged;
    unparseable values become NaN (surfaced, never silently zeroed).
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        t = str(v).strip()
        if not t:
            return np.nan
        pct = bool(_PCT.match(t))
        neg = t.startswith("(") and t.endswith(")")
        t = t.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            x = float(t)
        except ValueError:
            return np.nan
        if neg:
            x = -abs(x)
        return x / 100.0 if pct else x

    return s.map(one)


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Excel portfolio file (BOM-tolerant; xlsx needs openpyxl)."""
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise RuntimeError(
                f"Reading {path.name} requires openpyxl (pip install openpyxl), "
                "or save the file as CSV."
            ) from e
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_excel(path)  # mislabeled .csv that is really xlsx


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case-insensitive column match from a candidate list; None if absent."""
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


PORT_W_CANDIDATES = ["Portfolio Weight", "Port Weight", "Port Wt", "port_weight", "wP", "weight"]
BENCH_W_CANDIDATES = ["Cap Weight", "Benchmark Weight", "Bench Weight", "Univ Weight",
                      "Universe Weight", "wB", "wU"]
RETURN_CANDIDATES = ["Return", "Ret", "Perf", "Performance", "r"]
GROUP_CANDIDATES = ["Sector", "Industry", "Country", "Region", "Cap Group", "Style"]


def _prep(df: pd.DataFrame, group_col: str, port_w: str, bench_w: str, ret: str) -> pd.DataFrame:
    work = df[[group_col, port_w, bench_w, ret]].copy()
    work.columns = ["bucket", "wP", "wU", "r"]
    for c in ("wP", "wU", "r"):
        work[c] = coerce_numeric(work[c])
    bad = work[["wP", "wU", "r"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} row(s) have non-numeric weight/return values after "
            f"coercion (rows {list(work.index[bad])[:5]}...). Fix the file; the "
            "engine never silently drops or zeroes data."
        )
    return work


# ------------------------------------------------------------ attribution ---

def brinson_fachler(df: pd.DataFrame, group_col: str, port_w: str,
                    bench_w: str, ret: str) -> tuple[pd.DataFrame, dict]:
    """Classic Brinson-Fachler relative attribution (long-only)."""
    work = _prep(df, group_col, port_w, bench_w, ret)

    rows = []
    for bucket, g in work.groupby("bucket", sort=True):
        wP, wB = float(g["wP"].sum()), float(g["wU"].sum())
        rP = float(np.average(g["r"], weights=g["wP"])) if g["wP"].sum() else 0.0
        rB = float(np.average(g["r"], weights=g["wU"])) if g["wU"].sum() else 0.0
        rows.append({"bucket": bucket, "wP": wP, "wB": wB, "rP": rP, "rB": rB})
    grp = pd.DataFrame(rows)

    rB_total = float((grp["wB"] * grp["rB"]).sum())
    rP_total = float((grp["wP"] * grp["rP"]).sum())
    grp["Allocation"] = (grp["wP"] - grp["wB"]) * (grp["rB"] - rB_total)
    grp["Selection"] = grp["wB"] * (grp["rP"] - grp["rB"])
    grp["Interaction"] = (grp["wP"] - grp["wB"]) * (grp["rP"] - grp["rB"])
    grp["Total"] = grp["Allocation"] + grp["Selection"] + grp["Interaction"]

    summary = {
        "portfolio_return": rP_total,
        "benchmark_return": rB_total,
        "active_return": rP_total - rB_total,
        "total_allocation": float(grp["Allocation"].sum()),
        "total_selection": float(grp["Selection"].sum()),
        "total_interaction": float(grp["Interaction"].sum()),
    }
    # Effects must reconcile to active return; a residual means bad inputs
    # (e.g. weights not summing sensibly), and we surface it rather than hide it.
    resid = summary["active_return"] - (summary["total_allocation"]
                                        + summary["total_selection"]
                                        + summary["total_interaction"])
    summary["reconciliation_residual"] = float(resid)
    return grp, summary


def cooper_li(df: pd.DataFrame, group_col: str, port_w: str, univ_w: str,
              ret: str, notional: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Cooper-Li (2012) long/short absolute P&L attribution."""
    work = _prep(df, group_col, port_w, univ_w, ret)

    if not work["wU"].sum():
        raise ValueError("universe weights sum to zero; cannot form universe return")
    rU_total = float(np.average(work["r"], weights=work["wU"]))

    rows = []
    for bucket, g in work.groupby("bucket", sort=True):
        wU_sum = float(g["wU"].sum())
        rU_b = float(np.average(g["r"], weights=g["wU"])) if wU_sum else 0.0
        longs, shorts = g[g["wP"] > 0], g[g["wP"] < 0]
        wL, wS = float(longs["wP"].sum()), float(shorts["wP"].sum())

        NE = notional * (wL + wS) * rU_total
        BE = notional * (wL + wS) * (rU_b - rU_total)
        LSE = notional * float((longs["wP"] * (longs["r"] - rU_b)).sum())
        SSE = notional * float((shorts["wP"] * (shorts["r"] - rU_b)).sum())
        rows.append({
            "bucket": bucket, "wP": wL + wS, "wU": wU_sum, "rU": rU_b,
            "NE": NE, "BE": BE, "LSE": LSE, "SSE": SSE,
            "bucket_pnl": notional * float((g["wP"] * g["r"]).sum()),
        })
    grp = pd.DataFrame(rows)

    total_pnl = notional * float((work["wP"] * work["r"]).sum())
    summary = {
        "notional": notional,
        "universe_return": rU_total,
        "NE_total": float(grp["NE"].sum()),
        "BE_total": float(grp["BE"].sum()),
        "LSE_total": float(grp["LSE"].sum()),
        "SSE_total": float(grp["SSE"].sum()),
        "total_pnl": total_pnl,
        "gross_exposure": float(work["wP"].abs().sum()),
        "net_exposure": float(work["wP"].sum()),
    }
    resid = total_pnl - (summary["NE_total"] + summary["BE_total"]
                         + summary["LSE_total"] + summary["SSE_total"])
    summary["reconciliation_residual"] = float(resid)
    return grp, summary


def market_making_effect(df: pd.DataFrame, dollars_col: str, shares_col: str,
                         end_price_col: str) -> float:
    """Cooper-Li MM effect: sum(D_i + N_i * P_end); D is signed cash to the
    portfolio (negative buys, positive sells)."""
    d = coerce_numeric(df[dollars_col])
    n = coerce_numeric(df[shares_col])
    p = coerce_numeric(df[end_price_col])
    return float((d + n * p).sum())


def arbitrage_effect(etf_start: float, etf_end: float, stock_start: float,
                     stock_end: float, beta: float) -> float:
    """Cooper-Li ETF arbitrage effect (eqn 9B):
    AE = d(ETF) - beta * d(Stock) * (ETF_start / Stock_start)."""
    return float((etf_end - etf_start)
                 - beta * (stock_end - stock_start) * (etf_start / stock_start))


# --------------------------------------------------------------- profiling ---

def profile(df: pd.DataFrame) -> dict:
    """Data-shape profile the planner uses to pick a methodology."""
    out: dict = {"rows": int(len(df)), "columns": list(df.columns)}
    pw = resolve_col(df, PORT_W_CANDIDATES)
    bw = resolve_col(df, BENCH_W_CANDIDATES)
    rc = resolve_col(df, RETURN_CANDIDATES)
    out["portfolio_weight_col"] = pw
    out["benchmark_weight_col"] = bw
    out["return_col"] = rc
    out["grouping_columns"] = [c for c in df.columns if c in GROUP_CANDIDATES]
    if pw:
        w = coerce_numeric(df[pw])
        out["has_short_positions"] = bool((w < 0).any())
        out["gross_exposure"] = float(w.abs().sum())
        out["net_exposure"] = float(w.sum())
        out["suggested_method"] = ("cooper_li" if out["has_short_positions"]
                                   else "brinson_fachler")
    return out


# ------------------------------------------- full Cooper-Li model (v11.7) ---
# Extends the base NE/BE/LSE/SSE decomposition to the paper's full portfolio
# dollar return, eqn (11): PDR = NE* + BE* + (LSE*+SSE*) + MM + AE*.
#   - Market-making (eqn 8) captures intra-period trades: MM = sum(D + N*P_end),
#     with D SIGNED (negative = dollars spent buying, positive = received
#     selling). Paper example: buy 100 sh at $15, end $15.02 ->
#     D=-1500, N=+100, P=15.02 -> MM=+2.00.
#   - ETF handling is the paper's full Chart-6 treatment: each ETF is
#     decomposed into beta-scaled constituent stock exposures that MERGE INTO
#     the holdings weights (so NE/BE/LSE/SSE see through the ETF), plus an
#     Arbitrage Effect (eqn 9B) per ETF for the ETF-vs-basket dislocation:
#       implied $ exposure in stock i = beta * (ETF_start/Basket_start)
#                                        * shares_i * start_i * shares_held
#       AE* = (dETF - beta * dBasket * ETF_start/Basket_start) * shares_held
#     Constituent tickers must exist in the holdings/universe file (their
#     sector, universe weight, and return come from there); an unknown ticker
#     raises rather than silently attributing to nothing.

def _group_key(df: pd.DataFrame, group_cols):
    """Single- or multi-column bucketing (the paper uses Sector x Cap)."""
    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    if len(cols) == 1:
        return df[cols[0]].astype(str), cols
    return df[cols].astype(str).agg(" | ".join, axis=1), cols


def market_making(trades: pd.DataFrame, dollars_col: str = "Dollars",
                  shares_col: str = "Shares", end_price_col: str = "EndPrice") -> float:
    """Eqn (8) over a trades table; thin named-column wrapper."""
    return market_making_effect(trades, dollars_col, shares_col, end_price_col)


def etf_decompose(etfs: pd.DataFrame, constituents: pd.DataFrame,
                  notional: float) -> tuple[pd.Series, float, pd.DataFrame]:
    """Chart-6 decomposition of ETF positions.

    Returns (implied portfolio-weight adjustments indexed by Ticker,
             total dollar Arbitrage Effect,
             per-ETF detail frame).
    """
    for c in ("Beta", "SharesHeld", "StartPrice", "EndPrice"):
        etfs[c] = coerce_numeric(etfs[c])
    for c in ("SharesPerETF", "StartPrice", "EndPrice"):
        constituents[c] = coerce_numeric(constituents[c])

    adj: dict[str, float] = {}
    detail = []
    ae_total = 0.0
    for e in etfs.itertuples():
        cons = constituents[constituents["ETF"] == e.ETF]
        if cons.empty:
            raise ValueError(f"ETF {e.ETF!r} has no constituent rows")
        basket_start = float((cons["SharesPerETF"] * cons["StartPrice"]).sum())
        basket_end = float((cons["SharesPerETF"] * cons["EndPrice"]).sum())
        if basket_start <= 0:
            raise ValueError(f"ETF {e.ETF!r} basket start value must be positive")
        factor = e.Beta * e.StartPrice / basket_start          # per $1 of basket
        for c in cons.itertuples():
            dollars = factor * c.SharesPerETF * c.StartPrice * e.SharesHeld
            adj[c.Ticker] = adj.get(c.Ticker, 0.0) + dollars / notional
        d_etf_ps = e.EndPrice - e.StartPrice
        d_basket = (basket_end - basket_start)
        # Eqn (9B) per ETF share: dislocation of the ETF move vs the
        # beta-scaled basket move, scaled by shares held.
        ae = (d_etf_ps - e.Beta * d_basket * e.StartPrice / basket_start) * e.SharesHeld
        ae_total += ae
        detail.append({"ETF": e.ETF, "beta": e.Beta, "shares": e.SharesHeld,
                       "basket_start": basket_start, "d_basket": d_basket,
                       "d_etf_per_share": d_etf_ps, "AE": ae})
    return pd.Series(adj, name="w_adj"), float(ae_total), pd.DataFrame(detail)


def cooper_li_full(holdings: pd.DataFrame, group_cols, port_w: str, univ_w: str,
                   ret: str, notional: float = 1.0,
                   trades: pd.DataFrame | None = None,
                   etfs: pd.DataFrame | None = None,
                   constituents: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Full Cooper-Li PDR (eqn 11) with optional trades and ETF positions.

    With trades/etfs absent this reduces EXACTLY to cooper_li (the base
    decomposition), preserving v11.5/v11.6 behavior and eval ground truth.
    """
    df = holdings.copy()
    bucket, cols = _group_key(df, group_cols)
    df = df.assign(__bucket=bucket)

    ae_total, etf_detail = 0.0, None
    if etfs is not None and len(etfs):
        if constituents is None or not len(constituents):
            raise ValueError("ETF positions supplied without a constituents table")
        adj, ae_total, etf_detail = etf_decompose(etfs.copy(), constituents.copy(), notional)
        unknown = set(adj.index) - set(df["Ticker"].astype(str))
        if unknown:
            raise ValueError(
                f"ETF constituents {sorted(unknown)} not present in holdings/universe - "
                "sector, universe weight, and return are unknown; add them as rows "
                "(zero portfolio weight is fine).")
        w = coerce_numeric(df[port_w]).astype(float)
        w = w.add(df["Ticker"].astype(str).map(adj).fillna(0.0), fill_value=0.0)
        df[port_w] = w

    grp, s = cooper_li(df, "__bucket", port_w, univ_w, ret, notional=notional)
    grp = grp.rename(columns={"bucket": " | ".join(cols)})

    mm_total = 0.0
    if trades is not None and len(trades):
        t = trades.copy()
        for c in ("Dollars", "Shares", "EndPrice"):
            t[c] = coerce_numeric(t[c])
        mm_total = market_making(t)

    s = dict(s)
    s["MM"] = float(mm_total)
    s["AE"] = float(ae_total)
    s["base_pnl"] = s.pop("total_pnl")
    s["PDR"] = s["base_pnl"] + s["MM"] + s["AE"]
    if etf_detail is not None:
        s["etf_detail"] = etf_detail.to_dict("records")
    return grp, s
