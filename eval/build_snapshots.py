#!/usr/bin/env python3
"""Build frozen price snapshots for the golden set, fully offline.

Why this exists: load_prices() replays data/api_cache/yfinance_<T>_<asof>_<lb>d.csv
for any *past* as-of date instead of hitting yfinance. If we drop snapshot CSVs
with the exact filenames the quant tools compute, the whole price/indicator path
runs deterministically with no network. We then compute the ground-truth values
by calling the SAME indicator functions the agent uses, so the golden expected
numbers can never drift from the implementation.

Run from the project root:  python eval/build_snapshots.py
It writes:
  - eval/golden_dataset/snapshots/*.csv   (the frozen price frames)
  - eval/golden_dataset/expected_quant.json (ground-truth values + source_ids)

The harness copies snapshots into the live api_cache before running each case.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agent.tools.quant.indicators import (
    ema,
    rsi,
    realized_vol,
    sma,
)

HERE = Path(__file__).resolve().parent
SNAP = HERE / "golden_dataset" / "snapshots"
SNAP.mkdir(parents=True, exist_ok=True)

# Fixed as-of date used across the quant golden cases. Must be in the past so
# load_prices treats the snapshot as immutable and replays it.
AS_OF = date(2025, 3, 14)  # a Friday

# Lookback windows each tool computes (must match indicators.py exactly):
#   sma/ema/rsi/realized_vol(window w) -> load_prices(..., max(380, w*6))
#   max_drawdown(w=252)               -> load_prices(..., int(w*1.6)+30) = 433
#   correlation(w=252)                -> load_prices(..., w*2) = 504 per ticker
LB_INDICATOR = 380
LB_DRAWDOWN = 433
LB_CORR = 504


def _business_days_back(end: date, n_calendar_days: int) -> pd.DatetimeIndex:
    """yfinance returns trading days; we approximate with pandas business days,
    truncated to <= as_of exactly like load_prices does."""
    start = end - timedelta(days=n_calendar_days)
    idx = pd.bdate_range(start=start, end=end)
    return idx[idx.date <= end]


def _synth_close(idx: pd.DatetimeIndex, seed: int, start_price: float,
                 drift: float, vol: float, shock_at: float | None = None) -> pd.Series:
    """Deterministic synthetic close series: geometric random walk with fixed
    seed so values are reproducible. shock_at injects a drawdown for the
    drawdown test (a multiplicative dip partway through)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=len(idx))
    if shock_at is not None:
        k = int(len(idx) * shock_at)
        rets[k:k + 10] -= 0.04  # ten sharp down days -> a clean, large drawdown
    close = start_price * np.exp(np.cumsum(rets))
    return pd.Series(close, index=idx, name="Close")


def _write_snapshot(ticker: str, lookback: int, close: pd.Series) -> Path:
    """Write a CSV matching load_prices' cache format (index + OHLCV columns).
    The tools only read Close, but we emit a full OHLCV frame for realism."""
    df = pd.DataFrame({
        "Open": close.values,
        "High": close.values * 1.01,
        "Low": close.values * 0.99,
        "Close": close.values,
        "Volume": 1_000_000,
    }, index=close.index)
    path = SNAP / f"yfinance_{ticker.upper()}_{AS_OF}_{lookback}d.csv"
    df.to_csv(path)
    return path


def build() -> dict:
    expected: dict = {"as_of": AS_OF.isoformat(), "cases": {}}

    # --- Single-ticker series for sma/ema/rsi/realized_vol (lookback 380) ---
    idx_ind = _business_days_back(AS_OF, LB_INDICATOR)
    syn = _synth_close(idx_ind, seed=42, start_price=150.0, drift=0.0004, vol=0.012)
    _write_snapshot("SYN", LB_INDICATOR, syn)

    for ind_name, fn, w in [
        ("sma", sma, 50),
        ("ema", ema, 21),
        ("rsi", rsi, 14),
        ("realized_vol", realized_vol, 21),
    ]:
        series = fn(syn, w).dropna()
        val = float(series.iloc[-1])
        last_date = series.index[-1].date()
        expected["cases"][f"{ind_name}_SYN_{w}"] = {
            "ticker": "SYN", "indicator": ind_name, "window": w,
            "expected_value": round(val, 4),
            "source_id": f"quant:{ind_name}:SYN:{w}:{AS_OF}",
            "content_value_str": f"{val:.4f}",
            "last_date": last_date.isoformat(),
        }

    # --- Drawdown series (lookback 433, window 252, with an injected shock) ---
    idx_dd = _business_days_back(AS_OF, LB_DRAWDOWN)
    dds = _synth_close(idx_dd, seed=7, start_price=200.0, drift=0.0003,
                       vol=0.011, shock_at=0.55)
    _write_snapshot("DDN", LB_DRAWDOWN, dds)
    close_dd = dds.tail(252)
    dd_val = float((close_dd / close_dd.cummax() - 1).min())
    expected["cases"]["max_drawdown_DDN_252"] = {
        "ticker": "DDN", "indicator": "max_drawdown", "window": 252,
        "expected_value": round(dd_val * 100, 2),  # percent, matches content
        "source_id": f"quant:max_drawdown:DDN:252:{AS_OF}",
        "content_value_str": f"{dd_val * 100:.2f}%",
        "sessions": len(close_dd),
    }

    # --- Correlation pair (lookback 504 each, window 252) ---
    idx_corr = _business_days_back(AS_OF, LB_CORR)
    a = _synth_close(idx_corr, seed=1, start_price=100.0, drift=0.0003, vol=0.010)
    # b is correlated to a plus idiosyncratic noise -> a stable, knowable corr
    rng = np.random.default_rng(99)
    a_ret = a.pct_change().fillna(0.0).values
    b_ret = 0.7 * a_ret + 0.3 * rng.normal(0.0003, 0.010, size=len(a))
    b = pd.Series(100.0 * np.exp(np.cumsum(b_ret)), index=idx_corr, name="Close")
    _write_snapshot("CRA", LB_CORR, a)
    _write_snapshot("CRB", LB_CORR, b)
    r1 = a.pct_change()
    r2 = b.pct_change()
    joined = pd.concat([r1, r2], axis=1, join="inner").dropna().tail(252)
    corr_val = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
    expected["cases"]["correlation_CRA_CRB_252"] = {
        "ticker": "CRA,CRB", "indicator": "correlation", "window": 252,
        "expected_value": round(corr_val, 3),
        "source_id": f"quant:correlation:CRA-CRB:252:{AS_OF}",
        "content_value_str": f"{corr_val:.3f}",
        "sessions": len(joined),
    }

    out = HERE / "golden_dataset" / "expected_quant.json"
    out.write_text(json.dumps(expected, indent=2))

    # --- Extra SYN snapshots for non-indicator lookbacks ---
    # price_history default lookback = 90 ; make_chart default = 252 + 30 = 282
    for lb in (90, 282):
        idx_extra = _business_days_back(AS_OF, lb)
        syn_extra = _synth_close(idx_extra, seed=42, start_price=150.0,
                                 drift=0.0004, vol=0.012)
        _write_snapshot("SYN", lb, syn_extra)

    return expected


if __name__ == "__main__":
    exp = build()
    print(f"Wrote {len(list(SNAP.glob('*.csv')))} snapshot CSVs to {SNAP}")
    print(f"Ground-truth quant values for {len(exp['cases'])} cases:")
    for k, v in exp["cases"].items():
        print(f"  {k:28s} -> {v['content_value_str']:>10s}  [{v['source_id']}]")
