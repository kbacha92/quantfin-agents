#!/usr/bin/env python3
"""Snapshot generator (reusable pipeline piece).

Slices the raw per-ticker price store (eval/raw_prices/<TICKER>.csv, produced by
acquire_data.py) into correctly-named, as-of-truncated snapshots that the price
tools replay offline. You never name a snapshot by hand: names are derived from
(ticker, as_of, lookback) using the SAME cache-path format the tool uses.

Two modes:
  --from-eval <jsonl>   : read every case's as_of, infer the lookback per
                          indicator, and produce exactly the snapshots the eval
                          needs. (Auto-derive; the default workflow.)
  --asof D --lookback N : produce one specific snapshot.

    python eval/make_snapshots.py --ticker AAPL --from-eval eval/aapl/aapl.jsonl
    python eval/make_snapshots.py --ticker AAPL --asof 2024-06-14 --lookback 380

Lookback per indicator (matches the tool's internal logic):
  rsi/sma/ema/realized_vol -> 380 ; max_drawdown -> 433 ; correlation -> 504
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_PRICES = HERE / "raw_prices"
API_CACHE = ROOT / "data" / "api_cache"

LOOKBACK = {"rsi": 380, "sma": 380, "ema": 380, "realized_vol": 380,
            "max_drawdown": 433, "correlation": 504}


def snapshot_name(ticker: str, as_of: str, lookback: int) -> str:
    return f"yfinance_{ticker.upper()}_{as_of}_{int(lookback)}d.csv"


def make_one(ticker: str, as_of: str, lookback: int, out_dir: Path) -> Path:
    raw = RAW_PRICES / f"{ticker.upper()}.csv"
    if not raw.exists():
        raise SystemExit(f"raw price store missing: {raw}. Run acquire_data.py first.")
    df = pd.read_csv(raw, index_col=0, parse_dates=True)
    as_of_d = date.fromisoformat(as_of)
    # truncate to <= as_of, take trailing rows the tool would have seen
    df = df[[d.date() <= as_of_d for d in df.index]]
    if df.empty:
        raise SystemExit(f"no price rows on/before {as_of} for {ticker}")
    # the tool requests `lookback` calendar days; keep a generous trailing slice
    df = df.tail(max(int(lookback), 400))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / snapshot_name(ticker, as_of, lookback)
    df.to_csv(path)
    return path


def lookbacks_for_case(case: dict) -> set[int]:
    """Infer which lookbacks a case needs from its query/assertions. Matches both
    the canonical indicator key and common natural-language phrasings."""
    text = (case.get("query", "") + " " + json.dumps(case.get("assertion_sketch", ""))
            + " " + json.dumps(case.get("expect_citation_source_ids", ""))).lower()
    # phrase -> lookback
    PHRASES = {
        380: ["rsi", "sma", "ema", "simple moving average", "exponential moving average",
              "realized vol", "realized volatility", "volatility"],
        433: ["max_drawdown", "max drawdown", "maximum drawdown", "drawdown"],
        504: ["correlation", "correl"],
    }
    needed = set()
    for lb, phrases in PHRASES.items():
        if any(p in text for p in phrases):
            needed.add(lb)
    if "price" in text or "closing" in text:
        needed.add(90)
    return needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--from-eval", help="eval JSONL to auto-derive snapshots from")
    ap.add_argument("--asof", help="single snapshot: as-of date")
    ap.add_argument("--lookback", type=int, help="single snapshot: lookback days")
    ap.add_argument("--out", default=str(API_CACHE),
                    help="output dir (default: the agent's api_cache)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    made = []

    if args.from_eval:
        cases = [json.loads(l) for l in Path(args.from_eval).read_text().splitlines() if l.strip()]
        for c in cases:
            as_of = c.get("as_of", "")
            if not re.match(r"\d{4}-\d{2}-\d{2}$", as_of):
                print(f"  skip {c.get('id')}: as_of not a concrete date ({as_of!r})")
                continue
            for lb in lookbacks_for_case(c):
                p = make_one(args.ticker, as_of, lb, out_dir)
                made.append(p.name)
                print(f"  {c.get('id')}: {p.name}")
    elif args.asof and args.lookback:
        p = make_one(args.ticker, args.asof, args.lookback, out_dir)
        made.append(p.name)
        print(f"  {p.name}")
    else:
        raise SystemExit("provide either --from-eval, or both --asof and --lookback")

    print(f"\nWrote {len(set(made))} snapshot(s) to {out_dir}")


if __name__ == "__main__":
    main()
