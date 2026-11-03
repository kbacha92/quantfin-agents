#!/usr/bin/env python3
"""Build eval cases anchored to the REAL filing dates in a ticker's manifest.

This is the payoff of the acquisition pipeline: because acquire_data.py recorded
the actual 10-K filing dates, this generator can construct point-in-time cases
whose as-of logic is correct BY CONSTRUCTION - no date guessing, no manual
[VERIFY] on the as-of windows.

    python eval/build_eval_from_manifest.py --ticker AAPL

What it generates from N real 10-K filing dates:
  PIT - future filing invisible: for each filing F, an as-of one day BEFORE F's
        date asks about F's content -> expected insufficient_data, and F must NOT
        be cited (forbid). (Needs >=1 filing.)
  PIT - visible filing: an as-of shortly AFTER F asks for the most recent annual
        figures -> expected ok, F should be citable. (Needs >=1 filing.)
  NUMERICAL - rsi/drawdown/vol at a price-data date -> exact-value cases. These
        still need the exact value filled in from the tool (one [VERIFY] each),
        but the as-of and snapshot naming are handled by make_snapshots.py.
  PROVENANCE - control: a basic figure from the most recent visible filing.

Facts that STILL need verification are limited to figures inside filings (which
require reading the document) - clearly marked. The DATE logic, which is where
manual work was most error-prone, is now automatic.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest"


def iso(d: date) -> str:
    return d.isoformat()


def build(ticker: str, manifest: dict) -> list[dict]:
    T = ticker.upper()
    filings = sorted(manifest.get("filings", []), key=lambda f: f["filing_date"])
    prices = manifest.get("prices", {})
    cases = []

    def add(**c):
        c.setdefault("ticker", T)
        cases.append(c)

    # ---- PIT cases anchored to real filing dates ----
    for i, f in enumerate(filings):
        fdate = date.fromisoformat(f["filing_date"])
        sid = f["source_id"]
        # future-invisible: as-of one day before this filing
        as_of_before = iso(fdate - timedelta(days=1))
        add(id=f"{T.lower()}_pit_future_{f['filing_date']}",
            capability="pit",
            query=f"What did {T} report in its annual (10-K) filing dated {f['filing_date']}? "
                  f"Answer based only on what is public as of the date below.",
            as_of=as_of_before,
            expected_status="insufficient_data",
            forbid_citation_source_ids=[sid],
            _anchor=f"as-of {as_of_before} is one day before this 10-K was filed ({f['filing_date']})",
            _status="READY",  # date logic is correct by construction; no figure asserted
            note="PIT: the 10-K is not yet public as of the as-of date; agent must not cite it.")

        # visible: as-of one week after this filing
        as_of_after = iso(fdate + timedelta(days=7))
        add(id=f"{T.lower()}_pit_visible_{f['filing_date']}",
            capability="pit",
            query=f"As of the date below, what is the most recent annual (10-K) filing {T} has made, "
                  f"and roughly when was it filed?",
            as_of=as_of_after,
            expected_status="ok",
            expect_citation_source_ids=[sid],
            _anchor=f"as-of {as_of_after} is one week after this 10-K ({f['filing_date']})",
            _status="VERIFY_VISIBLE",  # confirm this is the most-recent as of that date
            note="PIT: the 10-K is visible as of the as-of date and should be citable. "
                 "VERIFY there isn't a later 10-K before the as-of that would be 'most recent' instead.")

    # ---- NUMERICAL cases (need a price-data date with lookback) ----
    last_price = prices.get("last")
    if last_price:
        # use a date ~30 days before the last available price, so lookback exists
        anchor_d = date.fromisoformat(last_price) - timedelta(days=30)
        as_of = iso(anchor_d)
        for ind, lb, label in [("rsi", 14, "14-day RSI"),
                               ("max_drawdown", 252, "maximum drawdown over the last year"),
                               ("realized_vol", 21, "21-day annualized realized volatility")]:
            ind_key = "rsi" if ind == "rsi" else ind
            # snapshot lookback the price tool will request for this indicator
            snap_lb = {"rsi": 380, "max_drawdown": 433, "realized_vol": 380}[ind]
            add(id=f"{T.lower()}_num_{ind}",
                capability="numerical",
                query=f"What is {T}'s {label} as of the date below?",
                as_of=as_of,
                snapshots=[f"yfinance_{T}_{as_of}_{snap_lb}d.csv"],
                expected_status="ok",
                expect_citation_source_ids=[f"quant:{ind}:{T}:{lb}:{as_of}"],
                expected_value="[VERIFY: run make_snapshots then read the tool's exact value]",
                value_tolerance=0.5 if ind != "realized_vol" else 0.01,
                _status="VERIFY_VALUE",
                note=f"NUMERICAL: snapshot listed in 'snapshots' (staged from data/api_cache). Fill "
                     f"expected_value by running the tool once on the generated snapshot.")

    # ---- PROVENANCE control: a basic figure from the most recent filing ----
    if filings:
        f = filings[-1]
        fdate = date.fromisoformat(f["filing_date"])
        add(id=f"{T.lower()}_prov_control",
            capability="provenance",
            query=f"According to {T}'s most recent 10-K as of the date below, what was total net revenue "
                  f"for the fiscal year? Cite the filing.",
            as_of=iso(fdate + timedelta(days=7)),
            expected_status="ok",
            expect_citation_source_ids=[f["source_id"]],
            must_contain=["[VERIFY: total net revenue figure from this 10-K]"],
            _status="VERIFY_VALUE",
            note="PROVENANCE control: fill in the revenue figure from the filing; date logic is automatic.")

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    T = args.ticker.upper()
    mpath = MANIFEST / f"{T}.json"
    if not mpath.exists():
        raise SystemExit(f"manifest not found: {mpath}. Run acquire_data.py --ticker {T} first.")
    manifest = json.loads(mpath.read_text())
    cases = build(T, manifest)
    out = Path(args.out) if args.out else HERE / T.lower() / f"{T.lower()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    from collections import Counter
    by_status = Counter(c.get("_status") for c in cases)
    by_cap = Counter(c.get("capability") for c in cases)
    print(f"Wrote {len(cases)} cases for {T} -> {out}")
    print(f"  by capability: {dict(by_cap)}")
    print(f"  by readiness:  {dict(by_status)}")
    print(f"\n  READY            = date logic correct, runnable as-is (PIT future-invisible)")
    print(f"  VERIFY_VISIBLE   = confirm 'most recent' filing as of the as-of date")
    print(f"  VERIFY_VALUE     = fill one figure/number from filing or tool")
    print(f"\n  Next: python eval/make_snapshots.py --ticker {T} --from-eval {out}")


if __name__ == "__main__":
    main()
