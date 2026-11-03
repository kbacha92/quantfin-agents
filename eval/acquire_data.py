#!/usr/bin/env python3
"""Eval data-acquisition pipeline (v6).

Pulls the real-world data an evaluation needs for a given ticker and date range:
  1. 10-K filings from SEC EDGAR across the range -> saved into the docs corpus
     with correct YYYY-MM-DD pub-date filename prefixes (drives the PIT filter).
  2. Daily price history from yfinance -> a raw per-ticker store, from which
     correctly-named as-of snapshots are sliced on demand.

It then records the REAL filing dates it pulled into a manifest, so the case
generator can build point-in-time cases whose as-of logic is anchored to actual
dates rather than guesses.

    python eval/acquire_data.py --ticker AAPL --start 2020-12-31 --end 2024-12-31

Outputs:
    data/docs/<YYYY-MM-DD>_<TICKER>_10-K.txt        (indexed into the corpus)
    eval/raw_prices/<TICKER>.csv                    (full daily history)
    eval/manifest/<TICKER>.json                     (filing dates + price coverage)

Network: needs SEC EDGAR + yfinance access. Set SEC_IDENTITY (e.g. "Name email")
for SEC's fair-access policy. Reuses the agent's EDGAR CIK lookup.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RAW_PRICES = HERE / "raw_prices"
MANIFEST = HERE / "manifest"


def _headers() -> dict:
    ident = os.environ.get("SEC_IDENTITY", "quant-agent-eval research@example.com")
    return {"User-Agent": ident}


def _cik(ticker: str):
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}", row["title"]
    raise RuntimeError(f"Ticker {ticker} not found on SEC EDGAR.")


def fetch_10ks(ticker: str, start: str, end: str, docs_dir: Path) -> list[dict]:
    """Enumerate all 10-K filings in [start, end], download each filing's primary
    document text, and save with a pub-date filename prefix. Returns metadata
    list: [{filing_date, accession, path}]."""
    cik, title = _cik(ticker)
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                       headers=_headers(), timeout=30)
    sub.raise_for_status()
    recent = sub.json()["filings"]["recent"]
    forms = recent["form"]
    dates = recent["filingDate"]
    accns = recent["accessionNumber"]
    prim = recent["primaryDocument"]

    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    docs_dir.mkdir(parents=True, exist_ok=True)
    for form, fdate, accn, doc in zip(forms, dates, accns, prim):
        if form != "10-K":
            continue
        fd = date.fromisoformat(fdate)
        if not (start_d <= fd <= end_d):
            continue
        accn_nodash = accn.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{doc}"
        time.sleep(0.2)  # SEC fair-access throttle
        try:
            r = requests.get(url, headers=_headers(), timeout=60)
            r.raise_for_status()
            text = r.text
        except Exception as e:
            print(f"  WARN: could not fetch {accn} ({fdate}): {e}")
            continue
        # crude HTML strip; the corpus indexer tolerates raw text
        import re as _re
        clean = _re.sub(r"<[^>]+>", " ", text)
        clean = _re.sub(r"\s+", " ", clean).strip()
        fname = f"{fdate}_{ticker.upper()}_10-K.txt"
        path = docs_dir / fname
        path.write_text(clean[:500000], encoding="utf-8")  # cap size
        out.append({"filing_date": fdate, "accession": accn, "path": str(path),
                    "source_id": f"doc:{fname}#chunk0"})
        print(f"  pulled 10-K filed {fdate} -> {fname}")
    if not out:
        print(f"  (no 10-K filings found for {ticker} in {start}..{end})")
    return out


def fetch_prices(ticker: str, start: str, end: str) -> dict:
    """Download daily history into the raw per-ticker store. Returns coverage."""
    import yfinance as yf

    RAW_PRICES.mkdir(parents=True, exist_ok=True)
    # pad start back so as-of snapshots near the range start still have lookback
    df = yf.download(ticker, start="2018-01-01", end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No price data for {ticker}.")
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] for c in df.columns]
    raw_path = RAW_PRICES / f"{ticker.upper()}.csv"
    df.to_csv(raw_path)
    cov = {"raw_path": str(raw_path), "rows": len(df),
           "first": str(df.index[0].date()), "last": str(df.index[-1].date())}
    print(f"  prices: {cov['rows']} rows {cov['first']}..{cov['last']} -> {raw_path.name}")
    return cov


def main():
    ap = argparse.ArgumentParser(description="Acquire EDGAR 10-Ks + yfinance prices for an eval.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=True, help="ISO date, e.g. 2020-12-31")
    ap.add_argument("--end", required=True, help="ISO date, e.g. 2024-12-31")
    ap.add_argument("--docs-dir", default=str(ROOT / "data" / "docs"),
                    help="where to write filings (default: the agent's docs corpus)")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-filings", action="store_true")
    args = ap.parse_args()

    T = args.ticker.upper()
    print(f"Acquiring data for {T}, {args.start}..{args.end}")
    manifest = {"ticker": T, "start": args.start, "end": args.end,
                "acquired_at": datetime.now().isoformat(timespec="seconds")}

    if not args.skip_filings:
        print("EDGAR 10-Ks:")
        manifest["filings"] = fetch_10ks(T, args.start, args.end, Path(args.docs_dir))
    if not args.skip_prices:
        print("yfinance prices:")
        manifest["prices"] = fetch_prices(T, args.start, args.end)

    MANIFEST.mkdir(parents=True, exist_ok=True)
    mpath = MANIFEST / f"{T}.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest -> {mpath}")
    nf = len(manifest.get("filings", []))
    print(f"Done: {nf} filing(s) pulled. Next: python eval/build_eval_from_manifest.py --ticker {T}")


if __name__ == "__main__":
    main()
