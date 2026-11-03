#!/usr/bin/env python3
"""Live yfinance options smoke test (v11.3) - UNSCORED capability check.

Downloads the CURRENT option chain for a ticker from yfinance and exercises the
options tools on real market data: solve implied vol from real mid-prices and
assemble a volatility curve across strikes. This tests that the download +
compute machinery works end to end on live data.

    python eval/live_options_smoketest.py --ticker AAPL

WHY THIS IS NOT A SCORED EVAL (important):
- yfinance exposes only the CURRENT chain, not historical/as-of chains, so results
  are NOT reproducible - run it twice on different days and the data differs.
- Live prices have NO frozen ground truth to score against; we can only sanity-
  check (IV solves within bounds, ATM vol is plausible, smile has a shape).
- The scored options eval (eval/options/options.jsonl) uses SYNTHETIC chains with
  analytic ground truth for exactly these reasons. This script is the separate,
  unscored counterpart that verifies the live path.

SANITY CHECKS (not pass/fail scoring): each solved IV is finite and within
plausible bounds; the ATM implied vol is in a sane range; the curve has >= 3
points so a smile is visible. Yahoo occasionally returns NaN bids/asks for
illiquid strikes - those are skipped and reported, not treated as failures.

MODEL CAVEAT (v11.4): pricing is European Black-Scholes-Merton with a
continuous dividend yield (--dividend-yield). US single-name options are
AMERICAN; on dividend payers, early exercise adds a premium the model does
not capture, so solved IVs are approximate - worst for deep-ITM calls near
ex-dividend dates. Fine for a smoke test; do not treat these IVs as exact.

NETWORK: needs internet access to Yahoo Finance. Cannot run in a sandbox without
egress to Yahoo - run it on your own machine.
"""
from __future__ import annotations

import argparse
import math
from datetime import date, datetime

from quant_agent.tools.quant.options import implied_vol


def _year_fraction(expiry_str: str) -> float:
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (exp - date.today()).days
    return max(days, 1) / 365.0


def run(ticker: str, rate: float, max_expiries: int, moneyness: float, q: float = 0.0):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed. pip install yfinance")

    tk = yf.Ticker(ticker)
    spot = None
    try:
        # fast_info is the current lightweight quote path
        spot = float(tk.fast_info["last_price"])
    except Exception:
        try:
            spot = float(tk.history(period="1d")["Close"].iloc[-1])
        except Exception:
            raise SystemExit(f"Could not fetch spot price for {ticker}.")

    expiries = list(tk.options)[:max_expiries]
    if not expiries:
        raise SystemExit(f"No option expiries returned for {ticker}.")

    print(f"{ticker}  spot={spot:.2f}  rate={rate}  div_yield={q}  (as of {date.today()}, LIVE - not reproducible)")
    if q == 0.0:
        print("NOTE: dividend_yield=0. For dividend payers (AAPL etc), pass --dividend-yield")
        print("      or ITM IVs will be biased low (v11.4). Model is still EUROPEAN, so")
        print("      American early-exercise premium remains a small residual approximation.")
    print("=" * 66)

    for expiry in expiries:
        T = _year_fraction(expiry)
        try:
            chain = tk.option_chain(expiry)
        except Exception as e:
            print(f"[{expiry}] could not fetch chain: {e}")
            continue
        calls = chain.calls
        lo, hi = spot * (1 - moneyness), spot * (1 + moneyness)
        near = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        if near.empty:
            print(f"[{expiry}] no strikes within +/-{moneyness:.0%} of spot")
            continue

        print(f"[{expiry}]  T={T:.3f}y  (strike, mid, solved_iv)")
        curve = []
        skipped = 0
        for _, row in near.iterrows():
            K = float(row["strike"])
            bid, ask = row.get("bid"), row.get("ask")
            # prefer mid; fall back to lastPrice if bid/ask missing
            if bid and ask and bid > 0 and ask > 0:
                mid = 0.5 * (float(bid) + float(ask))
            else:
                last = row.get("lastPrice")
                if not last or float(last) <= 0 or (isinstance(last, float) and math.isnan(last)):
                    skipped += 1
                    continue
                mid = float(last)
            try:
                iv = implied_vol(mid, spot, K, T, rate, "call", q=q)
                if not math.isfinite(iv) or iv <= 0 or iv > 5:
                    skipped += 1
                    continue
                curve.append((K, iv))
                print(f"    K={K:>8.2f}  mid={mid:>8.2f}  iv={iv:.4f}")
            except Exception:
                skipped += 1

        # sanity summary (NOT scoring)
        if curve:
            atm = min(curve, key=lambda kv: abs(kv[0] - spot))
            print(f"    -> {len(curve)} points, {skipped} skipped; ATM(~K={atm[0]:.0f}) IV={atm[1]:.4f}")
            sane = 0.02 <= atm[1] <= 2.0
            print(f"    -> ATM IV plausible (2%-200%): {sane}; smile visible (>=3 pts): {len(curve) >= 3}")
        else:
            print(f"    -> no solvable strikes ({skipped} skipped) - illiquid or bad quotes")
        print()

    print("Reminder: LIVE data - not reproducible, not scored. For the reproducible")
    print("scored eval, use eval/options/options.jsonl (synthetic, analytic truth).")


def main():
    ap = argparse.ArgumentParser(description="Live yfinance options smoke test (unscored).")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--rate", type=float, default=0.04, help="risk-free rate assumption")
    ap.add_argument("--max-expiries", type=int, default=2)
    ap.add_argument("--moneyness", type=float, default=0.10,
                    help="strike window as +/- fraction of spot (0.10 = +/-10%%)")
    ap.add_argument("--dividend-yield", type=float, default=0.0, dest="q",
                    help="continuous dividend yield of the underlying (v11.4; e.g. 0.005 for AAPL)")
    args = ap.parse_args()
    run(args.ticker.upper(), args.rate, args.max_expiries, args.moneyness, args.q)


if __name__ == "__main__":
    main()
