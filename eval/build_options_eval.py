#!/usr/bin/env python3
"""Build the synthetic options eval (v11.2).

Generates cases whose answers are computed by the REAL options tools from a known
volatility smile, so every case has exact ground truth - the same discipline as
the SYN/DDN quant cases. Tests the computational core a vanilla model cannot do
by hand: Black-Scholes pricing, Greeks, implied-vol solving, and vol-curve
recovery.

    python eval/build_options_eval.py

Output: eval/options/options.jsonl  (cases with tool-verified expected_value)

Ground truth is computed HERE by calling the tools, so there are no [VERIFY]
placeholders - the eval is self-consistent by construction. A synthetic scenario
(spot/rate/expiry/smile) is fixed below; change it to regenerate a fresh set.
"""
from __future__ import annotations

import json
from pathlib import Path

from quant_agent.tools.quant.options import bs_price, greeks, implied_vol
from quant_agent.tools.quant.option_chain import build_chain, smile_vol

HERE = Path(__file__).resolve().parent
OUT = HERE / "options" / "options.jsonl"

# Fixed synthetic scenario (the "as-of" is nominal; options math is analytic).
AS_OF = "2025-03-14"
SPOT = 100.0
RATE = 0.04
EXPIRY = 0.5           # years
ATM_VOL, SKEW, CURV = 0.25, -0.15, 0.80


def build():
    cases = []

    def add(**c):
        c.setdefault("as_of", AS_OF)
        c.setdefault("capability", "options")
        cases.append(c)

    # 1) PRICING: exact BS price at a known vol.
    K = 105.0
    px = bs_price(SPOT, K, EXPIRY, RATE, ATM_VOL, "call")
    add(id="opt_price_call",
        query=(f"Using Black-Scholes, what is the price of a European CALL with spot={SPOT}, "
               f"strike={K}, expiry={EXPIRY} years, rate={RATE}, volatility={ATM_VOL}? "
               f"Use the option_calc tool."),
        expected_status="ok",
        expect_citation_source_ids=[f"option:price:call:{SPOT}:{K}:{EXPIRY}:{RATE}:{ATM_VOL}"],
        expected_value=round(px, 4), value_tolerance=0.02)

    # 2) GREEKS: delta at a known vol.
    g = greeks(SPOT, K, EXPIRY, RATE, ATM_VOL, "call")
    add(id="opt_greeks_delta",
        query=(f"What is the delta of a European CALL with spot={SPOT}, strike={K}, "
               f"expiry={EXPIRY} years, rate={RATE}, volatility={ATM_VOL}? Use option_calc."),
        expected_status="ok",
        expect_citation_source_ids=[f"option:greeks:call:{SPOT}:{K}:{EXPIRY}:{RATE}:{ATM_VOL}"],
        expected_value=round(g["delta"], 4), value_tolerance=0.01)

    # 3) IMPLIED VOL: recover sigma from a price (the decisive vanilla-vs-agent case).
    true_sigma = 0.32
    market_px = bs_price(SPOT, 110.0, EXPIRY, RATE, true_sigma, "call")
    add(id="opt_implied_vol",
        query=(f"A European CALL with spot={SPOT}, strike=110, expiry={EXPIRY} years, "
               f"rate={RATE} trades at {round(market_px,4)}. What is its implied volatility? "
               f"Use option_calc."),
        expected_status="ok",
        expect_citation_source_ids=[f"option:implied_vol:call:{SPOT}:110.0:{EXPIRY}:{RATE}:{round(market_px,4)}"],
        expected_value=round(true_sigma, 4), value_tolerance=0.01)

    # 4) VOL CURVE: recover the smile vol at one wing strike from its price.
    Kw = 90.0
    wing_vol = smile_vol(Kw, SPOT, ATM_VOL, SKEW, CURV)
    wing_px = bs_price(SPOT, Kw, EXPIRY, RATE, wing_vol, "call")
    add(id="opt_vol_curve_wing",
        query=(f"On a volatility smile, a European CALL with spot={SPOT}, strike={Kw}, "
               f"expiry={EXPIRY} years, rate={RATE} is priced at {round(wing_px,4)}. "
               f"What implied volatility does that imply? Use option_calc."),
        expected_status="ok",
        expect_citation_source_ids=[f"option:implied_vol:call:{SPOT}:{Kw}:{EXPIRY}:{RATE}:{round(wing_px,4)}"],
        expected_value=round(wing_vol, 4), value_tolerance=0.01,
        note="Wing of the smile; downside vol should exceed ATM (skew).")

    # 5) PUT pricing (kind coverage).
    Kp = 95.0
    put_px = bs_price(SPOT, Kp, EXPIRY, RATE, ATM_VOL, "put")
    add(id="opt_price_put",
        query=(f"Black-Scholes price of a European PUT: spot={SPOT}, strike={Kp}, "
               f"expiry={EXPIRY} years, rate={RATE}, volatility={ATM_VOL}? Use option_calc."),
        expected_status="ok",
        expect_citation_source_ids=[f"option:price:put:{SPOT}:{Kp}:{EXPIRY}:{RATE}:{ATM_VOL}"],
        expected_value=round(put_px, 4), value_tolerance=0.02)

    return cases


def main():
    cases = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(cases)} options cases -> {OUT}")
    print("All ground-truth values computed by the real tools (no [VERIFY] placeholders).")
    for c in cases:
        print(f"  {c['id']:22s} expected={c['expected_value']}")


if __name__ == "__main__":
    main()
