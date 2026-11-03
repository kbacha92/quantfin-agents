"""Synthetic option chain + volatility curve construction (eval ground truth).

A synthetic chain is generated from a KNOWN volatility smile, so every contract's
price is exact and the true implied vol per strike is known by construction. This
is what makes an options eval reproducible and scorable - the agent must recover
the smile we put in, the same way the RSI cases check a known computed value.

No market data, no PIT-sourcing problem: the chain is fabricated from chosen
parameters, frozen, and the answers are analytic.
"""
from __future__ import annotations

from .options import bs_price, implied_vol


def smile_vol(strike: float, spot: float, atm_vol: float, skew: float, curvature: float) -> float:
    """A simple quadratic smile in log-moneyness.

    vol(K) = atm_vol + skew * m + curvature * m^2,  m = ln(K/S).
    skew tilts the smile (negative = equity-style downside skew); curvature sets
    how much it smiles. Floored at a small positive vol.
    """
    import math
    m = math.log(strike / spot)
    v = atm_vol + skew * m + curvature * m * m
    return max(v, 0.01)


def build_chain(spot: float, strikes: list[float], T: float, r: float,
                atm_vol: float, skew: float, curvature: float,
                kind: str = "call") -> list[dict]:
    """Build a synthetic chain: for each strike, the true vol from the smile and
    the exact BS price at that vol. Returns rows the eval/agent can consume."""
    rows = []
    for K in strikes:
        true_vol = smile_vol(K, spot, atm_vol, skew, curvature)
        price = bs_price(spot, K, T, r, true_vol, kind)
        rows.append({
            "strike": K, "expiry_years": T, "kind": kind,
            "price": round(price, 6),
            "true_vol": round(true_vol, 6),  # ground truth (hidden from the agent in eval)
        })
    return rows


def recover_curve(spot: float, chain: list[dict], r: float) -> list[dict]:
    """Given a chain of (strike, price), recover implied vol per strike by
    inverting BS - this is what the agent should do to rebuild the smile."""
    out = []
    for row in chain:
        iv = implied_vol(row["price"], spot, row["strike"], row["expiry_years"],
                         r, row["kind"])
        out.append({"strike": row["strike"], "implied_vol": iv})
    return out
