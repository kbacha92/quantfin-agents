"""Agent-facing options tool: price, Greeks, and implied-vol solving.

Wraps the closed-form options math as a Tool the planner can call. Unlike the
price-based indicators, options inputs (spot, strike, expiry, rate, vol) are
passed directly - the synthetic/analytic design means there's no market data to
load, so no PIT price-loader is involved. The computed result is exact and is
returned as Evidence with a deterministic source_id, so the eval can assert on it
like any other computed value.
"""
from __future__ import annotations

from datetime import date

from ...schemas import Evidence
from ..base import Tool
from .options import bs_price, greeks, implied_vol


def _q_sid(q: float) -> str:
    """Source-id suffix for dividend yield: empty at q=0 (v11.2 compat)."""
    return f":q={q}" if q else ""


class OptionTool(Tool):
    name = "option_calc"
    needs_query = False
    description = (
        "option_calc(op: str, spot: float, strike: float, expiry_years: float, rate: float, "
        "vol: float = None, price: float = None, kind: str = 'call', dividend_yield: float = 0) "
        "- European Black-Scholes(-Merton). op='price' needs vol; op='greeks' needs vol (returns "
        "delta/gamma/vega/theta/rho; vega per 1.00 vol, theta per year); op='implied_vol' needs "
        "price (solves for vol). kind is 'call' or 'put'. dividend_yield is the continuous yield "
        "for dividend-paying underlyings (0 = classic BS)."
    )

    def run(self, as_of: date, op, spot, strike, expiry_years, rate,
            vol=None, price=None, kind="call", dividend_yield=0.0, **_):
        op = str(op).lower().strip()
        kind = str(kind).lower().strip()
        S, K, T, r = float(spot), float(strike), float(expiry_years), float(rate)
        q = float(dividend_yield or 0.0)
        qs = _q_sid(q)
        qtxt = f", div_yield={q}" if q else ""

        if op == "price":
            if vol is None:
                raise ValueError("op='price' requires vol")
            val = bs_price(S, K, T, r, float(vol), kind, q)
            content = (f"Black-Scholes {kind} price for spot={S}, strike={K}, T={T}y, r={r}, "
                       f"vol={vol}{qtxt}: {val:.6f}.")
            sid = f"option:price:{kind}:{S}:{K}:{T}:{r}:{vol}{qs}"
        elif op == "greeks":
            if vol is None:
                raise ValueError("op='greeks' requires vol")
            g = greeks(S, K, T, r, float(vol), kind, q)
            content = (f"Greeks for {kind} spot={S}, strike={K}, T={T}y, r={r}, vol={vol}{qtxt}: "
                       + ", ".join(f"{k}={v:.6f}" for k, v in g.items())
                       + ". Units: vega per 1.00 vol, theta per year, rho per 1.00 rate.")
            sid = f"option:greeks:{kind}:{S}:{K}:{T}:{r}:{vol}{qs}"
        elif op == "implied_vol":
            if price is None:
                raise ValueError("op='implied_vol' requires price")
            iv = implied_vol(float(price), S, K, T, r, kind, q=q)
            content = (f"Implied volatility for {kind} priced at {price} (spot={S}, strike={K}, "
                       f"T={T}y, r={r}{qtxt}): {iv:.6f}.")
            sid = f"option:implied_vol:{kind}:{S}:{K}:{T}:{r}:{price}{qs}"
        else:
            raise ValueError("op must be 'price', 'greeks', or 'implied_vol'")

        return [Evidence(source_id=sid, store="quant", content=content, pub_date=as_of,
                         pit_safe=False, meta={"pit_caveat": "analytic"})]
