"""Options pricing layer: Black-Scholes price, Greeks, and implied-vol solving.

Pure functions with closed-form ground truth, mirroring the indicators module.
European options, continuous compounding. Implied vol is solved by Newton-Raphson
with a bisection fallback so it converges even where vega is tiny.

These are exact and deterministic - given (S, K, T, r, q, sigma, type) the price
and Greeks have one correct value, which is what lets the options eval score
against known answers the same way the RSI/vol cases do.

UNITS: vega is per 1.00 of vol (divide by 100 for per-vol-point); theta is per
YEAR (divide by 365 for per-day); rho is per 1.00 of rate.
"""
from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, sigma must be positive for Black-Scholes")
    if q < 0:
        raise ValueError("dividend yield q must be >= 0")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", q: float = 0.0) -> float:
    """Black-Scholes(-Merton) price of a European option.

    S spot, K strike, T years to expiry, r risk-free rate, sigma vol,
    kind call/put, q continuous dividend yield (default 0 = classic BS).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if kind == "call":
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    elif kind == "put":
        return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           kind: str = "call", q: float = 0.0) -> dict:
    """Return delta, gamma, vega, theta, rho.

    UNITS (raw analytic forms): vega per 1.00 vol, theta per YEAR, rho per
    1.00 rate. q=0 reproduces the classic no-dividend Greeks exactly.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf = _norm_pdf(d1)
    sqrtT = math.sqrt(T)
    gamma = disc_q * pdf / (S * sigma * sqrtT)
    vega = S * disc_q * pdf * sqrtT
    if kind == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sqrtT)
                 - r * K * disc_r * _norm_cdf(d2)
                 + q * S * disc_q * _norm_cdf(d1))
        rho = K * T * disc_r * _norm_cdf(d2)
    elif kind == "put":
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta = (-S * disc_q * pdf * sigma / (2 * sqrtT)
                 + r * K * disc_r * _norm_cdf(-d2)
                 - q * S * disc_q * _norm_cdf(-d1))
        rho = -K * T * disc_r * _norm_cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                kind: str = "call", tol: float = 1e-6, max_iter: int = 100,
                q: float = 0.0) -> float:
    """Recover sigma from a market price by inverting Black-Scholes(-Merton).

    Newton-Raphson (fast where vega is healthy) with a bisection fallback that
    guarantees convergence on a bracketed interval. Raises if the price is
    outside no-arbitrage bounds (no sigma can reproduce it). q defaults to 0;
    for dividend payers pass the continuous yield or ITM IVs will be biased.
    """
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    # No-arbitrage bounds with dividends:
    #   call: [max(0, S e^{-qT} - K e^{-rT}),  S e^{-qT}]
    #   put:  [max(0, K e^{-rT} - S e^{-qT}),  K e^{-rT}]
    if kind == "call":
        lo_bound, hi_bound = max(0.0, S * disc_q - K * disc_r), S * disc_q
    elif kind == "put":
        lo_bound, hi_bound = max(0.0, K * disc_r - S * disc_q), K * disc_r
    else:
        raise ValueError("kind must be 'call' or 'put'")
    if price < lo_bound - 1e-9 or price > hi_bound + 1e-9:
        raise ValueError(f"price {price} outside no-arbitrage bounds [{lo_bound}, {hi_bound}]")

    # Newton-Raphson seeded at a reasonable vol.
    sigma = 0.2
    for _ in range(max_iter):
        try:
            model = bs_price(S, K, T, r, sigma, kind, q)
            v = greeks(S, K, T, r, sigma, kind, q)["vega"]
        except ValueError:
            break
        diff = model - price
        if abs(diff) < tol:
            return sigma
        if v < 1e-8:
            break  # vega too small; fall back to bisection
        sigma = sigma - diff / v
        if sigma <= 0 or sigma > 10:
            break  # diverged; fall back to bisection

    # Bisection fallback on a wide vol bracket.
    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        model = bs_price(S, K, T, r, mid, kind, q)
        if abs(model - price) < tol:
            return mid
        if model > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
