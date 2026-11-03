
import math
from datetime import date

from quant_agent.tools.quant.options import bs_price, greeks, implied_vol
from quant_agent.tools.quant.option_tool import OptionTool


def test_q_zero_regression_matches_v11_1():
    # Same references as test_options.py - q default must not move them.
    assert abs(bs_price(100, 100, 1, 0.05, 0.20, "call") - 10.4506) < 1e-3
    assert abs(bs_price(100, 100, 1, 0.05, 0.20, "put") - 5.5735) < 1e-3
    g = greeks(100, 100, 1, 0.05, 0.20, "call")
    assert abs(g["delta"] - 0.6368) < 1e-3
    assert abs(g["gamma"] - 0.0188) < 1e-3
    # explicit q=0.0 is identical to omitting q
    assert bs_price(100, 100, 1, 0.05, 0.20, "call", q=0.0) == bs_price(100, 100, 1, 0.05, 0.20, "call")


def test_hull_index_option_reference():
    # Hull's European index call: S=930, K=900, r=0.08, q=0.03, sigma=0.20, T=2/12.
    c = bs_price(930, 900, 2 / 12, 0.08, 0.20, "call", q=0.03)
    assert abs(c - 51.83) < 0.05


def test_dividend_put_call_parity():
    # c - p = S e^{-qT} - K e^{-rT}
    S, K, T, r, q, sig = 100, 105, 0.75, 0.04, 0.025, 0.3
    c = bs_price(S, K, T, r, sig, "call", q=q)
    p = bs_price(S, K, T, r, sig, "put", q=q)
    assert abs((c - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-9


def test_iv_roundtrip_with_dividends():
    # Includes ITM, where ignoring q biases IV the most.
    for K in (70, 100, 140):
        for kind in ("call", "put"):
            px = bs_price(100, K, 0.5, 0.04, 0.35, kind, q=0.025)
            iv = implied_vol(px, 100, K, 0.5, 0.04, kind, q=0.025)
            assert abs(iv - 0.35) < 1e-4, (K, kind, iv)


def test_iv_bounds_respect_dividends():
    # A deep-ITM call's fair price under q can sit BELOW the q=0 intrinsic
    # floor (S - K e^{-rT}); the q-aware bounds must accept it, and the
    # solved vol must round-trip. (The old q=0 bounds would reject it.)
    S, K, T, r, q = 100, 60, 1.0, 0.05, 0.06
    px = bs_price(S, K, T, r, 0.15, "call", q=q)
    assert px < S - K * math.exp(-r * T)  # below old no-dividend lower bound
    iv = implied_vol(px, S, K, T, r, "call", q=q)
    assert abs(iv - 0.15) < 1e-4


def test_tool_source_id_backward_compatible():
    t = OptionTool()
    # q omitted / q=0: v11.2 source_id byte-for-byte (shipped eval depends on it)
    ev = t.run(as_of=date(2025, 3, 14), op="price", spot=100.0, strike=105.0,
               expiry_years=0.5, rate=0.04, vol=0.25)
    assert ev[0].source_id == "option:price:call:100.0:105.0:0.5:0.04:0.25"
    ev0 = t.run(as_of=date(2025, 3, 14), op="price", spot=100.0, strike=105.0,
                expiry_years=0.5, rate=0.04, vol=0.25, dividend_yield=0.0)
    assert ev0[0].source_id == ev[0].source_id
    # nonzero q: distinct, deterministic id (different inputs != same evidence)
    ev2 = t.run(as_of=date(2025, 3, 14), op="price", spot=100.0, strike=105.0,
                expiry_years=0.5, rate=0.04, vol=0.25, dividend_yield=0.02)
    assert ev2[0].source_id == "option:price:call:100.0:105.0:0.5:0.04:0.25:q=0.02"
    assert ev2[0].source_id != ev[0].source_id


def test_tool_greeks_content_states_units():
    t = OptionTool()
    ev = t.run(as_of=date(2025, 3, 14), op="greeks", spot=100.0, strike=105.0,
               expiry_years=0.5, rate=0.04, vol=0.25)
    assert "vega per 1.00 vol" in ev[0].content and "theta per year" in ev[0].content
