"""options pricing, Greeks, IV solving, and synthetic chain round-trip."""
import math
from datetime import date

from quant_agent.tools.quant.options import bs_price, greeks, implied_vol
from quant_agent.tools.quant.option_chain import build_chain, recover_curve
from quant_agent.tools.quant.option_tool import OptionTool


def test_textbook_call_put():
    assert abs(bs_price(100, 100, 1, 0.05, 0.20, "call") - 10.4506) < 1e-3
    assert abs(bs_price(100, 100, 1, 0.05, 0.20, "put") - 5.5735) < 1e-3


def test_put_call_parity():
    c = bs_price(100, 100, 1, 0.05, 0.20, "call")
    p = bs_price(100, 100, 1, 0.05, 0.20, "put")
    assert abs((c - p) - (100 - 100 * math.exp(-0.05))) < 1e-6


def test_greeks_known_values():
    g = greeks(100, 100, 1, 0.05, 0.20, "call")
    assert abs(g["delta"] - 0.6368) < 1e-3
    assert abs(g["gamma"] - 0.0188) < 1e-3
    assert abs(g["vega"] - 37.52) < 1e-1


def test_iv_roundtrip_call_and_put():
    for kind, sig in (("call", 0.27), ("put", 0.42)):
        px = bs_price(100, 105, 0.5, 0.03, sig, kind)
        assert abs(implied_vol(px, 100, 105, 0.5, 0.03, kind) - sig) < 1e-4


def test_iv_rejects_arbitrage_price():
    # price above spot is impossible for a call
    try:
        implied_vol(200, 100, 100, 1, 0.05, "call")
        assert False, "should have raised"
    except ValueError:
        pass


def test_synthetic_chain_recovers_smile():
    spot, T, r = 100, 0.5, 0.03
    chain = build_chain(spot, [80, 90, 100, 110, 120], T, r,
                        atm_vol=0.25, skew=-0.15, curvature=0.8, kind="call")
    recovered = recover_curve(spot, chain, r)
    for orig, rec in zip(chain, recovered):
        assert abs(orig["true_vol"] - rec["implied_vol"]) < 1e-4
    # equity-style skew: downside strike vol exceeds ATM
    assert chain[0]["true_vol"] > chain[2]["true_vol"]


def test_option_tool_wrapper():
    t = OptionTool()
    ev = t.run(as_of=date(2025, 1, 1), op="price", spot=100, strike=100,
               expiry_years=1, rate=0.05, vol=0.20, kind="call")
    assert ev and "10.45" in ev[0].content
    ev2 = t.run(as_of=date(2025, 1, 1), op="implied_vol", spot=100, strike=100,
                expiry_years=1, rate=0.05, price=10.4506, kind="call")
    assert "0.20" in ev2[0].content or "0.199" in ev2[0].content
