
import json
from pathlib import Path

from quant_agent.tools.quant.options import bs_price, greeks, implied_vol

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "eval" / "options" / "options.jsonl"


def _cases():
    if not DS.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "eval" / "build_options_eval.py")],
                       cwd=str(ROOT), check=True)
    return [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]


def test_all_cases_have_numeric_ground_truth():
    for c in _cases():
        assert isinstance(c["expected_value"], (int, float)), c["id"]
        assert not str(c["expected_value"]).startswith("[VERIFY"), c["id"]


def test_cases_are_self_consistent():
    # Recompute each case's answer independently and confirm it matches.
    for c in _cases():
        sid = c["expect_citation_source_ids"][0]
        parts = sid.split(":")
        if parts[1] == "price":
            _, _, kind, S, K, T, r, vol = parts
            val = bs_price(float(S), float(K), float(T), float(r), float(vol), kind)
            assert abs(val - c["expected_value"]) < 0.02, c["id"]
        elif parts[1] == "greeks":
            _, _, kind, S, K, T, r, vol = parts
            val = greeks(float(S), float(K), float(T), float(r), float(vol), kind)["delta"]
            assert abs(val - c["expected_value"]) < 0.01, c["id"]
        elif parts[1] == "implied_vol":
            _, _, kind, S, K, T, r, px = parts
            val = implied_vol(float(px), float(S), float(K), float(T), float(r), kind)
            assert abs(val - c["expected_value"]) < 0.01, c["id"]
