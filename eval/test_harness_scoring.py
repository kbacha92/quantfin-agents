#!/usr/bin/env python3
"""Offline self-test for the harness SCORER (no LLM, no network).

Verifies that run_harness.score() and store_of() behave correctly against
crafted Answer/case pairs covering a pass and every failure branch. This does
NOT run the agent; it guards the scoring logic itself from regressions.

Run:  PYTHONPATH=. python eval/test_harness_scoring.py   (exit 0 = all good)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_harness", HERE / "run_harness.py")
rh = importlib.util.module_from_spec(spec)
sys.argv = ["run_harness"]  # argparse not invoked on import, but be safe
spec.loader.exec_module(rh)


def _res(ans=None, tools=None, err=None):
    return {"answer": ans, "tools_fired": tools or [], "trace_id": "t", "error": err}


def _ans(status="ok", text="", citations=None, artifacts=None):
    return {"status": status, "text": text, "citations": citations or [],
            "artifacts": artifacts or [], "iterations": 1, "trace_id": "t"}


def check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok  {name}")


def main() -> int:
    # store_of mapping
    check("store_of doc->doc_kb", rh.store_of("doc:f#chunk0") == "doc_kb")
    check("store_of yfinance->prices", rh.store_of("yfinance:SYN:2025-03-14:380d") == "prices")
    check("store_of quant->quant", rh.store_of("quant:rsi:SYN:14:2025-03-14") == "quant")
    check("store_of knowledge->knowledge", rh.store_of("knowledge:x.txt#chunk0") == "knowledge")

    # clean pass
    case = {"area": "pit", "expected_status": "ok", "expect_tools_any": ["doc_search"],
            "expect_citation_source_ids": ["doc:f#chunk0"], "must_contain": ["38", "customer"],
            "must_not_contain": ["Beacon"]}
    p, r = rh.score(case, _res(_ans("ok", "customer was 38% [doc:f#chunk0]", ["doc:f#chunk0"]), ["doc_search"]))
    check("clean pass", p and not r)

    # status mismatch
    p, r = rh.score(case, _res(_ans("insufficient_data", "x")))
    check("status mismatch fails", (not p) and any("status" in x for x in r))

    # forbidden store
    cf = {"area": "pit", "expected_status": "insufficient_data", "forbid_citation_store": ["doc_kb"]}
    p, r = rh.score(cf, _res(_ans("insufficient_data", "x", ["doc:g#chunk0"])))
    check("forbidden store fails", (not p) and any("forbidden" in x for x in r))

    # must_not_contain trips
    cn = {"area": "pit", "expected_status": "insufficient_data", "must_not_contain": ["Beacon"]}
    p, r = rh.score(cn, _res(_ans("insufficient_data", "mentions Beacon")))
    check("must_not_contain fails", (not p) and any("must_not_contain" in x for x in r))

    # required citation missing
    cc = {"area": "quant", "expected_status": "ok",
          "expect_citation_source_ids": ["quant:rsi:SYN:14:2025-03-14"], "must_contain": ["52.3673"]}
    p, r = rh.score(cc, _res(_ans("ok", "RSI is 52.3673"), ["quant_indicator"]))
    check("missing citation fails", (not p) and any("required citation" in x for x in r))

    # expect_citation_store_any
    cs = {"area": "methodology", "expected_status": "ok", "expect_citation_store_any": ["knowledge"]}
    p, _ = rh.score(cs, _res(_ans("ok", "x", ["knowledge:r.txt#chunk0"]), ["knowledge_search"]))
    check("store_any pass", p)
    p, _ = rh.score(cs, _res(_ans("ok", "x", ["quant:y:1:1:1"])))
    check("store_any fail", not p)

    # expect_tools_all
    ct = {"area": "quant", "expected_status": "ok", "expect_tools_all": ["quant_indicator"]}
    p, _ = rh.score(ct, _res(_ans("ok", "x"), ["make_chart"]))
    check("tools_all missing fails", not p)

    # expected_value tolerance
    cv = {"area": "quant", "expected_status": "ok", "expected_value": 52.3673, "value_tolerance": 0.01}
    p, _ = rh.score(cv, _res(_ans("ok", "RSI = 52.3673"), ["quant_indicator"]))
    check("value in tolerance pass", p)
    p, r = rh.score(cv, _res(_ans("ok", "RSI = 60.0"), ["quant_indicator"]))
    check("value out of tolerance fails", (not p) and any("expected_value" in x for x in r))

    # artifact
    ca = {"area": "quant", "expected_status": "ok", "expect_artifact": True}
    p, _ = rh.score(ca, _res(_ans("ok", "saved", artifacts=["/x.png"]), ["make_chart"]))
    check("artifact present pass", p)
    p, _ = rh.score(ca, _res(_ans("ok", "no chart"), ["make_chart"]))
    check("artifact missing fails", not p)

    # forbidden exact citation (PIT leak)
    cfs = {"area": "pit", "expected_status": "insufficient_data",
           "forbid_citation_source_ids": ["doc:future.txt#chunk0"]}
    p, _ = rh.score(cfs, _res(_ans("insufficient_data", "x", ["doc:visible.txt#chunk0"])))
    check("forbid exact id: visible doc cited is fine", p)
    p, r = rh.score(cfs, _res(_ans("insufficient_data", "x", ["doc:future.txt#chunk0"])))
    check("forbid exact id: future doc cited fails", (not p) and any("forbidden citation" in x for x in r))

    # runner error
    p, r = rh.score(case, _res(err="boom"))
    check("runner error fails", (not p) and "runner error" in r[0])

    print("\nALL HARNESS SCORING SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
