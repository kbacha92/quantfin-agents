#!/usr/bin/env python3
"""Dump the agent's full answer (status, tools, citations, text) for one or more
golden case ids, for triaging a harness failure. No scoring - just shows what
the agent actually produced so you can decide if a failure is a real agent gap
or an over-strict assertion.

Usage:
    PYTHONPATH=. python eval/inspect_cases.py pit_05_acquisition_future quant_07_correlation_one_ticker
    PYTHONPATH=. python eval/inspect_cases.py --all-failed eval/last_run.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE / "golden_dataset"


def load_cases() -> dict:
    return {json.loads(l)["id"]: json.loads(l)
            for l in (DS / "golden.jsonl").read_text().splitlines() if l.strip()}


def run_one(case: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        env = os.environ.copy()
        env["QUANT_DATA_DIR"] = str(d)
        env["QUANT_KNOWLEDGE_DIR"] = str(d / "knowledge")
        env["PYTHONPATH"] = str(HERE.parent)
        cf = Path(tmp) / "c.json"
        cf.write_text(json.dumps(case))
        p = subprocess.run(
            [sys.executable, str(HERE / "_run_one_case.py"), str(cf), str(DS)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        if not lines:
            return {"error": p.stderr[-400:]}
        return json.loads(lines[-1])


def main() -> int:
    args = sys.argv[1:]
    cases = load_cases()
    if args and args[0] == "--all-failed":
        report = json.loads(Path(args[1]).read_text())
        ids = [c["id"] for c in report["cases"] if not c["passed"]]
    else:
        ids = args
    if not ids:
        print("Give case ids, or --all-failed <report.json>")
        return 2

    for cid in ids:
        if cid not in cases:
            print(f"\n=== {cid}: NOT FOUND ===")
            continue
        case = cases[cid]
        print("\n" + "=" * 70)
        print(f"CASE: {cid}   [{case['area']}]   as_of={case['as_of']}")
        print(f"QUERY: {case['query']}")
        print(f"EXPECTED status={case['expected_status']}")
        res = run_one(case)
        if res.get("error") or not res.get("answer"):
            print(f"RUNNER ERROR: {res.get('error')}")
            continue
        ans = res["answer"]
        print(f"GOT status={ans['status']}  tools={res.get('tools_fired')}")
        print(f"CITATIONS: {ans.get('citations')}")
        print("--- ANSWER TEXT ---")
        print(ans.get("text", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
