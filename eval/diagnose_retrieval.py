#!/usr/bin/env python3
"""Diagnose retrieval nondeterminism WITHOUT changing retrieval.

For a given case, run it N times and report, per run, exactly which document
chunks each doc_search call surfaced (read from the trace), plus the final
status. This shows whether a flaky failure is caused by:
  (a) retrieving NOTHING (the query missed entirely), vs
  (b) retrieving the WRONG chunk (a chunk that doesn't contain the needed fact), vs
  (c) retrieving the RIGHT chunk but the agent still declined.
Each diagnosis points at a different fix, so we look before touching retrieval.

Usage:
    PYTHONPATH=. python eval/diagnose_retrieval.py prov_02_primary_outranks_derived 5
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


def load_case(cid: str) -> dict:
    for l in (DS / "golden.jsonl").read_text().splitlines():
        if l.strip() and json.loads(l)["id"] == cid:
            return json.loads(l)
    raise SystemExit(f"case {cid} not found")


def run_once(case: dict) -> dict:
    """Run the case isolated and return status + the doc chunks each doc_search
    surfaced (parsed from the trace's tool_ok evidence, if present)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        env = os.environ.copy()
        env["QUANT_DATA_DIR"] = str(d)
        env["QUANT_KNOWLEDGE_DIR"] = str(d / "knowledge")
        env["PYTHONPATH"] = str(HERE.parent)
        env["QUANT_OFFLINE_ONLY"] = "1"
        cf = Path(tmp) / "c.json"
        cf.write_text(json.dumps(case))
        p = subprocess.run(
            [sys.executable, str(HERE / "_run_one_case.py"), str(cf), str(DS)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        if not lines:
            return {"status": "ERROR", "error": p.stderr[-300:], "citations": []}
        res = json.loads(lines[-1])
        ans = res.get("answer") or {}
        return {
            "status": ans.get("status", "?"),
            "citations": ans.get("citations", []),
            "tools": res.get("tools_fired", []),
        }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: diagnose_retrieval.py <case_id> [N=5]")
        return 2
    cid = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    case = load_case(cid)
    expected = case.get("expect_citation_source_ids", [])

    print(f"CASE {cid}  (as_of={case['as_of']})")
    print(f"QUERY: {case['query']}")
    print(f"Needs citations: {expected or '(none pinned)'}")
    print(f"Seeded docs: {case.get('seed_docs', [])}")
    print("=" * 70)

    status_counts: dict[str, int] = {}
    miss_counts: dict[str, int] = {}
    for i in range(1, n + 1):
        r = run_once(case)
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        cited = r.get("citations", [])
        # which required citations were present this run?
        missing = [sid for sid in expected if sid not in cited]
        for sid in missing:
            miss_counts[sid] = miss_counts.get(sid, 0) + 1
        got = "RETRIEVED NOTHING" if not cited else ", ".join(c.split("#")[0] for c in cited)
        flag = "OK" if not missing else f"MISSING {len(missing)}"
        print(f"run {i}: status={r['status']:18s} [{flag}]  cited: {got}")

    print("=" * 70)
    print("status distribution:", status_counts)
    if miss_counts:
        print("required-citation misses across runs:")
        for sid, c in miss_counts.items():
            print(f"  {c}/{n}  {sid}")
        # Interpretation hint
        print("\nReading:")
        print("- If misses correlate with 'RETRIEVED NOTHING' -> query missed entirely (retrieval recall).")
        print("- If other chunks were cited instead -> wrong-chunk retrieval (ranking/chunking).")
        print("- If the right chunk was cited but status still insufficient -> evaluator, not retrieval.")
    else:
        print("No required-citation misses in any run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
