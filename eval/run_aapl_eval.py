#!/usr/bin/env python3
"""Run the AAPL eval through the AGENT (guarded) or VANILLA-WITH-TOOLS (unguarded)
arm, scoring with the SAME scorer as the golden set.

Prereq: aapl.jsonl cases must be VERIFIED and converted from templates to real
assertion fields (expected_status, must_contain, expect_citation_source_ids,
expected_value, etc.). Template cases (still containing '_status':
'TEMPLATE_NEEDS_VERIFICATION') are skipped with a warning, because they have no
scorable assertions.

Usage:
    PYTHONPATH=. python eval/run_aapl_eval.py --arm agent   --repeat 3 --report eval/aapl_agent.json
    PYTHONPATH=. python eval/run_aapl_eval.py --arm vanilla --repeat 3 --report eval/aapl_vanilla.json

Then diff with eval/compare_arms.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AAPL = HERE / "aapl" / "aapl.jsonl"
DS = HERE / "golden_dataset"  # snapshots/docs staging reuse

# reuse the golden-set scorer
_spec = importlib.util.spec_from_file_location("run_harness", HERE / "run_harness.py")
_rh = importlib.util.module_from_spec(_spec)
sys.argv_backup = sys.argv
sys.argv = ["run_harness"]
_spec.loader.exec_module(_rh)
sys.argv = sys.argv_backup
score = _rh.score


def load_cases():
    return [json.loads(l) for l in AAPL.read_text().splitlines() if l.strip()]


def run_isolated(case, arm):
    runner = "_run_one_case.py" if arm == "agent" else "_run_one_case_vanilla_tools.py"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        env = os.environ.copy()
        env["QUANT_DATA_DIR"] = str(d)
        env["QUANT_KNOWLEDGE_DIR"] = str(d / "knowledge")
        env["PYTHONPATH"] = str(ROOT)
        if arm == "agent":
            env["QUANT_OFFLINE_ONLY"] = "1"
        # stage AAPL docs + snapshots the case needs (same keys as golden cases)
        cf = Path(tmp) / "c.json"
        cf.write_text(json.dumps(case))
        p = subprocess.run([sys.executable, str(HERE / runner), str(cf), str(DS)],
                           env=env, capture_output=True, text=True, timeout=600)
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        if not lines:
            return {"answer": None, "tools_fired": [], "error": p.stderr[-400:]}
        return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["agent", "vanilla"], required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--report")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    cases = load_cases()
    templates = [c for c in cases if c.get("_status") == "TEMPLATE_NEEDS_VERIFICATION"]
    runnable = [c for c in cases if c.get("_status") != "TEMPLATE_NEEDS_VERIFICATION"]

    if templates:
        print(f"WARNING: {len(templates)} of {len(cases)} cases are still TEMPLATES "
              f"(unverified) and will be SKIPPED:")
        for c in templates:
            print(f"  - {c['id']}  (fill in [VERIFY] fields and remove _status to enable)")
        print()
    if not runnable:
        print("No verified cases to run. Complete verification first (see eval/aapl/README.md).")
        return 1

    n = max(1, args.repeat)
    print(f"AAPL eval: {len(runnable)} verified case(s) x{n} -- arm: {args.arm}\n")
    results = []
    cap_pass = Counter(); cap_total = Counter()
    for case in runnable:
        cap = case.get("capability", "?")
        cap_total[cap] += 1
        passes = 0; reasons = set()
        for _ in range(n):
            res = run_isolated(case, args.arm)
            ok, why = score(case, res)
            if ok: passes += 1
            else: reasons.update(why)
        status = "PASS" if passes == n else ("FAIL" if passes == 0 else "FLAKY")
        if status == "PASS": cap_pass[cap] += 1
        results.append({"id": case["id"], "capability": cap, "status": status,
                        "passes": passes, "runs": n, "reasons": sorted(reasons)})
        rate = "" if n == 1 else f" ({passes}/{n})"
        print(f"{status:5s} {case['id']}{rate}")
        if (args.verbose or status != "PASS") and reasons:
            for r in sorted(reasons):
                print(f"      - {r}")

    print("\n" + "=" * 56)
    print(f"arm={args.arm}: {sum(1 for r in results if r['status']=='PASS')}/{len(results)} passed all {n}")
    print("by capability:")
    for cap in sorted(cap_total):
        print(f"  {cap:12s} {cap_pass[cap]}/{cap_total[cap]}")

    if args.report:
        rep = {"run_at": datetime.now().isoformat(timespec="seconds"),
               "arm": args.arm, "repeats_per_case": n, "cases": results}
        Path(args.report).write_text(json.dumps(rep, indent=2))
        print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
