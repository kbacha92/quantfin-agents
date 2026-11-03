#!/usr/bin/env python3
"""Run the LOOK-AHEAD test set against the agent and/or a vanilla model, and
score each answer for whether it LEAKED a post-as-of event.

This is the head-to-head that actually tests the agent's reason to exist: a
vanilla frontier model has real future events in its weights and tends to leak
them when asked about a past as-of date; the agent, with point-in-time
enforcement and no corpus knowledge of these events, should refuse or say "not
as of this date."

Scoring (leak detection, not the golden-set scorer):
  - LEAK (fail): the answer asserts the post-as-of event as having happened.
  - SAFE (pass): the answer says the event has not happened / is unknown as of
    the as-of date, or refuses.
  - For the CONTROL case (event precedes as-of), SAFE means correctly affirming
    the event - a system that refuses the control is being uselessly evasive.

Usage:
    PYTHONPATH=. python eval/run_lookahead.py --arm agent   --repeat 3 --report eval/la_agent.json
    PYTHONPATH=. python eval/run_lookahead.py --arm baseline --repeat 3 --report eval/la_vanilla.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LA = HERE / "lookahead" / "lookahead.jsonl"


def load_cases() -> list[dict]:
    return [json.loads(l) for l in LA.read_text().splitlines() if l.strip()]


def run_agent(case: dict) -> dict:
    """Run the case through the real agent in an isolated, offline-strict,
    EMPTY-corpus workspace. The agent has no document about these real events,
    so the correct behavior is to report insufficient / not-known."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        env = os.environ.copy()
        env["QUANT_DATA_DIR"] = str(d)
        env["QUANT_KNOWLEDGE_DIR"] = str(d / "knowledge")
        env["PYTHONPATH"] = str(ROOT)
        env["QUANT_OFFLINE_ONLY"] = "1"
        # minimal case shape the agent runner expects
        ac = {"id": case["id"], "query": case["query"], "as_of": case["as_of"],
              "seed_docs": [], "seed_knowledge": [], "snapshots": []}
        cf = Path(tmp) / "c.json"
        cf.write_text(json.dumps(ac))
        p = subprocess.run(
            [sys.executable, str(HERE / "_run_one_case.py"), str(cf), str(HERE / "golden_dataset")],
            env=env, capture_output=True, text=True, timeout=600,
        )
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        if not lines:
            return {"text": "", "error": p.stderr[-300:]}
        res = json.loads(lines[-1])
        ans = res.get("answer") or {}
        return {"text": ans.get("text", ""), "status": ans.get("status", "?")}


def run_vanilla(case: dict) -> dict:
    """Run the case through a bare model call - no tools, no corpus, no PIT
    enforcement. This is the look-ahead-prone arm."""
    from quant_agent import llm

    system = (
        "You are a research assistant answering as of a stated historical date. "
        "Answer ONLY based on what was publicly known as of that date. Do not use "
        "any knowledge of events after the as-of date. If something had not yet "
        "happened as of that date, say so plainly."
    )
    user = f"AS-OF DATE: {case['as_of']}\n\nQUESTION: {case['query']}"
    try:
        text = llm.complete(system, user, max_tokens=1000)
        return {"text": text}
    except Exception as e:
        return {"text": "", "error": f"{type(e).__name__}: {e}"}


def score_leak(case: dict, text: str) -> tuple[str, str]:
    """Return (verdict, reason). verdict in {SAFE, LEAK, UNCLEAR}.

    Leak detection is marker-based and deliberately transparent. Because markers
    can be fooled, the runner also stores the full text so a human can audit any
    verdict - especially UNCLEAR ones."""
    t = text.lower()
    is_control = not case["leak_markers"]

    if is_control:
        # Control: event precedes as-of. SAFE = correctly affirms it.
        if any(m in t for m in case["safe_markers"]):
            return "SAFE", "control: correctly affirmed a past (pre-as-of) event"
        return "UNCLEAR", "control: did not clearly affirm the known past event"

    leaked = [m for m in case["leak_markers"] if m in t]
    safe = [m for m in case["safe_markers"] if m in t]

    # A strong safe signal ("not yet", "no public", "as of <date>") + no specific
    # leaked detail = SAFE. A specific leaked detail (date, dollar amount, "yes
    # ... acquired") = LEAK. The specific-detail markers are the reliable ones.
    specific_leak = [m for m in leaked if m not in ("yes",)]
    if specific_leak:
        return "LEAK", f"asserted post-as-of detail(s): {specific_leak}"
    if safe:
        return "SAFE", f"correct point-in-time answer: {safe}"
    if "yes" in leaked:
        return "LEAK", "affirmed the event ('yes') without a point-in-time qualifier"
    return "UNCLEAR", "no clear leak or safe marker; inspect text"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["agent", "baseline"], required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--report")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    cases = load_cases()
    n = max(1, args.repeat)
    print(f"LOOK-AHEAD test: {len(cases)} cases x{n} -- arm: {args.arm}\n")

    results = []
    leak_total = 0
    for case in cases:
        verdicts = Counter()
        sample_text = ""
        for _ in range(n):
            r = run_agent(case) if args.arm == "agent" else run_vanilla(case)
            text = r.get("text", "")
            if not sample_text:
                sample_text = text
            v, reason = score_leak(case, text)
            verdicts[v] += 1
        leaks = verdicts.get("LEAK", 0)
        if leaks:
            leak_total += 1
        # dominant verdict
        dom = verdicts.most_common(1)[0][0]
        kind = "CONTROL" if not case["leak_markers"] else "LEAK-TEST"
        line = f"[{kind}] {case['id']:24s} {dict(verdicts)}"
        print(line)
        if args.verbose or "LEAK" in verdicts:
            print(f"    as_of={case['as_of']} event={case['event_date']}")
            print(f"    sample: {sample_text[:200].replace(chr(10),' ')}")
        results.append({
            "id": case["id"], "kind": kind, "as_of": case["as_of"],
            "event_date": case["event_date"], "verdicts": dict(verdicts),
            "dominant": dom, "sample_text": sample_text,
        })

    leak_test_cases = [c for c in cases if c["leak_markers"]]
    print("\n" + "=" * 64)
    print(f"arm={args.arm}: {leak_total} of {len(cases)} cases leaked at least once "
          f"({len([r for r in results if r['kind']=='LEAK-TEST' and 'LEAK' in r['verdicts']])} "
          f"of {len(leak_test_cases)} leak-test cases).")

    if args.report:
        rep = {"run_at": datetime.now().isoformat(timespec="seconds"),
               "arm": args.arm, "repeats": n, "cases": results}
        Path(args.report).write_text(json.dumps(rep, indent=2))
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
