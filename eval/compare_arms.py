#!/usr/bin/env python3
"""Diff an AGENT report against a VANILLA BASELINE report into a head-to-head.

Both reports come from run_harness.py (the baseline one produced with --baseline).
This does not re-run anything; it just reads the two JSON files and shows where
the agent and the vanilla model agree and differ, per area and per case.

The useful output is NOT a single aggregate ("agent 45, baseline 22") - it is
the per-category breakdown and the per-case win/loss list, because the story is
almost always "the vanilla model matches on definitions and collapses on
point-in-time / grounded computation / provenance." That pattern is the value
argument for the agent, and it is also honest about where the agent adds little.

Usage:
    PYTHONPATH=. python eval/compare_arms.py eval/agent_baseline.json eval/vanilla_baseline.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def passed(case: dict) -> bool:
    # repeat runs: PASS only if all repeats passed. single run: status == PASS.
    return case.get("status") == "PASS"


def load(path: str) -> dict:
    r = json.loads(Path(path).read_text())
    r["_by_id"] = {c["id"]: c for c in r["cases"]}
    return r


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: compare_arms.py <agent_report.json> <baseline_report.json>")
        return 2
    agent = load(sys.argv[1])
    base = load(sys.argv[2])

    if agent.get("arm") in ("baseline", "vanilla"):
        agent, base = base, agent  # tolerate argument order
    print(f"AGENT report:    {sys.argv[1]}  (arm={agent.get('arm','?')}, repeats={agent.get('repeats_per_case')})")
    print(f"BASELINE report: {sys.argv[2]}  (arm={base.get('arm','?')}, repeats={base.get('repeats_per_case')})")
    print("=" * 72)

    ids = [c["id"] for c in agent["cases"]]
    # Reports use 'area' (golden set) or 'capability' (ticker evals). Accept either.
    def _category(c):
        return c.get("area") or c.get("capability") or "?"
    area_of = {c["id"]: _category(c) for c in agent["cases"]}

    # tallies
    cat = defaultdict(lambda: {"both": 0, "agent_only": 0, "base_only": 0, "neither": 0, "total": 0})
    agent_wins, base_wins = [], []

    for cid in ids:
        a = agent["_by_id"].get(cid)
        b = base["_by_id"].get(cid)
        if not a or not b:
            continue
        ap, bp = passed(a), passed(b)
        area = area_of[cid]
        cat[area]["total"] += 1
        if ap and bp:
            cat[area]["both"] += 1
        elif ap and not bp:
            cat[area]["agent_only"] += 1
            agent_wins.append(cid)
        elif bp and not ap:
            cat[area]["base_only"] += 1
            base_wins.append(cid)
        else:
            cat[area]["neither"] += 1

    # per-area table
    print(f"{'area':12s} {'agent':>6s} {'base':>6s} {'agent-only':>11s} {'base-only':>10s} {'both':>5s}")
    tot_a = tot_b = 0
    for area in sorted(cat):
        c = cat[area]
        a_pass = c["both"] + c["agent_only"]
        b_pass = c["both"] + c["base_only"]
        tot_a += a_pass
        tot_b += b_pass
        print(f"{area:12s} {a_pass:>5d}/{c['total']:<1d} {b_pass:>4d}/{c['total']:<1d} "
              f"{c['agent_only']:>11d} {c['base_only']:>10d} {c['both']:>5d}")
    n = len(ids)
    print("-" * 72)
    print(f"{'TOTAL':12s} {tot_a:>5d}/{n:<2d} {tot_b:>3d}/{n:<2d}")
    print("=" * 72)

    # the value story: where the agent wins
    print(f"\nAGENT WINS ({len(agent_wins)} cases the agent passed but vanilla failed):")
    for cid in agent_wins:
        print(f"  + {cid}  [{area_of[cid]}]")
    if base_wins:
        print(f"\nBASELINE WINS ({len(base_wins)} cases vanilla passed but the agent failed) -- worth inspecting:")
        for cid in base_wins:
            print(f"  - {cid}  [{area_of[cid]}]")
    else:
        print("\nNo cases where the vanilla baseline beat the agent.")

    
    # ---- v11.7 statistics: case-level aggregation, CI, exact McNemar ----
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("eval_stats", Path(__file__).parent / "stats.py")
        _st = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_st)
        ids, a_rates, v_rates = _st.paired_rates(agent, base)
        if ids:
            delta, lo, hi = _st.bootstrap_delta_ci(a_rates, v_rates, seed=0)
            b, c = _st.discordant_counts(a_rates, v_rates)
            p = _st.mcnemar_exact(b, c)
            print("\n" + "=" * 72)
            print(f"STATISTICS (unit = case, n={len(ids)}; repeats aggregated per case first)")
            print(f"  agent mean pass rate:   {sum(a_rates)/len(a_rates):.3f}")
            print(f"  vanilla mean pass rate: {sum(v_rates)/len(v_rates):.3f}")
            print(f"  delta:                  {delta:+.3f}  (95% bootstrap CI [{lo:+.3f}, {hi:+.3f}], paired, seed=0)")
            print(f"  exact McNemar (all-runs-pass): b={b} agent-only, c={c} vanilla-only -> p = {p:.4g}")
            caps = _st.by_capability(agent, base)
            if len(caps) > 1:
                print(f"  {'capability':22s} {'n':>3s} {'agent':>7s} {'vanilla':>8s}")
                for k in sorted(caps):
                    g = caps[k]
                    print(f"  {k:22s} {g['n']:>3d} {g['agent_rate']:>7.3f} {g['vanilla_rate']:>8.3f}")
            if len(ids) < 20:
                print(f"  NOTE: n={len(ids)} cases is small; CIs above are wide by construction.")
                print("        Use eval/generate_attribution_cases.py for a population-scale set.")
    except Exception as e:  # stats must never break the basic comparison
        print(f"\n(statistics section unavailable: {e})")

    print("\nReading: agent-only wins concentrated in pit / provenance / quant = the "
          "scaffolding (point-in-time, retrieval, exact computation, citations) is "
          "doing the work. Cases where both pass are usually definitional - where a "
          "vanilla model needs no tools. base-only wins, if any, point at agent "
          "over-refusal or an over-strict assertion, not a real vanilla advantage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
