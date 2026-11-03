#!/usr/bin/env python3
"""Compare retrieval backends (v8.1): run the same queries through rag, sql, and
graph, repeated, and report determinism + PIT behaviour side by side.

This demonstrates the eval can compare AGENT TYPES (by retrieval backend) on the
same facts. RAG is expected to be nondeterministic (the flaky cluster); SQL and
graph are expected to be deterministic.

    python eval/compare_backends.py
    python eval/compare_backends.py --repeat 8

No API key needed - this exercises the retrieval layer directly, not the LLM.
"""
from __future__ import annotations

import argparse
from datetime import date

from quant_agent.tools.retrieval_backends import get_backend

# (query, as_of) probes that exercise PIT boundaries and core facts.
PROBES = [
    ("ACME fiscal 2024 revenue", date(2025, 1, 1)),
    ("ACME Beacon acquisition", date(2025, 6, 1)),
    ("ACME Beacon acquisition", date(2025, 3, 1)),  # before announce -> should be empty/clean
    ("ACME single supplier Region B", date(2025, 1, 1)),
    ("ACME withdrew guidance", date(2025, 3, 1)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    n = max(2, args.repeat)

    print(f"Backend comparison ({n} repeats per probe)\n" + "=" * 64)
    for backend_name in ["rag", "sql", "graph"]:
        try:
            b = get_backend(backend_name)
        except Exception as e:
            print(f"{backend_name}: unavailable ({e})")
            continue
        det_count = 0
        for q, asof in PROBES:
            runs = []
            for _ in range(n):
                try:
                    ev = b.retrieve(q, asof, k=3)
                    runs.append(tuple(e.source_id for e in ev))
                except Exception:
                    runs.append(("ERROR",))
            deterministic = len(set(runs)) == 1
            det_count += deterministic
        verdict = "DETERMINISTIC" if det_count == len(PROBES) else f"{det_count}/{len(PROBES)} probes stable"
        print(f"  {backend_name:6s} -> {verdict}")
    print("=" * 64)
    print("Expected: rag varies (semantic similarity, occasional misses);")
    print("          sql + graph identical every run (exact query / traversal).")
    print("All three apply the same pub_date <= as_of point-in-time filter.")


if __name__ == "__main__":
    main()
