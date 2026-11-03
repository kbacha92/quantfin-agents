#!/usr/bin/env python3
"""Analyze retrieval probes to settle: cold-start race vs inherent ranking variance.

Reads retrieval_probe records from trace JSONL files (written when QA traces are
saved, e.g. via run_harness --save-trace DIR) and produces the two decisive
distributions:

  1. Empties by QUERY POSITION in the process (query_index).
     front-loaded (positions 1-2) -> cold-start race
     scattered across positions    -> inherent variance

  2. COLLECTION COUNT at the moment of emptiness.
     zero / low  -> index not ready when queried (race)
     full        -> populated index, query just returned nothing (ranking variance)

Usage:
    python eval/analyze_retrieval_probes.py path/to/traces_dir
    python eval/analyze_retrieval_probes.py path/to/traces_dir/*.jsonl
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_probes(paths):
    probes = []
    for p in paths:
        try:
            for line in Path(p).read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("step") == "retrieval_probe":
                    probes.append(rec)
        except Exception:
            continue
    return probes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    args = sys.argv[1:]
    paths = []
    for a in args:
        if Path(a).is_dir():
            paths += glob.glob(str(Path(a) / "*.jsonl"))
        else:
            paths += glob.glob(a)
    probes = load_probes(paths)
    if not probes:
        print("No retrieval_probe records found. Run with --save-trace and ensure v9.2+.")
        return 1

    total = len(probes)
    by_result = Counter(p.get("result") for p in probes)
    empties = [p for p in probes if p.get("result") in ("empty_correct", "empty_despite_populated")]
    hits = [p for p in probes if p.get("result") == "hit"]
    retried_hits = [p for p in hits if p.get("retried")]

    print(f"Total retrievals probed: {total}")
    print(f"  hits:                    {by_result.get('hit', 0)}")
    print(f"    (of which succeeded only after a retry: {len(retried_hits)})")
    print(f"  empty (correct/empty col): {by_result.get('empty_correct', 0)}")
    print(f"  empty DESPITE populated:   {by_result.get('empty_despite_populated', 0)}")
    print("=" * 60)

    # Signal 1: position-in-run for the diagnostic empties + retry-resolved hits
    print("\n[1] Empties + retry-resolved hits by QUERY POSITION in process:")
    pos = Counter()
    for p in empties:
        pos[p.get("query_index", "?")] += 1
    for p in retried_hits:
        pos[p.get("query_index", "?")] += 1
    if pos:
        for idx in sorted(pos, key=lambda x: (x == "?", x)):
            print(f"    position {idx:>3}: {'#' * pos[idx]} ({pos[idx]})")
        front = sum(v for k, v in pos.items() if isinstance(k, int) and k <= 2)
        back = sum(v for k, v in pos.items() if isinstance(k, int) and k > 2)
        print(f"  front-loaded (pos<=2): {front}   later (pos>2): {back}")
        print("  -> front-loaded suggests COLD-START RACE; scattered suggests INHERENT VARIANCE")
    else:
        print("    (no empties or retry-resolved hits - retrieval was clean)")

    # Signal 2: collection count at the moment of emptiness
    print("\n[2] COLLECTION COUNT at moment of emptiness:")
    zero_low = sum(1 for p in empties if isinstance(p.get("collection_count"), int) and p["collection_count"] <= 0)
    populated = sum(1 for p in empties if isinstance(p.get("collection_count"), int) and p["collection_count"] > 0)
    print(f"    empty while collection_count <= 0 (not ready):   {zero_low}")
    print(f"    empty while collection_count  > 0 (populated):   {populated}")
    if populated > zero_low:
        print("  -> mostly POPULATED at emptiness => inherent ranking variance (router would help)")
    elif zero_low > populated:
        print("  -> mostly NOT-READY at emptiness => cold-start race (warm/retry is the cheaper fix)")
    else:
        print("  -> mixed / inconclusive; collect more runs")

    # Signal 3: did retry resolve it? (race resolves between attempts; variance doesn't)
    print("\n[3] Retry effect:")
    print(f"    hits that needed a retry: {len(retried_hits)}  "
          f"(if many, an immediate retry resolves emptiness => race-like)")
    print(f"    empty_despite_populated (retry did NOT resolve): {by_result.get('empty_despite_populated', 0)}  "
          f"(if many, retry doesn't help => inherent-like)")


if __name__ == "__main__":
    sys.exit(main())
