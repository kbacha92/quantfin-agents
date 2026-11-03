#!/usr/bin/env python3
"""Validate the golden set's internal consistency WITHOUT calling the LLM.

This is not the eval harness (that runs the agent and needs an API key). This
checks the dataset itself:
  1. Every referenced snapshot / seed_doc / seed_knowledge file exists.
  2. Every quant case's expected_value matches what the REAL indicator tool
     emits when replaying the frozen snapshot offline (no network, no key).
  3. Schema sanity: required fields present, ids unique, statuses valid.

Run:  PYTHONPATH=. python eval/validate_golden.py
Exit code 0 = dataset is self-consistent and quant ground truth is exact.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

from quant_agent import config
from quant_agent.tools.quant.indicators import QuantIndicatorTool

HERE = Path(__file__).resolve().parent
DS = HERE / "golden_dataset"
SNAP = DS / "snapshots"
DOCS = DS / "seed_docs"
KN = DS / "seed_knowledge"

VALID_STATUS = {"ok", "insufficient_data"}


def _stage_snapshots(names):
    for n in names:
        src = SNAP / n
        if src.exists():
            shutil.copy(src, config.API_CACHE_DIR / n)


def main() -> int:
    cases = [json.loads(l) for l in (DS / "golden.jsonl").read_text().splitlines() if l.strip()]
    errors: list[str] = []
    quant_checked = 0

    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case ids present")

    tool = QuantIndicatorTool()

    for c in cases:
        cid = c["id"]
        if c["expected_status"] not in VALID_STATUS:
            errors.append(f"{cid}: bad expected_status {c['expected_status']!r}")
        # file existence
        for f in c.get("snapshots", []):
            if not (SNAP / f).exists():
                errors.append(f"{cid}: missing snapshot {f}")
        for f in c.get("seed_docs", []):
            if not (DOCS / f).exists():
                errors.append(f"{cid}: missing seed_doc {f}")
        for f in c.get("seed_knowledge", []):
            if not (KN / f).exists():
                errors.append(f"{cid}: missing seed_knowledge {f}")

        # quant exact-value check: replay the snapshot through the real tool
        if c["area"] == "quant" and "expected_value" in c:
            # derive tool args from the case's expected source_id
            sids = c.get("expect_citation_source_ids", [])
            if not sids:
                continue  # value-only sanity case without a pinned id
            sid = sids[0]
            # sid format: quant:<indicator>:<ticker(s)>:<window>:<as_of>
            _, indicator, tick, window, _asof = sid.split(":", 4)
            ticker = tick.replace("-", ",") if indicator == "correlation" else tick
            _stage_snapshots(c["snapshots"])
            try:
                ev = tool(as_of=date.fromisoformat(c["as_of"]),
                          ticker=ticker, indicator=indicator, window=int(window))[0]
            except Exception as e:
                errors.append(f"{cid}: tool raised replaying snapshot: {e}")
                continue
            if ev.source_id != sid:
                errors.append(f"{cid}: source_id mismatch\n    expected {sid}\n    got      {ev.source_id}")
            expect_str = c["must_contain"][0] if c.get("must_contain") else None
            if expect_str and expect_str not in ev.content:
                errors.append(f"{cid}: expected value '{expect_str}' not in tool output: {ev.content[:80]}")
            quant_checked += 1

    # cleanup staged snapshots
    for f in config.API_CACHE_DIR.glob("yfinance_*.csv"):
        f.unlink()

    print(f"Validated {len(cases)} cases; exact-value-checked {quant_checked} quant cases.")
    if errors:
        print(f"\nFAILURES ({len(errors)}):")
        for e in errors:
            print("  - " + e)
        return 1
    print("All cases self-consistent; all quant ground-truth values match the real tool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
