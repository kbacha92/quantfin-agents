#!/usr/bin/env python3
"""Run ONE golden case in a fully isolated workspace and print its result JSON.

Invoked as a subprocess by run_harness.py so that QUANT_DATA_DIR /
QUANT_KNOWLEDGE_DIR are set BEFORE quant_agent.config binds its paths at import
time. Each case therefore gets its own docs/, knowledge/, chroma/, api_cache/,
and traces/ with zero cross-case leakage.

Usage (normally not called directly):
    QUANT_DATA_DIR=... QUANT_KNOWLEDGE_DIR=... \
    python eval/_run_one_case.py <case_json_path> <dataset_dir>

Prints a single JSON object to stdout:
    {"answer": {...}, "tools_fired": [...], "trace_id": "...", "error": null}
"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from datetime import date
from pathlib import Path


def main() -> int:
    case_path = Path(sys.argv[1])
    dataset_dir = Path(sys.argv[2])
    case = json.loads(case_path.read_text())

    # Import AFTER env vars are set (the parent process set them). config binds
    # DATA_DIR / KNOWLEDGE_DIR from the env at import time and mkdir's them.
    from quant_agent import config
    from quant_agent.orchestrator import run_query
    from quant_agent.tools.doc_kb import ingest, ingest_knowledge

    snap_src = dataset_dir / "snapshots"
    docs_src = dataset_dir / "seed_docs"
    kn_src = dataset_dir / "seed_knowledge"

    # 1) Seed the FACTS corpus
    for fname in case.get("seed_docs", []):
        shutil.copy(docs_src / fname, config.DOCS_DIR / fname)
    # 2) Seed the METHODOLOGY corpus
    for fname in case.get("seed_knowledge", []):
        shutil.copy(kn_src / fname, config.KNOWLEDGE_DIR / fname)
    # 3) Index both corpora (idempotent)
    if case.get("seed_docs"):
        ingest()
    if case.get("seed_knowledge"):
        ingest_knowledge()
    # 4) Stage price snapshots into api_cache so load_prices replays offline.
    #    Look first in the dataset's snapshots/ dir (golden set), then fall back
    #    to the repo's real api_cache (where make_snapshots.py writes ticker
    #    snapshots for manifest-built evals). Skip silently if already staged
    #    (run_eval.py may have copied it) or not found - never crash on a missing
    #    snapshot here.
    repo_cache = Path(__file__).resolve().parent.parent / "data" / "api_cache"
    for fname in case.get("snapshots", []):
        dst = config.API_CACHE_DIR / fname
        if dst.exists():
            continue
        for src in (snap_src / fname, repo_cache / fname):
            if src.exists():
                shutil.copy(src, dst)
                break

    # 5) Run the agent
    from quant_agent import llm
    llm.reset_usage()  # start per-case token accounting from zero
    try:
        ans = run_query(case["query"], date.fromisoformat(case["as_of"]))
    except Exception as e:  # never let a single case kill the run
        print(json.dumps({
            "answer": None,
            "tools_fired": [],
            "trace_id": None,
            "usage": llm.get_usage(),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        return 0

    # 6) Recover which tools actually fired from the trace JSONL
    tools_fired: list[str] = []
    trace_path = config.TRACES_DIR / f"{ans.trace_id}.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("step") == "tool_ok":
                tools_fired.append(rec["tool"])

    # Persist the trace into the repo's data/traces so it survives the temp
    # workspace (the harness can copy it out for --save-trace). Best-effort.
    trace_copy = None
    try:
        if trace_path.exists():
            repo_traces = Path(__file__).resolve().parent.parent / "data" / "traces"
            repo_traces.mkdir(parents=True, exist_ok=True)
            dst = repo_traces / f"{ans.trace_id}.jsonl"
            shutil.copy(trace_path, dst)
            trace_copy = str(dst)
    except Exception:
        pass

    print(json.dumps({
        "answer": ans.model_dump(mode="json"),
        "tools_fired": tools_fired,
        "trace_id": ans.trace_id,
        "trace_path": trace_copy,
        "usage": llm.get_usage(),
        "error": None,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
