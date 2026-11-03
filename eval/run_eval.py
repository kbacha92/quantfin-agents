#!/usr/bin/env python3
"""Run a stock eval (any ticker) through the AGENT (guarded) or VANILLA-WITH-TOOLS
(unguarded) arm, scoring with the SAME scorer as the golden set.

Ticker-agnostic: point --dataset at the JSONL produced by build_eval.py.

    PYTHONPATH=. python run_eval.py --dataset eval/msft/msft.jsonl --arm agent   --repeat 3 --report eval/msft_agent.json
    PYTHONPATH=. python run_eval.py --dataset eval/msft/msft.jsonl --arm vanilla --repeat 3 --report eval/msft_vanilla.json
    PYTHONPATH=. python eval/compare_arms.py eval/msft_agent.json eval/msft_vanilla.json

Prereq: cases must be VERIFIED and converted from templates to real assertion
fields. Cases still marked TEMPLATE_NEEDS_VERIFICATION are skipped with a warning.

Drop this file (and _run_one_case_vanilla_tools.py) into the agent's eval/ folder.
It reuses the existing _run_one_case.py (agent arm), run_harness.score (scorer),
and golden_dataset/ for snapshot/doc staging.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DS = HERE / "golden_dataset"

_spec = importlib.util.spec_from_file_location("run_harness", HERE / "run_harness.py")
_rh = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = ["run_harness"]
_spec.loader.exec_module(_rh)
sys.argv = _argv
score = _rh.score


def prebuild_shared_index(shared_dir: Path):
    """Build the docs+knowledge Chroma index ONCE into a persistent shared dir,
    so every case reuses it instead of re-embedding the corpus per case. The
    point-in-time guarantee is unaffected: the as-of filter is applied at QUERY
    time (pub_date <= as_of), so a shared index over all filings still gives each
    case the correct as-of-filtered view.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["QUANT_DATA_DIR"] = str(shared_dir)
    env["QUANT_KNOWLEDGE_DIR"] = str(shared_dir / "knowledge")
    env["PYTHONPATH"] = str(ROOT)
    # stage the real docs into the shared workspace, then ingest once
    real_docs = ROOT / "data" / "docs"
    shared_docs = shared_dir / "docs"
    shared_docs.mkdir(parents=True, exist_ok=True)
    if real_docs.exists():
        for f in real_docs.glob("*"):
            if f.is_file():
                shutil.copy(f, shared_docs / f.name)
    code = (
        "import os; from quant_agent.tools.doc_kb import ingest, ingest_knowledge; "
        "n=ingest(); k=ingest_knowledge(); print(f'shared index built: {n} doc chunks, {k} knowledge chunks')"
    )
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=3600)
    print(p.stdout.strip() or p.stderr[-300:])
    return shared_dir


def run_isolated(case, arm, shared_dir=None):
    runner = "_run_one_case.py" if arm == "agent" else "_run_one_case_vanilla_tools.py"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        env = os.environ.copy()
        if shared_dir is not None:
            # Reuse the prebuilt shared index: point DATA_DIR (-> CHROMA_DIR,
            # DOCS_DIR) at the shared workspace. No per-case re-ingest.
            env["QUANT_DATA_DIR"] = str(shared_dir)
            env["QUANT_KNOWLEDGE_DIR"] = str(shared_dir / "knowledge")
            env["QUANT_EVAL_SKIP_INGEST"] = "1"
            cache_root = Path(shared_dir) / "api_cache"
        else:
            env["QUANT_DATA_DIR"] = str(d)
            env["QUANT_KNOWLEDGE_DIR"] = str(d / "knowledge")
            cache_root = d / "api_cache"
        env["PYTHONPATH"] = str(ROOT)
        if arm == "agent":
            env["QUANT_OFFLINE_ONLY"] = "1"

        # Stage price snapshots the case needs into the active api_cache.
        cache_root.mkdir(parents=True, exist_ok=True)
        real_cache = ROOT / "data" / "api_cache"
        for snap in case.get("snapshots", []):
            src = real_cache / snap
            if src.exists():
                shutil.copy(src, cache_root / snap)

        # Stage portfolio files the case needs (v11.6 attribution eval) into the
        # active data root's portfolios/ dir - same pattern as snapshots. The
        # attribution tool's own PIT gate then decides visibility per as-of, so
        # staging a post-as-of file is safe (and deliberate for lookahead cases).
        portfolios = case.get("portfolios", [])
        if portfolios:
            port_root = (Path(shared_dir) if shared_dir is not None else d) / "portfolios"
            port_root.mkdir(parents=True, exist_ok=True)
            real_port = ROOT / "data" / "portfolios"
            for pf in portfolios:
                src = real_port / pf
                if src.exists():
                    shutil.copy(src, port_root / pf)

        # Stage filings per-case ONLY when not using the shared index (the shared
        # index already contains them).
        if shared_dir is None:
            real_docs = ROOT / "data" / "docs"
            tmp_docs = d / "docs"
            tmp_docs.mkdir(parents=True, exist_ok=True)
            if real_docs.exists():
                for f in real_docs.glob("*"):
                    if f.is_file():
                        shutil.copy(f, tmp_docs / f.name)

        cf = Path(tmp) / "c.json"
        cf.write_text(json.dumps(case))
        timeout_s = int(os.environ.get("QUANT_EVAL_TIMEOUT", "1800"))
        p = subprocess.run([sys.executable, str(HERE / runner), str(cf), str(DS)],
                           env=env, capture_output=True, text=True, timeout=timeout_s)
        lines = [l for l in p.stdout.splitlines() if l.strip()]
        if not lines:
            return {"answer": None, "tools_fired": [], "error": p.stderr[-400:]}
        return json.loads(lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to the ticker JSONL from build_eval.py")
    ap.add_argument("--arm", choices=["agent", "vanilla"], required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--report")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-shared-index", action="store_true",
                    help="disable the shared prebuilt index (re-ingest per case; slower, full isolation)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"ERROR: dataset not found: {ds_path}", file=sys.stderr)
        return 2
    cases = [json.loads(l) for l in ds_path.read_text().splitlines() if l.strip()]
    templates = [c for c in cases if c.get("_status") == "TEMPLATE_NEEDS_VERIFICATION"]
    runnable = [c for c in cases if c.get("_status") != "TEMPLATE_NEEDS_VERIFICATION"]

    if templates:
        print(f"WARNING: {len(templates)} of {len(cases)} cases are still TEMPLATES and will be SKIPPED:")
        for c in templates:
            print(f"  - {c['id']}")
        print()
    if not runnable:
        print("No verified cases to run. Complete verification first (see eval_README.md).")
        return 1

    n = max(1, args.repeat)
    ticker = runnable[0].get("ticker", "?")
    print(f"{ticker} eval: {len(runnable)} verified case(s) x{n} -- arm: {args.arm}\n")

    # Build the corpus index ONCE and reuse it across all cases (v7.1). This
    # turns a per-case re-embed of full 10-Ks (minutes each) into a single
    # up-front ingest. Disabled with --no-shared-index. The vanilla arm uses its
    # own unguarded retrieval, but a shared index still saves its ingest too.
    shared_dir = None
    if not args.no_shared_index:
        import tempfile as _tf
        shared_dir = Path(_tf.mkdtemp(prefix="quant_shared_index_"))
        print("Prebuilding shared corpus index (once)...")
        prebuild_shared_index(shared_dir)
        print()

    results = []
    cap_pass = Counter(); cap_total = Counter()
    for case in runnable:
        cap = case.get("capability", "?")
        cap_total[cap] += 1
        passes = 0; reasons = set()
        for _ in range(n):
            try:
                res = run_isolated(case, args.arm, shared_dir=shared_dir)
            except subprocess.TimeoutExpired:
                res = {"answer": None, "tools_fired": [],
                       "error": f"case timed out (>{os.environ.get('QUANT_EVAL_TIMEOUT','1800')}s)"}
            except Exception as e:
                res = {"answer": None, "tools_fired": [], "error": f"runner exception: {e}"}
            ok, why = score(case, res)
            if ok: passes += 1
            else:
                # v11.7.1: when the RUN errored (no answer produced), the
                # scorer's assertion failures are symptoms; the runner's
                # captured error text is the diagnosis. Surface it - a report
                # that says only "status: got 'error'" forces trace archaeology
                # (lesson from the gen42 tail failures, 2026-07-03).
                if res.get("error"):
                    why = list(why) + [f"runner_error: {str(res['error']).strip()[:300]}"]
                reasons.update(why)
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
               "ticker": ticker, "arm": args.arm, "repeats_per_case": n, "cases": results}
        Path(args.report).write_text(json.dumps(rep, indent=2))
        print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
