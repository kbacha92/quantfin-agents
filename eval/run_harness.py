#!/usr/bin/env python3
"""Golden-set scoring harness for the quant research agent.

Runs each case in eval/golden_dataset/golden.jsonl against the live agent and
scores the returned Answer against the case's assertions. Each case runs in its
own isolated workspace (separate corpus, chroma index, price cache, traces) via
a subprocess, so there is no cross-case state leakage.

THIS CALLS THE LLM. It needs a working ANTHROPIC_API_KEY in the environment
(launch the project once via run.py, or export the key). Token cost scales with
the number of cases x iterations.

Usage:
    PYTHONPATH=. python eval/run_harness.py                 # all 50 cases
    PYTHONPATH=. python eval/run_harness.py --area pit      # one area
    PYTHONPATH=. python eval/run_harness.py --id pit_02_future_8k_invisible
    PYTHONPATH=. python eval/run_harness.py --limit 5 --verbose
    PYTHONPATH=. python eval/run_harness.py --report eval/last_run.json

Exit code 0 iff every selected case passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DS = HERE / "golden_dataset"

# source_id prefix -> Evidence.store. Citations in Answer are source_id strings;
# the store assertions reference store names, so we map prefix to store here.
# (Confirmed against the tool source: doc->doc_kb, yfinance->prices, etc.)
PREFIX_TO_STORE = {
    "doc": "doc_kb",
    "knowledge": "knowledge",
    "yfinance": "prices",
    "quant": "quant",
    "edgar": "edgar",
    "chart": "chart",
    "model": "model",
}


def store_of(source_id: str) -> str:
    prefix = source_id.split(":", 1)[0]
    return PREFIX_TO_STORE.get(prefix, prefix)


def load_cases() -> list[dict]:
    return [json.loads(l) for l in (DS / "golden.jsonl").read_text().splitlines() if l.strip()]


def run_case_isolated(case: dict, baseline: bool = False) -> dict:
    """Spawn the single-case runner in a fresh temp workspace. Returns the
    runner's result dict (answer / tools_fired / error).

    If baseline=True, runs the VANILLA model arm (single LLM call, no tools, no
    corpus, no as-of enforcement) for head-to-head comparison."""
    runner = "_run_one_case_baseline.py" if baseline else "_run_one_case.py"
    with tempfile.TemporaryDirectory(prefix="goldeneval_") as tmp:
        data_dir = Path(tmp) / "data"
        kn_dir = data_dir / "knowledge"
        env = os.environ.copy()
        env["QUANT_DATA_DIR"] = str(data_dir)
        env["QUANT_KNOWLEDGE_DIR"] = str(kn_dir)
        env["PYTHONPATH"] = str(ROOT)
        env["QUANT_OFFLINE_ONLY"] = "1"
        case_file = Path(tmp) / "case.json"
        case_file.write_text(json.dumps(case))
        proc = subprocess.run(
            [sys.executable, str(HERE / runner), str(case_file), str(DS)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"answer": None, "tools_fired": [], "trace_id": None,
                    "error": f"runner failed (rc={proc.returncode}): {proc.stderr[-500:]}"}
        last = [l for l in proc.stdout.splitlines() if l.strip()][-1]
        return json.loads(last)


def score(case: dict, result: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failure_reasons). Empty list == pass."""
    fails: list[str] = []

    if result.get("error"):
        return False, [f"runner error: {result['error']}"]
    ans = result["answer"]
    if ans is None:
        return False, ["no answer returned"]

    text = (ans.get("text") or "")
    text_l = text.lower()
    citations = ans.get("citations") or []
    artifacts = ans.get("artifacts") or []
    tools_fired = result.get("tools_fired") or []
    cited_stores = {store_of(c) for c in citations}

    # 1) status
    if ans.get("status") != case["expected_status"]:
        fails.append(f"status: expected {case['expected_status']!r}, got {ans.get('status')!r}")

    # 2) tools
    for t in case.get("expect_tools_all", []):
        if t not in tools_fired:
            fails.append(f"tool '{t}' expected to fire (all) but did not; fired={tools_fired}")
    any_tools = case.get("expect_tools_any", [])
    if any_tools and not any(t in tools_fired for t in any_tools):
        fails.append(f"none of expect_tools_any {any_tools} fired; fired={tools_fired}")

    # 3) required citations (exact source_ids)
    #    Document-level tolerance: a generated assertion often pins '...#chunk0'
    #    as a stand-in for "this document". A real multi-chunk filing answers
    #    from whichever chunk holds the fact (e.g. #chunk112), so for a required
    #    citation ending in '#chunk0' we accept ANY chunk of the same document.
    #    Exact non-chunk0 source_ids (e.g. quant:...) still match exactly.
    def _citation_satisfied(sid: str) -> bool:
        if sid in citations:
            return True
        if sid.endswith("#chunk0"):
            doc_prefix = sid.rsplit("#", 1)[0]  # e.g. doc:2024-11-01_AAPL_10-K.txt
            return any(c.split("#", 1)[0] == doc_prefix for c in citations)
        return False

    for sid in case.get("expect_citation_source_ids", []):
        if not _citation_satisfied(sid):
            fails.append(f"required citation '{sid}' missing; got={citations}")

    # 3b) forbidden exact citations (e.g. a specific future document that must
    #     never be cited at this as-of date). Precise counterpart to the
    #     store-level forbid below.
    for sid in case.get("forbid_citation_source_ids", []):
        if sid in citations:
            fails.append(f"forbidden citation '{sid}' present (point-in-time leak)")

    # 4) citation store constraints
    for store in case.get("forbid_citation_store", []):
        if store in cited_stores:
            offenders = [c for c in citations if store_of(c) == store]
            fails.append(f"forbidden citation store '{store}' present: {offenders}")
    any_store = case.get("expect_citation_store_any", [])
    if any_store and not any(s in cited_stores for s in any_store):
        fails.append(f"expected a citation from one of {any_store}; got stores={sorted(cited_stores)}")

    # 5) text substring asserts (case-insensitive)
    for sub in case.get("must_contain", []):
        if sub.strip().startswith("[VERIFY"):
            fails.append(f"must_contain not filled in (still {sub!r}); verify this case")
        elif sub.lower() not in text_l:
            fails.append(f"must_contain '{sub}' not in answer text")
    for sub in case.get("must_not_contain", []):
        if sub.lower() in text_l:
            fails.append(f"must_not_contain '{sub}' present in answer text")

    # 6) artifact
    if case.get("expect_artifact") and not artifacts:
        fails.append("expected a chart artifact but answer.artifacts is empty")

    # 7) numeric value (only meaningful when status ok and a value was pinned)
    if "expected_value" in case and ans.get("status") == "ok":
        ev = case["expected_value"]
        # Guard: an unverified template may still carry a placeholder string
        # (e.g. "[VERIFY: ...]"). Don't attempt arithmetic on it - flag the case
        # as needing verification rather than crashing the whole run.
        if not isinstance(ev, (int, float)):
            fails.append(f"expected_value not filled in (still {ev!r}); verify this case")
        else:
            tol = case.get("value_tolerance", 0.01)
            import re
            # v11.6.1: sign-aware extraction. The first live attribution run
            # (the suite's first NEGATIVE expected values) showed the old
            # ASCII-adjacent regex is sign-blind two ways: LLM answers carry
            # negatives as typographic dashes (en dash U+2013, minus U+2212)
            # and/or with a currency symbol between sign and digits ("-$793.70"
            # extracted as +793.7). Normalize LOCALLY (must_contain checks etc.
            # still see the original text), then extract. Sign matching stays
            # STRICT - sign-insensitive matching would defeat sign-convention
            # cases like attr_cl_sse.
            norm = text
            norm = re.sub(r"[\u2212\u2012\u2013\u2014]", "-", norm)  # minus/figure/en/em dash
            norm = re.sub(r"-\s*\$\s*", "-", norm)                   # -$793.70 -> -793.70
            norm = re.sub(r"\$\s*-\s*", "-", norm)                   # $-793.70 -> -793.70
            norm = re.sub(r"\(\s*\$\s*([\d,]+(?:\.\d+)?)\s*\)", r"-\1", norm)  # ($793.70) -> -793.70
            nums = [float(n.replace(",", "")) for n in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", norm)]
            candidates = [(ev, tol), (ev * 100, tol * 100), (ev / 100, tol / 100)]
            if not any(abs(n - target) <= t for n in nums for target, t in candidates):
                fails.append(f"expected_value {ev} (±{tol}, also accepting %/fraction forms) not found among answer numbers")

    return (len(fails) == 0), fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", help="run only one area: pit|provenance|quant|methodology")
    ap.add_argument("--id", help="run only the case with this id")
    ap.add_argument("--ids", help="comma-separated list of case ids to run")
    ap.add_argument("--rerun-failed", metavar="REPORT_JSON",
                    help="run only the FAIL/FLAKY cases from a prior report file")
    ap.add_argument("--limit", type=int, help="cap number of cases (after filtering)")
    ap.add_argument("--verbose", action="store_true", help="print each case result")
    ap.add_argument("--report", help="write a JSON report to this path")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each case N times and report pass-rate (detects nondeterminism). Default 1.")
    ap.add_argument("--baseline", action="store_true",
                    help="run the VANILLA model arm (single LLM call, no tools/corpus/as-of) instead of the agent, for head-to-head comparison. Scored by the same scorer.")
    ap.add_argument("--show-answer", action="store_true",
                    help="print each case's answer text, status, and citations (diagnostic for FAIL/insufficient_data cases)")
    ap.add_argument("--save-trace", metavar="DIR",
                    help="copy each case's trace JSONL into DIR for inspection")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. This harness calls the LLM.\n"
              "Set the key (or run the project once via run.py) and retry.", file=sys.stderr)
        return 2

    cases = load_cases()
    if args.area:
        cases = [c for c in cases if c["area"] == args.area]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.rerun_failed:
        prior = json.loads(Path(args.rerun_failed).read_text())
        bad = {c["id"] for c in prior["cases"] if c.get("status") in ("FAIL", "FLAKY")
               or c.get("passed") is False}
        cases = [c for c in cases if c["id"] in bad]
        print(f"Re-running {len(cases)} FAIL/FLAKY case(s) from {args.rerun_failed}")
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases matched the filter.", file=sys.stderr)
        return 2

    n = max(1, args.repeat)
    arm = "VANILLA BASELINE (no tools/corpus/as-of)" if args.baseline else "AGENT"
    print(f"Running {len(cases)} case(s){'' if n == 1 else f' x{n} repeats'} -- arm: {arm}\n")
    results = []
    area_pass = Counter()      # cases that passed ALL repeats (strict)
    area_total = Counter()
    flaky = []                 # cases with mixed pass/fail across repeats
    grand = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}  # run-wide totals

    if args.save_trace:
        Path(args.save_trace).mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        area_total[case["area"]] += 1
        run_passes = 0
        all_reasons: list[str] = []
        last_res = None
        case_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        for rep in range(n):
            try:
                res = run_case_isolated(case, baseline=args.baseline)
            except subprocess.TimeoutExpired:
                res = {"answer": None, "tools_fired": [], "error": "timeout (600s)"}
            last_res = res
            # accumulate token usage across repeats for this case
            u = res.get("usage") or {}
            case_usage["input_tokens"] += u.get("input_tokens", 0)
            case_usage["output_tokens"] += u.get("output_tokens", 0)
            case_usage["cost_usd"] += u.get("cost_usd", 0.0)
            ok, reasons = score(case, res)
            if ok:
                run_passes += 1
            else:
                all_reasons.extend(reasons)

            # --save-trace: copy THIS repeat's trace out, uniquely named per repeat
            # (rep index + trace id) so all N runs are kept, not just the last.
            # This is what lets analyze_retrieval_probes see every run's probes.
            if args.save_trace and res.get("trace_path"):
                try:
                    src = Path(res["trace_path"])
                    if src.exists():
                        dst = Path(args.save_trace) / f"{cid}__rep{rep+1:02d}__{src.name}"
                        shutil.copy(src, dst)
                except Exception:
                    pass

        grand["input_tokens"] += case_usage["input_tokens"]
        grand["output_tokens"] += case_usage["output_tokens"]
        grand["cost_usd"] += case_usage["cost_usd"]

        # Classify: PASS = all repeats passed, FAIL = none, FLAKY = mixed.
        if run_passes == n:
            status = "PASS"
            area_pass[case["area"]] += 1
        elif run_passes == 0:
            status = "FAIL"
        else:
            status = "FLAKY"
            flaky.append(cid)
        # unique reasons only, for readability
        uniq_reasons = sorted(set(all_reasons))
        results.append({"id": cid, "area": case["area"], "status": status,
                        "passes": run_passes, "runs": n, "reasons": uniq_reasons,
                        "usage": {"input_tokens": case_usage["input_tokens"],
                                  "output_tokens": case_usage["output_tokens"],
                                  "cost_usd": round(case_usage["cost_usd"], 6)}})

        rate = "" if n == 1 else f" ({run_passes}/{n})"
        # per-case token + cost line
        cu = case_usage
        cost_str = (f"  [in {cu['input_tokens']:>6d} tok | out {cu['output_tokens']:>5d} tok "
                    f"| ${cu['cost_usd']:.4f}{'' if n == 1 else f' over {n} runs'}]")
        print(f"[{i:2d}/{len(cases)}] {status:5s} {cid}{rate}{cost_str}")
        if (args.verbose or status != "PASS") and uniq_reasons:
            for r in uniq_reasons:
                print(f"          - {r}")

        # --show-answer: print the agent's actual answer for diagnosis
        if args.show_answer and last_res and last_res.get("answer"):
            a = last_res["answer"]
            print(f"          STATUS: {a.get('status')}")
            print(f"          CITATIONS: {a.get('citations')}")
            txt = (a.get("text") or "").strip().replace("\n", "\n          ")
            print(f"          ANSWER: {txt[:1200]}")
        elif args.show_answer and last_res and last_res.get("error"):
            print(f"          ERROR: {last_res['error']}")

    strict_pass = sum(1 for r in results if r["status"] == "PASS")
    print("\n" + "=" * 60)
    print(f"OVERALL: {strict_pass}/{len(results)} passed all {n} run(s)")
    if flaky:
        print(f"FLAKY ({len(flaky)} cases vary run-to-run): {', '.join(flaky)}")
    print("By area (cases passing all repeats):")
    for area in sorted(area_total):
        print(f"  {area:12s} {area_pass[area]:2d}/{area_total[area]:2d}")

    # Cost summary (token counts exact; $ via config pricing table).
    price = None
    try:
        from quant_agent import config as _cfg
        price = _cfg.pricing_for(_cfg.MODEL)
        model_name = _cfg.MODEL
    except Exception:
        model_name = "?"
    print("\nToken usage & cost (model: {}):".format(model_name))
    print(f"  input  {grand['input_tokens']:>9d} tok")
    print(f"  output {grand['output_tokens']:>9d} tok")
    print(f"  TOTAL COST ${grand['cost_usd']:.4f}"
          + (f"  (avg ${grand['cost_usd']/max(1,len(results)):.4f}/case)" if results else ""))
    if price:
        print(f"  (priced at ${price['input']}/${price['output']} per Mtok in/out; "
              f"verify current rates at anthropic.com/pricing)")

    if args.report:
        report = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "arm": "baseline" if args.baseline else "agent",
            "repeats_per_case": n,
            "total": len(results),
            "passed_all_runs": strict_pass,
            "flaky": flaky,
            "by_area": {a: {"passed": area_pass[a], "total": area_total[a]} for a in area_total},
            "cost": {
                "model": model_name,
                "input_tokens": grand["input_tokens"],
                "output_tokens": grand["output_tokens"],
                "total_cost_usd": round(grand["cost_usd"], 6),
                "avg_cost_per_case_usd": round(grand["cost_usd"] / max(1, len(results)), 6),
            },
            "cases": results,
        }
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.report}")

    # Exit non-zero if anything didn't pass cleanly (FAIL or FLAKY).
    return 0 if strict_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
