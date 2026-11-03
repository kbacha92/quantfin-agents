#!/usr/bin/env python3
"""Run ONE golden case through a VANILLA model - no tools, no corpus, no as-of
enforcement - and emit the same result JSON shape as _run_one_case.py so the
SAME scorer can grade it. This is the baseline arm for head-to-head comparison.

The point of the comparison is to isolate what the agent scaffolding adds. So
the baseline is deliberately bare:
  - one LLM call, no planner/evaluate/synthesize loop
  - no document retrieval (the model never sees the seeded corpus)
  - no point-in-time guardrail (the model is only TOLD the as-of date, the way a
    person might mention it; nothing enforces it)
  - no tools, so no real price/quant computation and no real source_ids

To keep the fight fair on the dimensions that should be fair, the baseline is
given a system prompt that tells it the as-of date, asks for grounded answers
with [source_id] citations if it can, and to say so when it lacks information -
i.e. we let the vanilla model do its best at the SAME task, then score identically.

Usage (invoked by run_harness --baseline):
    python eval/_run_one_case_baseline.py <case_json> <dataset_dir>
Prints: {"answer": {...}, "tools_fired": [], "trace_id": null, "error": null}
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path


BASELINE_SYSTEM = (
    "You are a quantitative investment research assistant answering as of a stated date. "
    "Answer the user's question as accurately and honestly as you can. "
    "Critical rules:\n"
    "- Treat the AS-OF DATE as the present moment. Do not use any information that postdates it.\n"
    "- Only state facts you are confident are true as of that date. If you do not have the "
    "information, say so plainly and do not guess or fabricate figures.\n"
    "- If you reference a source, cite it inline in [source_id] form. If you have no sources, "
    "do not invent citations.\n"
    "- For numerical questions, give the precise value if you can compute it from data provided; "
    "otherwise say you cannot compute it without the underlying data."
)


def _infer_status(text: str) -> str:
    """The vanilla model returns free text with no status field. Infer one the
    way the agent would label itself, so the status assertion is comparable:
    if the answer is dominated by 'I don't have / cannot determine' language,
    call it insufficient_data; otherwise ok. Deliberately simple and transparent."""
    t = text.lower()
    insufficient_markers = (
        "i don't have", "i do not have", "cannot determine", "can't determine",
        "no information", "not able to", "unable to", "insufficient", "cannot answer",
        "i don't know", "do not have access", "no data", "cannot compute", "can't compute",
        "i cannot provide", "not available", "no available", "lack the",
    )
    hits = sum(1 for m in insufficient_markers if m in t)
    # If it hedges heavily and is short, treat as insufficient.
    if hits >= 1 and len(text) < 600:
        return "insufficient_data"
    if hits >= 2:
        return "insufficient_data"
    return "ok"


def _extract_citations(text: str) -> list[str]:
    """Pull any [bracketed] tokens the model emitted as citations. A vanilla
    model usually can't produce real source_ids (it never retrieved anything),
    so this is typically empty or fabricated - which is exactly what the
    citation assertions will (correctly) catch."""
    return [m.strip() for m in re.findall(r"\[([^\[\]]+)\]", text) if ":" in m or "#" in m]


def main() -> int:
    case_path = Path(sys.argv[1])
    case = json.loads(case_path.read_text())

    # Import only the LLM wrapper - NOT the orchestrator/tools. The baseline must
    # not touch retrieval or the corpus.
    from quant_agent import llm

    user = (
        f"AS-OF DATE: {case['as_of']}\n\n"
        f"QUESTION: {case['query']}\n\n"
        "Answer as of the as-of date above."
    )
    try:
        text = llm.complete(BASELINE_SYSTEM, user, max_tokens=2000)
    except Exception as e:
        print(json.dumps({
            "answer": None, "tools_fired": [], "trace_id": None,
            "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
        }))
        return 0

    answer = {
        "status": _infer_status(text),
        "text": text,
        "citations": _extract_citations(text),
        "artifacts": [],
        "iterations": 1,
        "trace_id": "",
    }
    print(json.dumps({"answer": answer, "tools_fired": [], "trace_id": None, "error": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
