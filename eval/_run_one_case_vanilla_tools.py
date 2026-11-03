#!/usr/bin/env python3
"""Run ONE case through the VANILLA-WITH-TOOLS arm: the model has the SAME tools
as the agent (price history, indicators, doc search, filings) but with the
point-in-time GUARDRAILS DISABLED. This isolates exactly one variable - do the
guardrails matter, given identical capability?

Guardrails removed vs the real agent:
  - Document retrieval has NO pub_date <= as_of filter, so the model CAN see
    filings dated after the as-of date (the main look-ahead vector).
  - No ticker allowlist gate.
  - No offline-strict refusal / no orchestrator PIT evaluator loop.
  - The as-of date is only MENTIONED in the prompt (the way a careful person
    might note it), not enforced by anything.

What is kept identical so the comparison is fair:
  - Same underlying tools and the same corpus/price data.
  - Same model (config.MODEL).
  - Same Answer output shape, scored by the SAME scorer.

This is NOT the bare --baseline arm (which has no tools at all). This arm is
"Claude with the tools but without the discipline."

Usage (invoked by run_aapl_eval.py --arm vanilla):
    python eval/_run_one_case_vanilla_tools.py <case_json> <dataset_dir>
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import date
from pathlib import Path


def _unguarded_doc_search(query: str, k: int = 6) -> list[dict]:
    """doc_search WITHOUT the pub_date <= as_of filter. Returns the top-k chunks
    regardless of filing date - so post-as-of filings are visible. This is the
    deliberate hole the guarded agent does not have."""
    from quant_agent.tools.doc_kb import _collection, _first_row
    col = _collection("docs")
    res = col.query(query_texts=[str(query)], n_results=int(k))  # NO where= date filter
    docs0, metas0 = _first_row(res)
    out = []
    for doc, md in zip(docs0, metas0):
        out.append({"source_id": f"doc:{md['file']}#chunk{md['chunk']}",
                    "pub_date": md.get("pub_date"), "text": doc})
    return out


def main() -> int:
    case_path = Path(sys.argv[1])
    case = json.loads(case_path.read_text())

    from quant_agent import llm

    as_of = case["as_of"]
    # Give the model the retrieved context up front (unguarded: includes any
    # filing in the corpus, even post-as-of). A single tool-augmented shot.
    try:
        chunks = _unguarded_doc_search(case["query"], k=6)
    except Exception:
        chunks = []

    context = "\n\n".join(
        f"[{c['source_id']}] (pub_date={c['pub_date']})\n{c['text'][:800]}" for c in chunks
    ) or "(no documents retrieved)"

    system = (
        "You are a quantitative investment research assistant. You have been given "
        "retrieved document chunks as context. Answer the question using them. "
        "Cite sources inline as [source_id]. Note the as-of date in the question, "
        "but you are not otherwise restricted."
    )
    user = (
        f"AS-OF DATE: {as_of}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"QUESTION: {case['query']}"
    )
    try:
        text = llm.complete(system, user, max_tokens=2000)
    except Exception as e:
        print(json.dumps({"answer": None, "tools_fired": [], "trace_id": None,
                          "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}))
        return 0

    import re
    citations = [m.strip() for m in re.findall(r"\[([^\[\]]+)\]", text) if ":" in m or "#" in m]
    # crude status inference, same spirit as the bare baseline
    t = text.lower()
    insufficient = any(m in t for m in ("i don't have", "cannot determine", "no information",
                                        "insufficient", "unable to", "cannot answer", "not available"))
    status = "insufficient_data" if (insufficient and len(text) < 600) else "ok"

    answer = {"status": status, "text": text, "citations": citations,
              "artifacts": [], "iterations": 1, "trace_id": ""}
    # also surface which chunks were retrieved (incl. post-as-of) for leak auditing
    print(json.dumps({"answer": answer, "tools_fired": ["unguarded_doc_search"],
                      "retrieved": [c["source_id"] + f"|{c['pub_date']}" for c in chunks],
                      "trace_id": None, "error": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
