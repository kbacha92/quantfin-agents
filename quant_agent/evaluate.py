"""Evaluation layer: verdict(state) -> Verdict. One cheap LLM call.

Checks coverage, conflicts, and grounding against the evidence and returns a
structured pass/fail plus a gap list the planner can act on. The budget guard
(loop vs stop) deliberately lives in the orchestrator, not here.
"""
from . import llm, prompts
from .schemas import Gap, State, Verdict


def verdict(state: State) -> Verdict:
    data = llm.complete_json(prompts.EVAL_SYSTEM, prompts.eval_user(state))
    gaps = []
    for g in data.get("gaps", [])[:5]:
        try:
            gaps.append(Gap(**g))
        except Exception:
            continue
    return Verdict(
        sufficient=bool(data.get("sufficient", False)),
        gaps=gaps,
        notes=str(data.get("notes", "")),
    )
