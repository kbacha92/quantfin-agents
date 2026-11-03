"""Synthesis layer: synthesize(state) -> Answer. One LLM call.

Drafts the final answer from evidence only, with [source_id] citations.
Numbers are reported, never recomputed - that constraint lives in the prompt
and is auditable via the citations list returned alongside the text.
"""
import re

from . import llm, prompts
from .schemas import Answer, State

# Pull whatever sits inside square brackets in the drafted answer, e.g.
# "[doc:2025-02-12_AAPL_10-K.txt#chunk3]". The synthesis prompt asks the model
# to echo source ids verbatim, but it sometimes lightly reformats spacing or
# casing; matching on a normalized bracket token recovers those instead of
# silently dropping the citation from the structured list (fix #6).
_BRACKET = re.compile(r"\[([^\[\]]+)\]")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _cited_ids(text: str, state: State) -> list[str]:
    ids = {e.source_id for e in state.evidence}
    found: set[str] = set()
    # 1) exact substring (cheap, catches the common verbatim case)
    for sid in ids:
        if sid in text:
            found.add(sid)
    # 2) normalized match against bracketed tokens (recovers reformatted ids)
    bracket_tokens = {_normalize(t) for t in _BRACKET.findall(text)}
    norm_to_id = {_normalize(sid): sid for sid in ids}
    for norm, sid in norm_to_id.items():
        if norm in bracket_tokens:
            found.add(sid)
    return sorted(found)


def synthesize(state: State) -> Answer:
    text = llm.complete(prompts.SYNTH_SYSTEM, prompts.synth_user(state), max_tokens=2000)
    cited = _cited_ids(text, state)
    artifacts = [e.meta["path"] for e in state.evidence if e.store == "chart" and e.meta.get("path")]
    return Answer(
        status="ok",
        text=text,
        citations=cited,
        artifacts=artifacts,
        iterations=state.iteration + 1,
        trace_id=state.trace_id,
    )
