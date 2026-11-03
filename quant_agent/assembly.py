"""Context assembly layer: deterministic merge of new evidence into state.

Dedupes by source_id, then orders by source precedence (config.PRECEDENCE,
primary sources first) and recency. Provenance fields are never touched.
"""
from . import config
from .schemas import Evidence, State


def merge(state: State, new: list[Evidence]) -> int:
    seen = {e.source_id for e in state.evidence}
    added = 0
    for ev in new:
        if ev.source_id in seen:
            continue
        state.evidence.append(ev)
        seen.add(ev.source_id)
        added += 1
    state.evidence.sort(
        key=lambda e: (
            config.PRECEDENCE.get(e.store, config.DEFAULT_PRECEDENCE),
            -(int(e.pub_date.strftime("%Y%m%d")) if e.pub_date else 0),
        )
    )
    return added
