"""Typed contracts shared by every layer.

Nothing crosses a layer boundary except these objects. The Evidence model is
the carrier for both guardrails: provenance (source_id is required) and
point-in-time integrity (pub_date + pit_safe travel with every datum).
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class Evidence(BaseModel):
    source_id: str
    store: str
    content: str
    pub_date: Optional[date] = None
    pit_safe: bool = True
    kind: Literal["fact", "method"] = "fact"
    meta: dict[str, Any] = Field(default_factory=dict)

    def digest(self, max_chars: int = 700) -> str:
        text = self.content
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        d = self.pub_date.isoformat() if self.pub_date else "undated"
        flags = ""
        if not self.pit_safe:
            # Distinguish *why* the datum isn't strictly point-in-time so the
            # synthesizer flags it correctly. "retroactive" data (e.g. yfinance
            # adjusted closes) is reproducible via the as-of snapshot but uses
            # adjustments unknown at the as-of date; "live" data is not
            # reproducible at all. Default to the stronger live-data warning.
            reason = self.meta.get("pit_caveat", "live")
            if reason == "retroactive":
                flags += " [NON-PIT: retroactively adjusted - reproducible via snapshot but uses post-as-of corporate-action adjustments]"
            else:
                flags += " [NON-PIT LIVE DATA - not reproducible for backtesting]"
        if self.kind == "method":
            flags += " [METHOD - cite for methods/definitions only, never period facts]"
        if self.meta.get("postdates_as_of"):
            flags += " [POSTDATES AS-OF - hindsight risk]"
        return f"[{self.source_id}] ({self.store}, {d}){flags}\n{text}"


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        return self.tool + "|" + "|".join(f"{k}={self.args[k]}" for k in sorted(self.args))


class Plan(BaseModel):
    thoughts: str = ""
    actions: list[ToolCall] = Field(default_factory=list)


class Gap(BaseModel):
    sub_question: str
    suggestion: str = ""


class Verdict(BaseModel):
    sufficient: bool
    gaps: list[Gap] = Field(default_factory=list)
    notes: str = ""


class Answer(BaseModel):
    status: Literal["ok", "insufficient_data", "error"]
    text: str
    citations: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)  # saved files (charts etc.)
    iterations: int = 0
    trace_id: str = ""


class State(BaseModel):
    """Shared state: created once per query by the orchestrator, passed into
    every layer, accumulated into, never replaced."""
    trace_id: str = Field(default_factory=new_trace_id)
    query: str
    as_of: date
    evidence: list[Evidence] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    action_history: list[ToolCall] = Field(default_factory=list)
    iteration: int = 0

    def evidence_digest(self, max_total_chars: int = 12000) -> str:
        if not self.evidence:
            return "(no evidence collected yet)"
        parts: list[str] = []
        used = 0
        for ev in self.evidence:
            d = ev.digest()
            if used + len(d) > max_total_chars:
                parts.append(f"...and {len(self.evidence) - len(parts)} more evidence items omitted.")
                break
            parts.append(d)
            used += len(d)
        return "\n\n".join(parts)

    def seen(self, call: ToolCall) -> bool:
        """Novelty check: has this exact action already been executed?"""
        return any(c.key() == call.key() for c in self.action_history)
