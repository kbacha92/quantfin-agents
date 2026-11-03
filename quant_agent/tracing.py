"""Tracing rail: one JSONL file per query under data/traces/.

Every layer call is logged by the orchestrator through this; no layer opts in
or out. The trace is the audit trail your evals and debugging run on.
"""
import json
import time

from . import config


class Trace:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.path = config.TRACES_DIR / f"{trace_id}.jsonl"

    def log(self, step: str, **payload) -> None:
        rec = {"ts": time.time(), "trace": self.trace_id, "step": step, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
