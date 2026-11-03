
from datetime import date

from quant_agent import orchestrator
from quant_agent.schemas import Evidence, Plan, ToolCall, Verdict


def _patch(monkeypatch, plan_fn, verdict_fn, registry):
    monkeypatch.setattr(orchestrator.planner, "plan", plan_fn)
    monkeypatch.setattr(orchestrator.evaluate, "verdict", verdict_fn)
    monkeypatch.setattr(orchestrator, "REGISTRY", registry)


def test_stranded_planner_stops_early(monkeypatch):
    calls = {"tool": 0}

    def fake_tool(as_of, **kw):
        calls["tool"] += 1
        return [Evidence(source_id="quant:x", store="quant", content="c", pub_date=as_of)]

    registry = {"t": fake_tool}
    plan = lambda state: Plan(thoughts="", actions=[ToolCall(tool="t", args={"a": 1})])
    verdict = lambda state: Verdict(sufficient=False, gaps=[])

    _patch(monkeypatch, plan, verdict, registry)
    ans = orchestrator.run_query("q", date(2025, 1, 1), max_iters=5)

    assert ans.status == "insufficient_data"
    assert calls["tool"] == 1           # action executed once, never repeated
    assert ans.iterations < 5           # stopped early, did not burn the cap
    assert "repeat" in ans.text.lower() # message explains the early stop


def test_genuine_progress_runs_until_sufficient(monkeypatch):
    """A normal run still reaches a sufficient verdict and synthesizes."""
    def fake_tool(as_of, **kw):
        return [Evidence(source_id=f"quant:{kw.get('a')}", store="quant", content="c", pub_date=as_of)]

    registry = {"t": fake_tool}
    seq = iter([
        Plan(thoughts="", actions=[ToolCall(tool="t", args={"a": 1})]),
        Plan(thoughts="", actions=[ToolCall(tool="t", args={"a": 2})]),
    ])
    plan = lambda state: next(seq)
    verdicts = iter([Verdict(sufficient=False, gaps=[]), Verdict(sufficient=True, gaps=[])])
    verdict = lambda state: next(verdicts)

    monkeypatch.setattr(orchestrator.synthesize, "synthesize",
                        lambda state: __import__("quant_agent.schemas", fromlist=["Answer"]).Answer(
                            status="ok", text="done", iterations=state.iteration + 1, trace_id=state.trace_id))
    _patch(monkeypatch, plan, verdict, registry)
    ans = orchestrator.run_query("q", date(2025, 1, 1), max_iters=5)
    assert ans.status == "ok"
