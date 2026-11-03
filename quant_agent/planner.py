"""Planning layer: plan(state) -> Plan. One LLM call.

On iteration 1 it decomposes the query; on later iterations it targets the
gap list written into state by the evaluation layer. It selects tools but
never executes them - execution belongs to the orchestrator.
"""
from . import llm, prompts
from .schemas import Plan, State, ToolCall
from .tools import describe_tools


def plan(state: State) -> Plan:
    data = llm.complete_json(prompts.PLANNER_SYSTEM, prompts.planner_user(state, describe_tools()))
    actions = []
    for a in data.get("actions", [])[:4]:
        try:
            actions.append(ToolCall(**a))
        except Exception:
            continue
    return Plan(thoughts=str(data.get("thoughts", "")), actions=actions)
