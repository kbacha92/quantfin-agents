"""The loop. Deliberately dumb: invoke planner, validate + execute tools,
merge via assembly, evaluate, then branch (loop / stop / synthesize).

All intelligence lives in the layers; all budgets and tracing live here.
"""
from datetime import date
from typing import Callable, Optional

from . import assembly, config, evaluate, planner, synthesize
from .schemas import Answer, Gap, State, ToolCall
from .tools import REGISTRY
from .tracing import Trace


# Retrieval tools whose first positional argument is the search query. If the
# planner emits one of these without a query, the call crashes ("missing required
# argument 'query'"). These are recoverable: the user's question IS a reasonable
# query, so we inject it rather than letting the call fail and strand the run.
_QUERY_TOOLS = {"doc_search": "query", "knowledge_search": "query"}


def _repair_action(a: "ToolCall", state: State, trace: Trace) -> "ToolCall":
    """Return a possibly-repaired copy of an action.

    Currently repairs one proven failure mode: a retrieval tool emitted with no
    query (or an empty/blank query). We inject the user's question as the query.
    Any repair is logged ('plan_repair') so the planner's omission rate stays
    visible - this fixes the symptom without hiding how often it happens.
    Non-recoverable malformed actions are returned unchanged (they will error and
    be recorded as before).
    """
    if a.tool not in _QUERY_TOOLS:
        return a
    qkey = _QUERY_TOOLS[a.tool]
    current = a.args.get(qkey)
    if isinstance(current, str) and current.strip():
        return a  # query present and non-empty; nothing to repair
    # Inject the user's question as the query.
    new_args = dict(a.args)
    new_args[qkey] = state.query
    try:
        trace.log("plan_repair", tool=a.tool, missing=qkey,
                  injected_query=state.query, original_args=a.args)
    except Exception:
        pass
    return a.model_copy(update={"args": new_args})


def run_query(
    query: str,
    as_of: date,
    status_cb: Optional[Callable[[str], None]] = None,
    max_iters: Optional[int] = None,
) -> Answer:
    state = State(query=query, as_of=as_of)
    trace = Trace(state.trace_id)
    trace.log("start", query=query, as_of=str(as_of))
    say = status_cb or (lambda _msg: None)
    cap = max_iters or config.MAX_ITERS

    try:
        for i in range(cap):
            state.iteration = i
            say(f"Iteration {i + 1}/{cap}: planning...")
            p = planner.plan(state)
            trace.log("plan", iteration=i, thoughts=p.thoughts, actions=[a.model_dump() for a in p.actions])

            # Validation + novelty filter (deterministic, no LLM).
            runnable = [a for a in p.actions if a.tool in REGISTRY]
            new_actions = [a for a in runnable if not state.seen(a)]
            # "Stranded" = the planner proposed work, but every proposed action
            # was already tried. Re-running them is futile, so we don't loop on
            # them; we give the evaluator one look at existing evidence and then
            # stop early rather than burning iterations to the cap (fix #5).
            stranded = bool(runnable) and not new_actions

            if not runnable and not state.evidence:
                state.gaps.append(Gap(sub_question="Planner produced no runnable actions", suggestion="rephrase the query"))

            for a in new_actions:
                # Plan-validation repair (v10.1): the planner intermittently emits
                # a retrieval action with no query (trace showed doc_search args={}
                # -> "missing required argument 'query'", crashing the call and
                # stranding the run). Rather than execute a malformed call, repair
                # the recoverable case: a query-less doc_search/knowledge_search
                # gets the user's question injected as the query. Logged so the
                # planner's omission rate stays measurable.
                repaired = _repair_action(a, state, trace)
                arg_str = ", ".join(f"{k}={v}" for k, v in repaired.args.items())
                say(f"Running {repaired.tool}({arg_str})...")
                try:
                    tool = REGISTRY[repaired.tool]
                    extra = {}
                    # Tools that gate on the user's intent (e.g. edgar_fetch's
                    # ticker allowlist) receive the original query as guardrail
                    # context. Plain data tools don't, keeping their signature clean.
                    if getattr(tool, "needs_query", False):
                        extra["_query"] = state.query
                    ev = tool(as_of=state.as_of, **repaired.args, **extra)
                    added = assembly.merge(state, ev)
                    trace.log("tool_ok", tool=repaired.tool, args=repaired.args, evidence_added=added)
                    # Drain any retrieval probes this tool produced into the trace,
                    # so empties are self-explaining (collection count, query index).
                    try:
                        from .tools import doc_kb as _dk
                        for _p in _dk.get_probes():
                            trace.log("retrieval_probe", tool=repaired.tool, **_p)
                        _dk.clear_probes()
                    except Exception:
                        pass
                except Exception as e:
                    trace.log("tool_error", tool=repaired.tool, args=repaired.args, error=str(e))
                    state.gaps.append(Gap(sub_question=f"Tool {repaired.tool} failed", suggestion=str(e)[:300]))
                state.action_history.append(a)

            say("Evaluating evidence...")
            v = evaluate.verdict(state)
            trace.log("verdict", iteration=i, sufficient=v.sufficient,
                      gaps=[g.model_dump() for g in v.gaps], notes=v.notes)

            if v.sufficient:
                say("Synthesizing answer...")
                ans = synthesize.synthesize(state)
                trace.log("answer", status=ans.status, citations=ans.citations)
                return ans

            state.gaps = v.gaps  # the planner targets these next iteration

            # Early stop: the only plan the planner can produce repeats actions
            # already taken, and the evidence still isn't sufficient. Looping
            # would re-propose the same dead plan until the cap. Break now and
            # report honestly (fix #5).
            if stranded:
                say("No new actions available; stopping early.")
                trace.log("stranded_stop", iteration=i,
                          repeated=[a.key() for a in runnable])
                break

        # Reached here either by hitting the cap or by an early stranded-stop.
        # Report what was found and what is still missing, accurately either way.
        iters_done = state.iteration + 1
        hit_cap = iters_done >= cap
        found = "\n".join(f"- {e.source_id}" for e in state.evidence) or "- nothing"
        missing = "\n".join(f"- {g.sub_question}" for g in state.gaps) or "- unspecified"
        if hit_cap:
            lead = f"I could not assemble enough point-in-time evidence within {cap} iterations."
            tip = "You can raise the iteration cap, add documents to the knowledge base, or narrow the question."
        else:
            lead = (
                f"I stopped after {iters_done} iteration(s): the planner could only "
                "repeat steps already taken, and the evidence so far is not sufficient."
            )
            tip = "Try rephrasing the question, adding documents to the knowledge base, or pulling a relevant filing."
        ans = Answer(
            status="insufficient_data",
            text=(
                f"{lead}\n\n"
                f"**Found:**\n{found}\n\n**Still missing:**\n{missing}\n\n"
                f"{tip}"
            ),
            citations=[e.source_id for e in state.evidence],
            artifacts=[e.meta["path"] for e in state.evidence if e.store == "chart" and e.meta.get("path")],
            iterations=iters_done,
            trace_id=state.trace_id,
        )
        trace.log("answer", status=ans.status)
        return ans

    except Exception as e:
        trace.log("error", error=str(e))
        return Answer(status="error", text=f"System error: {e}", trace_id=state.trace_id)
