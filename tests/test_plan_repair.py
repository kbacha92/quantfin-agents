
from datetime import date

from quant_agent.orchestrator import _repair_action
from quant_agent.schemas import State, ToolCall


class _FakeTrace:
    def log(self, *a, **k):
        pass


def _state():
    return State(query="What is ACME revenue across filings?", as_of=date(2025, 3, 1))


def test_empty_doc_search_gets_query_injected():
    out = _repair_action(ToolCall(tool="doc_search", args={}), _state(), _FakeTrace())
    assert out.args.get("query") == _state().query


def test_blank_query_repaired():
    out = _repair_action(ToolCall(tool="doc_search", args={"query": "   "}), _state(), _FakeTrace())
    assert out.args.get("query") == _state().query


def test_real_query_untouched():
    out = _repair_action(ToolCall(tool="doc_search", args={"query": "revenue"}), _state(), _FakeTrace())
    assert out.args.get("query") == "revenue"


def test_knowledge_search_repaired():
    out = _repair_action(ToolCall(tool="knowledge_search", args={}), _state(), _FakeTrace())
    assert out.args.get("query") == _state().query


def test_non_retrieval_tool_untouched():
    out = _repair_action(ToolCall(tool="edgar_fetch", args={}), _state(), _FakeTrace())
    assert out.args == {}


def test_original_action_not_mutated():
    a = ToolCall(tool="doc_search", args={})
    _repair_action(a, _state(), _FakeTrace())
    assert a.args == {}
