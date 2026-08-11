"""Usage Protection M2.1 Part D: tests for the standalone agent's usage
bounds in research_agent/agent.py. No live LLM or network call anywhere
in this file -- either research_agent.agent.create_agent is mocked to
inspect exactly what construction/config it was called with, or a real
langchain create_agent() is exercised end-to-end against a small
in-process fake chat model (never touches the network) to prove the
real middleware/recursion_limit behavior, not just that the right
arguments were passed.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.tools import tool

import research_agent.agent as agent_module
import research_agent.telemetry as telemetry
from research_agent.agent import ResearchSession, run_research_agent
from research_agent.config import get_usage_policy
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
_REAL_USAGE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB_PATH)


class FakeToolCallingModel(BaseChatModel):
    """Minimal in-process fake: cycles through a fixed list of AIMessage
    responses, never touches the network. bind_tools() is a required
    seam create_agent's model node calls -- BaseChatModel's own default
    raises NotImplementedError, so it must be overridden here even
    though this fake doesn't need to do anything with the tool schemas."""

    responses: list
    i: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self):
        return "fake-tool-calling-model"


@tool
def _dummy_tool(query: str) -> str:
    """A dummy tool for bounded-execution tests."""
    return f"result for {query}"


# --- Construction/configuration: mock create_agent, inspect the call ------

class TestConstructionWiring:
    def _run_with_mocked_create_agent(self, monkeypatch):
        fake_compiled_agent = MagicMock()
        fake_compiled_agent.stream.return_value = iter(
            [{"messages": [{"role": "user", "content": "topic"}]}]
        )
        mock_create_agent = MagicMock(return_value=fake_compiled_agent)
        monkeypatch.setattr(agent_module, "create_agent", mock_create_agent)
        monkeypatch.setattr(agent_module, "build_tools", lambda session: ())
        run_research_agent("a topic", top_k=5)
        return mock_create_agent, fake_compiled_agent

    def test_model_call_limit_middleware_installed_with_configured_run_limit(self, monkeypatch):
        mock_create_agent, _ = self._run_with_mocked_create_agent(monkeypatch)
        policy = get_usage_policy()
        middleware = mock_create_agent.call_args.kwargs["middleware"]
        model_limiters = [m for m in middleware if isinstance(m, ModelCallLimitMiddleware)]
        assert len(model_limiters) == 1
        assert model_limiters[0].run_limit == policy.agent_model_call_limit_per_run
        assert model_limiters[0].exit_behavior == "end"

    def test_tool_call_limit_middleware_installed_with_configured_run_limit(self, monkeypatch):
        mock_create_agent, _ = self._run_with_mocked_create_agent(monkeypatch)
        policy = get_usage_policy()
        middleware = mock_create_agent.call_args.kwargs["middleware"]
        tool_limiters = [m for m in middleware if isinstance(m, ToolCallLimitMiddleware)]
        assert len(tool_limiters) == 1
        assert tool_limiters[0].run_limit == policy.agent_tool_call_limit_per_run
        # "continue", not "end" -- deliberate, see agent.py's own comment:
        # "end" raises NotImplementedError when a batch has other pending
        # parallel tool calls, which this agent's own documented usage
        # pattern (searching multiple sources in one turn) triggers.
        assert tool_limiters[0].exit_behavior == "continue"

    def test_no_other_middleware_was_accidentally_added(self, monkeypatch):
        mock_create_agent, _ = self._run_with_mocked_create_agent(monkeypatch)
        middleware = mock_create_agent.call_args.kwargs["middleware"]
        assert len(middleware) == 2
        allowed_types = (ModelCallLimitMiddleware, ToolCallLimitMiddleware)
        assert all(isinstance(m, allowed_types) for m in middleware)
        type_names = {type(m).__name__ for m in middleware}
        forbidden = {
            "SummarizationMiddleware", "HumanInTheLoopMiddleware",
            "PIIMiddleware", "ContextEditingMiddleware", "RetryMiddleware",
        }
        assert type_names.isdisjoint(forbidden)

    def test_recursion_limit_passed_to_stream_invocation(self, monkeypatch):
        _, fake_compiled_agent = self._run_with_mocked_create_agent(monkeypatch)
        policy = get_usage_policy()
        config = fake_compiled_agent.stream.call_args.kwargs["config"]
        assert config["recursion_limit"] == policy.agent_recursion_limit

    def test_recursion_limit_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("USAGE_AGENT_RECURSION_LIMIT", "4")
        _, fake_compiled_agent = self._run_with_mocked_create_agent(monkeypatch)
        config = fake_compiled_agent.stream.call_args.kwargs["config"]
        assert config["recursion_limit"] == 4


# --- Real bounded execution: real create_agent + real middleware, fake model ---

def _build_bounded_agent(model, *, model_run_limit, tool_run_limit):
    return create_agent(
        model,
        tools=[_dummy_tool],
        system_prompt="test",
        middleware=[
            ModelCallLimitMiddleware(run_limit=model_run_limit, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=tool_run_limit, exit_behavior="continue"),
        ],
    )


class TestRealBoundedExecution:
    def test_model_call_limit_stops_a_runaway_loop_predictably(self):
        """A model that always calls a tool, forever, must not loop
        unboundedly -- ModelCallLimitMiddleware(exit_behavior="end")
        must terminate the run once the run_limit is reached."""
        looping_call = AIMessage(
            content="", tool_calls=[{"name": "_dummy_tool", "args": {"query": "x"}, "id": "call_1"}]
        )
        model = FakeToolCallingModel(responses=[looping_call])
        agent = _build_bounded_agent(model, model_run_limit=3, tool_run_limit=50)

        steps = list(agent.stream(
            {"messages": [{"role": "user", "content": "go"}]},
            stream_mode="values", config={"recursion_limit": 15},
        ))

        final_messages = steps[-1]["messages"]
        assert model.i == 3  # the model was never called a 4th time
        # The run ends with a synthetic AI message noting the limit, not a crash.
        assert isinstance(final_messages[-1], AIMessage)
        assert "limit" in final_messages[-1].content.lower()

    def test_tool_call_limit_does_not_crash_on_parallel_tool_calls(self):
        """The confirmed material-mismatch check: a model that issues
        multiple tool calls in the same turn (agent.py's own documented
        common case) must not raise NotImplementedError once the tool
        call budget is exhausted -- this is exactly why exit_behavior=
        "continue" was chosen over "end" for ToolCallLimitMiddleware."""
        parallel_call = AIMessage(content="", tool_calls=[
            {"name": "_dummy_tool", "args": {"query": "a"}, "id": "call_a"},
            {"name": "_dummy_tool", "args": {"query": "b"}, "id": "call_b"},
        ])
        final = AIMessage(content="done", tool_calls=[])
        model = FakeToolCallingModel(responses=[parallel_call, parallel_call, final])
        agent = _build_bounded_agent(model, model_run_limit=10, tool_run_limit=3)

        # Must not raise.
        steps = list(agent.stream(
            {"messages": [{"role": "user", "content": "go"}]},
            stream_mode="values", config={"recursion_limit": 15},
        ))

        final_messages = steps[-1]["messages"]
        assert final_messages[-1].content == "done"

    def test_normal_execution_completes_unchanged_when_under_every_limit(self):
        """A short, ordinary run (well under both limits) must reach a
        normal final answer -- the limits must not perturb the common case."""
        final = AIMessage(content="the answer", tool_calls=[])
        model = FakeToolCallingModel(responses=[final])
        agent = _build_bounded_agent(model, model_run_limit=10, tool_run_limit=10)

        steps = list(agent.stream(
            {"messages": [{"role": "user", "content": "go"}]},
            stream_mode="values", config={"recursion_limit": 15},
        ))

        final_messages = steps[-1]["messages"]
        assert final_messages[-1].content == "the answer"
        assert model.i == 1


# --- End-to-end sanity: run_research_agent still returns a normal session ---

def test_run_research_agent_returns_normal_session_when_under_every_limit(monkeypatch, tmp_path):
    # run_research_agent wraps its own loop in telemetry.timed_child_call --
    # currently a no-op with no active telemetry.paid_action() around it
    # (confirmed: this test's own before/after DB fingerprint check below
    # never changes), but redirected explicitly anyway so this test stays
    # correct even if that call site's telemetry behavior changes later.
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", tmp_path / "usage_telemetry.sqlite")

    final = AIMessage(content="the answer", tool_calls=[])
    fake_model = FakeToolCallingModel(responses=[final])
    monkeypatch.setattr(agent_module, "AGENT_MODEL", fake_model)

    session = run_research_agent("parameter-efficient fine-tuning", top_k=5)

    assert isinstance(session, ResearchSession)
    assert session.papers == []  # the fake model never called a search tool


def test_real_usage_db_path_untouched():
    """Does NOT assert nonexistence -- a legitimate local
    usage_telemetry.sqlite from real dev-server use is normal, valid
    state. Proves nothing in this file's test run created, deleted, or
    modified it (or its -wal/-shm sidecars)."""
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
