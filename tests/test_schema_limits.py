"""Usage Protection M2.2C Part B: tests for the constrained request
schema fields (research_agent/api_app/constrained_types.py applied in
research_agent/api_app/schemas.py). Pure Pydantic validation, no HTTP
layer, no network, no telemetry DB involved.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError

from research_agent.api_app.schemas import (
    ChatRequest,
    ChatTurnIn,
    CurationChatAddToReportRequest,
    CurationChatDeleteRequest,
    CurationChatEditRequest,
    CurationChatRequest,
    CurationPicksRequest,
    CurationStartRequest,
    SearchRequest,
)
from research_agent.config import get_usage_policy

_TEXT_LIMIT = get_usage_policy().max_text_length
_ID_LIMIT = get_usage_policy().max_picked_ids_per_mutation


# --- Every constrained user-text request field -----------------------

_TEXT_FIELD_CASES = [
    (SearchRequest, {"topic": None}, "topic"),
    (CurationStartRequest, {"topic": None}, "topic"),
    (ChatRequest, {"search_id": 1, "question": None}, "question"),
    (CurationChatRequest, {"message": None}, "message"),
    (CurationChatEditRequest, {"exchange_id": "e1", "question": None}, "question"),
]


@pytest.mark.parametrize("model_cls,base_kwargs,field_name", _TEXT_FIELD_CASES)
def test_text_field_exactly_at_limit_passes(model_cls, base_kwargs, field_name):
    kwargs = {**base_kwargs, field_name: "x" * _TEXT_LIMIT}
    instance = model_cls(**kwargs)
    assert len(getattr(instance, field_name)) == _TEXT_LIMIT


@pytest.mark.parametrize("model_cls,base_kwargs,field_name", _TEXT_FIELD_CASES)
def test_text_field_one_over_limit_rejects(model_cls, base_kwargs, field_name):
    kwargs = {**base_kwargs, field_name: "x" * (_TEXT_LIMIT + 1)}
    with pytest.raises(ValidationError):
        model_cls(**kwargs)


@pytest.mark.parametrize("model_cls,base_kwargs,field_name", _TEXT_FIELD_CASES)
def test_text_field_unicode_counts_code_points_not_bytes(model_cls, base_kwargs, field_name):
    # A 4-byte-UTF-8 emoji, well over 2000 bytes if length were counted
    # in bytes -- must be counted in Unicode characters (code points).
    kwargs = {**base_kwargs, field_name: "\U0001F600" * _TEXT_LIMIT}
    instance = model_cls(**kwargs)
    assert len(getattr(instance, field_name)) == _TEXT_LIMIT

    kwargs_over = {**base_kwargs, field_name: "\U0001F600" * (_TEXT_LIMIT + 1)}
    with pytest.raises(ValidationError):
        model_cls(**kwargs_over)


def test_optional_refinement_field_still_accepts_none():
    req = CurationPicksRequest(picked_paper_ids=[], refinement=None)
    assert req.refinement is None


def test_optional_refinement_field_respects_the_same_limit():
    CurationPicksRequest(picked_paper_ids=[], refinement="x" * _TEXT_LIMIT)
    with pytest.raises(ValidationError):
        CurationPicksRequest(picked_paper_ids=[], refinement="x" * (_TEXT_LIMIT + 1))


# --- Every constrained ID-list request field ---------------------------

_ID_LIST_FIELD_CASES = [
    (CurationPicksRequest, {}, "picked_paper_ids"),
    (CurationChatDeleteRequest, {}, "exchange_ids"),
    (CurationChatAddToReportRequest, {}, "exchange_ids"),
]


@pytest.mark.parametrize("model_cls,base_kwargs,field_name", _ID_LIST_FIELD_CASES)
def test_id_list_exactly_at_limit_passes(model_cls, base_kwargs, field_name):
    kwargs = {**base_kwargs, field_name: [f"id{i}" for i in range(_ID_LIMIT)]}
    instance = model_cls(**kwargs)
    assert len(getattr(instance, field_name)) == _ID_LIMIT


@pytest.mark.parametrize("model_cls,base_kwargs,field_name", _ID_LIST_FIELD_CASES)
def test_id_list_one_over_limit_rejects(model_cls, base_kwargs, field_name):
    kwargs = {**base_kwargs, field_name: [f"id{i}" for i in range(_ID_LIMIT + 1)]}
    with pytest.raises(ValidationError):
        model_cls(**kwargs)


def test_picked_paper_ids_omitted_defaults_to_empty_list():
    req = CurationPicksRequest()
    assert req.picked_paper_ids == []


# --- Client-supplied chat history (ChatRequest.history) ----------------

def test_chat_turn_in_content_respects_the_text_limit():
    ChatTurnIn(role="user", content="x" * _TEXT_LIMIT)
    with pytest.raises(ValidationError):
        ChatTurnIn(role="user", content="x" * (_TEXT_LIMIT + 1))


def test_chat_request_history_omitted_defaults_to_empty_list():
    req = ChatRequest(search_id=1, question="q")
    assert req.history == []


def test_chat_request_history_list_size_bound_matches_turn_policy():
    policy = get_usage_policy()
    limit = policy.max_chat_turns_per_session * 2
    turns = [{"role": "user", "content": "hi"}] * limit
    ChatRequest(search_id=1, question="q", history=turns)

    with pytest.raises(ValidationError):
        ChatRequest(search_id=1, question="q", history=turns + [{"role": "user", "content": "one more"}])


def test_chat_turn_in_backward_compatible_with_bare_role_content_dict():
    """A pre-metadata {role, content} dict (no exchange_id/cited_papers/
    etc.) still constructs cleanly via ChatTurnIn(**turn), same
    additive/defaulted convention ChatTurn itself already documents."""
    turn = ChatTurnIn(**{"role": "assistant", "content": "an old-shape reply"})
    assert turn.exchange_id is None
    assert turn.cited_web_articles == []
    assert turn.cited_papers == []
    assert turn.added_to_report is False
    assert turn.web_relevance_verified is None


# --- Generated response models are unaffected ---------------------------

def test_response_models_do_not_constrain_long_generated_text():
    from research_agent.api_app.schemas import ChatResponse, ChatTurn, ReportSectionOut

    long_text = "x" * (_TEXT_LIMIT * 3)  # a real answer/report section can be long
    turn = ChatTurn(role="assistant", content=long_text)
    assert len(turn.content) == len(long_text)

    section = ReportSectionOut(content=long_text, cited_papers=[])
    assert len(section.content) == len(long_text)

    resp = ChatResponse(answer=long_text, answerable=True, cited_papers=[], cited_web_articles=[], history=[])
    assert len(resp.answer) == len(long_text)


# --- Normal Pydantic validation still returns useful 422 details --------

def test_validation_error_still_carries_field_level_detail():
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(topic="x" * (_TEXT_LIMIT + 1))
    errors = exc_info.value.errors()
    assert any(e["loc"] == ("topic",) for e in errors)
    assert any(e["type"] == "string_too_long" for e in errors)
