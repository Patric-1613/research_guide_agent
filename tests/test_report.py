"""Deterministic tests for report.py's non-LLM logic: schema
construction, per-section citation attachment, and skipped-paper
detection. The OpenAI call itself is mocked here (a real, live
generation was already run manually for Phase 4b's "real output, not a
mock" requirement) so these run without network access or billing. The
critical adversarial grounding test (Phase 4c) lives in
tests/test_report_grounding.py, separated out since it's the one
property this whole phase exists to prove.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.query_expansion import PaperPoolSession
from research_agent.report import _build_report_schema, generate_report, generate_report_for_session
from research_agent.schema import Paper


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _mock_parsed_response(parsed):
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def test_report_schema_rejects_unknown_paper_id():
    schema = _build_report_schema(["a", "b"])
    section_cls = schema.model_fields["findings"].annotation

    section_cls(content="fine", cited_paper_ids=["a"])  # known id: should not raise
    try:
        section_cls(content="fabricated", cited_paper_ids=["not-a-real-id"])
        assert False, "expected a validation error for an unknown paper_id"
    except Exception:
        pass


def test_generate_report_attaches_citations_per_section_and_flags_skipped():
    papers = [_paper("1111", "Paper One"), _paper("2222", "Paper Two"), _paper("3333", "Paper Three")]
    schema = _build_report_schema([p.paper_id for p in papers])
    section_cls = schema.model_fields["findings"].annotation

    # Different sections cite different, non-overlapping subsets;
    # "3333" is never cited anywhere -> must show up as skipped.
    parsed = schema(
        findings=section_cls(content="Findings text.", cited_paper_ids=["1111", "2222"]),
        limitations=section_cls(content="Limitations text.", cited_paper_ids=["1111"]),
        future_scope=section_cls(content="Future scope text.", cited_paper_ids=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", papers, client=mock_client)

    assert [p.paper_id for p in result["findings"]["cited_papers"]] == ["1111", "2222"]
    assert [p.paper_id for p in result["limitations"]["cited_papers"]] == ["1111"]
    assert result["future_scope"]["cited_papers"] == []
    assert [p.paper_id for p in result["skipped_papers"]] == ["3333"]


def test_generate_report_returns_empty_for_no_selected_papers():
    result = generate_report("topic", [], client=MagicMock())
    assert result["findings"] == {"content": "", "cited_papers": []}
    assert result["limitations"] == {"content": "", "cited_papers": []}
    assert result["future_scope"] == {"content": "", "cited_papers": []}
    assert result["skipped_papers"] == []


def test_generate_report_for_session_refuses_when_stage_is_not_synthesize():
    session = PaperPoolSession(topic="q", stage="curate", selected_papers=[_paper("1111", "Paper One")])
    try:
        generate_report_for_session(session, client=MagicMock())
        assert False, "expected a ValueError for a session not yet in the synthesize stage"
    except ValueError as e:
        assert "curate" in str(e)


# --- Phase 4d: edge cases ---

def test_generate_report_with_just_one_selected_paper():
    papers = [_paper("1111", "Paper One")]
    schema = _build_report_schema([p.paper_id for p in papers])
    section_cls = schema.model_fields["findings"].annotation

    parsed = schema(
        findings=section_cls(content="Findings text.", cited_paper_ids=["1111"]),
        limitations=section_cls(content="Limitations text.", cited_paper_ids=["1111"]),
        future_scope=section_cls(content="Future scope text.", cited_paper_ids=["1111"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", papers, client=mock_client)

    for section in ("findings", "limitations", "future_scope"):
        assert [p.paper_id for p in result[section]["cited_papers"]] == ["1111"]
    assert result["skipped_papers"] == []


def test_report_schema_construction_near_the_30_paper_cap():
    """Pure Python/pydantic mechanics, not an LLM-output question — does
    building a Literal from a large id set even work? Real cap from the
    product's own UI design (max 30 papers per literature review)."""
    paper_ids = [f"p{i}" for i in range(30)]
    schema = _build_report_schema(paper_ids)
    section_cls = schema.model_fields["findings"].annotation

    section_cls(content="fine", cited_paper_ids=paper_ids)  # all 30, should not raise
    try:
        section_cls(content="bad", cited_paper_ids=["p999"])
        assert False, "expected a validation error for an id outside the 30"
    except Exception:
        pass


def test_generate_report_falls_back_to_placeholder_text_for_missing_abstract():
    """Mirrors summarize.py's own established convention exactly:
    '(no abstract available)' when a selected paper's abstract is
    missing, rather than crashing or silently omitting the paper from
    the prompt."""
    thin_paper = Paper(
        title="Paper With No Abstract", authors=["A"], year=2024, venue="X",
        abstract=None, url=None, doi=None, citation_count=None,
        source="arxiv", paper_id="thin-1",
    )
    schema = _build_report_schema(["thin-1"])
    section_cls = schema.model_fields["findings"].annotation
    parsed = schema(
        findings=section_cls(content="Findings text.", cited_paper_ids=["thin-1"]),
        limitations=section_cls(content="Limitations text.", cited_paper_ids=[]),
        future_scope=section_cls(content="Future scope text.", cited_paper_ids=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", [thin_paper], client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "(no abstract available)" in user_content
    assert [p.paper_id for p in result["findings"]["cited_papers"]] == ["thin-1"]


if __name__ == "__main__":
    test_report_schema_rejects_unknown_paper_id()
    test_generate_report_attaches_citations_per_section_and_flags_skipped()
    test_generate_report_returns_empty_for_no_selected_papers()
    test_generate_report_for_session_refuses_when_stage_is_not_synthesize()
    test_generate_report_with_just_one_selected_paper()
    test_report_schema_construction_near_the_30_paper_cap()
    test_generate_report_falls_back_to_placeholder_text_for_missing_abstract()
    print("All report tests passed.")
