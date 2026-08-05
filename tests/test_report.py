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
from research_agent.report import (
    ANALYTICAL_SECTION_NAMES,
    GENERATION_REASON_INITIAL,
    GENERATION_REASON_REGENERATE,
    REPORT_SECTION_DEFINITIONS,
    REPORT_TEMPLATES,
    _build_references_and_renumber,
    _build_regeneration_system_prompt,
    _build_report_schema,
    _build_report_system_prompt,
    _cleanup_marker_stripped_whitespace,
    activate_report_version,
    append_report_version,
    build_references_and_renumber,
    derive_legacy_references,
    derive_sections_from_legacy_report,
    ReportEvaluation,
    evaluate_report,
    generate_report,
    generate_report_for_session,
    get_active_report_version,
    refine_report_if_requested,
    regenerate_report_with_approved_web_sources,
    regenerate_report_with_new_sources,
    revise_report,
)
from research_agent.schema import Paper, WebArticle


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


def _analytical_parsed(schema, web_urls_used: bool = False, **section_overrides):
    """report-quality Phase R2B: builds a full 8-field parsed report
    response for a REPORT_SECTION_DEFINITIONS-shaped schema, defaulting
    every section not explicitly passed in section_overrides to empty
    content/no citations -- lets each test only spell out the section(s)
    it actually cares about, matching this file's existing convention of
    hand-building `parsed` via the schema/section_cls rather than a raw
    dict."""
    section_cls = schema.model_fields["executive_summary"].annotation
    kwargs = {}
    for key in ANALYTICAL_SECTION_NAMES:
        if key in section_overrides:
            kwargs[key] = section_overrides[key]
        else:
            empty_kwargs = {"content": "", "cited_paper_ids": []}
            if web_urls_used:
                empty_kwargs["cited_web_urls"] = []
            kwargs[key] = section_cls(**empty_kwargs)
    return schema(**kwargs)


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
    schema = _build_report_schema([p.paper_id for p in papers], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation

    # Different sections cite different, non-overlapping subsets;
    # "3333" is never cited anywhere -> must show up as skipped.
    parsed = _analytical_parsed(
        schema,
        thematic_findings=section_cls(content="Findings text.", cited_paper_ids=["1111", "2222"]),
        contradictions_open_debates=section_cls(content="Limitations text.", cited_paper_ids=["1111"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", papers, client=mock_client)

    assert [p.paper_id for p in result["thematic_findings"]["cited_papers"]] == ["1111", "2222"]
    assert [p.paper_id for p in result["contradictions_open_debates"]["cited_papers"]] == ["1111"]
    assert result["future_research_directions"]["cited_papers"] == []
    assert [p.paper_id for p in result["skipped_papers"]] == ["3333"]
    # legacy fields are straight projections of their mapped Analytical section
    assert result["findings"] == result["thematic_findings"]
    assert result["limitations"] == result["contradictions_open_debates"]
    assert result["future_scope"] == result["future_research_directions"]


def test_generate_report_returns_empty_for_no_selected_papers():
    result = generate_report("topic", [], client=MagicMock())
    assert result["findings"] == {"content": "", "cited_papers": [], "reference_numbers": []}
    assert result["limitations"] == {"content": "", "cited_papers": [], "reference_numbers": []}
    assert result["future_scope"] == {"content": "", "cited_papers": [], "reference_numbers": []}
    assert result["skipped_papers"] == []
    assert result["references"] == []


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
    schema = _build_report_schema([p.paper_id for p in papers], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation

    parsed = _analytical_parsed(
        schema,
        **{key: section_cls(content=f"{key} text.", cited_paper_ids=["1111"]) for key in ANALYTICAL_SECTION_NAMES},
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", papers, client=mock_client)

    for section in ANALYTICAL_SECTION_NAMES:
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
    schema = _build_report_schema(["thin-1"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, thematic_findings=section_cls(content="Findings text.", cited_paper_ids=["thin-1"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", [thin_paper], client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "(no abstract available)" in user_content
    assert [p.paper_id for p in result["findings"]["cited_papers"]] == ["thin-1"]


# --- Phase 5d: report regeneration with web sources + citation preservation ---

def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet=f"Snippet for {title}.", published_date=None, source_domain="example.com")


def test_report_schema_with_web_urls_rejects_unknown_url():
    schema = _build_report_schema(["a"], ["https://real.com"])
    section_cls = schema.model_fields["findings"].annotation
    section_cls(content="fine", cited_paper_ids=["a"], cited_web_urls=["https://real.com"])
    try:
        section_cls(content="bad", cited_paper_ids=[], cited_web_urls=["https://not-retrieved.com"])
        assert False, "expected a validation error for an unretrieved url"
    except Exception:
        pass


def test_report_schema_without_web_urls_has_no_cited_web_urls_field():
    schema = _build_report_schema(["a"])
    section_cls = schema.model_fields["findings"].annotation
    assert "cited_web_urls" not in section_cls.model_fields


def test_generate_report_with_web_articles_attaches_cited_web_articles_per_section():
    papers = [_paper("1111", "Paper One")]
    web = _web_article("https://x.com/a", "Article A")
    schema = _build_report_schema(["1111"], ["https://x.com/a"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="f", cited_paper_ids=["1111"], cited_web_urls=["https://x.com/a"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("topic", papers, web_articles=[web], client=mock_client)

    assert [a.url for a in result["findings"]["cited_web_articles"]] == ["https://x.com/a"]
    assert result["contradictions_open_debates"]["cited_web_articles"] == []


def test_generate_report_without_web_articles_omits_cited_web_articles_key():
    """Backward compatibility at the per-section level, not just the
    empty-papers early-return path already covered above -- existing
    callers that never pass web_articles must see byte-identical shape."""
    papers = [_paper("1111", "Paper One")]
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(schema, thematic_findings=section_cls(content="f", cited_paper_ids=["1111"]))
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("topic", papers, client=mock_client)

    assert "cited_web_articles" not in result["findings"]
    assert "cited_web_articles" not in result["thematic_findings"]


def test_regenerate_report_refuses_when_no_existing_report():
    session = PaperPoolSession(topic="q", stage="synthesize", selected_papers=[_paper("1111", "Paper One")], report=None)
    try:
        regenerate_report_with_new_sources(session, client=MagicMock())
        assert False, "expected a ValueError when there's no existing report to regenerate"
    except ValueError as e:
        assert "no existing report" in str(e).lower()


def test_regenerate_report_refuses_when_stage_is_not_synthesize():
    p1 = _paper("1111", "Paper One")
    existing_report = {
        "findings": {"content": "x", "cited_papers": [p1]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(topic="q", stage="curate", selected_papers=[p1], report=existing_report)
    try:
        regenerate_report_with_new_sources(session, client=MagicMock())
        assert False, "expected a ValueError for a session not in the synthesize stage"
    except ValueError as e:
        assert "curate" in str(e)


def test_regenerate_report_preserves_all_original_citations_and_adds_web_citations():
    """Normal-case regeneration: the model (mocked) behaves and honors
    the prompt instruction -- confirms the two layers agree and every
    original citation plus the new web citation survives."""
    p1, p2, p3 = _paper("1111", "Paper One"), _paper("2222", "Paper Two"), _paper("3333", "Paper Three")
    web = _web_article("https://x.com/a", "New Article")

    existing_report = {
        "findings": {"content": "old findings", "cited_papers": [p1, p2]},
        "limitations": {"content": "old limitations", "cited_papers": [p1]},
        "future_scope": {"content": "old future", "cited_papers": []},
        "skipped_papers": [p3],
    }
    session = PaperPoolSession(
        topic="some topic", stage="synthesize",
        selected_papers=[p1, p2, p3], selected_paper_ids=["1111", "2222", "3333"],
        report=existing_report, web_articles_added=[web],
    )

    schema = _build_report_schema(["1111", "2222", "3333"], ["https://x.com/a"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="new findings", cited_paper_ids=["1111", "2222"], cited_web_urls=["https://x.com/a"]),
        contradictions_open_debates=section_cls(content="new limitations", cited_paper_ids=["1111"], cited_web_urls=[]),
        future_research_directions=section_cls(content="new future, now covers paper three", cited_paper_ids=["3333"], cited_web_urls=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    assert [p.paper_id for p in result["findings"]["cited_papers"]] == ["1111", "2222"]
    assert [a.url for a in result["findings"]["cited_web_articles"]] == ["https://x.com/a"]
    assert [p.paper_id for p in result["limitations"]["cited_papers"]] == ["1111"]
    assert [p.paper_id for p in result["future_scope"]["cited_papers"]] == ["3333"]
    assert result["skipped_papers"] == []


def test_regenerate_report_defensive_layer_restores_citation_the_prompt_instruction_failed_to_preserve():
    """The isolation test explicitly requested for Phase 5d: defeats the
    PROMPT layer on purpose by mocking a regeneration that legitimately
    omits "2222" from findings -- exactly as if the model ignored the
    preservation instruction outright, not just imperfectly honored it
    -- to prove the defensive re-append layer alone restores the missing
    citation, independent of whether the prompt instruction was ever
    followed. This is proof the structural safety net works on its own,
    not just proof the two layers happened to agree in the normal case
    covered above."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")

    existing_report = {
        "findings": {"content": "old findings", "cited_papers": [p1, p2]},
        "limitations": {"content": "old limitations", "cited_papers": [p1]},
        "future_scope": {"content": "old future", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="some topic", stage="synthesize",
        selected_papers=[p1, p2], selected_paper_ids=["1111", "2222"],
        report=existing_report, web_articles_added=[],
    )

    schema = _build_report_schema(["1111", "2222"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    # The mocked regeneration DROPS "2222" from findings entirely -- this
    # IS the prompt instruction failing, not a simulation of it: the mock
    # never even sees the real prompt text, so this result is reachable
    # regardless of what the instruction says.
    parsed = _analytical_parsed(
        schema,
        thematic_findings=section_cls(content="new findings, missing a citation", cited_paper_ids=["1111"]),
        contradictions_open_debates=section_cls(content="new limitations", cited_paper_ids=["1111"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    assert [p.paper_id for p in result["findings"]["cited_papers"]] == ["1111", "2222"]
    assert [p.paper_id for p in result["limitations"]["cited_papers"]] == ["1111"]
    assert result["skipped_papers"] == []  # "2222" restored -> no longer skipped


# --- curation-chat-add-to-report Phase 4: regenerate_report_with_approved_web_sources ---

def test_regenerate_with_approved_web_sources_only_reaches_the_model_for_the_passed_in_articles():
    """The core Phase 4 correctness proof: an article sitting in
    session.web_articles_added that ISN'T in the approved_web_articles
    argument must never appear in the prompt sent to the model, even
    though it's real, present session data."""
    p1 = _paper("1111", "Paper One")
    approved = _web_article("https://approved.com", "Approved Article")
    unapproved = _web_article("https://unapproved.com", "Unapproved Article")

    existing_report = {
        "findings": {"content": "old", "cited_papers": [p1]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="some topic", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report,
        # The raw pool has BOTH -- proves the function doesn't fall back
        # to reading this at all, since only `approved` is passed below.
        web_articles_added=[approved, unapproved],
    )

    schema = _build_report_schema(["1111"], ["https://approved.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="new", cited_paper_ids=["1111"], cited_web_urls=["https://approved.com"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [approved], client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "approved.com" in joined
    assert "unapproved.com" not in joined
    assert [a.url for a in result["findings"]["cited_web_articles"]] == ["https://approved.com"]


def test_regenerate_with_approved_web_sources_ignores_web_articles_added_entirely():
    """Even more direct: approved_web_articles shares NO overlap at all
    with session.web_articles_added -- proves the function truly never
    reads session.web_articles_added, not just that it filters it."""
    p1 = _paper("1111", "Paper One")
    pool_only_article = _web_article("https://pool-only.com", "Pool Only")
    approved_article = _web_article("https://approved-elsewhere.com", "Approved Elsewhere")

    existing_report = {
        "findings": {"content": "old", "cited_papers": [p1]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="some topic", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report,
        web_articles_added=[pool_only_article],
    )

    schema = _build_report_schema(["1111"], ["https://approved-elsewhere.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="new", cited_paper_ids=["1111"], cited_web_urls=["https://approved-elsewhere.com"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    regenerate_report_with_approved_web_sources(session, [approved_article], client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "pool-only.com" not in joined
    assert "approved-elsewhere.com" in joined


def test_regenerate_with_approved_web_sources_refuses_when_no_existing_report():
    session = PaperPoolSession(topic="q", stage="synthesize", selected_papers=[_paper("1111", "Paper One")], report=None)
    try:
        regenerate_report_with_approved_web_sources(session, [], client=MagicMock())
        assert False, "expected a ValueError when there's no existing report to regenerate"
    except ValueError as e:
        assert "no existing report" in str(e).lower()


def test_regenerate_with_approved_web_sources_refuses_when_stage_is_not_synthesize():
    p1 = _paper("1111", "Paper One")
    existing_report = {
        "findings": {"content": "x", "cited_papers": [p1]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(topic="q", stage="curate", selected_papers=[p1], report=existing_report)
    try:
        regenerate_report_with_approved_web_sources(session, [], client=MagicMock())
        assert False, "expected a ValueError for a session not in the synthesize stage"
    except ValueError as e:
        assert "curate" in str(e)


# --- report-quality Phase R1: inline numbered citations + References ---

def _sections_out(findings: dict, limitations: dict | None = None, future_scope: dict | None = None) -> dict:
    return {
        "findings": findings,
        "limitations": limitations or {"content": "", "cited_papers": []},
        "future_scope": future_scope or {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }


# --- report-quality Phase R3.2 Chunk 2: public wrapper reuse ---

def test_build_references_and_renumber_public_wrapper_matches_private_helper_exactly():
    """build_references_and_renumber is a pure pass-through to
    _build_references_and_renumber -- same input, same output, proving
    the wrapper adds no behavior of its own for chat (or any other
    future caller) to accidentally diverge from."""
    p1, p2 = _paper("p1", "Paper One"), _paper("p2", "Paper Two")
    web = _web_article("https://w.com", "Web One")

    def _sections():
        return _sections_out(
            findings={
                "content": "Per [Paper 1] and [Paper 2], X. Also [Web 1].",
                "cited_papers": [p1, p2], "cited_web_articles": [web],
            },
        )

    via_private = _build_references_and_renumber(_sections())
    via_public = build_references_and_renumber(_sections())

    assert via_public == via_private
    assert via_public["findings"]["content"] == "Per [1] and [2], X. Also [3]."


def test_build_references_and_renumber_public_wrapper_accepts_arbitrary_section_names():
    """Not just a report-shaped call -- proves the wrapper works over an
    arbitrary section_names tuple the way derive_chat_references will
    call it (keyed by exchange_id, not ANALYTICAL_SECTION_NAMES)."""
    p1 = _paper("p1", "Paper One")
    sections_out = {
        "exchange-a": {"content": "Per [Paper 1].", "cited_papers": [p1], "cited_web_articles": []},
        "exchange-b": {"content": "No citation here.", "cited_papers": [], "cited_web_articles": []},
    }

    result = build_references_and_renumber(sections_out, ("exchange-a", "exchange-b"))

    assert result["exchange-a"]["content"] == "Per [1]."
    assert len(result["references"]) == 1


def test_build_references_and_renumber_converts_section_local_markers_to_one_global_sequence():
    p1, p2 = _paper("p1", "Paper One"), _paper("p2", "Paper Two")
    web = _web_article("https://w.com", "Web One")
    sections_out = _sections_out(
        findings={
            "content": "Per [Paper 1] and [Paper 2], X. Also [Web 1].",
            "cited_papers": [p1, p2], "cited_web_articles": [web],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Per [1] and [2], X. Also [3]."
    assert result["findings"]["reference_numbers"] == [1, 2, 3]
    assert [r["number"] for r in result["references"]] == [1, 2, 3]


def test_build_references_and_renumber_same_source_cited_in_multiple_sections_keeps_one_number():
    p1 = _paper("p1", "Paper One")
    sections_out = _sections_out(
        findings={"content": "Per [Paper 1], X.", "cited_papers": [p1]},
        limitations={"content": "Also [Paper 1] shows Y.", "cited_papers": [p1]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Per [1], X."
    assert result["limitations"]["content"] == "Also [1] shows Y."
    assert len(result["references"]) == 1
    assert result["references"][0]["number"] == 1
    assert result["findings"]["reference_numbers"] == [1]
    assert result["limitations"]["reference_numbers"] == [1]


def test_build_references_and_renumber_strips_out_of_range_marker():
    p1 = _paper("p1", "Paper One")
    sections_out = _sections_out(
        findings={
            "content": "Per [Paper 1], X. But [Paper 9] is invented.",
            "cited_papers": [p1],
        },
    )

    result = _build_references_and_renumber(sections_out)

    # report-quality Phase R2B.1: the stripped marker's now-orphaned
    # double space is collapsed by the deterministic whitespace cleanup.
    assert result["findings"]["content"] == "Per [1], X. But is invented."
    assert len(result["references"]) == 1  # the invented one never got a reference


def test_build_references_and_renumber_structurally_cited_but_unmarked_source_appears_in_references():
    p1, p2 = _paper("p1", "Paper One"), _paper("p2", "Paper Two")
    sections_out = _sections_out(
        findings={"content": "Only [Paper 1] is marked.", "cited_papers": [p1, p2]},
    )

    result = _build_references_and_renumber(sections_out)

    # p2 was structurally cited (in cited_papers) but never bracketed --
    # still gets a trailing reference, not silently dropped.
    assert result["findings"]["content"] == "Only [1] is marked."
    assert len(result["references"]) == 2
    assert {r["paper_id"] for r in result["references"]} == {"p1", "p2"}
    assert result["findings"]["reference_numbers"] == [1, 2]


def test_build_references_and_renumber_includes_paper_and_web_entries():
    p1 = _paper("p1", "Paper One")
    web = _web_article("https://w.com", "Web One")
    sections_out = _sections_out(
        findings={
            "content": "Per [Paper 1] and [Web 1].",
            "cited_papers": [p1], "cited_web_articles": [web],
        },
    )

    result = _build_references_and_renumber(sections_out)

    kinds = {r["kind"] for r in result["references"]}
    assert kinds == {"paper", "web"}


# --- report-quality Phase R2C: grouped citation-marker parsing fix ---
# The model sometimes bundles more than one citation into a single
# bracket (e.g. "[Paper 6, Paper 8]") instead of one marker per
# citation as prompted -- these must still be converted to global
# numbers, not leak through as raw, unresolved text.

def test_build_references_and_renumber_converts_grouped_paper_marker_to_separate_global_markers():
    p1, p2 = _paper("p1", "Paper One"), _paper("p2", "Paper Two")
    sections_out = _sections_out(
        findings={"content": "See [Paper 1, Paper 2] for details.", "cited_papers": [p1, p2]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "See [1][2] for details."
    assert result["findings"]["reference_numbers"] == [1, 2]
    assert len(result["references"]) == 2
    assert "Paper" not in result["findings"]["content"]


def test_build_references_and_renumber_converts_grouped_web_marker_to_separate_global_markers():
    web1, web2 = _web_article("https://w1.com", "Web One"), _web_article("https://w2.com", "Web Two")
    sections_out = _sections_out(
        findings={
            "content": "See [Web 1, Web 2] for coverage.",
            "cited_papers": [], "cited_web_articles": [web1, web2],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "See [1][2] for coverage."
    assert len(result["references"]) == 2
    assert {r["kind"] for r in result["references"]} == {"web"}
    assert "Web" not in result["findings"]["content"]


def test_build_references_and_renumber_converts_mixed_paper_and_web_grouped_marker():
    p1 = _paper("p1", "Paper One")
    web = _web_article("https://w.com", "Web One")
    sections_out = _sections_out(
        findings={
            "content": "See [Paper 1, Web 1] together.",
            "cited_papers": [p1], "cited_web_articles": [web],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "See [1][2] together."
    kinds_by_number = {r["number"]: r["kind"] for r in result["references"]}
    assert kinds_by_number == {1: "paper", 2: "web"}


def test_build_references_and_renumber_grouped_marker_source_reused_elsewhere_keeps_same_number():
    p1, p2 = _paper("p1", "Paper One"), _paper("p2", "Paper Two")
    sections_out = _sections_out(
        findings={"content": "First [Paper 1] alone.", "cited_papers": [p1]},
        limitations={"content": "Then [Paper 1, Paper 2] together.", "cited_papers": [p1, p2]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "First [1] alone."
    assert result["limitations"]["content"] == "Then [1][2] together."
    assert len(result["references"]) == 2


def test_build_references_and_renumber_grouped_marker_drops_only_the_invalid_entry():
    """Only one paper is actually cited in this section -- the "[Paper
    9]" entry inside the group is out of range and must be dropped on
    its own, leaving the valid "[Paper 1]" entry resolved normally
    rather than stripping the whole bracket."""
    p1 = _paper("p1", "Paper One")
    sections_out = _sections_out(
        findings={"content": "See [Paper 1, Paper 9] here.", "cited_papers": [p1]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "See [1] here."
    assert len(result["references"]) == 1
    assert "Paper" not in result["findings"]["content"]


def test_build_references_and_renumber_grouped_marker_all_entries_invalid_leaves_no_raw_text():
    sections_out = _sections_out(
        findings={"content": "See [Paper 1, Paper 2] here.", "cited_papers": []},
    )

    result = _build_references_and_renumber(sections_out)

    # report-quality Phase R2B.1: whitespace cleanup collapses the
    # stripped group's now-orphaned double space.
    assert result["findings"]["content"] == "See here."
    assert result["references"] == []
    assert "Paper" not in result["findings"]["content"]


# --- report-quality Phase R2C: raw-source-id citation-hardening fix ---
# Root cause: the model sometimes ignores the instructed [Paper N]/
# [Web N] marker format and cites a source using its own real
# identifier instead (observed in Foundational-template output) -- e.g.
# "[2308.06821v1]" (arXiv id) or "[abd1c342495432171beb7ca8fd9551ef13cbd0ff]"
# (a Semantic-Scholar-style hash id), which the old parser had no way to
# recognize at all, leaking raw, unresolved identifier text straight
# into the rendered report.

def test_build_references_and_renumber_converts_arxiv_style_raw_paper_id_marker():
    p1 = _paper("2308.06821v1", "Paper One")
    sections_out = _sections_out(
        findings={"content": "Per [2308.06821v1], X is shown.", "cited_papers": [p1]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Per [1], X is shown."
    assert len(result["references"]) == 1
    assert result["references"][0]["paper_id"] == "2308.06821v1"


def test_build_references_and_renumber_converts_hash_style_raw_paper_id_marker():
    p1 = _paper("abd1c342495432171beb7ca8fd9551ef13cbd0ff", "Paper One")
    sections_out = _sections_out(
        findings={
            "content": "Per [abd1c342495432171beb7ca8fd9551ef13cbd0ff], X is shown.",
            "cited_papers": [p1],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Per [1], X is shown."
    assert len(result["references"]) == 1


def test_build_references_and_renumber_raw_paper_id_marker_matches_a_web_url_too():
    """The same backstop, applied to a raw web article url exact-matched
    against that section's own cited_web_articles -- the minimum-risk
    extension the fix's own investigation covers, not just paper_ids."""
    web = _web_article("https://example.com/survey", "A Survey")
    sections_out = _sections_out(
        findings={
            "content": "Per [https://example.com/survey], X is shown.",
            "cited_papers": [], "cited_web_articles": [web],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Per [1], X is shown."
    assert result["references"][0]["kind"] == "web"


def test_build_references_and_renumber_raw_paper_id_and_paper_n_marker_for_same_source_keep_one_number():
    p1, p2 = _paper("2308.06821v1", "Paper One"), _paper("2222", "Paper Two")
    sections_out = _sections_out(
        findings={
            "content": "First [Paper 1] then also [2308.06821v1], and separately [Paper 2].",
            "cited_papers": [p1, p2],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "First [1] then also [1], and separately [2]."
    assert len(result["references"]) == 2


def test_build_references_and_renumber_unrecognized_raw_id_marker_does_not_leak_into_content():
    p1 = _paper("2308.06821v1", "Paper One")
    sections_out = _sections_out(
        findings={"content": "Per [2308.06821v1] and [totally-unknown-id], X.", "cited_papers": [p1]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Per [1] and, X."
    assert "totally-unknown-id" not in result["findings"]["content"]
    assert len(result["references"]) == 1


def test_build_references_and_renumber_raw_id_backstop_never_misreads_a_bare_digit_bracket():
    """Safety guard: a bare digit string inside brackets (the shape of
    an already-final [N] marker) is never treated as a raw-id candidate,
    even if -- purely hypothetically -- a paper_id happened to be a
    small integer string. Regular [Paper N] markers remain the only way
    to cite by position; this backstop only ever fires on an exact,
    non-numeric identifier match."""
    p1 = _paper("1", "Paper One")
    sections_out = _sections_out(
        findings={"content": "See [1] here.", "cited_papers": []},
    )

    result = _build_references_and_renumber(sections_out)

    # [1] is left completely untouched by the raw-id backstop (it's
    # digit-only) -- and since it's not a valid [Paper N]/[Web N] marker
    # either, the regular pass also leaves it alone (no cited_papers to
    # even attempt a match against).
    assert result["findings"]["content"] == "See [1] here."
    assert result["references"] == []


def test_build_references_and_renumber_existing_single_and_grouped_paper_n_markers_still_work():
    """Non-regression: the raw-id backstop must not interfere with
    ordinary [Paper N]/grouped [Paper N, Paper M] resolution at all."""
    p1, p2 = _paper("p1", "Paper One"), _paper("p2", "Paper Two")
    sections_out = _sections_out(
        findings={"content": "Solo [Paper 1]. Grouped [Paper 1, Paper 2].", "cited_papers": [p1, p2]},
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Solo [1]. Grouped [1][2]."
    assert len(result["references"]) == 2


def test_generate_report_end_to_end_raw_paper_id_marker_never_leaks_for_any_template():
    """Proves the wiring for all three templates, not just the isolated
    post-processing function -- a mocked model response using a raw
    paper_id marker (the actual observed Foundational bug) comes back
    through the real generate_report() fully resolved regardless of
    which template generated it."""
    for template in _ALL_TEMPLATES:
        p1 = _paper("2308.06821v1", "Paper One")
        schema = _build_report_schema(["2308.06821v1"], None, REPORT_TEMPLATES[template])
        section_cls = schema.model_fields["executive_summary"].annotation
        parsed = _analytical_parsed(
            schema,
            thematic_findings=section_cls(
                content="Per [2308.06821v1], strong results.", cited_paper_ids=["2308.06821v1"],
            ),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

        result = generate_report("some topic", [p1], client=mock_client, report_template=template)

        assert result["thematic_findings"]["content"] == "Per [1], strong results."
        assert "2308.06821v1" not in result["thematic_findings"]["content"]
        assert len(result["references"]) == 1


# --- report-quality Phase R2B.1: deterministic whitespace cleanup ---

def test_cleanup_marker_stripped_whitespace_collapses_repeated_spaces():
    assert _cleanup_marker_stripped_whitespace("But  is invented.") == "But is invented."
    assert _cleanup_marker_stripped_whitespace("See   here.") == "See here."


def test_cleanup_marker_stripped_whitespace_removes_space_before_punctuation():
    assert _cleanup_marker_stripped_whitespace("classification ,") == "classification,"
    assert _cleanup_marker_stripped_whitespace("training .") == "training."
    assert _cleanup_marker_stripped_whitespace("a claim ; another") == "a claim; another"
    assert _cleanup_marker_stripped_whitespace("a question ?") == "a question?"


def test_cleanup_marker_stripped_whitespace_is_a_no_op_on_already_clean_text():
    clean = "Per [1] and [2], X. Also [3] shows Y -- Z's own claim, still fine."
    assert _cleanup_marker_stripped_whitespace(clean) == clean


def test_build_references_and_renumber_end_to_end_strips_marker_before_punctuation_cleanly():
    """Root-cause repro for the observed "classification ," / "training
    ." bug: an out-of-range marker sitting directly before punctuation
    must not leave an orphaned space once stripped, through the real
    _build_references_and_renumber path (not just the isolated helper
    above)."""
    # No papers actually cited -- both distinct raw markers below (which
    # densify still numbers 1 and 2, since densify only cares about
    # DISTINCT raw values, not validity) are out of range and stripped.
    sections_out = _sections_out(
        findings={
            "content": "Strong results for classification [Paper 1], and training [Paper 2].",
            "cited_papers": [],
        },
    )

    result = _build_references_and_renumber(sections_out)

    assert result["findings"]["content"] == "Strong results for classification, and training."
    assert " ," not in result["findings"]["content"]
    assert " ." not in result["findings"]["content"]


def test_generate_report_end_to_end_converts_grouped_model_markers_to_global_numbers():
    """Proves the wiring, not just the isolated post-processing function
    -- a mocked model response using a grouped section-local marker
    comes back through the real generate_report() fully resolved."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    schema = _build_report_schema(["1111", "2222"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema,
        thematic_findings=section_cls(
            content="Both approaches agree [Paper 1, Paper 2].", cited_paper_ids=["1111", "2222"],
        ),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", [p1, p2], client=mock_client)

    assert result["thematic_findings"]["content"] == "Both approaches agree [1][2]."
    assert len(result["references"]) == 2


def test_reference_entry_link_url_prefers_doi_then_falls_back_to_paper_url_and_uses_article_url_for_web():
    with_doi = Paper(
        title="Has DOI", authors=["A"], year=2024, venue="X", abstract="a",
        url="http://arxiv.org/abs/withdoi", doi="10.1234/xyz", citation_count=None,
        source="arxiv", paper_id="withdoi",
    )
    without_doi = _paper("nodoi", "No DOI")
    web = _web_article("https://w.com", "Web One")
    sections_out = _sections_out(
        findings={
            "content": "Per [Paper 1], [Paper 2], and [Web 1].",
            "cited_papers": [with_doi, without_doi], "cited_web_articles": [web],
        },
    )

    result = _build_references_and_renumber(sections_out)

    by_id = {r.get("paper_id") or r.get("url"): r for r in result["references"]}
    assert by_id["withdoi"]["link_url"] == "https://doi.org/10.1234/xyz"
    assert by_id["nodoi"]["link_url"] == without_doi.url
    assert by_id["https://w.com"]["link_url"] == "https://w.com"


def test_derive_legacy_references_does_not_rewrite_content_and_builds_references():
    """The backward-compat path: an old, pre-R1 report dict has no
    inline markers at all in its content -- derive_legacy_references
    must never invent any, only build the References list."""
    p1 = _paper("p1", "Paper One")
    old_report = {
        "findings": {"content": "Old prose with no markers at all.", "cited_papers": [p1]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }

    result = derive_legacy_references(old_report)

    assert result["findings"]["content"] == "Old prose with no markers at all."
    assert result["findings"]["reference_numbers"] == [1]
    assert len(result["references"]) == 1
    assert result["references"][0]["paper_id"] == "p1"


# --- report-quality Phase R2A: dynamic section model (schema/rendering
# migration only, no generation change) ---

def test_derive_sections_from_legacy_report_maps_the_three_fixed_sections_in_order():
    report = {
        "findings": {"content": "F content", "cited_papers": [], "reference_numbers": [1]},
        "limitations": {"content": "L content", "cited_papers": [], "reference_numbers": []},
        "future_scope": {"content": "S content", "cited_papers": [], "reference_numbers": [2]},
        "skipped_papers": [],
    }

    sections = derive_sections_from_legacy_report(report)

    assert [s["key"] for s in sections] == ["findings", "limitations", "future_scope"]
    assert [s["title"] for s in sections] == ["Findings", "Limitations", "Future Scope"]
    assert [s["content"] for s in sections] == ["F content", "L content", "S content"]
    assert sections[0]["reference_numbers"] == [1]
    assert sections[2]["reference_numbers"] == [2]


def test_derive_sections_from_legacy_report_defaults_missing_reference_numbers_to_empty():
    """A report predating even R1's reference_numbers field (older than
    the references field itself) still derives a sections list safely --
    reference_numbers degrades to [] rather than a KeyError."""
    report = {
        "findings": {"content": "F", "cited_papers": []},
        "limitations": {"content": "L", "cited_papers": []},
        "future_scope": {"content": "S", "cited_papers": []},
        "skipped_papers": [],
    }

    sections = derive_sections_from_legacy_report(report)

    assert all(s["reference_numbers"] == [] for s in sections)


def test_generate_report_end_to_end_converts_section_local_model_markers_to_global_numbers():
    """Proves the wiring, not just the isolated post-processing function:
    a mocked model response using section-local [Paper N]/[Web N]
    markers comes back through the real generate_report() with global
    [N] markers and a populated references list."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    web = _web_article("https://x.com/a", "Article A")
    schema = _build_report_schema(["1111", "2222"], ["https://x.com/a"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(
            content="Per [Paper 1] and [Web 1], X.", cited_paper_ids=["1111"], cited_web_urls=["https://x.com/a"],
        ),
        contradictions_open_debates=section_cls(content="Also [Paper 1] shows Y.", cited_paper_ids=["1111"], cited_web_urls=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("some topic", [p1, p2], web_articles=[web], client=mock_client)

    assert result["findings"]["content"] == "Per [1] and [2], X."
    assert result["limitations"]["content"] == "Also [1] shows Y."
    assert len(result["references"]) == 2
    assert result["references"][0]["kind"] == "paper"
    assert result["references"][1]["kind"] == "web"


# --- report-quality Phase R2B: Analytical dynamic section generation ---

def test_analytical_report_schema_requires_all_eight_sections():
    """Named required pydantic fields (not a Literal-constrained list)
    make a missing section structurally impossible -- pydantic rejects
    an incomplete section set outright rather than silently defaulting."""
    schema = _build_report_schema(["a"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    section = section_cls(content="x", cited_paper_ids=[])
    incomplete_kwargs = {key: section for key in ANALYTICAL_SECTION_NAMES if key != "conclusion"}
    try:
        schema(**incomplete_kwargs)
        assert False, "expected a validation error for a missing required section"
    except Exception:
        pass


def test_generate_report_produces_sections_list_with_all_eight_analytical_keys_in_order():
    papers = [_paper("1111", "Paper One")]
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("topic", papers, client=mock_client)

    assert [s["key"] for s in result["sections"]] == list(ANALYTICAL_SECTION_NAMES)
    assert [s["title"] for s in result["sections"]] == [d["title"] for d in REPORT_SECTION_DEFINITIONS]


def test_generate_report_for_empty_selection_still_produces_all_eight_sections():
    result = generate_report("topic", [], client=MagicMock())

    assert [s["key"] for s in result["sections"]] == list(ANALYTICAL_SECTION_NAMES)
    assert all(s["content"] == "" for s in result["sections"])
    assert result["findings"] == {"content": "", "cited_papers": [], "reference_numbers": []}
    assert result["limitations"] == {"content": "", "cited_papers": [], "reference_numbers": []}
    assert result["future_scope"] == {"content": "", "cited_papers": [], "reference_numbers": []}
    assert result["skipped_papers"] == []
    assert result["references"] == []


def test_legacy_fields_are_projections_of_their_mapped_analytical_section_not_independent_text():
    papers = [_paper("1111", "Paper One")]
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema,
        thematic_findings=section_cls(content="thematic findings text", cited_paper_ids=["1111"]),
        contradictions_open_debates=section_cls(content="contradictions text", cited_paper_ids=[]),
        future_research_directions=section_cls(content="future directions text", cited_paper_ids=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("topic", papers, client=mock_client)

    assert result["findings"]["content"] == "thematic findings text"
    assert result["limitations"]["content"] == "contradictions text"
    assert result["future_scope"]["content"] == "future directions text"


def test_build_references_and_renumber_works_across_all_eight_analytical_sections():
    """Same global-renumbering guarantee as the 3-section tests above,
    proven again over the full 8-section ANALYTICAL_SECTION_NAMES set --
    a source cited in the first and last section still keeps one
    number."""
    p1 = _paper("p1", "Paper One")
    sections_out = {key: {"content": "", "cited_papers": []} for key in ANALYTICAL_SECTION_NAMES}
    sections_out["executive_summary"] = {"content": "Per [Paper 1], X.", "cited_papers": [p1]}
    sections_out["conclusion"] = {"content": "Again, [Paper 1] shows Y.", "cited_papers": [p1]}
    sections_out["skipped_papers"] = []

    result = _build_references_and_renumber(sections_out, ANALYTICAL_SECTION_NAMES)

    assert result["executive_summary"]["content"] == "Per [1], X."
    assert result["conclusion"]["content"] == "Again, [1] shows Y."
    assert len(result["references"]) == 1
    assert result["references"][0]["number"] == 1


def test_generate_report_for_session_produces_all_eight_analytical_sections():
    p1 = _paper("1111", "Paper One")
    session = PaperPoolSession(topic="q", stage="synthesize", selected_papers=[p1])
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report_for_session(session, client=mock_client)

    assert set(s["key"] for s in result["sections"]) == set(ANALYTICAL_SECTION_NAMES)


def test_regenerate_report_cross_version_maps_legacy_priors_and_never_crashes_on_new_only_sections():
    """report-quality Phase R2B: a session's existing report may still
    predate R2B entirely (only has findings/limitations/future_scope,
    never thematic_findings/contradictions_open_debates/future_research_
    directions at all) -- regenerating it under the new 8-section schema
    must not KeyError, and must resolve prior citations for the three
    mapped sections from their legacy counterpart, while every other new
    section (no legacy analogue at all) simply has nothing to restore."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    existing_report = {
        "findings": {"content": "old", "cited_papers": [p1, p2]},
        "limitations": {"content": "old", "cited_papers": []},
        "future_scope": {"content": "old", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_papers=[p1, p2], selected_paper_ids=["1111", "2222"],
        report=existing_report, web_articles_added=[],
    )

    schema = _build_report_schema(["1111", "2222"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    # The mocked regeneration drops BOTH prior citations from
    # thematic_findings entirely, and writes nothing for the brand-new
    # gap_analysis section -- neither should crash or lose data.
    parsed = _analytical_parsed(schema, thematic_findings=section_cls(content="new", cited_paper_ids=[]))
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    assert [p.paper_id for p in result["thematic_findings"]["cited_papers"]] == ["1111", "2222"]
    assert result["gap_analysis"]["cited_papers"] == []
    assert set(s["key"] for s in result["sections"]) == set(ANALYTICAL_SECTION_NAMES)


# --- report-quality Phase R2D: revoked web citation must not resurrect ---
#
# session.revoked_web_article_urls (query_expansion.py) is the persistent
# record curation_chat.py's delete_chat_exchanges/edit_chat_exchange
# populate whenever a web source loses its only live chat backing --
# these tests exercise report.py's OWN consumption of that field
# (regenerate_report_with_new_sources/regenerate_report_with_approved_
# web_sources excluding anything in it from the model's candidates), not
# how the field itself gets populated -- see tests/test_curation_chat.py
# for that half.

def test_regenerate_report_with_new_sources_excludes_revoked_web_citation_from_candidates():
    """Root-cause repro: a web article marked revoked (session.revoked_
    web_article_urls) must not be offered to the model as a citable
    candidate on the next whole-pool regeneration -- even though
    session.web_articles_added (the raw, unfiltered discovery pool)
    still contains it, since it's deliberately never pruned. Checked at
    the prompt level (what's actually sent to the model), not just the
    parsed response, since the leak is in what's OFFERED as a candidate."""
    p1 = _paper("1111", "Paper One")
    revoked = _web_article("https://revoked.com", "Revoked Article")

    existing_report = {
        "thematic_findings": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [],
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report,
        web_articles_added=[revoked],  # Phase B never prunes this raw pool
        revoked_web_article_urls={"https://revoked.com"},
    )

    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "revoked.com" not in joined
    assert "Revoked Article" not in joined
    assert "cited_web_articles" not in result["thematic_findings"]
    assert result["references"] == []


def test_regenerate_report_with_new_sources_still_offers_a_non_revoked_web_article():
    """The counterpart to the revocation test above: a web article NOT
    in session.revoked_web_article_urls must still be offered as a
    candidate -- the fix must not become so aggressive that it excludes
    every web source, only ones actually marked revoked."""
    p1 = _paper("1111", "Paper One")
    still_live = _web_article("https://still-live.com", "Still Live Article")

    existing_report = {
        "thematic_findings": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [],
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report,
        web_articles_added=[still_live],
        revoked_web_article_urls=set(),
    )

    schema = _build_report_schema(["1111"], ["https://still-live.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(
            content="Per [Web 1].", cited_paper_ids=[], cited_web_urls=["https://still-live.com"],
        ),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "still-live.com" in joined
    assert [a.url for a in result["thematic_findings"]["cited_web_articles"]] == ["https://still-live.com"]


def test_regenerate_report_with_new_sources_repeated_regeneration_does_not_resurrect_revoked_citation():
    """Requirement: the fix must hold across repeated regenerations, not
    just the first one. session.revoked_web_article_urls is a PERSISTENT
    session-level record -- unlike inferring revocation from the prior
    report's own references (which would self-heal to "nothing to
    revoke" the moment the revoked source drops out of the report,
    letting a LATER regeneration re-offer it) -- so it stays excluded
    across arbitrarily many regeneration cycles, not just the first."""
    p1 = _paper("1111", "Paper One")
    revoked = _web_article("https://revoked.com", "Revoked Article")

    existing_report = {
        "thematic_findings": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [],
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[revoked],
        revoked_web_article_urls={"https://revoked.com"},
    )

    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    first = regenerate_report_with_new_sources(session, client=mock_client)
    assert first["references"] == []

    # Second regeneration starts from the report the FIRST call produced
    # -- revoked_web_article_urls itself is untouched by regeneration
    # (only curation_chat.py's delete/edit/accept flows ever change it),
    # so the exclusion holds again here with no extra bookkeeping needed.
    session.report = first
    second = regenerate_report_with_new_sources(session, client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "revoked.com" not in joined
    assert second["references"] == []


def test_regenerate_report_with_approved_web_sources_never_offers_a_url_outside_the_approved_list():
    """The selective path's own scoping: only approved_web_articles
    actually passed into this call are ever offered as candidates -- a
    web reference the PRIOR report cited that isn't in THIS call's
    approved list is excluded, matching curation_chat.py's own resolve_
    approved_web_articles_for_regeneration filtering upstream (this is a
    defense-in-depth confirmation, not a replacement for it)."""
    p1 = _paper("1111", "Paper One")
    approved = _web_article("https://approved.com", "Approved Article")
    no_longer_approved = _web_article("https://no-longer-approved.com", "No Longer Approved")

    existing_report = {
        "thematic_findings": {
            "content": "Per [1] and [2].", "cited_papers": [], "cited_web_articles": [approved, no_longer_approved],
            "reference_numbers": [1, 2],
        },
        "references": [
            {"number": 1, "kind": "web", "paper_id": None, "url": "https://approved.com", "title": "Approved Article", "formatted": "x", "link_url": "https://approved.com"},
            {"number": 2, "kind": "web", "paper_id": None, "url": "https://no-longer-approved.com", "title": "No Longer Approved", "formatted": "x", "link_url": "https://no-longer-approved.com"},
        ],
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[approved, no_longer_approved],
    )

    schema = _build_report_schema(["1111"], ["https://approved.com"], REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema, web_urls_used=True)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [approved], client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "no-longer-approved.com" not in joined
    assert "approved.com" in joined


def test_regenerate_report_with_new_sources_old_report_without_references_key_stays_compatible():
    """Old reports without references/sections remain compatible: an
    existing_report predating R1 (no "references" key at all), combined
    with a session that predates this fix entirely (revoked_web_article_
    urls defaults to an empty set), must not crash and must not
    spuriously exclude a normal, never-revoked candidate."""
    p1 = _paper("1111", "Paper One")
    web = _web_article("https://x.com", "Article X")
    existing_report = {
        "findings": {"content": "old", "cited_papers": []},
        "limitations": {"content": "old", "cited_papers": []},
        "future_scope": {"content": "old", "cited_papers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[web],
    )

    schema = _build_report_schema(["1111"], ["https://x.com"], REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema, web_urls_used=True)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "x.com" in joined


# --- report-quality Phase R2C: report templates (Foundational/Analytical/Expert) ---

_ALL_TEMPLATES = ("foundational", "analytical", "expert")


def test_report_templates_all_share_the_same_eight_keys_and_titles():
    """The deliberate, low-risk design constraint this whole phase rests
    on: no template ever introduces/renames/reorders a section, only
    varies its own description/word-budget text."""
    reference = [(d["key"], d["title"]) for d in REPORT_SECTION_DEFINITIONS]
    for template in _ALL_TEMPLATES:
        assert [(d["key"], d["title"]) for d in REPORT_TEMPLATES[template]] == reference


def test_build_report_system_prompt_omitted_template_matches_explicit_analytical():
    """Non-regression guard: Analytical's generated prompt must be
    byte-identical whether template is omitted or passed explicitly --
    _TEMPLATE_DEPTH_GUIDANCE maps "analytical" to "", so nothing is
    appended in either case."""
    omitted = _build_report_system_prompt(REPORT_SECTION_DEFINITIONS)
    explicit = _build_report_system_prompt(REPORT_SECTION_DEFINITIONS, "analytical")
    assert omitted == explicit


def test_generate_report_omitted_template_defaults_to_analytical():
    papers = [_paper("1111", "Paper One")]
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = generate_report("topic", papers, client=mock_client)

    assert result["report_template"] == "analytical"


def test_generate_report_stamps_report_template_and_produces_all_eight_sections_for_every_template():
    for template in _ALL_TEMPLATES:
        papers = [_paper("1111", "Paper One")]
        schema = _build_report_schema(["1111"], None, REPORT_TEMPLATES[template])
        parsed = _analytical_parsed(schema)
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

        result = generate_report("topic", papers, client=mock_client, report_template=template)

        assert result["report_template"] == template
        assert set(s["key"] for s in result["sections"]) == set(ANALYTICAL_SECTION_NAMES)
        assert len(result["sections"]) == 8


def test_generate_report_empty_selection_stamps_report_template_for_every_template():
    for template in _ALL_TEMPLATES:
        result = generate_report("topic", [], client=MagicMock(), report_template=template)
        assert result["report_template"] == template
        assert len(result["sections"]) == 8


def test_legacy_field_projection_works_for_every_template():
    for template in _ALL_TEMPLATES:
        papers = [_paper("1111", "Paper One")]
        schema = _build_report_schema(["1111"], None, REPORT_TEMPLATES[template])
        section_cls = schema.model_fields["executive_summary"].annotation
        parsed = _analytical_parsed(
            schema, thematic_findings=section_cls(content=f"{template} findings", cited_paper_ids=["1111"]),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

        result = generate_report("topic", papers, client=mock_client, report_template=template)

        assert result["findings"]["content"] == f"{template} findings"
        assert result["findings"] == result["thematic_findings"]


def test_regenerate_report_with_new_sources_omitted_template_preserves_existing():
    p1 = _paper("1111", "Paper One")
    existing_report = {
        "thematic_findings": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [], "report_template": "expert",
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[],
    )
    schema = _build_report_schema(["1111"], None, REPORT_TEMPLATES["expert"])
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    assert result["report_template"] == "expert"


def test_regenerate_report_with_new_sources_explicit_template_switches():
    p1 = _paper("1111", "Paper One")
    existing_report = {
        "thematic_findings": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [], "report_template": "analytical",
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[],
    )
    schema = _build_report_schema(["1111"], None, REPORT_TEMPLATES["foundational"])
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client, report_template="foundational")

    assert result["report_template"] == "foundational"


def test_regenerate_report_with_approved_web_sources_never_accepts_an_explicit_template_from_its_caller():
    """report-quality Phase R2C decision 8: chat's add-to-report flow
    never passes report_template, so this always preserves the existing
    report's template -- proven here by simply never passing it, same
    as curation_chat_service.py's real call site."""
    p1 = _paper("1111", "Paper One")
    approved = _web_article("https://a.com", "A")
    existing_report = {
        "thematic_findings": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [], "report_template": "foundational",
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[approved],
    )
    schema = _build_report_schema(["1111"], ["https://a.com"], REPORT_TEMPLATES["foundational"])
    parsed = _analytical_parsed(schema, web_urls_used=True)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [approved], client=mock_client)

    assert result["report_template"] == "foundational"


def test_regenerate_report_with_new_sources_old_report_without_report_template_defaults_to_analytical():
    p1 = _paper("1111", "Paper One")
    existing_report = {
        "findings": {"content": "old", "cited_papers": []},
        "limitations": {"content": "old", "cited_papers": []},
        "future_scope": {"content": "old", "cited_papers": []},
        "skipped_papers": [],
    }  # no "report_template" key at all -- predates R2C
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[],
    )
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    assert result["report_template"] == "analytical"


# --- report-quality Phase R2D/R3.1 (superseded by R3.1b below): approved
# web sources must survive regeneration ---
# Original root cause: paper citations get force-restored across a
# regeneration (_restore_dropped_citations), but web citations had no
# equivalent -- a web source approved into the report via "Add to
# report" would silently vanish from References if the model simply
# never chose to cite it. R2D/R3.1 fixed that with a force-include/
# restore guarantee -- which then produced its OWN bug (R3.1b below):
# a References entry with no inline marker anywhere in the body reads
# as a broken/decorative reference to an actual reader. R3.1b removed
# that guarantee; the tests below reflect the corrected behavior.

# --- report-quality Phase R3.1b: no orphan References entries for
# approved web sources ---
# Product rule: a web reference should not appear in References unless
# at least one inline [N] marker points to it in the report body. The
# CURRENT round's own model output (cited_web_urls) is the sole source
# of truth for which web sources a section cites -- no restoration, no
# force-inclusion, regardless of approval status or prior-report history.

def test_regenerate_report_with_approved_web_sources_never_cited_url_does_not_appear_as_orphan():
    """The core R3.1b fix: an approved web URL the model doesn't cite in
    any section this round must not appear in References at all."""
    p1 = _paper("1111", "Paper One")
    web = _web_article("https://new-source.com", "New Source")
    existing_report = {
        "thematic_findings": {"content": "old", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"], report=existing_report,
    )
    schema = _build_report_schema(["1111"], ["https://new-source.com"], REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema, web_urls_used=True)  # model cites nothing anywhere
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [web], client=mock_client)

    assert result["references"] == []
    assert result["thematic_findings"]["cited_web_articles"] == []


def test_regenerate_report_with_approved_web_sources_omitted_url_stays_absent_alongside_a_cited_one():
    """An approved-but-uncited URL stays absent even when a DIFFERENT
    web source IS genuinely cited that same round -- no deterministic
    fallback-section injection anymore."""
    p1 = _paper("1111", "Paper One")
    cited = _web_article("https://cited.com", "Cited")
    never_cited = _web_article("https://never-cited.com", "Never Cited")
    existing_report = {
        "gap_analysis": {"content": "old", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"], report=existing_report,
    )
    schema = _build_report_schema(["1111"], ["https://cited.com", "https://never-cited.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        gap_analysis=section_cls(content="Cites one [Web 1].", cited_paper_ids=[], cited_web_urls=["https://cited.com"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [cited, never_cited], client=mock_client)

    assert [a.url for a in result["gap_analysis"]["cited_web_articles"]] == ["https://cited.com"]
    urls_in_references = {r["url"] for r in result["references"]}
    assert urls_in_references == {"https://cited.com"}
    assert "https://never-cited.com" not in urls_in_references


def test_regenerate_report_with_approved_web_sources_organically_cited_url_appears_inline_and_in_references():
    """The positive path: an approved URL the model genuinely cites --
    structurally (cited_web_urls) AND with a real inline [Web N] marker
    -- still appears in References, with the marker resolved to a
    final [N] in the rendered content. Proves R3.1b didn't break the
    ordinary, organic citation path."""
    p1 = _paper("1111", "Paper One")
    web = _web_article("https://organic.com", "Organic")
    existing_report = {
        "thematic_findings": {"content": "old", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"], report=existing_report,
    )
    schema = _build_report_schema(["1111"], ["https://organic.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="A claim [Web 1].", cited_paper_ids=[], cited_web_urls=["https://organic.com"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [web], client=mock_client)

    assert len(result["references"]) == 1
    assert result["references"][0]["url"] == "https://organic.com"
    assert result["thematic_findings"]["content"] == "A claim [1]."


def test_regenerate_report_with_new_sources_revoked_approved_url_never_appears():
    """A URL that's BOTH previously approved AND now revoked must never
    appear -- not offered to the model, not in References."""
    p1 = _paper("1111", "Paper One")
    web = _web_article("https://revoked-but-approved.com", "Revoked But Approved")
    existing_report = {
        "thematic_findings": {
            "content": "Per [1].", "cited_papers": [], "cited_web_articles": [web], "reference_numbers": [1],
        },
        "references": [{
            "number": 1, "kind": "web", "paper_id": None, "url": "https://revoked-but-approved.com",
            "title": "Revoked But Approved", "formatted": "x", "link_url": "https://revoked-but-approved.com",
        }],
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[web],
        # Approved (so it WOULD be named in the approved-sources prompt
        # paragraph) but also revoked -- revocation must win regardless.
        report_approved_web_article_urls={"https://revoked-but-approved.com"},
        revoked_web_article_urls={"https://revoked-but-approved.com"},
    )
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in sent_messages)
    assert "revoked-but-approved.com" not in joined
    assert result["references"] == []
    assert "cited_web_articles" not in result["thematic_findings"]


def test_regenerate_report_with_approved_web_sources_previously_marked_citation_dropped_by_model_disappears():
    """R3.1b Option B: a URL that had a genuine inline marker in the
    PRIOR report, but the model's new draft doesn't cite this round,
    must disappear entirely -- not survive as a metadata-only orphan
    Reference. The current round's own model output is the sole source
    of truth; there is no restoration step anymore."""
    p1 = _paper("1111", "Paper One")
    web = _web_article("https://was-cited.com", "Was Cited")
    existing_report = {
        "gap_analysis": {
            "content": "Per [1].", "cited_papers": [], "cited_web_articles": [web], "reference_numbers": [1],
        },
        "references": [{
            "number": 1, "kind": "web", "paper_id": None, "url": "https://was-cited.com",
            "title": "Was Cited", "formatted": "x", "link_url": "https://was-cited.com",
        }],
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"], report=existing_report,
    )
    schema = _build_report_schema(["1111"], ["https://was-cited.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    # The model's new draft drops the citation entirely from gap_analysis.
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        gap_analysis=section_cls(content="New prose, no citation.", cited_paper_ids=[], cited_web_urls=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [web], client=mock_client)

    assert result["gap_analysis"]["cited_web_articles"] == []
    assert result["gap_analysis"]["content"] == "New prose, no citation."
    assert result["references"] == []


def test_regenerate_report_with_new_sources_never_cited_sources_stay_absent_regardless_of_approval():
    """Whole-pool regeneration: since force-inclusion is gone (R3.1b),
    an approved-but-uncited source and a merely-discovered-but-never-
    approved source are now treated identically when the model cites
    neither -- both stay absent, not just the unapproved one."""
    p1 = _paper("1111", "Paper One")
    approved = _web_article("https://approved.com", "Approved")
    merely_discovered = _web_article("https://merely-discovered.com", "Merely Discovered")
    existing_report = {
        "thematic_findings": {"content": "old", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"],
        report=existing_report, web_articles_added=[approved, merely_discovered],
        report_approved_web_article_urls={"https://approved.com"},  # NOT merely_discovered
    )
    schema = _build_report_schema(["1111"], ["https://approved.com", "https://merely-discovered.com"], REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema, web_urls_used=True)  # model cites nothing
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    assert result["references"] == []


def test_regenerate_report_with_approved_web_sources_paper_citation_preservation_still_works():
    """Paper citation preservation (_restore_dropped_citations) is
    unaffected by R3.1b -- a paper the model's new draft drops is still
    force-restored, same as before, even alongside a never-cited
    approved web source in the same call (which correctly stays absent
    now, not force-included)."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    web = _web_article("https://new.com", "New")
    existing_report = {
        "thematic_findings": {
            "content": "Per [Paper 1].", "cited_papers": [p1, p2], "cited_web_articles": [], "reference_numbers": [],
        },
        "references": [], "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1, p2], selected_paper_ids=["1111", "2222"],
        report=existing_report,
    )
    schema = _build_report_schema(["1111", "2222"], ["https://new.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    # The model's new draft drops paper 2222 entirely and cites no web source.
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="New prose.", cited_paper_ids=["1111"], cited_web_urls=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_approved_web_sources(session, [web], client=mock_client)

    assert {p.paper_id for p in result["thematic_findings"]["cited_papers"]} == {"1111", "2222"}
    assert result["thematic_findings"]["cited_web_articles"] == []
    assert result["skipped_papers"] == []


def test_build_regeneration_system_prompt_includes_approved_sources_paragraph_when_present():
    web = _web_article("https://approved.com", "Approved Source")
    existing_report = {"thematic_findings": {"content": "", "cited_papers": []}, "skipped_papers": []}

    prompt = _build_regeneration_system_prompt(
        existing_report, REPORT_SECTION_DEFINITIONS, "analytical",
        allowed_web_urls={"https://approved.com"}, web_by_url={"https://approved.com": web},
    )

    assert "approved by the user via chat" in prompt
    assert "Approved Source" in prompt
    assert "[Web N]" in prompt


def test_build_regeneration_system_prompt_omits_approved_sources_paragraph_when_none_given():
    existing_report = {"thematic_findings": {"content": "", "cited_papers": []}, "skipped_papers": []}

    prompt = _build_regeneration_system_prompt(existing_report, REPORT_SECTION_DEFINITIONS, "analytical")

    assert "approved by the user via chat" not in prompt


# --- report-quality Phase R3: report versioning ---

def _report_dict(content: str = "content", report_template: str = "analytical") -> dict:
    return {
        "findings": {"content": content, "cited_papers": []},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [], "report_template": report_template,
    }


def test_append_report_version_creates_version_1_for_initial_generation():
    session = PaperPoolSession(topic="q", stage="synthesize")
    report = _report_dict("first report")

    version = append_report_version(session, report, GENERATION_REASON_INITIAL)

    assert version["version_number"] == 1
    assert version["generation_reason"] == "initial"
    assert version["report_template"] == "analytical"
    assert version["report"] is report
    assert version["version_id"]  # non-empty
    assert version["created_at"]  # non-empty ISO string
    assert len(session.report_versions) == 1
    assert session.report_versions[0] is version
    assert session.active_report_version_id == version["version_id"]
    assert session.report is report


def test_append_report_version_appends_version_2_for_regenerate_without_mutating_version_1():
    session = PaperPoolSession(topic="q", stage="synthesize")
    v1_report = _report_dict("first report")
    v1 = append_report_version(session, v1_report, GENERATION_REASON_INITIAL)

    v2_report = _report_dict("second report", report_template="expert")
    v2 = append_report_version(session, v2_report, GENERATION_REASON_REGENERATE)

    assert v2["version_number"] == 2
    assert v2["generation_reason"] == "regenerate"
    assert len(session.report_versions) == 2
    assert session.report_versions[0] is v1
    assert session.report_versions[1] is v2
    # version 1's own dict is completely untouched -- same object,
    # same content, never rewritten by the later append.
    assert session.report_versions[0]["report"]["findings"]["content"] == "first report"
    assert session.report_versions[0]["version_number"] == 1
    assert session.active_report_version_id == v2["version_id"]
    assert session.report is v2_report


def test_append_report_version_version_ids_are_unique():
    session = PaperPoolSession(topic="q", stage="synthesize")
    v1 = append_report_version(session, _report_dict("a"), GENERATION_REASON_INITIAL)
    v2 = append_report_version(session, _report_dict("b"), GENERATION_REASON_REGENERATE)

    assert v1["version_id"] != v2["version_id"]


def test_get_active_report_version_returns_none_for_a_session_with_no_report_yet():
    session = PaperPoolSession(topic="q", stage="synthesize")
    assert get_active_report_version(session) is None


def test_get_active_report_version_returns_the_currently_active_version():
    session = PaperPoolSession(topic="q", stage="synthesize")
    append_report_version(session, _report_dict("a"), GENERATION_REASON_INITIAL)
    v2 = append_report_version(session, _report_dict("b"), GENERATION_REASON_REGENERATE)

    assert get_active_report_version(session) is v2


def test_activate_report_version_switches_session_report_to_an_older_version():
    session = PaperPoolSession(topic="q", stage="synthesize")
    v1 = append_report_version(session, _report_dict("first report"), GENERATION_REASON_INITIAL)
    append_report_version(session, _report_dict("second report"), GENERATION_REASON_REGENERATE)
    assert session.report["findings"]["content"] == "second report"

    activated = activate_report_version(session, v1["version_id"])

    assert activated is v1
    assert session.active_report_version_id == v1["version_id"]
    assert session.report["findings"]["content"] == "first report"
    assert session.report is v1["report"]
    # Nothing about report_versions itself changed -- still both entries, in order.
    assert len(session.report_versions) == 2


def test_activate_report_version_unknown_id_returns_none_and_does_not_mutate_session():
    session = PaperPoolSession(topic="q", stage="synthesize")
    append_report_version(session, _report_dict("only report"), GENERATION_REASON_INITIAL)
    active_before = session.active_report_version_id

    result = activate_report_version(session, "does-not-exist")

    assert result is None
    assert session.active_report_version_id == active_before
    assert session.report["findings"]["content"] == "only report"


def test_regenerate_builds_from_the_active_version_not_always_latest():
    """report-quality Phase R3 decision 8: regenerate must build from
    whichever version is currently ACTIVE, not whatever's most recently
    appended -- proven here via the citation-preservation mechanism
    itself: activating an OLDER version, then regenerating, must
    preserve THAT version's prior citations, not the latest version's."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    session = PaperPoolSession(
        topic="q", stage="synthesize", selected_papers=[p1, p2], selected_paper_ids=["1111", "2222"],
        web_articles_added=[],
    )
    v1_report = {
        "thematic_findings": {"content": "v1", "cited_papers": [p1], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [], "report_template": "analytical",
    }
    v1 = append_report_version(session, v1_report, GENERATION_REASON_INITIAL)
    v2_report = {
        "thematic_findings": {"content": "v2", "cited_papers": [p2], "cited_web_articles": [], "reference_numbers": []},
        "references": [], "skipped_papers": [], "report_template": "analytical",
    }
    append_report_version(session, v2_report, GENERATION_REASON_REGENERATE)

    # Activate the OLDER version (which cited p1, not p2) before regenerating.
    activate_report_version(session, v1["version_id"])

    schema = _build_report_schema(["1111", "2222"], None, REPORT_SECTION_DEFINITIONS)
    parsed = _analytical_parsed(schema)  # model drops all citations this time
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = regenerate_report_with_new_sources(session, client=mock_client)

    # p1 (v1's own prior citation) is restored -- proves the regeneration
    # built from the ACTIVE (v1) version, not the latest (v2) one, which
    # would have restored p2 instead.
    assert [p.paper_id for p in result["thematic_findings"]["cited_papers"]] == ["1111"]


# --- report-quality Phase R4.1: optional, bounded refinement loop ---

def _clean_analytical_draft(papers: list, template: str = "analytical") -> dict:
    """Hand-builds a report dict in exactly the shape generate_report's
    own pipeline produces (raw section-local [Paper N] markers in, real
    resolved [N] markers + references out via _build_references_and_
    renumber -- same convention this file's own R3.2 tests already use
    for _sections_out/_build_references_and_renumber directly) -- and
    passes every _deterministic_report_checks hard gate on its own:
    every REPORT_TEMPLATES section has non-empty content, the first
    paper is genuinely cited with a resolved marker, references/
    reference_numbers are internally consistent. The "nothing
    structurally wrong with this draft" baseline several R4.1 tests
    below build on."""
    section_defs = REPORT_TEMPLATES[template]
    sections_out = {}
    for d in section_defs:
        key = d["key"]
        if key == "thematic_findings" and papers:
            sections_out[key] = {"content": "A finding [Paper 1].", "cited_papers": [papers[0]]}
        else:
            sections_out[key] = {"content": f"{d['title']} text.", "cited_papers": []}
    result = _build_references_and_renumber({**sections_out, "skipped_papers": []}, ANALYTICAL_SECTION_NAMES)
    result["report_template"] = template
    return result


def _evaluation(
    overall_score: int = 90, needs_revision: bool = False, issues: list[str] | None = None,
    revision_instructions: str = "", section_scores: dict[str, int] | None = None,
) -> ReportEvaluation:
    return ReportEvaluation(
        overall_score=overall_score, needs_revision=needs_revision, issues=issues or [],
        revision_instructions=revision_instructions, section_scores=section_scores,
    )


def test_refine_report_if_requested_off_or_omitted_is_a_pure_passthrough_with_zero_extra_calls():
    """Required test 1: refinement_mode omitted/off -> existing
    behavior, zero evaluator/revision calls, no refinement metadata at
    all (not even {"enabled": False, ...}) -- the strongest possible
    reading of "identical to current behavior"."""
    draft = _clean_analytical_draft([_paper("1111", "Paper One")])
    mock_client = MagicMock()

    result_off = refine_report_if_requested(draft, "topic", [], [], "analytical", "off", mock_client)
    result_none = refine_report_if_requested(draft, "topic", [], [], "analytical", None, mock_client)

    assert result_off is draft
    assert result_none is draft
    assert "refinement" not in result_off
    mock_client.chat.completions.parse.assert_not_called()


def test_refine_single_mode_no_revision_needed_finalizes_draft():
    """Required test 2: evaluator says no revision needed -> draft
    becomes final, rounds 0, no revision call (exactly one LLM call
    total -- the evaluation)."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=88, needs_revision=False),
    )

    result = refine_report_if_requested(draft, "topic", [p1], [], "analytical", "single", mock_client)

    assert mock_client.chat.completions.parse.call_count == 1
    assert result is draft
    assert result["refinement"] == {
        "enabled": True, "rounds": 0, "initial_score": 88, "final_score": 88,
        "issues": [], "revision_instructions": "", "section_scores": None,
    }


def test_refine_single_mode_revision_needed_revises_exactly_once_and_stops():
    """Required tests 3 + 4: evaluator says revision needed -> exactly
    one revision occurs (rounds=1); nothing re-evaluates the revision
    afterward -- call_count stays at 2 (evaluate + revise), never a
    3rd call, proving there is no loop back to evaluate_report even
    though the revised report was never itself judged."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    mock_client = MagicMock()

    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    revised_parsed = _analytical_parsed(
        schema,
        thematic_findings=section_cls(content="A better finding [Paper 1].", cited_paper_ids=["1111"]),
    )
    mock_client.chat.completions.parse.side_effect = [
        _mock_parsed_response(_evaluation(
            overall_score=40, needs_revision=True, issues=["too shallow"], revision_instructions="add more depth",
        )),
        _mock_parsed_response(revised_parsed),
    ]

    result = refine_report_if_requested(draft, "topic", [p1], [], "analytical", "single", mock_client)

    assert mock_client.chat.completions.parse.call_count == 2
    assert result["refinement"] == {
        "enabled": True, "rounds": 1, "initial_score": 40, "final_score": None,
        "issues": ["too shallow"], "revision_instructions": "add more depth", "section_scores": None,
    }
    assert result["thematic_findings"]["content"] == "A better finding [1]."
    assert len(result["references"]) == 1


def test_refine_report_if_requested_stamps_section_scores_when_the_evaluator_returns_them():
    """report-quality Phase R4.2: a non-null section_scores from the
    evaluator survives into the final refinement metadata unchanged --
    proven for both branches (no revision needed / revision needed),
    since the CURRENT code stamps the same `evaluation` dict's
    section_scores either way."""
    p1 = _paper("1111", "Paper One")
    scores = {"thematic_findings": 70, "conclusion": 85}

    draft = _clean_analytical_draft([p1])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=82, needs_revision=False, section_scores=scores),
    )
    no_revision_result = refine_report_if_requested(draft, "topic", [p1], [], "analytical", "single", mock_client)
    assert no_revision_result["refinement"]["section_scores"] == scores

    draft2 = _clean_analytical_draft([p1])
    mock_client2 = MagicMock()
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    revised_parsed = _analytical_parsed(
        schema, thematic_findings=section_cls(content="Revised [Paper 1].", cited_paper_ids=["1111"]),
    )
    mock_client2.chat.completions.parse.side_effect = [
        _mock_parsed_response(_evaluation(
            overall_score=35, needs_revision=True, revision_instructions="deepen it", section_scores=scores,
        )),
        _mock_parsed_response(revised_parsed),
    ]
    revision_result = refine_report_if_requested(draft2, "topic", [p1], [], "analytical", "single", mock_client2)
    assert revision_result["refinement"]["section_scores"] == scores


def test_evaluate_report_deterministic_hard_gate_forces_revision_even_if_llm_says_no():
    """Required test 5: a deterministic hard gate (here: a required
    section left empty) forces needs_revision=True even though the
    mocked LLM evaluator itself says no revision is needed."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    draft = {**draft, "gap_analysis": {**draft["gap_analysis"], "content": ""}}
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=95, needs_revision=False),
    )

    result = evaluate_report(draft, "topic", [p1], [], "analytical", mock_client)

    assert result["needs_revision"] is True
    assert any("Missing or empty required section" in issue for issue in result["issues"])


def test_evaluate_report_unresolved_marker_leak_is_a_hard_gate():
    """_deterministic_report_checks' raw-marker-leak regression
    tripwire: a bracket containing letters (never a properly-resolved
    final [N] marker) forces revision regardless of the LLM's verdict."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    draft = {**draft, "gap_analysis": {**draft["gap_analysis"], "content": "See [Paper 3] for details."}}
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=95, needs_revision=False),
    )

    result = evaluate_report(draft, "topic", [p1], [], "analytical", mock_client)

    assert result["needs_revision"] is True
    assert any("Unresolved/raw citation marker" in issue for issue in result["issues"])


def test_evaluate_report_orphan_reference_is_a_hard_gate():
    """_deterministic_report_checks' orphan-reference regression
    tripwire (redundant with R3.1b's own structural guarantee in
    normal operation, but cheap insurance against a future regression
    of it): a References entry no section's reference_numbers points
    to forces revision regardless of the LLM's verdict."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    orphan_ref = {
        "number": 99, "kind": "web", "paper_id": None, "url": "https://orphan.com",
        "title": "Orphan", "formatted": "x", "link_url": "https://orphan.com",
    }
    draft = {**draft, "references": [*draft["references"], orphan_ref]}
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=95, needs_revision=False),
    )

    result = evaluate_report(draft, "topic", [p1], [], "analytical", mock_client)

    assert result["needs_revision"] is True
    assert any("Orphan reference" in issue for issue in result["issues"])


def test_evaluate_report_skipped_papers_warning_alone_does_not_force_revision():
    """Required test 6: skipped_papers is included as an issue (context
    for both the evaluator and a human reading the final metadata) but
    must NOT by itself force needs_revision when the LLM says the
    report is otherwise fine."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    draft = _clean_analytical_draft([p1])
    draft = {**draft, "skipped_papers": [p2]}
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=85, needs_revision=False),
    )

    result = evaluate_report(draft, "topic", [p1, p2], [], "analytical", mock_client)

    assert result["needs_revision"] is False
    assert any("never cited in any section" in issue for issue in result["issues"])


def test_revise_report_produces_a_normal_full_report_dict():
    """Required test 7: the revised report is a normal report dict --
    real resolved citation markers, populated References, the full
    sections/legacy-field projection every other generation path
    produces -- not a distinct "revision" shape."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    evaluation = _evaluation(
        overall_score=50, needs_revision=True, issues=["shallow"], revision_instructions="deepen it",
    ).model_dump()

    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    parsed = _analytical_parsed(
        schema, thematic_findings=section_cls(content="Deeper finding [Paper 1].", cited_paper_ids=["1111"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = revise_report("topic", [p1], [], draft, evaluation, mock_client, report_template="analytical")

    assert result["thematic_findings"]["content"] == "Deeper finding [1]."
    assert len(result["references"]) == 1
    assert result["report_template"] == "analytical"
    assert len(result["sections"]) == len(REPORT_SECTION_DEFINITIONS)
    assert result["findings"] == result["thematic_findings"]  # legacy projection still works


def test_revise_report_preserves_a_paper_citation_the_revision_drops():
    """revise_report reuses _restore_dropped_citations exactly like
    regeneration does -- a paper the draft cited stays cited even if
    the revision's own output drops it."""
    p1, p2 = _paper("1111", "Paper One"), _paper("2222", "Paper Two")
    draft = _clean_analytical_draft([p1, p2])
    draft["thematic_findings"]["cited_papers"] = [p1, p2]
    evaluation = _evaluation(overall_score=50, needs_revision=True, revision_instructions="tighten it").model_dump()

    schema = _build_report_schema(["1111", "2222"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    # The revision's own output drops paper 2222 entirely.
    parsed = _analytical_parsed(
        schema, thematic_findings=section_cls(content="Tighter finding.", cited_paper_ids=["1111"]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = revise_report("topic", [p1, p2], [], draft, evaluation, mock_client, report_template="analytical")

    assert {p.paper_id for p in result["thematic_findings"]["cited_papers"]} == {"1111", "2222"}


def test_revise_report_never_restores_or_force_includes_a_dropped_web_citation():
    """R3.1b's product rule carries over into revision unchanged: the
    revision's own cited_web_urls is the sole source of truth for web
    citations -- a web source the draft cited but the revision drops
    simply disappears, never becomes a metadata-only orphan."""
    p1 = _paper("1111", "Paper One")
    web = _web_article("https://was-cited.com", "Was Cited")
    draft = _clean_analytical_draft([p1])
    draft["thematic_findings"]["cited_web_articles"] = [web]
    evaluation = _evaluation(overall_score=50, needs_revision=True, revision_instructions="tighten it").model_dump()

    schema = _build_report_schema(["1111"], ["https://was-cited.com"], REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    # The revision's own output cites no web source at all.
    parsed = _analytical_parsed(
        schema, web_urls_used=True,
        thematic_findings=section_cls(content="Tighter finding [Paper 1].", cited_paper_ids=["1111"], cited_web_urls=[]),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed)

    result = revise_report("topic", [p1], [web], draft, evaluation, mock_client, report_template="analytical")

    assert result["thematic_findings"]["cited_web_articles"] == []
    assert all(r["kind"] != "web" for r in result["references"])


def test_refine_report_if_requested_preserves_report_template_in_both_branches():
    """Required test 8: report_template survives refinement -- checked
    here on the no-revision branch with a non-default template
    (expert), since that's the cheapest path to prove nothing along the
    way silently resets it to the default."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1], template="expert")
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=70, needs_revision=False),
    )

    result = refine_report_if_requested(draft, "topic", [p1], [], "expert", "single", mock_client)

    assert result["report_template"] == "expert"


def test_refine_report_if_requested_output_appends_as_a_normal_report_version():
    """Required test 9: the refined report becomes a normal R3 report
    version via the existing, unmodified append_report_version --
    generation_reason keeps its existing vocabulary (no "refined"
    value was added; refinement is orthogonal metadata on the report
    body, not a new reason)."""
    p1 = _paper("1111", "Paper One")
    draft = _clean_analytical_draft([p1])
    session = PaperPoolSession(topic="q", stage="synthesize", selected_papers=[p1], selected_paper_ids=["1111"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parsed_response(
        _evaluation(overall_score=77, needs_revision=False),
    )

    refined = refine_report_if_requested(draft, session.topic, [p1], [], "analytical", "single", mock_client)
    version = append_report_version(session, refined, GENERATION_REASON_INITIAL)

    assert version["generation_reason"] == "initial"
    assert version["report"]["refinement"]["enabled"] is True
    assert session.report_versions[0] is version
    assert session.report["refinement"]["initial_score"] == 77


if __name__ == "__main__":
    test_report_schema_rejects_unknown_paper_id()
    test_generate_report_attaches_citations_per_section_and_flags_skipped()
    test_generate_report_returns_empty_for_no_selected_papers()
    test_generate_report_for_session_refuses_when_stage_is_not_synthesize()
    test_generate_report_with_just_one_selected_paper()
    test_report_schema_construction_near_the_30_paper_cap()
    test_generate_report_falls_back_to_placeholder_text_for_missing_abstract()
    test_report_schema_with_web_urls_rejects_unknown_url()
    test_report_schema_without_web_urls_has_no_cited_web_urls_field()
    test_generate_report_with_web_articles_attaches_cited_web_articles_per_section()
    test_generate_report_without_web_articles_omits_cited_web_articles_key()
    test_regenerate_report_refuses_when_no_existing_report()
    test_regenerate_report_refuses_when_stage_is_not_synthesize()
    test_regenerate_report_preserves_all_original_citations_and_adds_web_citations()
    test_regenerate_report_defensive_layer_restores_citation_the_prompt_instruction_failed_to_preserve()
    test_regenerate_with_approved_web_sources_only_reaches_the_model_for_the_passed_in_articles()
    test_regenerate_with_approved_web_sources_ignores_web_articles_added_entirely()
    test_regenerate_with_approved_web_sources_refuses_when_no_existing_report()
    test_regenerate_with_approved_web_sources_refuses_when_stage_is_not_synthesize()
    print("All report tests passed.")
