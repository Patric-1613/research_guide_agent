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
    REPORT_SECTION_DEFINITIONS,
    _build_references_and_renumber,
    _build_report_schema,
    derive_legacy_references,
    derive_sections_from_legacy_report,
    generate_report,
    generate_report_for_session,
    regenerate_report_with_approved_web_sources,
    regenerate_report_with_new_sources,
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

    assert result["findings"]["content"] == "Per [1], X. But  is invented."
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

    assert result["findings"]["content"] == "See  here."
    assert result["references"] == []
    assert "Paper" not in result["findings"]["content"]


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
