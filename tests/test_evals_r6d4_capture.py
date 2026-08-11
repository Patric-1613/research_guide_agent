"""R6D.4a: tests for `research_agent.evals.r6d4_capture` -- the
developer-only, in-memory helper that captures a genuine R4 draft/
refined report pair. Every test here mocks either `client.chat.
completions.parse` directly (same convention `tests/test_report.py`
already establishes for exercising the REAL `report.py` functions) or
`report_module.generate_report_for_session`/`refine_report_if_
requested` themselves (to prove call shape/argument identity, and to
drive `refine_report_if_requested`'s two branches precisely without
depending on report.py's own internal prompt-building details). No
real OpenAI client is ever constructed and no network call is ever
made anywhere in this file.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from unittest.mock import MagicMock, patch

import pytest

from research_agent.evals import r6d4_capture
from research_agent.evals.r6d4_capture import (
    R6D4CaptureError,
    capture_real_refinement_pair,
    find_forbidden_keys,
    validate_r6d4_capture,
)
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import REPORT_TEMPLATES, ReportEvaluation
from research_agent import report as report_module
from research_agent.schema import Paper


# --- Shared helpers (same conventions as tests/test_report.py) -----------

def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _session(topic: str = "Citation grounding", selected_papers: list[Paper] | None = None, stage: str = "synthesize") -> PaperPoolSession:
    return PaperPoolSession(topic=topic, stage=stage, selected_papers=selected_papers or [])


def _mock_parsed_response(parsed):
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def _full_analytical_parsed(schema, paper_id: str, **section_overrides):
    """Every one of the 8 sections gets real, non-empty, citing content
    -- unlike test_report.py's own `_analytical_parsed` (which defaults
    to empty), this file needs every section non-empty so `_
    deterministic_report_checks`'s hard gates never force `needs_
    revision=True` behind a test's back, contaminating a deliberately-
    "no revision" scenario."""
    section_cls = schema.model_fields["executive_summary"].annotation
    kwargs = {}
    for key in report_module.ANALYTICAL_SECTION_NAMES:
        if key in section_overrides:
            kwargs[key] = section_overrides[key]
        else:
            kwargs[key] = section_cls(content=f"A finding about {key} [Paper 1].", cited_paper_ids=[paper_id])
    return schema(**kwargs)


def _evaluation(overall_score=90, needs_revision=False, issues=None, revision_instructions="", section_scores=None) -> ReportEvaluation:
    return ReportEvaluation(
        overall_score=overall_score, needs_revision=needs_revision, issues=issues or [],
        revision_instructions=revision_instructions, section_scores=section_scores,
    )


def _no_revision_client(paper_id: str = "1111", template: str = "analytical"):
    """A MagicMock client whose first call answers report generation,
    whose second answers R4's own evaluation with needs_revision=False
    -- exactly 2 calls total, never a 3rd (no revision call)."""
    schema = report_module._build_report_schema([paper_id], None, REPORT_TEMPLATES[template])
    parsed = _full_analytical_parsed(schema, paper_id)
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(parsed),
        _mock_parsed_response(_evaluation(overall_score=92, needs_revision=False)),
    ]
    return client


def _revision_client(paper_id: str = "1111", template: str = "analytical"):
    """A MagicMock client whose first call answers generation, second
    answers evaluation with needs_revision=True, third answers the one
    permitted revision -- exactly 3 calls, with a genuinely different
    thematic_findings section so draft/refined bodies differ."""
    schema = report_module._build_report_schema([paper_id], None, REPORT_TEMPLATES[template])
    draft_parsed = _full_analytical_parsed(schema, paper_id)
    section_cls = schema.model_fields["executive_summary"].annotation
    revised_parsed = _full_analytical_parsed(
        schema, paper_id,
        thematic_findings=section_cls(content="A substantially revised finding [Paper 1].", cited_paper_ids=[paper_id]),
    )
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(draft_parsed),
        _mock_parsed_response(_evaluation(
            overall_score=35, needs_revision=True, issues=["too shallow"], revision_instructions="add depth",
        )),
        _mock_parsed_response(revised_parsed),
    ]
    return client


# --- Production generation/refinement wrapper reuse -----------------------

class TestProductionWrapperReuse:
    def test_generate_report_for_session_is_the_generation_function_called(self):
        """Proves `generate_report_for_session` (not `generate_report`
        directly) is what this module calls, with exactly the
        arguments `get_or_create_report`'s own initial-generation call
        uses."""
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        with patch.object(report_module, "generate_report_for_session", wraps=report_module.generate_report_for_session) as spy:
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )

        spy.assert_called_once_with(session, client=client, report_template="analytical")

    def test_refine_report_if_requested_receives_the_same_inputs_get_or_create_report_uses(self):
        """`refine_report_if_requested` must be called with exactly:
        the draft, session.topic, session.selected_papers (the SAME
        list object, never re-resolved), an empty web_articles list
        (matching get_or_create_report's own hardcoded `[]`, never
        `session.web_articles_added`), refinement_mode="single"."""
        p1 = _paper("1111", "Paper One")
        session = _session(topic="my topic", selected_papers=[p1])
        client = _no_revision_client()

        with patch.object(report_module, "refine_report_if_requested", wraps=report_module.refine_report_if_requested) as spy:
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )

        assert spy.call_count == 1
        call = spy.call_args
        assert call.args[1] == "my topic"
        assert call.args[2] is session.selected_papers
        assert call.args[3] == []
        assert call.kwargs["refinement_mode"] == "single"
        assert call.kwargs["client"] is client

    def test_never_uses_generate_report_directly(self):
        """Structural guarantee: `generate_report` (session-independent,
        different evidence-resolution semantics than the session-aware
        wrapper) is never referenced anywhere in this module's own
        source -- checked via AST rather than by patching the
        production symbol, since `generate_report_for_session` itself
        legitimately delegates to `generate_report` internally, and
        patching it would corrupt that unrelated call instead of
        proving anything about THIS module."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(r6d4_capture))
        called_names = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        assert "generate_report" not in called_names
        assert "generate_report_for_session" in called_names

    def test_session_stage_validation_still_applies(self):
        """generate_report_for_session's own readiness check (stage !=
        'synthesize') is never bypassed -- this module adds no
        alternate path around it."""
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1], stage="curate")
        client = MagicMock()

        with pytest.raises(ValueError, match="not ready for report synthesis"):
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )
        client.chat.completions.parse.assert_not_called()


# --- Draft preservation / mutation handling --------------------------------

class TestDraftPreservation:
    def test_generated_draft_object_is_the_object_passed_into_refinement(self):
        draft = {"report_template": "analytical"}
        captured_args = {}

        def fake_generate(session, client, report_template):
            return draft

        def fake_refine(passed_draft, *args, **kwargs):
            captured_args["passed_draft"] = passed_draft
            passed_draft["refinement"] = {
                "enabled": True, "rounds": 0, "initial_score": 90, "final_score": 90,
                "issues": [], "revision_instructions": "", "section_scores": None,
            }
            return passed_draft

        with patch.object(report_module, "generate_report_for_session", side_effect=fake_generate), \
             patch.object(report_module, "refine_report_if_requested", side_effect=fake_refine):
            capture_real_refinement_pair(
                _session(), MagicMock(), report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )

        assert captured_args["passed_draft"] is draft

    def test_deepcopy_snapshot_survives_no_revision_in_place_mutation(self):
        """The single most important guarantee in this module: even
        though the real no-revision branch mutates its own `draft`
        argument in place, the artifact's own `draft_report` must never
        show that mutation."""
        original_draft = {"report_template": "analytical", "thematic_findings": {"content": "original"}}

        def fake_generate(session, client, report_template):
            return original_draft

        def fake_refine(passed_draft, *args, **kwargs):
            # Mimics refine_report_if_requested's real no-revision branch exactly:
            # `final = draft; final["refinement"] = {...}` -- same object, mutated in place.
            passed_draft["refinement"] = {
                "enabled": True, "rounds": 0, "initial_score": 90, "final_score": 90,
                "issues": [], "revision_instructions": "", "section_scores": None,
            }
            return passed_draft

        with patch.object(report_module, "generate_report_for_session", side_effect=fake_generate), \
             patch.object(report_module, "refine_report_if_requested", side_effect=fake_refine):
            artifact = capture_real_refinement_pair(
                _session(), MagicMock(), report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )

        assert "refinement" not in artifact["draft_report"]
        assert artifact["draft_report"]["thematic_findings"]["content"] == "original"
        # The ORIGINAL dict really was mutated (proving the scenario is realistic) --
        # the artifact's own snapshot is a genuinely separate object, not a live view of it.
        assert original_draft["refinement"]["rounds"] == 0
        assert artifact["draft_report"] is not original_draft

    def test_draft_snapshot_never_mutated_after_capture(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        before = copy.deepcopy(artifact["draft_report"])
        artifact["refined_report"]["thematic_findings"]["content"] = "tampered"
        assert artifact["draft_report"] == before


# --- No-revision / revision equality semantics -----------------------------

class TestRevisionSemantics:
    def test_no_revision_bodies_equal_after_refinement_stripping(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )

        assert artifact["draft_report"] == artifact["refined_report"]
        assert artifact["refinement_context"]["revision_applied"] is False
        assert "refinement" not in artifact["draft_report"]
        assert "refinement" not in artifact["refined_report"]
        assert client.chat.completions.parse.call_count == 2

    def test_revised_bodies_differ(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )

        assert artifact["draft_report"] != artifact["refined_report"]
        assert artifact["refinement_context"]["revision_applied"] is True
        assert "refinement" not in artifact["draft_report"]
        assert "refinement" not in artifact["refined_report"]
        assert client.chat.completions.parse.call_count == 3

    def test_refinement_metadata_moved_into_refinement_context(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )

        meta = artifact["refinement_context"]["r4_refinement_metadata"]
        assert meta["rounds"] == 1
        assert meta["initial_score"] == 35
        assert meta["final_score"] is None  # never fabricated -- R4 never re-evaluates the revision
        assert meta["issues"] == ["too shallow"]
        assert meta["revision_instructions"] == "add depth"

    def test_contradictory_rounds_gt_zero_but_bodies_equal_is_rejected(self):
        draft = {"report_template": "analytical", "thematic_findings": {"content": "same"}}

        def fake_generate(session, client, report_template):
            return draft

        def fake_refine(passed_draft, *args, **kwargs):
            result = copy.deepcopy(passed_draft)
            result["refinement"] = {
                "enabled": True, "rounds": 1, "initial_score": 40, "final_score": None,
                "issues": ["x"], "revision_instructions": "y", "section_scores": None,
            }
            return result

        with patch.object(report_module, "generate_report_for_session", side_effect=fake_generate), \
             patch.object(report_module, "refine_report_if_requested", side_effect=fake_refine):
            with pytest.raises(R6D4CaptureError, match="contradictory provenance"):
                capture_real_refinement_pair(
                    _session(), MagicMock(), report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
                )

    def test_contradictory_rounds_zero_but_bodies_differ_is_rejected(self):
        draft = {"report_template": "analytical", "thematic_findings": {"content": "draft"}}

        def fake_generate(session, client, report_template):
            return draft

        def fake_refine(passed_draft, *args, **kwargs):
            result = {"report_template": "analytical", "thematic_findings": {"content": "DIFFERENT"}}
            result["refinement"] = {
                "enabled": True, "rounds": 0, "initial_score": 90, "final_score": 90,
                "issues": [], "revision_instructions": "", "section_scores": None,
            }
            return result

        with patch.object(report_module, "generate_report_for_session", side_effect=fake_generate), \
             patch.object(report_module, "refine_report_if_requested", side_effect=fake_refine):
            with pytest.raises(R6D4CaptureError, match="contradictory provenance"):
                capture_real_refinement_pair(
                    _session(), MagicMock(), report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
                )


# --- Evidence snapshot ------------------------------------------------------

class TestEvidenceSnapshot:
    def test_exact_evidence_snapshot(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )

        assert artifact["selected_papers"] == [p1.to_dict()]
        assert artifact["approved_web_articles"] == []

    def test_approved_web_articles_empty_even_when_session_has_discovered_web_sources(self):
        """The initial-Generate production path never offers web
        sources -- `approved_web_articles` must be `[]` even when
        `session.web_articles_added` (a field this module never even
        reads) is non-empty."""
        from research_agent.schema import WebArticle

        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        session.web_articles_added = [
            WebArticle(title="Some article", url="https://example.com/a", snippet="s", published_date=None, source_domain="example.com"),
        ]
        client = _no_revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        assert artifact["approved_web_articles"] == []

    def test_selected_papers_not_independently_re_resolved(self):
        """The exact same `session.selected_papers` list object is what
        gets serialized -- never re-derived from `session.reserve` or
        anything else."""
        p1 = _paper("1111", "Paper One")
        p2 = _paper("2222", "Paper Two")
        session = _session(selected_papers=[p1, p2])
        session.reserve = [(p1, 0.9)]  # deliberately does NOT include p2 -- proves reserve is never consulted
        client = MagicMock()
        schema = report_module._build_report_schema(["1111", "2222"], None, REPORT_TEMPLATES["analytical"])
        section_cls = schema.model_fields["executive_summary"].annotation
        parsed = schema(**{
            key: section_cls(content=f"Finding [Paper 1] [Paper 2].", cited_paper_ids=["1111", "2222"])
            for key in report_module.ANALYTICAL_SECTION_NAMES
        })
        client.chat.completions.parse.side_effect = [
            _mock_parsed_response(parsed),
            _mock_parsed_response(_evaluation(needs_revision=False)),
        ]

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        assert {p["paper_id"] for p in artifact["selected_papers"]} == {"1111", "2222"}

    def test_template_and_topic_preserved(self):
        p1 = _paper("1111", "Paper One")
        session = _session(topic="A very specific topic", selected_papers=[p1])
        client = _no_revision_client(template="expert")

        artifact = capture_real_refinement_pair(
            session, client, report_template="expert", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        assert artifact["topic"] == "A very specific topic"
        assert artifact["template"] == "expert"
        assert artifact["draft_report"]["report_template"] == "expert"
        assert artifact["refined_report"]["report_template"] == "expert"


# --- Session/production-state immutability ---------------------------------

class TestSessionImmutability:
    def test_session_remains_deep_equal_before_and_after_capture(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        before = copy.deepcopy(session)
        client = _revision_client()  # exercise the "more invasive" branch

        capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )

        assert session == before
        assert session.report is None
        assert session.report_versions == []
        assert session.active_report_version_id is None
        assert session.stage == "synthesize"
        assert session.selected_papers is not None and session.selected_papers[0] is p1

    def test_no_report_version_appended(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        with patch.object(report_module, "append_report_version") as spy:
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )
        spy.assert_not_called()
        assert session.report_versions == []

    def test_no_save_or_checkpointer_call(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        with patch("research_agent.curation_session.save_curation_session") as spy:
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )
        spy.assert_not_called()

    def test_module_never_imports_curation_session_save_or_append_version(self):
        """Structural guarantee: it is not merely untested, it is
        impossible for this module to call either function, since
        neither is imported at all."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(r6d4_capture))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
        assert not any("save_curation_session" in n for n in names)
        assert not any("append_report_version" in n for n in names)
        assert not any(n == "research_agent.curation_session" or n.startswith("research_agent.curation_session.") for n in names)


# --- Privacy: no chat/turn/raw-identifier capture --------------------------

class TestPrivacy:
    def test_no_chat_or_turn_history_or_raw_session_identifier_captured(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        session.chat_history = [{"role": "user", "content": "secret chat content"}]
        session.turn_history = [{"turn_number": 1, "batch": [], "refilled": False}]
        client = _no_revision_client()

        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )

        assert find_forbidden_keys(artifact) == []
        dumped = json.dumps(artifact)
        assert "secret chat content" not in dumped
        assert "chat_history" not in dumped
        assert "turn_history" not in dumped

    def test_find_forbidden_keys_recursively_detects_poisoned_structure(self):
        poisoned = {
            "outer": {"chat_history": ["x"]},
            "list": [{"api_key": "sk-secret"}, {"nested": {"OPENAI_API_KEY": "sk-secret"}}],
            "clean": {"topic": "fine"},
        }
        found = find_forbidden_keys(poisoned)
        assert any("chat_history" in f for f in found)
        assert any("api_key" in f.lower() for f in found)
        assert any("openai_api_key" in f.lower() for f in found)

    def test_find_forbidden_keys_reports_nothing_for_clean_structure(self):
        clean = {"topic": "x", "selected_papers": [{"paper_id": "1"}]}
        assert find_forbidden_keys(clean) == []

    def test_raw_session_id_never_a_capture_parameter(self):
        """Structural guarantee: the function signature has no
        `session_id` parameter at all -- there is no argument through
        which a raw session id could even be passed in."""
        import inspect
        sig = inspect.signature(capture_real_refinement_pair)
        assert "session_id" not in sig.parameters

    def test_source_session_ref_rejects_curation_thread_id_shape(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        with pytest.raises(R6D4CaptureError, match="opaque evaluation reference"):
            capture_real_refinement_pair(
                session, MagicMock(), report_template="analytical", pair_id="p1",
                source_session_ref="curation-session:abc123",
            )

    def test_source_session_ref_rejects_uuid4_hex_shape(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        with pytest.raises(R6D4CaptureError, match="opaque evaluation reference"):
            capture_real_refinement_pair(
                session, MagicMock(), report_template="analytical", pair_id="p1",
                source_session_ref="9f8e7d6c5b4a39281706f5e4d3c2b1a0",
            )

    def test_source_session_ref_accepts_a_normal_opaque_label(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-foundational-01",
        )
        assert artifact["refinement_context"]["source_session_ref"] == "real-pair-foundational-01"


# --- Capture schema / labelling ---------------------------------------------

class TestCaptureSchema:
    def test_schema_version_is_r6d4_capture_v1(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        assert artifact["schema_version"] == "r6d4-capture-v1"
        assert artifact["refinement_context"]["source_origin"] == "real_r4_generated"
        assert artifact["refinement_context"]["refinement_mode"] == "single"

    def test_no_expected_or_winner_or_acceptance_fields_anywhere(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        for forbidden in ("expected", "expected_dimension_directions", "overall_direction", "overall_score", "winner", "accept_refinement"):
            assert forbidden not in artifact
            assert forbidden not in artifact["refinement_context"]

    def test_no_evaluator_model_or_prompt_version_captured_yet(self):
        """Evaluator metadata belongs to a LATER live-run's own detail
        JSON, never to the capture artifact itself."""
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        for forbidden in ("evaluator_model", "claim_source_prompt_version", "pairwise_holistic_prompt_version"):
            assert forbidden not in artifact
            assert forbidden not in artifact["refinement_context"]

    def test_generation_model_and_commit_sha_recorded(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            source_commit_sha="abc1234",
        )
        assert artifact["refinement_context"]["generation_model"] == report_module.REPORT_MODEL
        assert artifact["refinement_context"]["capture_commit_sha"] == "abc1234"

    def test_commit_sha_defaults_to_none_never_fabricated(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        )
        assert artifact["refinement_context"]["capture_commit_sha"] is None

    def test_explicit_now_used_verbatim(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()
        artifact = capture_real_refinement_pair(
            session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            now="2026-08-11T12:00:00+00:00",
        )
        assert artifact["refinement_context"]["capture_timestamp"] == "2026-08-11T12:00:00+00:00"

    def test_malformed_now_rejected(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        with pytest.raises(R6D4CaptureError, match="capture_timestamp"):
            capture_real_refinement_pair(
                session, MagicMock(), report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
                now="not-a-real-timestamp",
            )

    def test_naive_now_without_tzinfo_rejected(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        with pytest.raises(R6D4CaptureError, match="timezone-aware"):
            capture_real_refinement_pair(
                session, MagicMock(), report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
                now="2026-08-11T12:00:00",
            )


# --- validate_r6d4_capture ---------------------------------------------------

def _valid_artifact_for_validator():
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    client = _no_revision_client()
    return capture_real_refinement_pair(
        session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
        now="2026-08-11T12:00:00+00:00",
    )


class TestValidateR6D4Capture:
    def test_a_real_captured_artifact_validates_cleanly(self):
        artifact = _valid_artifact_for_validator()
        validate_r6d4_capture(artifact)  # must not raise

    def test_wrong_schema_version_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["schema_version"] = "r6d1-v1"
        with pytest.raises(R6D4CaptureError, match="schema_version"):
            validate_r6d4_capture(artifact)

    def test_missing_evidence_reference_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["draft_report"]["references"].append({"number": 999, "kind": "paper", "paper_id": "does-not-exist"})
        with pytest.raises(R6D4CaptureError, match="not resolvable"):
            validate_r6d4_capture(artifact)

    def test_refinement_key_in_report_body_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["draft_report"]["refinement"] = {"enabled": True}
        with pytest.raises(R6D4CaptureError, match="refinement"):
            validate_r6d4_capture(artifact)

    def test_expected_field_anywhere_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["expected"] = {"dimension_directions": {}}
        with pytest.raises(R6D4CaptureError, match="expected"):
            validate_r6d4_capture(artifact)

    def test_winner_field_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["winner"] = "refined"
        with pytest.raises(R6D4CaptureError):
            validate_r6d4_capture(artifact)

    def test_forbidden_recursive_key_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["refinement_context"]["chat_history"] = []
        with pytest.raises(R6D4CaptureError, match="forbidden"):
            validate_r6d4_capture(artifact)

    def test_malformed_timestamp_rejected_by_validator(self):
        artifact = _valid_artifact_for_validator()
        artifact["refinement_context"]["capture_timestamp"] = "garbage"
        with pytest.raises(R6D4CaptureError, match="capture_timestamp"):
            validate_r6d4_capture(artifact)

    def test_raw_session_ref_rejected_by_validator(self):
        artifact = _valid_artifact_for_validator()
        artifact["refinement_context"]["source_session_ref"] = "curation-session:xyz"
        with pytest.raises(R6D4CaptureError, match="opaque evaluation reference"):
            validate_r6d4_capture(artifact)

    def test_missing_r4_refinement_metadata_rejected(self):
        artifact = _valid_artifact_for_validator()
        del artifact["refinement_context"]["r4_refinement_metadata"]
        with pytest.raises(R6D4CaptureError, match="r4_refinement_metadata"):
            validate_r6d4_capture(artifact)

    def test_contradictory_revision_applied_rejected_by_validator(self):
        artifact = _valid_artifact_for_validator()
        artifact["refinement_context"]["revision_applied"] = True  # bodies are actually equal
        with pytest.raises(R6D4CaptureError, match="contradictory provenance"):
            validate_r6d4_capture(artifact)

    def test_wrong_source_origin_rejected(self):
        artifact = _valid_artifact_for_validator()
        artifact["refinement_context"]["source_origin"] = "synthetic_handcrafted"
        with pytest.raises(R6D4CaptureError, match="source_origin"):
            validate_r6d4_capture(artifact)

    def test_kept_separate_from_r6d1_loader(self):
        """This module's own validator never imports from, and is
        never imported by, `report_refinement_inputs.py` -- the frozen
        synthetic schema stays completely untouched."""
        import ast
        import inspect
        from research_agent.evals import report_refinement_inputs

        tree = ast.parse(inspect.getsource(r6d4_capture))
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
        assert not any("report_refinement_inputs" in (n or "") for n in names)

        tree2 = ast.parse(inspect.getsource(report_refinement_inputs))
        names2 = [n.module for n in ast.walk(tree2) if isinstance(n, ast.ImportFrom) and n.module]
        assert not any("r6d4_capture" in (n or "") for n in names2)


# --- Failure handling: no partial artifact ---------------------------------

class TestFailureHandling:
    def test_generation_failure_returns_no_artifact(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = MagicMock()
        client.chat.completions.parse.side_effect = RuntimeError("generation exploded")

        with pytest.raises(RuntimeError, match="generation exploded"):
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )

    def test_evaluation_failure_returns_no_artifact(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        schema = report_module._build_report_schema(["1111"], None, REPORT_TEMPLATES["analytical"])
        parsed = _full_analytical_parsed(schema, "1111")
        client = MagicMock()
        client.chat.completions.parse.side_effect = [
            _mock_parsed_response(parsed),
            RuntimeError("evaluation exploded"),
        ]

        with pytest.raises(RuntimeError, match="evaluation exploded"):
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )

    def test_revision_failure_returns_no_artifact(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        schema = report_module._build_report_schema(["1111"], None, REPORT_TEMPLATES["analytical"])
        parsed = _full_analytical_parsed(schema, "1111")
        client = MagicMock()
        client.chat.completions.parse.side_effect = [
            _mock_parsed_response(parsed),
            _mock_parsed_response(_evaluation(needs_revision=True, issues=["x"], revision_instructions="y")),
            RuntimeError("revision exploded"),
        ]

        with pytest.raises(RuntimeError, match="revision exploded"):
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )


# --- No file writes / no network -------------------------------------------

class TestNoFileWritesOrNetwork:
    def test_zero_file_writes_during_capture(self):
        p1 = _paper("1111", "Paper One")
        session = _session(selected_papers=[p1])
        client = _no_revision_client()

        with patch("builtins.open") as mock_open:
            capture_real_refinement_pair(
                session, client, report_template="analytical", pair_id="p1", source_session_ref="real-pair-test-01",
            )
        mock_open.assert_not_called()

    def test_module_never_constructs_its_own_openai_client(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(r6d4_capture))
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert "OpenAI" not in calls

    def test_module_imports_openai_only_for_typing(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(r6d4_capture))
        import_froms = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        openai_imports = [node for node in import_froms if node.module == "openai"]
        assert len(openai_imports) == 1
        assert [alias.name for alias in openai_imports[0].names] == ["OpenAI"]
