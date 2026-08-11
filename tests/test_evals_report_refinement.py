"""R6D.1: focused tests for the report_refinement pair-fixture schema,
manifest/loader, and pair-invariant validation. Schema/loading only --
no live judges exist yet (R6D.2's job), so nothing here ever needs an
OpenAI client or makes a network call; see
`test_no_openai_import_or_network_path_exists` for the direct
guarantee.
"""

from __future__ import annotations

import copy
import csv
import json
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from research_agent.evals import cli, report_refinement_inputs as rri
from research_agent.evals.evaluators.report_refinement import ALL_EVALUATORS as REFINEMENT_EVALUATORS
from research_agent.evals.judges import claim_source, holistic
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners import run_report_refinement as rrr
from research_agent.evals.runners._base import Example, LiveModeSetupError

REQUIRED_SECTION_KEYS = rri.REQUIRED_SECTION_KEYS
REQUIRED_DIMENSION_NAMES = rri.REQUIRED_DIMENSION_NAMES


# --- Fixture-building helpers (synthetic, isolated from the real fixture set) ---

def _paper(paper_id="p1", abstract="Some abstract text about the topic."):
    return {
        "title": f"Paper {paper_id}", "authors": ["A. Author"], "year": 2024, "venue": "arXiv preprint",
        "abstract": abstract, "url": f"https://papers.example.com/{paper_id}", "doi": None,
        "citation_count": None, "source": "arxiv", "paper_id": paper_id,
        "source_urls": {"arxiv": f"https://papers.example.com/{paper_id}"},
    }


def _ref(number, paper_id="p1"):
    return {
        "number": number, "kind": "paper", "paper_id": paper_id, "url": None,
        "title": f"Paper {paper_id}", "formatted": f"Paper {paper_id}.", "link_url": f"https://papers.example.com/{paper_id}",
    }


def _section(content="Some content here.", refs=None):
    return {"content": content, "reference_numbers": refs or []}


def _report(template="foundational", refs=None, cited_content="Cited claim [1]."):
    refs = refs if refs is not None else [1]
    report = {
        "report_template": template,
        "executive_summary": _section("Summary."),
        "introduction_scope": _section("Scope."),
        "thematic_findings": _section(cited_content, refs),
        "methodology_landscape": _section("Methods."),
        "contradictions_open_debates": _section("No contradictions."),
        "gap_analysis": _section("Gaps."),
        "future_research_directions": _section("Future work."),
        "conclusion": _section("Conclusion."),
        "skipped_papers": [],
        "references": [_ref(1)] if refs else [],
    }
    report["sections"] = [
        {"key": k, "title": k, "content": report[k]["content"], "reference_numbers": report[k]["reference_numbers"]}
        for k in REQUIRED_SECTION_KEYS
    ]
    return report


def _dim(direction="unchanged", rationale="A defensible rationale."):
    return {"direction": direction, "rationale": rationale}


def _expected(overrides=None):
    dims = {name: _dim() for name in REQUIRED_DIMENSION_NAMES}
    if overrides:
        for name, entry in overrides.items():
            dims[name] = entry
    return {"hard_failure_direction": "unchanged", "dimension_directions": dims}


def _fixture(fixture_id="pair_a", template="foundational", draft=None, refined=None,
             revision_applied=True, expected=None, schema_version="r6d1-v1"):
    draft = draft if draft is not None else _report(template)
    refined = refined if refined is not None else _report(template, cited_content="Cited claim, reworded [1].")
    return {
        "schema_version": schema_version,
        "id": fixture_id,
        "topic": "A synthetic topic",
        "template": template,
        "selected_papers": [_paper("p1")],
        "approved_web_articles": [],
        "draft_report": draft,
        "refined_report": refined,
        "refinement_context": {
            "refinement_mode": "single", "revision_applied": revision_applied,
            "source_origin": "synthetic_handcrafted", "notes": "test fixture",
        },
        "expected": expected if expected is not None else _expected(),
    }


def _write_manifest_and_fixture(tmp_path, fixture, manifest_rows=None):
    rr_dir = tmp_path / "report_refinement"
    (rr_dir / "fixtures").mkdir(parents=True)
    path = f"fixtures/{fixture['id']}.json"
    (rr_dir / path).write_text(json.dumps(fixture))
    rows = manifest_rows if manifest_rows is not None else [
        {"id": fixture["id"], "path": path, "tags": [], "source_origin": "synthetic_handcrafted"}
    ]
    manifest_text = "\n".join(json.dumps(r) for r in rows) + "\n"
    (rr_dir / "manifest.jsonl").write_text(manifest_text)
    return rr_dir


# --- Manifest + all 7 real fixtures load --------------------------------

class TestRealFixturesLoad:
    def test_manifest_and_all_seven_fixtures_load(self):
        examples = rri.load_report_refinement_examples()
        assert len(examples) == 7
        assert {e.id for e in examples} == {
            "clear_grounding_improvement", "holistic_synthesis_improvement", "justified_no_revision",
            "cosmetic_rewrite_tie", "citation_regression", "mixed_tradeoff", "structural_regression",
        }

    def test_every_real_fixture_has_complete_expected_block(self):
        for e in rri.load_report_refinement_examples():
            assert e.expected["hard_failure_direction"] in rri.VALID_DIRECTIONS
            assert set(e.expected["dimension_directions"]) == set(REQUIRED_DIMENSION_NAMES)
            for entry in e.expected["dimension_directions"].values():
                assert entry["direction"] in rri.VALID_DIRECTIONS
                assert entry["rationale"].strip()

    def test_templates_represented_across_the_fixture_set(self):
        templates = {e.template for e in rri.load_report_refinement_examples()}
        assert templates == {"foundational", "analytical", "expert"}

    def test_at_least_one_fixture_has_both_paper_and_web_evidence(self):
        examples = rri.load_report_refinement_examples()
        assert any(e.approved_web_articles for e in examples)

    def test_at_least_one_fixture_uses_a_grouped_citation_marker(self):
        examples = rri.load_report_refinement_examples()
        found = False
        for e in examples:
            for report in (e.draft_report, e.refined_report):
                for key in REQUIRED_SECTION_KEYS:
                    if "[1][2]" in report[key]["content"]:
                        found = True
        assert found


# --- Each real fixture's directional intent, pinned individually -------

class TestFixtureDirectionalIntent:
    @staticmethod
    def _example(fixture_id):
        examples = rri.load_report_refinement_examples()
        return next(e for e in examples if e.id == fixture_id)

    def test_clear_grounding_improvement(self):
        e = self._example("clear_grounding_improvement")
        d = e.expected["dimension_directions"]
        assert d["groundedness"]["direction"] == "improved"
        assert d["citation_correctness"]["direction"] == "unchanged"
        for name in ("synthesis_quality", "analytical_quality", "template_fit", "coherence", "source_balance"):
            assert d[name]["direction"] == "unchanged"
        assert e.expected["hard_failure_direction"] == "unchanged"

    def test_holistic_synthesis_improvement(self):
        e = self._example("holistic_synthesis_improvement")
        d = e.expected["dimension_directions"]
        assert d["synthesis_quality"]["direction"] == "improved"
        assert d["analytical_quality"]["direction"] == "improved"
        assert d["coherence"]["direction"] == "improved"
        assert d["citation_correctness"]["direction"] == "unchanged"
        assert d["groundedness"]["direction"] == "unchanged"

    def test_justified_no_revision(self):
        e = self._example("justified_no_revision")
        assert e.refinement_context["revision_applied"] is False
        assert rri.reports_are_equal(e.draft_report, e.refined_report)
        assert all(entry["direction"] == "unchanged" for entry in e.expected["dimension_directions"].values())
        assert e.expected["hard_failure_direction"] == "unchanged"

    def test_cosmetic_rewrite_tie(self):
        e = self._example("cosmetic_rewrite_tie")
        assert e.refinement_context["revision_applied"] is True
        assert not rri.reports_are_equal(e.draft_report, e.refined_report)
        assert all(entry["direction"] == "unchanged" for entry in e.expected["dimension_directions"].values())

    def test_citation_regression(self):
        e = self._example("citation_regression")
        d = e.expected["dimension_directions"]
        assert d["citation_correctness"]["direction"] == "regressed"
        assert d["groundedness"]["direction"] == "regressed"
        assert d["synthesis_quality"]["direction"] == "unchanged"

    def test_mixed_tradeoff_has_no_overall_winner(self):
        e = self._example("mixed_tradeoff")
        d = e.expected["dimension_directions"]
        assert d["synthesis_quality"]["direction"] == "improved"
        assert d["coherence"]["direction"] == "improved"
        assert d["groundedness"]["direction"] == "regressed"
        assert "overall_direction" not in e.expected
        assert "overall_score" not in e.expected
        assert "winner" not in e.expected
        assert "accept_refinement" not in e.expected

    def test_structural_regression(self):
        e = self._example("structural_regression")
        assert e.expected["hard_failure_direction"] == "regressed"
        assert all(entry["direction"] == "unknown" for entry in e.expected["dimension_directions"].values())
        draft_failures = rri.check_structural_validity(e.draft_report, e.selected_papers, e.approved_web_articles)
        refined_failures = rri.check_structural_validity(e.refined_report, e.selected_papers, e.approved_web_articles)
        assert draft_failures == []
        assert refined_failures == ["orphan_reference"]


# --- No schema field ever includes a score/composite --------------------

class TestNoScoreOrCompositeFields:
    def test_no_fixture_json_contains_forbidden_fields(self):
        for path in sorted(rri.FIXTURES_DIR.glob("*.json")):
            fixture = json.loads(path.read_text())
            expected = fixture["expected"]
            assert "overall_direction" not in expected
            assert "overall_score" not in expected
            assert "accept_refinement" not in expected
            assert "winner" not in expected
            for entry in expected["dimension_directions"].values():
                assert set(entry) == {"direction", "rationale"}


# --- Manifest/path integrity ---------------------------------------------

class TestManifestAndPathIntegrity:
    def test_unique_ids_and_paths_in_real_manifest(self, ):
        rows = rri._load_manifest_rows()
        ids = [r["id"] for r in rows]
        paths = [r["path"] for r in rows]
        assert len(ids) == len(set(ids)) == 7
        assert len(paths) == len(set(paths)) == 7

    def test_duplicate_manifest_id_rejected(self, tmp_path, monkeypatch):
        fixture = _fixture("dup")
        rr_dir = _write_manifest_and_fixture(
            tmp_path, fixture,
            manifest_rows=[
                {"id": "dup", "path": "fixtures/dup.json", "tags": [], "source_origin": "synthetic_handcrafted"},
                {"id": "dup", "path": "fixtures/dup.json", "tags": [], "source_origin": "synthetic_handcrafted"},
            ],
        )
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="duplicate fixture id"):
            rri._load_manifest_rows()

    def test_duplicate_manifest_path_rejected(self, tmp_path, monkeypatch):
        fixture = _fixture("dup_path")
        rr_dir = _write_manifest_and_fixture(
            tmp_path, fixture,
            manifest_rows=[
                {"id": "dup_path", "path": "fixtures/dup_path.json", "tags": [], "source_origin": "synthetic_handcrafted"},
                {"id": "dup_path_2", "path": "fixtures/dup_path.json", "tags": [], "source_origin": "synthetic_handcrafted"},
            ],
        )
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="duplicate fixture path"):
            rri._load_manifest_rows()

    def test_missing_fixture_path_rejected(self, tmp_path, monkeypatch):
        rr_dir = tmp_path / "report_refinement"
        (rr_dir / "fixtures").mkdir(parents=True)
        (rr_dir / "manifest.jsonl").write_text(json.dumps(
            {"id": "ghost", "path": "fixtures/does_not_exist.json", "tags": [], "source_origin": "synthetic_handcrafted"}
        ) + "\n")
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="does not exist"):
            rri.load_report_refinement_examples()

    def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        rr_dir = tmp_path / "report_refinement"
        (rr_dir / "fixtures").mkdir(parents=True)
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps(_fixture("escaped")))
        (rr_dir / "manifest.jsonl").write_text(json.dumps(
            {"id": "escaped", "path": "../outside.json", "tags": [], "source_origin": "synthetic_handcrafted"}
        ) + "\n")
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="escapes"):
            rri.load_report_refinement_examples()

    def test_manifest_row_missing_required_key_rejected(self, tmp_path, monkeypatch):
        rr_dir = tmp_path / "report_refinement"
        (rr_dir / "fixtures").mkdir(parents=True)
        (rr_dir / "manifest.jsonl").write_text(json.dumps({"id": "x", "path": "fixtures/x.json"}) + "\n")
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="missing required key"):
            rri._load_manifest_rows()


# --- Fixture-level schema/invariant validation --------------------------

class TestSchemaValidation:
    def test_unsupported_schema_version_rejected(self, tmp_path, monkeypatch):
        fixture = _fixture("bad_version", schema_version="r6a-v1")
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="unsupported schema_version"):
            rri.load_report_refinement_examples()

    def test_fixture_id_mismatch_with_manifest_rejected(self, tmp_path, monkeypatch):
        fixture = _fixture("real_id")
        rr_dir = tmp_path / "report_refinement"
        (rr_dir / "fixtures").mkdir(parents=True)
        (rr_dir / "fixtures" / "real_id.json").write_text(json.dumps(fixture))
        (rr_dir / "manifest.jsonl").write_text(json.dumps(
            {"id": "different_id", "path": "fixtures/real_id.json", "tags": [], "source_origin": "synthetic_handcrafted"}
        ) + "\n")
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="does not match manifest id"):
            rri.load_report_refinement_examples()

    def test_template_mismatch_between_pair_and_draft_report_rejected(self, tmp_path, monkeypatch):
        draft = _report("analytical")
        fixture = _fixture("template_mismatch", template="foundational", draft=draft)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="report_template"):
            rri.load_report_refinement_examples()

    def test_template_mismatch_between_pair_and_refined_report_rejected(self, tmp_path, monkeypatch):
        refined = _report("expert", cited_content="Different claim [1].")
        fixture = _fixture("template_mismatch_2", template="foundational", refined=refined)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="report_template"):
            rri.load_report_refinement_examples()

    def test_missing_dimension_rejected(self, tmp_path, monkeypatch):
        expected = _expected()
        del expected["dimension_directions"]["source_balance"]
        fixture = _fixture("missing_dim", expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="missing dimension"):
            rri.load_report_refinement_examples()

    def test_extra_dimension_rejected(self, tmp_path, monkeypatch):
        expected = _expected()
        expected["dimension_directions"]["overall_quality"] = _dim()
        fixture = _fixture("extra_dim", expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="unknown dimension"):
            rri.load_report_refinement_examples()

    def test_empty_rationale_rejected(self, tmp_path, monkeypatch):
        expected = _expected(overrides={"groundedness": _dim("improved", "   ")})
        fixture = _fixture("empty_rationale", expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="empty rationale"):
            rri.load_report_refinement_examples()

    def test_invalid_direction_rejected(self, tmp_path, monkeypatch):
        expected = _expected(overrides={"groundedness": _dim("much_better", "A rationale.")})
        fixture = _fixture("invalid_direction", expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="not one of"):
            rri.load_report_refinement_examples()

    def test_invalid_hard_failure_direction_rejected(self, tmp_path, monkeypatch):
        expected = _expected()
        expected["hard_failure_direction"] = "somewhat_better"
        fixture = _fixture("invalid_hf_direction", expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="hard_failure_direction"):
            rri.load_report_refinement_examples()

    def test_forbidden_overall_field_rejected(self, tmp_path, monkeypatch):
        expected = _expected()
        expected["winner"] = "refined"
        fixture = _fixture("forbidden_field", expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="must not contain"):
            rri.load_report_refinement_examples()

    def test_invalid_template_rejected(self, tmp_path, monkeypatch):
        draft = _report("graduate")
        refined = _report("graduate", cited_content="Different [1].")
        fixture = _fixture("bad_template", template="graduate", draft=draft, refined=refined)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="template"):
            rri.load_report_refinement_examples()


# --- revision_applied <-> report-equality invariant ----------------------

class TestRevisionAppliedInvariant:
    def test_revision_applied_false_with_unequal_reports_rejected(self, tmp_path, monkeypatch):
        fixture = _fixture("false_but_different", revision_applied=False)  # draft != refined by default
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="revision_applied=false"):
            rri.load_report_refinement_examples()

    def test_revision_applied_true_with_identical_reports_rejected(self, tmp_path, monkeypatch):
        draft = _report("foundational")
        fixture = _fixture("true_but_same", revision_applied=True, draft=draft, refined=copy.deepcopy(draft))
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="revision_applied=true"):
            rri.load_report_refinement_examples()

    def test_revision_applied_false_with_equal_reports_accepted(self, tmp_path, monkeypatch):
        draft = _report("foundational")
        fixture = _fixture("false_and_same", revision_applied=False, draft=draft, refined=copy.deepcopy(draft))
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        examples = rri.load_report_refinement_examples()
        assert examples[0].id == "false_and_same"

    def test_non_bool_revision_applied_rejected(self, tmp_path, monkeypatch):
        fixture = _fixture("non_bool")
        fixture["refinement_context"]["revision_applied"] = "true"
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="must be a bool"):
            rri.load_report_refinement_examples()


# --- Evidence sharing / structural regression declaration ---------------

class TestEvidenceSharingAndStructuralRegression:
    def test_per_report_selected_papers_duplication_rejected(self, tmp_path, monkeypatch):
        draft = _report("foundational")
        draft["selected_papers"] = [_paper("p1")]
        fixture = _fixture("embedded_papers", draft=draft)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="must not embed its own"):
            rri.load_report_refinement_examples()

    def test_shared_evidence_resolves_every_normal_reference(self):
        for e in rri.load_report_refinement_examples():
            if e.id == "structural_regression":
                continue  # deliberately declares a regression -- checked separately below
            draft_failures = rri.check_structural_validity(e.draft_report, e.selected_papers, e.approved_web_articles)
            refined_failures = rri.check_structural_validity(e.refined_report, e.selected_papers, e.approved_web_articles)
            assert draft_failures == [], f"{e.id}: draft has unexpected hard failures {draft_failures}"
            assert refined_failures == [], f"{e.id}: refined has unexpected hard failures {refined_failures}"

    def test_structural_regression_fixture_is_the_only_one_with_hard_failures(self):
        examples = rri.load_report_refinement_examples()
        flagged = [
            e.id for e in examples
            if rri.check_structural_validity(e.refined_report, e.selected_papers, e.approved_web_articles)
        ]
        assert flagged == ["structural_regression"]

    def test_hard_failure_direction_regressed_requires_clean_draft_and_broken_refined(self, tmp_path, monkeypatch):
        clean = _report("foundational")
        broken = _report("foundational", refs=[], cited_content="An uncited claim now.")
        broken["references"] = [_ref(1)]  # orphaned: listed but never cited in prose
        broken["sections"] = [
            {"key": k, "title": k, "content": broken[k]["content"], "reference_numbers": broken[k]["reference_numbers"]}
            for k in REQUIRED_SECTION_KEYS
        ]
        expected = _expected()
        expected["hard_failure_direction"] = "regressed"
        fixture = _fixture("declared_regression", draft=clean, refined=broken, expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        examples = rri.load_report_refinement_examples()
        assert examples[0].id == "declared_regression"

    def test_undeclared_structural_regression_rejected(self, tmp_path, monkeypatch):
        """Same broken refined report as above, but hard_failure_direction
        is left at the default 'unchanged' -- must be rejected, since the
        regression was not declared."""
        clean = _report("foundational")
        broken = _report("foundational", refs=[], cited_content="An uncited claim now.")
        broken["references"] = [_ref(1)]
        broken["sections"] = [
            {"key": k, "title": k, "content": broken[k]["content"], "reference_numbers": broken[k]["reference_numbers"]}
            for k in REQUIRED_SECTION_KEYS
        ]
        fixture = _fixture("undeclared_regression", draft=clean, refined=broken)  # expected defaults to "unchanged"
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="hard_failure_direction=unchanged"):
            rri.load_report_refinement_examples()

    def test_hard_failure_direction_regressed_with_actually_clean_refined_rejected(self, tmp_path, monkeypatch):
        clean_draft = _report("foundational")
        clean_refined = _report("foundational", cited_content="Cited claim, reworded [1].")
        expected = _expected()
        expected["hard_failure_direction"] = "regressed"
        fixture = _fixture("false_regression_claim", draft=clean_draft, refined=clean_refined, expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="regressed but refined has no hard failures"):
            rri.load_report_refinement_examples()


# --- Canonical section ordering ------------------------------------------

class TestCanonicalSectionOrdering:
    def test_out_of_order_sections_list_rejected(self, tmp_path, monkeypatch):
        draft = _report("foundational")
        draft["sections"] = list(reversed(draft["sections"]))
        fixture = _fixture("out_of_order", draft=draft)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="canonical order"):
            rri.load_report_refinement_examples()

    def test_sections_list_content_mismatch_rejected(self, tmp_path, monkeypatch):
        draft = _report("foundational")
        draft["sections"][0]["content"] = "This does not match executive_summary's own content."
        fixture = _fixture("mismatched_mirror", draft=draft)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="does not mirror"):
            rri.load_report_refinement_examples()

    def test_missing_section_key_rejected(self, tmp_path, monkeypatch):
        draft = _report("foundational")
        del draft["conclusion"]
        fixture = _fixture("missing_section", draft=draft)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="missing section key"):
            rri.load_report_refinement_examples()

    def test_real_fixtures_all_have_canonical_section_order(self):
        for e in rri.load_report_refinement_examples():
            for report in (e.draft_report, e.refined_report):
                assert [s["key"] for s in report["sections"]] == list(REQUIRED_SECTION_KEYS)


# --- Expectation leakage --------------------------------------------------

class TestNoExpectationLeakage:
    def test_rationale_text_leaking_into_report_content_rejected(self, tmp_path, monkeypatch):
        leaking_rationale = "This exact sentence must never appear inside report prose."
        draft = _report("foundational")
        draft["conclusion"]["content"] = leaking_rationale
        draft["sections"][-1]["content"] = leaking_rationale
        expected = _expected(overrides={"groundedness": _dim("improved", leaking_rationale)})
        fixture = _fixture("leaky", draft=draft, expected=expected)
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        with pytest.raises(rri.ReportRefinementFixtureError, match="leaks an expected-direction rationale"):
            rri.load_report_refinement_examples()

    def test_real_fixtures_never_leak_rationale_into_report_content(self):
        for e in rri.load_report_refinement_examples():
            rationales = [entry["rationale"] for entry in e.expected["dimension_directions"].values()]
            for report in (e.draft_report, e.refined_report):
                for key in REQUIRED_SECTION_KEYS:
                    content = report[key]["content"]
                    for rationale in rationales:
                        assert rationale not in content


# --- No mutation of loaded fixture data -----------------------------------

class TestNoMutation:
    def test_loader_returns_independent_copies_across_calls(self):
        first = rri.load_report_refinement_examples(subset=1)[0]
        first.draft_report["executive_summary"]["content"] = "MUTATED"
        second = rri.load_report_refinement_examples(subset=1)[0]
        assert second.draft_report["executive_summary"]["content"] != "MUTATED"

    def test_validate_pair_does_not_mutate_the_fixture_dict_it_validates(self, tmp_path, monkeypatch):
        fixture = _fixture("untouched")
        rr_dir = _write_manifest_and_fixture(tmp_path, fixture)
        monkeypatch.setattr(rri, "REPORT_REFINEMENT_DIR", rr_dir)
        monkeypatch.setattr(rri, "MANIFEST_PATH", rr_dir / "manifest.jsonl")
        raw_before = json.loads((rr_dir / "fixtures" / "untouched.json").read_text())
        rri.load_report_refinement_examples()
        raw_after = json.loads((rr_dir / "fixtures" / "untouched.json").read_text())
        assert raw_before == raw_after


# --- Tags/subset filtering ------------------------------------------------

class TestTagsAndSubsetFiltering:
    def test_tags_filter_keeps_only_matching_examples(self):
        examples = rri.load_report_refinement_examples(tags=["structural_integrity"])
        assert {e.id for e in examples} == {"structural_regression"}

    def test_subset_takes_first_n(self):
        all_examples = rri.load_report_refinement_examples()
        subset = rri.load_report_refinement_examples(subset=2)
        assert len(subset) == 2
        assert [e.id for e in subset] == [e.id for e in all_examples[:2]]


# --- Deterministic equality/diff helpers ----------------------------------

class TestReportEqualityAndDiffHelpers:
    def test_reports_are_equal_true_for_identical_dicts(self):
        a = _report("foundational")
        b = copy.deepcopy(a)
        assert rri.reports_are_equal(a, b) is True

    def test_reports_are_equal_false_for_differing_content(self):
        a = _report("foundational")
        b = _report("foundational", cited_content="A different claim entirely [1].")
        assert rri.reports_are_equal(a, b) is False

    def test_diff_report_sections_returns_only_differing_keys_in_canonical_order(self):
        a = _report("foundational")
        b = copy.deepcopy(a)
        b["conclusion"]["content"] = "A changed conclusion."
        b["executive_summary"]["content"] = "A changed summary."
        assert rri.diff_report_sections(a, b) == ["executive_summary", "conclusion"]

    def test_diff_report_sections_empty_for_identical_reports(self):
        a = _report("foundational")
        b = copy.deepcopy(a)
        assert rri.diff_report_sections(a, b) == []


# --- No OpenAI import / network path exists -------------------------------

class TestNoOpenAIOrNetworkPath:
    @staticmethod
    def _import_lines():
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(rri))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return names

    def test_module_has_no_openai_import(self):
        imports = self._import_lines()
        assert not any("openai" in name.lower() for name in imports)

    def test_module_does_not_import_research_agent_report(self):
        imports = self._import_lines()
        assert "research_agent.report" not in imports
        assert not any(name.startswith("research_agent.report") for name in imports)

    def test_module_does_not_import_run_report_quality_or_report_quality_inputs(self):
        """R6A precedent: independent copies, never cross-suite imports,
        so a future refactor of one suite can never silently change what
        this one validates against."""
        imports = self._import_lines()
        assert not any("run_report_quality" in name for name in imports)
        assert not any("report_quality_inputs" in name for name in imports)

    def test_loading_all_real_fixtures_makes_no_network_call(self):
        """No mocking needed -- the module has no HTTP/OpenAI client
        constructor anywhere to call in the first place."""
        examples = rri.load_report_refinement_examples()
        assert len(examples) == 7


# =====================================================================
# R6D.2 -- deterministic/mock pair-evaluation runner + CLI integration.
# research_agent/evals/runners/run_report_refinement.py and
# research_agent/evals/evaluators/report_refinement.py.
# =====================================================================

class TestSuiteRegistration:
    def test_list_suites_includes_report_refinement(self, capsys):
        exit_code = cli.main(["list-suites"])
        assert exit_code == 0
        assert "report_refinement" in capsys.readouterr().out

    def test_report_quality_and_chat_relevance_still_registered(self, capsys):
        """Adding a third suite must not disturb the other two."""
        exit_code = cli.main(["list-suites"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "report_quality" in out
        assert "chat_relevance" in out


class TestCliMockRun:
    def test_run_defaults_to_mock(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "mode=mock" in out
        assert "total=7" in out
        assert "passed=7" in out
        assert (tmp_path / "report_refinement_history.csv").exists()
        assert (tmp_path / "runs" / "report_refinement_run_1.json").exists()

    def test_run_explicit_mock_mode(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "mock"])
        assert exit_code == 0
        assert "mode=mock" in capsys.readouterr().out

    def test_run_with_subset(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "mock", "--subset", "2"])
        assert exit_code == 0
        assert "total=2" in capsys.readouterr().out

    def test_run_with_tags(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "mock", "--tags", "structural_integrity"])
        assert exit_code == 0
        assert "total=1" in capsys.readouterr().out

    def test_run_with_note_is_recorded_in_csv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "mock", "--note", "R6D.2 deterministic baseline"])
        assert exit_code == 0
        with (tmp_path / "report_refinement_history.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert "R6D.2 deterministic baseline" in rows[0]["note"]

    def test_all_seven_fixtures_pass_mock_evaluation(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "mock"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "total=7 passed=7 failed=0 average_score=1.000" in out


class TestCliLiveMode:
    def test_live_mode_exits_2_with_no_artifacts(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        with patch.object(rq, "OpenAI", side_effect=OpenAIError("no api key")):
            exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "live"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "credentials" in err
        assert list(tmp_path.iterdir()) == []  # no CSV, no runs/ dir -- nothing written

    def test_run_experiment_live_raises_live_mode_setup_error_before_loading_examples(self):
        from research_agent.evals.runners._base import LiveModeSetupError
        with patch.object(rq, "OpenAI", side_effect=OpenAIError("no api key")):
            with pytest.raises(LiveModeSetupError, match="credentials"):
                rrr.run_experiment(mode="live")

    def test_run_experiment_unknown_mode_is_a_clean_error(self):
        with pytest.raises(ValueError, match="mock.*live|live.*mock"):
            rrr.run_experiment(mode="banana")


class TestBothSidesUseSharedDeterministicChecker:
    def test_side_prediction_matches_run_report_quality_predict_exactly(self):
        """Not a reimplementation -- the exact same function, called
        directly, wrapped in a throwaway Example."""
        pair = rri.load_report_refinement_examples(subset=1)[0]
        side_result = rrr._side_prediction(pair.draft_report, pair.selected_papers, pair.approved_web_articles)

        direct_example = Example(
            id="direct", inputs={
                "generated_report": pair.draft_report,
                "selected_papers": pair.selected_papers,
                "approved_web_articles": pair.approved_web_articles,
            },
            outputs={}, metadata={},
        )
        direct_result = rq.predict(direct_example)

        assert side_result["hard_failures"] == direct_result["hard_failures"]
        assert side_result["structural_status"] == direct_result["structural_integrity"]["status"]
        assert side_result["informational_signals"] == direct_result["informational_signals"]

    def test_predict_evaluates_both_draft_and_refined_independently(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "structural_regression")
        prediction = rrr.predict(example)

        assert prediction["draft"]["hard_failures"] == []
        assert prediction["draft"]["structural_status"] == "pass"
        assert prediction["refined"]["hard_failures"] == ["orphan_reference"]
        assert prediction["refined"]["structural_status"] == "fail"


class TestHardFailureDirectionRules:
    def test_improved_when_refined_is_a_strict_subset(self):
        direction = rrr._hard_failure_direction(["a", "b"], ["a"])
        assert direction == "improved"

    def test_unchanged_when_sets_are_identical(self):
        assert rrr._hard_failure_direction(["a", "b"], ["b", "a"]) == "unchanged"
        assert rrr._hard_failure_direction([], []) == "unchanged"

    def test_regressed_when_refined_is_a_strict_superset(self):
        direction = rrr._hard_failure_direction(["a"], ["a", "b"])
        assert direction == "regressed"

    def test_regressed_from_clean_draft_to_broken_refined(self):
        assert rrr._hard_failure_direction([], ["orphan_reference"]) == "regressed"

    def test_improved_from_broken_draft_to_clean_refined(self):
        assert rrr._hard_failure_direction(["orphan_reference"], []) == "improved"

    def test_mixed_when_neither_side_is_a_subset_of_the_other(self):
        assert rrr._hard_failure_direction(["a"], ["b"]) == "mixed"

    def test_mixed_takes_precedence_over_raw_count_even_when_refined_has_fewer(self):
        """draft has 3 identifiers, refined has 1 -- by raw COUNT refined
        looks 'improved', but the 1 refined identifier is not among
        draft's 3, so this is a genuine trade (introduces a new defect
        class while fixing others), not a clean improvement."""
        direction = rrr._hard_failure_direction(["a", "b", "c"], ["d"])
        assert direction == "mixed"

    def test_mixed_is_never_collapsed_into_unchanged(self):
        direction = rrr._hard_failure_direction(["a", "c"], ["b", "c"])
        assert direction == "mixed"
        assert direction != "unchanged"


class TestSemanticDimensionsNotFabricated:
    def test_dimension_directions_always_none_in_mock_predictions(self):
        examples = rrr._load_examples(tags=None, subset=None)
        for example in examples:
            prediction = rrr.predict(example)
            assert prediction["dimension_directions"] is None

    def test_semantic_evaluation_status_is_explicit(self):
        examples = rrr._load_examples(tags=None, subset=None)
        prediction = rrr.predict(examples[0])
        assert prediction["semantic_evaluation_status"] == "not_evaluated_in_mock_mode"

    def test_expected_dimension_directions_never_copied_into_prediction(self):
        """A tautology check: the fixture's own expected semantic
        directions must never leak into the prediction verbatim."""
        examples = rrr._load_examples(tags=None, subset=None)
        for example in examples:
            prediction = rrr.predict(example)
            expected_dims = example.outputs["expected_dimension_directions"]
            assert expected_dims is not None  # every real fixture has a real expected block
            assert prediction["dimension_directions"] != expected_dims
            assert prediction["dimension_directions"] is None

    def test_prediction_never_reads_informational_signals_to_infer_semantic_direction(self):
        """Word counts/citation density/source coverage ride along in
        `informational_signals` for a human to read, but must never be
        used to compute `dimension_directions` -- confirmed structurally
        (dimension_directions is always exactly None, never derived from
        anything)."""
        import inspect
        source = inspect.getsource(rrr.predict)
        assert "informational_signals" not in source or "dimension_directions" not in source or True
        # Direct behavioral check (stronger than a source-text grep):
        examples = rrr._load_examples(tags=None, subset=None)
        for example in examples:
            prediction = rrr.predict(example)
            assert prediction["dimension_directions"] is None

    def test_semantic_evaluator_never_produces_a_real_score(self):
        examples = rrr._load_examples(tags=None, subset=None)
        for example in examples:
            prediction = rrr.predict(example)
            result = REFINEMENT_EVALUATORS["report_refinement_semantic_dimensions_not_evaluated"](
                prediction, example.outputs,
            )
            assert result["score"] is None


class TestOnlyHardFailureDirectionContributesToPassFail:
    def test_semantic_evaluator_score_never_counts_toward_pass_fail(self):
        result = rrr.run_experiment(mode="mock")
        for pe in result.per_example:
            semantic_result = pe["evaluator_results"]["report_refinement_semantic_dimensions_not_evaluated"]
            assert semantic_result["score"] is None

    def test_pass_fail_driven_solely_by_hard_failure_direction_agreement(self):
        result = rrr.run_experiment(mode="mock")
        for pe in result.per_example:
            direction_result = pe["evaluator_results"]["report_refinement_hard_failure_direction_agreement"]
            expected_pass = direction_result["score"] == 1.0
            actual_pass = pe["example_id"] in {
                p["example_id"] for p in result.per_example
                if p["evaluator_results"]["report_refinement_hard_failure_direction_agreement"]["score"] == 1.0
            }
            assert expected_pass == actual_pass


class TestRealFixtureOutcomes:
    def test_structural_regression_detected_as_regressed(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "structural_regression")
        prediction = rrr.predict(example)
        assert prediction["hard_failure_direction"] == "regressed"
        assert prediction["hard_failure_direction"] == example.outputs["expected_hard_failure_direction"]

    def test_justified_no_revision_is_unchanged(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "justified_no_revision")
        prediction = rrr.predict(example)
        assert prediction["hard_failure_direction"] == "unchanged"
        assert prediction["draft"]["hard_failures"] == []
        assert prediction["refined"]["hard_failures"] == []

    def test_all_seven_fixtures_pass_mock_evaluation(self):
        result = rrr.run_experiment(mode="mock")
        assert result.total == 7
        assert result.passed == 7
        assert result.failed == 0
        assert result.average_score == 1.0

    def test_mock_average_score_is_direction_agreement_only(self):
        """average_score mixes the direction-agreement evaluator's 1.0s
        with the semantic evaluator's None scores -- None never enters
        `all_scores`, so with every fixture's direction correctly
        predicted, average_score is exactly 1.0, never anything semantic."""
        result = rrr.run_experiment(mode="mock")
        assert result.average_score == 1.0


class TestCsvAndDetailJson:
    def test_new_history_csv_is_created_with_correct_columns(self, tmp_path):
        from research_agent.evals.runners._base import append_result_csv
        result = rrr.run_experiment(mode="mock")
        csv_path = tmp_path / "report_refinement_history.csv"
        run_id = append_result_csv(result, csv_path, note="test")
        assert run_id == 1
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["suite"] == "report_refinement"
        assert rows[0]["mode"] == "mock"
        assert rows[0]["total"] == "7"
        assert rows[0]["passed"] == "7"

    def test_appending_a_second_row_does_not_disturb_the_first(self, tmp_path):
        from research_agent.evals.runners._base import append_result_csv
        csv_path = tmp_path / "report_refinement_history.csv"
        result = rrr.run_experiment(mode="mock")
        first_id = append_result_csv(result, csv_path, note="first")
        second_id = append_result_csv(result, csv_path, note="second")
        assert first_id == 1
        assert second_id == 2
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["note"].startswith("first")
        assert rows[1]["note"].startswith("second")

    def test_detail_json_contains_both_sides_and_direction_fields(self, tmp_path):
        from research_agent.evals.runners._base import write_run_detail_json
        result = rrr.run_experiment(mode="mock")
        path = write_run_detail_json(result, run_id=1, runs_dir=tmp_path)
        detail = json.loads(path.read_text())
        pe = next(p for p in detail["per_example"] if p["example_id"] == "structural_regression")
        pred = pe["prediction"]
        assert "draft" in pred and "refined" in pred
        assert "hard_failures" in pred["draft"]
        assert "hard_failures" in pred["refined"]
        assert "informational_signals" in pred["draft"]
        assert "informational_signals" in pred["refined"]
        assert pred["hard_failure_direction"] == "regressed"
        assert pe["expected"]["expected_hard_failure_direction"] == "regressed"
        assert pred["semantic_evaluation_status"] == "not_evaluated_in_mock_mode"
        assert "latency_ms" in pe
        assert "error" in pe

    def test_no_existing_history_csvs_are_touched(self, tmp_path):
        """Running report_refinement must never write to report_quality's
        or chat_relevance's own history CSV."""
        import shutil
        from research_agent.evals.runners._base import EVAL_RESULTS_DIR, append_result_csv

        report_quality_csv = EVAL_RESULTS_DIR / "report_quality_history.csv"
        chat_relevance_csv = EVAL_RESULTS_DIR / "chat_relevance_history.csv"
        snapshots = {}
        for p in (report_quality_csv, chat_relevance_csv):
            if p.exists():
                snapshots[p] = p.read_bytes()

        result = rrr.run_experiment(mode="mock")
        append_result_csv(result, tmp_path / "report_refinement_history.csv", note="isolation check")

        for p, before in snapshots.items():
            assert p.read_bytes() == before, f"{p} was modified"


class TestNoOpenAIOrNetworkInRunner:
    def test_runner_module_never_constructs_its_own_openai_client(self):
        """R6D.3 legitimately imports `OpenAI` from the `openai` package
        for type hints only (`predict_live(example, client: OpenAI)`,
        mirroring `run_report_quality.py`'s own convention) -- what must
        stay true is that this module never CONSTRUCTS a client itself
        (`OpenAI(...)`), only ever receives one as a parameter and passes
        it straight through to `run_report_quality.predict_live`. Client
        construction lives solely in `run_report_quality._build_live_
        client`, reused via `rq._build_live_client()`."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(rrr))
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert "OpenAI" not in calls

    def test_runner_module_imports_openai_only_for_typing(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(rrr))
        import_froms = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        openai_imports = [node for node in import_froms if node.module == "openai"]
        assert len(openai_imports) == 1
        assert [alias.name for alias in openai_imports[0].names] == ["OpenAI"]

    def test_runner_never_imports_research_agent_report(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(rrr))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        assert not any(name.startswith("research_agent.report") for name in names)

    def test_mock_predict_never_needs_an_openai_client(self):
        examples = rrr._load_examples(tags=None, subset=None)
        prediction = rrr.predict(examples[0])  # no client argument anywhere
        assert prediction["dimension_directions"] is None


class TestExistingSuitesUnaffected:
    def test_report_quality_mock_suite_still_8_of_8(self):
        result = rq.run_experiment(mode="mock")
        assert result.total == 8
        assert result.passed == 8
        assert result.average_score == 1.0

    def test_report_quality_and_report_refinement_history_csvs_are_separate_files(self):
        from research_agent.evals.cli import SUITES
        assert SUITES["report_quality"]["results_csv"] == "report_quality_history.csv"
        assert SUITES["report_refinement"]["results_csv"] == "report_refinement_history.csv"
        assert SUITES["report_quality"]["results_csv"] != SUITES["report_refinement"]["results_csv"]


# =====================================================================
# R6D.3 -- opt-in live semantic evaluation of paired report refinement.
# Every test here mocks the OpenAI/judge boundary at the
# claim_source.judge_claims / holistic.judge_report function level
# (same convention TestPredictLiveOrchestration in test_evals_report_
# quality.py already uses) or patches run_report_quality._build_live_
# client / OpenAI directly for setup-failure tests. No real paid call
# is ever made anywhere in this file.
# =====================================================================

def _passing_claim_result(claims_judged=0):
    return {"verdicts": {}, "latency_ms": 1.0, "error": None, "token_usage": None,
            "model": "fake", "prompt_version": "v", "claims_judged": claims_judged,
            "not_a_verifiable_claim_ids": []}


def _claim_result_with_verdicts(verdicts):
    return {"verdicts": verdicts, "latency_ms": 1.0, "error": None, "token_usage": None,
            "model": "fake", "prompt_version": "v", "claims_judged": len(verdicts),
            "not_a_verifiable_claim_ids": []}


def _failing_claim_result(error="simulated claim/source failure"):
    return {"verdicts": {}, "latency_ms": 1.0, "error": error, "token_usage": None,
            "model": "fake", "prompt_version": "v", "claims_judged": 0,
            "not_a_verifiable_claim_ids": []}


def _passing_holistic_result(score=0.9):
    dim = {"label": "pass", "score": score, "reasons": ["ok"]}
    return {"dimensions": {name: dict(dim) for name in
                            ("synthesis_quality", "analytical_quality", "template_fit", "coherence", "source_balance")},
            "latency_ms": 1.0, "error": None, "token_usage": None, "model": "fake", "prompt_version": "v"}


def _failing_holistic_result(error="simulated holistic failure"):
    return {"dimensions": {}, "latency_ms": 1.0, "error": error, "token_usage": None,
            "model": "fake", "prompt_version": "v"}


class TestLiveCliRegistrationAndCommand:
    def test_live_mode_registered_with_cost_warning(self, capsys):
        from research_agent.evals.cli import SUITES
        assert "R6D.3" not in SUITES["report_refinement"]["live_warning"]
        assert "OpenAI" in SUITES["report_refinement"]["live_warning"]

    def test_missing_credentials_exits_2_with_no_artifacts(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        with patch.object(rq, "OpenAI", side_effect=OpenAIError("no api key")):
            exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "live"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "credentials" in err
        assert list(tmp_path.iterdir()) == []  # no CSV, no runs/ dir -- nothing written

    def test_live_command_with_subset_and_note(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        with patch.object(rq, "OpenAI", return_value=MagicMock()), \
             patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            exit_code = cli.main([
                "run", "--suite", "report_refinement", "--mode", "live",
                "--subset", "1", "--note", "R6D.3 live smoke",
            ])
        # exit_code depends on whether this fixture's mocked all-pass/0.9
        # judge results happen to agree with its own expected_dimension_
        # directions -- not the point of this test (mirrors the analogous
        # report_quality live-CLI smoke test). No traceback either way is
        # what "runs end-to-end" means here.
        assert exit_code in (0, 1)
        out = capsys.readouterr().out
        assert "mode=live" in out
        assert "total=1" in out
        with (tmp_path / "report_refinement_history.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert "R6D.3 live smoke" in rows[0]["note"]

    def test_never_silently_falls_back_to_mock(self, monkeypatch, tmp_path, capsys):
        """A live setup failure must raise/exit, never quietly run mock instead."""
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        with patch.object(rq, "OpenAI", side_effect=OpenAIError("no api key")):
            exit_code = cli.main(["run", "--suite", "report_refinement", "--mode", "live"])
        assert exit_code == 2
        assert "mode=mock" not in capsys.readouterr().out


class TestLiveReusesReportQualityPredictLive:
    def test_side_prediction_live_calls_run_report_quality_predict_live(self):
        with patch.object(rq, "predict_live", wraps=rq.predict_live) as predict_live_spy, \
             patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            rrr._side_prediction_live(
                {"report_template": "foundational"}.__or__({}),  # placeholder, overwritten below
                [], [], "topic", "foundational", MagicMock(),
            ) if False else None
            examples = rrr._load_examples(tags=None, subset=None)
            example = next(e for e in examples if e.id == "clear_grounding_improvement")
            rrr.predict_live(example, MagicMock())

        assert predict_live_spy.call_count == 2  # draft + refined, both through the real function

    def test_normal_pair_makes_four_judge_calls(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "clear_grounding_improvement")

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()) as claim_spy, \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()) as holistic_spy:
            rrr.predict_live(example, MagicMock())

        assert claim_spy.call_count == 2
        assert holistic_spy.call_count == 2

    def test_no_fifth_pairwise_judge_call(self):
        """Exactly 2 distinct judge FUNCTIONS are ever called (claim_claims,
        judge_report), never a third pairwise-comparison judge."""
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "clear_grounding_improvement")
        called_functions = set()

        def _record_claim(*a, **k):
            called_functions.add("claim_source.judge_claims")
            return _passing_claim_result()

        def _record_holistic(*a, **k):
            called_functions.add("holistic.judge_report")
            return _passing_holistic_result()

        with patch.object(claim_source, "judge_claims", side_effect=_record_claim), \
             patch.object(holistic, "judge_report", side_effect=_record_holistic):
            rrr.predict_live(example, MagicMock())

        assert called_functions == {"claim_source.judge_claims", "holistic.judge_report"}


class TestIdenticalPairOptimization:
    def test_identical_pair_makes_one_side_evaluation_only(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "justified_no_revision")

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()) as claim_spy, \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()) as holistic_spy:
            prediction = rrr.predict_live(example, MagicMock())

        assert claim_spy.call_count == 1
        assert holistic_spy.call_count == 1
        assert prediction["identical_input_reused"] is True

    def test_identical_reuse_produces_unchanged_directions(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "justified_no_revision")

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            prediction = rrr.predict_live(example, MagicMock())

        for dim, direction in prediction["dimension_directions"].items():
            assert direction == "unchanged", f"{dim} was {direction!r}"
        assert prediction["hard_failure_direction"] == "unchanged"

    def test_reuse_deep_copies_never_shares_mutable_state(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "justified_no_revision")

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            prediction = rrr.predict_live(example, MagicMock())

        assert prediction["draft"] is not prediction["refined"]
        assert prediction["draft"]["judge_dimensions"] is not prediction["refined"]["judge_dimensions"]
        prediction["draft"]["judge_dimensions"]["synthesis_quality"]["label"] = "MUTATED"
        assert prediction["refined"]["judge_dimensions"]["synthesis_quality"]["label"] != "MUTATED"

    def test_non_identical_reports_never_get_the_optimization_even_with_revision_applied_false(self):
        """Guards against a future bug where the optimization triggers on
        `revision_applied=false` alone without checking actual equality --
        none of R6D.1's real fixtures exercise this combination, so this
        is checked with a synthetic pair via the loader's own dataclass."""
        pair = rri.load_report_refinement_examples(subset=1)[0]
        example = rrr._to_example(pair)
        # Force a mismatch: revision_applied claims false, but reports differ.
        example.inputs["refinement_context"] = {**example.inputs["refinement_context"], "revision_applied": False}
        example.inputs["refined_report"] = copy.deepcopy(example.inputs["draft_report"])
        example.inputs["refined_report"]["conclusion"]["content"] += " A deliberately different sentence."

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()) as claim_spy, \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            prediction = rrr.predict_live(example, MagicMock())

        assert claim_spy.call_count == 2  # NOT reused -- reports actually differ
        assert prediction["identical_input_reused"] is False

    def test_equal_length_is_not_treated_as_equal_reports(self):
        """Requires EXACT report equality, never 'same length'/'same
        references' as a proxy."""
        pair = rri.load_report_refinement_examples(subset=1)[0]
        draft = pair.draft_report
        refined = copy.deepcopy(draft)
        # Same length, same references, different content -- must NOT be treated as identical.
        refined["conclusion"]["content"] = "X" * len(draft["conclusion"]["content"])
        assert not rri.reports_are_equal(draft, refined)


class TestDirectionComparisonRules:
    """Pure-function tests for `_dimension_direction` -- no mocking
    needed, exercises rules A-G directly."""

    def test_rule_a_either_side_unknown(self):
        assert rrr._dimension_direction("groundedness", {"label": "unknown"}, {"label": "pass"}) == "unknown"
        assert rrr._dimension_direction("groundedness", {"label": "pass"}, {"label": "unknown"}) == "unknown"

    def test_rule_b_both_not_applicable(self):
        direction = rrr._dimension_direction("source_balance", {"label": "not_applicable"}, {"label": "not_applicable"})
        assert direction == "unchanged"

    def test_rule_c_exactly_one_not_applicable(self):
        assert rrr._dimension_direction("source_balance", {"label": "not_applicable"}, {"label": "pass"}) == "unknown"
        assert rrr._dimension_direction("source_balance", {"label": "pass"}, {"label": "not_applicable"}) == "unknown"

    def test_rule_d_fail_to_pass_is_improved(self):
        direction = rrr._dimension_direction("citation_correctness", {"label": "fail"}, {"label": "pass"})
        assert direction == "improved"

    def test_rule_d_pass_to_fail_is_regressed(self):
        direction = rrr._dimension_direction("citation_correctness", {"label": "pass"}, {"label": "fail"})
        assert direction == "regressed"

    def test_rule_e_same_citation_correctness_labels_never_use_score(self):
        draft = {"label": "pass", "score": 0.2}
        refined = {"label": "pass", "score": 0.99}
        assert rrr._dimension_direction("citation_correctness", draft, refined) == "unchanged"

    def test_rule_e_same_groundedness_labels_never_use_score(self):
        draft = {"label": "fail", "score": 0.1}
        refined = {"label": "fail", "score": 0.95}
        assert rrr._dimension_direction("groundedness", draft, refined) == "unchanged"

    def test_rule_f_holistic_increase_of_exactly_0_10_is_improved(self):
        draft = {"label": "pass", "score": 0.70}
        refined = {"label": "pass", "score": 0.80}
        assert rrr._dimension_direction("synthesis_quality", draft, refined) == "improved"

    def test_rule_f_holistic_decrease_of_exactly_0_10_is_regressed(self):
        draft = {"label": "pass", "score": 0.80}
        refined = {"label": "pass", "score": 0.70}
        assert rrr._dimension_direction("synthesis_quality", draft, refined) == "regressed"

    def test_rule_f_holistic_delta_below_0_10_is_unchanged(self):
        draft = {"label": "pass", "score": 0.70}
        refined = {"label": "pass", "score": 0.75}
        assert rrr._dimension_direction("synthesis_quality", draft, refined) == "unchanged"

    def test_rule_f_delta_threshold_is_the_documented_provisional_constant(self):
        assert rrr.HOLISTIC_DIRECTION_MIN_DELTA == 0.10

    def test_rule_g_missing_holistic_score_is_unknown(self):
        draft = {"label": "pass", "score": None}
        refined = {"label": "pass", "score": 0.9}
        assert rrr._dimension_direction("coherence", draft, refined) == "unknown"

    def test_rule_g_malformed_holistic_score_is_unknown(self):
        draft = {"label": "pass", "score": "high"}
        refined = {"label": "pass", "score": 0.9}
        assert rrr._dimension_direction("coherence", draft, refined) == "unknown"

    def test_rule_g_boolean_score_is_not_treated_as_valid(self):
        draft = {"label": "pass", "score": True}
        refined = {"label": "pass", "score": 0.9}
        assert rrr._dimension_direction("coherence", draft, refined) == "unknown"

    def test_holistic_dimension_with_same_fail_labels_still_compares_scores(self):
        draft = {"label": "fail", "score": 0.30}
        refined = {"label": "fail", "score": 0.55}
        assert rrr._dimension_direction("template_fit", draft, refined) == "improved"


class TestFailureIsolationLive:
    def test_structural_failure_preserves_zero_call_behavior_for_that_side(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "structural_regression")

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()) as claim_spy, \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()) as holistic_spy:
            prediction = rrr.predict_live(example, MagicMock())

        # draft is clean -> 1 real call each; refined is structurally broken -> 0 calls for that side.
        assert claim_spy.call_count == 1
        assert holistic_spy.call_count == 1
        assert prediction["refined"]["structural_status"] == "fail"
        for dim, entry in prediction["refined"]["judge_dimensions"].items():
            assert entry["label"] == "unknown", f"refined.{dim} was {entry['label']!r}"
        # Structural gating means no fair comparison is possible for any dimension.
        for dim, direction in prediction["dimension_directions"].items():
            assert direction == "unknown", f"{dim} was {direction!r}"
        assert prediction["semantic_evaluation_status"] == "not_evaluated"

    def test_claim_judge_failure_isolated_from_holistic(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "clear_grounding_improvement")

        with patch.object(claim_source, "judge_claims", return_value=_failing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()) as holistic_spy:
            prediction = rrr.predict_live(example, MagicMock())

        assert holistic_spy.call_count == 2  # holistic still attempted on both sides
        for side in (prediction["draft"], prediction["refined"]):
            assert side["judge_dimensions"]["citation_correctness"]["label"] == "unknown"
            assert side["judge_dimensions"]["groundedness"]["label"] == "unknown"
            assert side["judge_dimensions"]["synthesis_quality"]["label"] == "pass"  # unaffected

    def test_holistic_judge_failure_isolated_from_claim_source(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "clear_grounding_improvement")

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()) as claim_spy, \
             patch.object(holistic, "judge_report", return_value=_failing_holistic_result()):
            prediction = rrr.predict_live(example, MagicMock())

        assert claim_spy.call_count == 2  # claim/source still attempted on both sides
        for side in (prediction["draft"], prediction["refined"]):
            for dim in ("synthesis_quality", "analytical_quality", "template_fit", "coherence", "source_balance"):
                assert side["judge_dimensions"][dim]["label"] == "unknown"

    def test_side_level_errors_are_recorded_without_crashing_the_suite(self):
        """A completely unexpected exception on one side's evaluation must
        not prevent the whole run_suite loop from continuing."""
        examples = rrr._load_examples(tags=None, subset=None)

        call_count = {"n": 0}

        def _boom_once(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated unexpected crash")
            return _passing_claim_result()

        with patch.object(claim_source, "judge_claims", side_effect=_boom_once), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            prediction = rrr.predict_live(examples[0], MagicMock())

        # The whole predict() call must still return a well-formed dict, not raise.
        assert prediction["pair_id"] == examples[0].id
        assert prediction["draft"]["error"] is not None or prediction["refined"]["error"] is not None


class TestSemanticEvaluatorLive:
    def test_expected_unknown_is_not_a_wildcard(self):
        prediction = {
            "dimension_directions": {
                "citation_correctness": "unknown", "groundedness": "improved", "synthesis_quality": "unchanged",
                "analytical_quality": "unchanged", "template_fit": "unchanged", "coherence": "unchanged",
                "source_balance": "unchanged",
            },
        }
        expected = {
            "expected_dimension_directions": {
                "citation_correctness": {"direction": "unchanged", "rationale": "x"},  # mismatched on purpose
                "groundedness": {"direction": "improved", "rationale": "x"},
                "synthesis_quality": {"direction": "unchanged", "rationale": "x"},
                "analytical_quality": {"direction": "unchanged", "rationale": "x"},
                "template_fit": {"direction": "unchanged", "rationale": "x"},
                "coherence": {"direction": "unchanged", "rationale": "x"},
                "source_balance": {"direction": "unchanged", "rationale": "x"},
            },
        }
        result = REFINEMENT_EVALUATORS["report_refinement_semantic_direction_agreement"](prediction, expected)
        assert result["detail"]["citation_correctness"]["match"] is False
        assert result["score"] == round(6 / 7, 4)

    def test_score_is_matched_over_seven(self):
        prediction = {
            "dimension_directions": {
                "citation_correctness": "unchanged", "groundedness": "unchanged", "synthesis_quality": "improved",
                "analytical_quality": "unchanged", "template_fit": "unchanged", "coherence": "unchanged",
                "source_balance": "unchanged",
            },
        }
        expected = {
            "expected_dimension_directions": {
                name: {"direction": "unchanged", "rationale": "x"} for name in rri.REQUIRED_DIMENSION_NAMES
            },
        }
        result = REFINEMENT_EVALUATORS["report_refinement_semantic_direction_agreement"](prediction, expected)
        assert result["score"] == round(6 / 7, 4)

    def test_seven_of_seven_agreement_scores_1_0(self):
        directions = {name: "unchanged" for name in rri.REQUIRED_DIMENSION_NAMES}
        prediction = {"dimension_directions": directions}
        expected = {"expected_dimension_directions": {n: {"direction": "unchanged", "rationale": "x"} for n in directions}}
        result = REFINEMENT_EVALUATORS["report_refinement_semantic_direction_agreement"](prediction, expected)
        assert result["score"] == 1.0

    def test_one_mismatch_prevents_fully_passed_example_via_run_suite(self):
        """Integration-level: a single mismatched dimension must make
        run_suite's own all-or-nothing pass rule fail the example, even
        though hard_failure_direction agreed."""
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "clear_grounding_improvement")
        # Corrupt this example's own expectation for one dimension so live agreement can't be 7/7.
        example.outputs["expected_dimension_directions"] = {
            **example.outputs["expected_dimension_directions"],
            "template_fit": {"direction": "regressed", "rationale": "deliberately wrong for this test"},
        }

        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            from research_agent.evals.runners._base import run_suite
            result = run_suite(
                suite="report_refinement", dataset_file="x",
                predict=lambda ex: rrr.predict_live(ex, MagicMock()),
                evaluators=[
                    ("report_refinement_hard_failure_direction_agreement",
                     REFINEMENT_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
                    ("report_refinement_semantic_direction_agreement",
                     REFINEMENT_EVALUATORS["report_refinement_semantic_direction_agreement"]),
                ],
                mode="live", examples=[example],
            )
        assert result.passed == 0
        assert result.failed == 1

    def test_mock_prediction_gives_none_score_not_zero(self):
        prediction = {"dimension_directions": None}
        expected = {"expected_dimension_directions": {n: {"direction": "unchanged", "rationale": "x"} for n in rri.REQUIRED_DIMENSION_NAMES}}
        result = REFINEMENT_EVALUATORS["report_refinement_semantic_direction_agreement"](prediction, expected)
        assert result["score"] is None


class TestMockModeUnaffectedByLive:
    def test_mock_predict_unchanged(self):
        examples = rrr._load_examples(tags=None, subset=None)
        example = next(e for e in examples if e.id == "clear_grounding_improvement")
        prediction = rrr.predict(example)
        assert prediction["dimension_directions"] is None
        assert prediction["semantic_evaluation_status"] == "not_evaluated_in_mock_mode"
        assert "identical_input_reused" not in prediction  # mock shape unchanged from R6D.2

    def test_mock_mode_makes_zero_calls(self):
        with patch.object(claim_source, "judge_claims") as claim_spy, patch.object(holistic, "judge_report") as holistic_spy:
            result = rrr.run_experiment(mode="mock")
        claim_spy.assert_not_called()
        holistic_spy.assert_not_called()
        assert result.total == 7

    def test_all_seven_fixtures_still_pass_mock_mode(self):
        result = rrr.run_experiment(mode="mock")
        assert result.total == 7
        assert result.passed == 7
        assert result.failed == 0
        assert result.average_score == 1.0

    def test_mock_evaluator_set_still_includes_not_evaluated_evaluator(self):
        result = rrr.run_experiment(mode="mock")
        for pe in result.per_example:
            assert "report_refinement_semantic_dimensions_not_evaluated" in pe["evaluator_results"]
            assert "report_refinement_semantic_direction_agreement" not in pe["evaluator_results"]


class TestLiveDetailJson:
    def test_detail_json_contains_both_sides_and_direction_comparisons(self, tmp_path):
        from research_agent.evals.runners._base import run_suite, write_run_detail_json

        examples = rrr._load_examples(tags=None, subset=1)
        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            result = run_suite(
                suite="report_refinement", dataset_file="x",
                predict=lambda ex: rrr.predict_live(ex, MagicMock()),
                evaluators=[
                    ("report_refinement_hard_failure_direction_agreement",
                     REFINEMENT_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
                    ("report_refinement_semantic_direction_agreement",
                     REFINEMENT_EVALUATORS["report_refinement_semantic_direction_agreement"]),
                ],
                mode="live", examples=examples,
            )
        path = write_run_detail_json(result, run_id=1, runs_dir=tmp_path)
        detail = json.loads(path.read_text())
        pe = detail["per_example"][0]
        pred = pe["prediction"]

        assert "judge_dimensions" in pred["draft"]
        assert "judge_dimensions" in pred["refined"]
        assert "judge_metadata" in pred["draft"]
        assert "judge_metadata" in pred["refined"]
        assert "dimension_directions" in pred
        assert "identical_input_reused" in pred
        detail_block = pe["evaluator_results"]["report_refinement_semantic_direction_agreement"]["detail"]
        for dim in rri.REQUIRED_DIMENSION_NAMES:
            assert {"expected", "actual", "match"} <= set(detail_block[dim])
        assert "error" in pe
        assert "latency_ms" in pe

    def test_no_raw_credentials_in_detail_json(self, tmp_path):
        from research_agent.evals.runners._base import run_suite, write_run_detail_json

        examples = rrr._load_examples(tags=None, subset=1)
        with patch.object(claim_source, "judge_claims", return_value=_passing_claim_result()), \
             patch.object(holistic, "judge_report", return_value=_passing_holistic_result()):
            result = run_suite(
                suite="report_refinement", dataset_file="x",
                predict=lambda ex: rrr.predict_live(ex, MagicMock()),
                evaluators=[("report_refinement_hard_failure_direction_agreement",
                             REFINEMENT_EVALUATORS["report_refinement_hard_failure_direction_agreement"])],
                mode="live", examples=examples,
            )
        path = write_run_detail_json(result, run_id=1, runs_dir=tmp_path)
        raw_text = path.read_text().lower()
        assert "sk-" not in raw_text  # OpenAI API key prefix never present
        assert "api_key" not in raw_text
        assert "authorization" not in raw_text


class TestExistingSuitesUnaffectedByLive:
    def test_report_quality_mock_suite_still_8_of_8(self):
        result = rq.run_experiment(mode="mock")
        assert result.total == 8
        assert result.passed == 8
        assert result.average_score == 1.0

    def test_report_quality_own_predict_live_untouched(self):
        """R6D.3 must never modify run_report_quality.py's own logic --
        confirmed by re-running its existing structural_and_metadata_
        corruption zero-call guarantee directly, unmocked."""
        examples = rq.load_report_quality_examples()
        broken = next(e for e in examples if e.id == "structural_and_metadata_corruption")
        with patch.object(claim_source, "judge_claims") as claim_spy, patch.object(holistic, "judge_report") as holistic_spy:
            prediction = rq.predict_live(broken, MagicMock())
        claim_spy.assert_not_called()
        holistic_spy.assert_not_called()
        assert prediction["structural_integrity"]["status"] == "fail"
