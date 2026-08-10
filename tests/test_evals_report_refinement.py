"""R6D.1: focused tests for the report_refinement pair-fixture schema,
manifest/loader, and pair-invariant validation. Schema/loading only --
no live judges exist yet (R6D.2's job), so nothing here ever needs an
OpenAI client or makes a network call; see
`test_no_openai_import_or_network_path_exists` for the direct
guarantee.
"""

from __future__ import annotations

import copy
import json

import pytest

from research_agent.evals import report_refinement_inputs as rri

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
