"""R6B: focused tests for the report_quality eval suite -- the manifest
+ fixture-file loader, each of the 6 frozen deterministic hard-failure
checks in isolation, citation-marker parsing precision, source-
availability checks, informational signals, the fixture-agreement
evaluator, and CLI/suite behavior (including the not-yet-implemented
live mode's clean failure).

R6B is deterministic/mock-only -- nothing in this file ever needs to
patch OpenAI (unlike test_evals_chat_relevance.py), since
research_agent.evals.runners.run_report_quality never imports it at
all; see test_predict_never_imports_openai below for the direct proof.
"""

from __future__ import annotations

import csv
import json

import pytest

from research_agent.evals import cli
from research_agent.evals.evaluators.report_quality import (
    ALL_EVALUATORS,
    report_quality_hard_failure_agreement,
)
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners._base import Example, LiveModeSetupError, append_result_csv, write_run_detail_json

REQUIRED_SECTION_KEYS = rq.REQUIRED_SECTION_KEYS


# --- Fixture-building helpers (synthetic, isolated from the real fixture set) ---

def _paper(paper_id="p1", abstract="Some abstract text about the topic."):
    return {
        "title": f"Paper {paper_id}", "authors": ["A. One"], "year": 2024, "venue": "arXiv preprint",
        "abstract": abstract, "url": f"https://papers.example.com/{paper_id}", "doi": None,
        "citation_count": None, "source": "arxiv", "paper_id": paper_id, "source_urls": {},
    }


def _web(url="https://example.com/a", snippet="Some snippet text."):
    return {"title": "Article", "url": url, "snippet": snippet, "published_date": None, "source_domain": "example.com"}


def _section(content, reference_numbers=None):
    return {"content": content, "cited_papers": [], "cited_web_articles": [], "reference_numbers": reference_numbers or []}


def _reference(number, kind="paper", paper_id=None, url=None, title="Ref"):
    return {
        "number": number, "kind": kind, "paper_id": paper_id, "url": url,
        "title": title, "formatted": f"{title}.", "link_url": url,
    }


def _clean_report(template="analytical"):
    """Every section cites [1] independently -- redundantly on purpose,
    so a test can remove/blank/corrupt ONE section without accidentally
    orphaning reference #1 (still cited by the other 7)."""
    report = {k: _section(f"Content for {k} citing [1].", reference_numbers=[1]) for k in REQUIRED_SECTION_KEYS}
    report["references"] = [_reference(1, kind="paper", paper_id="p1", title="Paper p1")]
    report["report_template"] = template
    report["skipped_papers"] = []
    return report


def _example(report, selected_papers=None, approved_web_articles=None, template="analytical"):
    return Example(
        id="synthetic",
        inputs={
            "topic": "synthetic topic", "template": template,
            "selected_papers": selected_papers if selected_papers is not None else [_paper("p1")],
            "approved_web_articles": approved_web_articles if approved_web_articles is not None else [],
            "generated_report": report,
        },
        outputs={}, metadata={},
    )


# --- Fixture loader --------------------------------------------------------

class TestFixtureLoader:
    def test_loads_all_8_real_fixtures(self):
        examples = rq.load_report_quality_examples()
        assert len(examples) == 8
        assert {e.id for e in examples} == {
            "good_foundational", "good_analytical", "good_expert",
            "citation_and_grounding_failure", "verbose_low_synthesis",
            "source_prompt_injection", "evaluator_injection_in_report",
            "structural_and_metadata_corruption",
        }
        for e in examples:
            assert "topic" in e.inputs and "template" in e.inputs
            assert "generated_report" in e.inputs
            assert "expected_hard_failures" in e.outputs
            assert "expected_dimension_labels" in e.outputs

    def test_tags_filter_keeps_only_matching_examples(self):
        examples = rq.load_report_quality_examples(tags=["security"])
        assert {e.id for e in examples} == {"source_prompt_injection", "evaluator_injection_in_report"}

    def test_subset_takes_first_n_after_tag_filtering(self):
        all_baseline = rq.load_report_quality_examples(tags=["baseline"])
        subset = rq.load_report_quality_examples(tags=["baseline"], subset=2)
        assert len(subset) == 2
        assert [e.id for e in subset] == [e.id for e in all_baseline[:2]]

    def test_duplicate_ids_in_manifest_are_rejected(self, tmp_path, monkeypatch):
        rq_dir = tmp_path / "report_quality"
        (rq_dir / "fixtures").mkdir(parents=True)
        manifest = rq_dir / "manifest.jsonl"
        manifest.write_text(
            '{"id": "dup", "path": "fixtures/a.json", "tags": [], "source_origin": "synthetic_handwritten"}\n'
            '{"id": "dup", "path": "fixtures/b.json", "tags": [], "source_origin": "synthetic_handwritten"}\n'
        )
        monkeypatch.setattr(rq, "REPORT_QUALITY_DIR", rq_dir)
        monkeypatch.setattr(rq, "MANIFEST_PATH", manifest)

        with pytest.raises(rq.FixtureLoadError, match="duplicate fixture id"):
            rq.load_report_quality_examples()

    def test_missing_fixture_path_is_rejected(self, tmp_path, monkeypatch):
        rq_dir = tmp_path / "report_quality"
        (rq_dir / "fixtures").mkdir(parents=True)
        manifest = rq_dir / "manifest.jsonl"
        manifest.write_text(
            '{"id": "ghost", "path": "fixtures/does_not_exist.json", "tags": [], "source_origin": "synthetic_handwritten"}\n'
        )
        monkeypatch.setattr(rq, "REPORT_QUALITY_DIR", rq_dir)
        monkeypatch.setattr(rq, "MANIFEST_PATH", manifest)

        with pytest.raises(rq.FixtureLoadError, match="does not exist"):
            rq.load_report_quality_examples()

    def test_path_traversal_outside_report_quality_root_is_rejected(self, tmp_path, monkeypatch):
        rq_dir = tmp_path / "report_quality"
        (rq_dir / "fixtures").mkdir(parents=True)
        # A real file that genuinely exists, but OUTSIDE rq_dir -- proves
        # rejection is about the path escaping the root, not just a
        # missing-file check.
        outside_secret = tmp_path / "outside_secret.json"
        outside_secret.write_text("{}")
        manifest = rq_dir / "manifest.jsonl"
        manifest.write_text(
            '{"id": "escape", "path": "../outside_secret.json", "tags": [], "source_origin": "synthetic_handwritten"}\n'
        )
        monkeypatch.setattr(rq, "REPORT_QUALITY_DIR", rq_dir)
        monkeypatch.setattr(rq, "MANIFEST_PATH", manifest)

        with pytest.raises(rq.FixtureLoadError, match="escapes"):
            rq.load_report_quality_examples()

    def test_unsupported_schema_version_is_rejected(self, tmp_path, monkeypatch):
        rq_dir = tmp_path / "report_quality"
        fixtures_dir = rq_dir / "fixtures"
        fixtures_dir.mkdir(parents=True)
        fixture = {
            "schema_version": "r9z-vNext", "topic": "t", "template": "analytical",
            "selected_papers": [], "approved_web_articles": [],
            "generated_report": _clean_report(), "expected": {
                "hard_failures": [], "dimension_labels": _dimension_labels(),
            },
            "human_annotations": [], "notes": "",
        }
        (fixtures_dir / "bad_version.json").write_text(json.dumps(fixture))
        manifest = rq_dir / "manifest.jsonl"
        manifest.write_text(
            '{"id": "bad_version", "path": "fixtures/bad_version.json", "tags": [], "source_origin": "synthetic_handwritten"}\n'
        )
        monkeypatch.setattr(rq, "REPORT_QUALITY_DIR", rq_dir)
        monkeypatch.setattr(rq, "MANIFEST_PATH", manifest)

        with pytest.raises(rq.FixtureLoadError, match="unsupported schema_version"):
            rq.load_report_quality_examples()

    def test_template_report_template_mismatch_is_rejected_as_identity_inconsistency(self, tmp_path, monkeypatch):
        rq_dir = tmp_path / "report_quality"
        fixtures_dir = rq_dir / "fixtures"
        fixtures_dir.mkdir(parents=True)
        report = _clean_report(template="expert")  # deliberately mismatched below
        fixture = {
            "schema_version": "r6a-v1", "topic": "t", "template": "analytical",  # <-- mismatch
            "selected_papers": [], "approved_web_articles": [],
            "generated_report": report, "expected": {
                "hard_failures": [], "dimension_labels": _dimension_labels(),
            },
            "human_annotations": [], "notes": "",
        }
        (fixtures_dir / "mismatch.json").write_text(json.dumps(fixture))
        manifest = rq_dir / "manifest.jsonl"
        manifest.write_text(
            '{"id": "mismatch", "path": "fixtures/mismatch.json", "tags": [], "source_origin": "synthetic_handwritten"}\n'
        )
        monkeypatch.setattr(rq, "REPORT_QUALITY_DIR", rq_dir)
        monkeypatch.setattr(rq, "MANIFEST_PATH", manifest)

        with pytest.raises(rq.FixtureLoadError, match="identity inconsistency"):
            rq.load_report_quality_examples()


def _dimension_labels():
    return {
        name: {"label": "pass", "rationale": "synthetic test rationale"}
        for name in rq.REQUIRED_DIMENSION_NAMES
    }


# --- Each hard-failure check, independently -------------------------------

class TestHardFailureChecks:
    def test_clean_report_produces_no_hard_failures(self):
        prediction = rq.predict(_example(_clean_report()))
        assert prediction["hard_failures"] == []
        assert prediction["structural_integrity"]["status"] == "pass"

    def test_missing_required_section(self):
        report = _clean_report()
        del report["conclusion"]
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == ["missing_required_section"]
        assert prediction["structural_integrity"]["checks"]["missing_sections"] == ["conclusion"]

    def test_empty_required_section(self):
        report = _clean_report()
        report["methodology_landscape"]["content"] = "   "  # whitespace-only counts as empty
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == ["empty_required_section"]
        assert prediction["structural_integrity"]["checks"]["empty_sections"] == ["methodology_landscape"]

    def test_unresolved_citation_marker(self):
        report = _clean_report()
        report["thematic_findings"]["content"] += " Also see [Paper 9]."
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == ["unresolved_citation_marker"]
        assert prediction["structural_integrity"]["checks"]["unresolved_markers"] == {
            "thematic_findings": ["[Paper 9]"],
        }

    def test_non_sequential_reference_numbering(self):
        report = _clean_report()
        report["thematic_findings"]["content"] = "Cites the third source [3]."
        report["thematic_findings"]["reference_numbers"] = [3]
        report["references"] = [
            _reference(1, kind="paper", paper_id="p1", title="Paper p1"),
            _reference(3, kind="paper", paper_id="p1", title="Paper p1 again"),
        ]
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == ["non_sequential_reference_numbering"]

    def test_orphan_reference(self):
        report = _clean_report()
        report["references"].append(_reference(2, kind="paper", paper_id="p2", title="Paper p2"))
        # p2 is never cited by any "[2]" marker in any section's content.
        prediction = rq.predict(_example(report, selected_papers=[_paper("p1"), _paper("p2")]))
        assert prediction["hard_failures"] == ["orphan_reference"]
        assert prediction["structural_integrity"]["checks"]["orphan_references"] == [2]

    def test_orphan_reference_check_ignores_reference_numbers_metadata(self):
        """Must not trust a section's own reference_numbers field --
        only a real inline [N] marker in prose counts."""
        report = _clean_report()
        report["references"].append(_reference(2, kind="paper", paper_id="p2", title="Paper p2"))
        # Metadata CLAIMS thematic_findings cites #2, but no "[2]" marker
        # actually appears in its content -- still an orphan.
        report["thematic_findings"]["reference_numbers"] = [1, 2]
        prediction = rq.predict(_example(report, selected_papers=[_paper("p1"), _paper("p2")]))
        assert prediction["hard_failures"] == ["orphan_reference"]

    def test_reference_source_unavailable_paper(self):
        report = _clean_report()
        report["references"] = [_reference(1, kind="paper", paper_id="ghost-paper", title="Ghost")]
        prediction = rq.predict(_example(report))  # selected_papers only has "p1"
        assert prediction["hard_failures"] == ["reference_source_unavailable"]
        assert prediction["structural_integrity"]["checks"]["unavailable_references"] == [1]

    def test_reference_source_unavailable_web(self):
        report = _clean_report()
        report["references"] = [_reference(1, kind="web", url="https://example.com/ghost", title="Ghost article")]
        prediction = rq.predict(_example(report, selected_papers=[], approved_web_articles=[_web("https://example.com/real")]))
        assert prediction["hard_failures"] == ["reference_source_unavailable"]

    def test_reference_source_unavailable_malformed_kind(self):
        report = _clean_report()
        report["references"] = [{"number": 1, "kind": None, "paper_id": None, "url": None, "title": "Malformed", "formatted": "x", "link_url": None}]
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == ["reference_source_unavailable"]

    def test_multiple_failures_have_deterministic_ordering(self):
        report = _clean_report()
        del report["conclusion"]  # missing_required_section
        report["references"].append(_reference(3, kind="paper", paper_id="p1", title="Paper p1"))  # non_sequential (1,3) + orphan (3 uncited)
        report["thematic_findings"]["content"] += " unresolved [Paper 9]"  # unresolved_citation_marker

        prediction = rq.predict(_example(report))

        assert prediction["hard_failures"] == [
            "missing_required_section",
            "unresolved_citation_marker",
            "non_sequential_reference_numbering",
            "orphan_reference",
        ]
        # Re-running is byte-identical -- no set/dict iteration-order flakiness.
        assert rq.predict(_example(report))["hard_failures"] == prediction["hard_failures"]

    def test_hard_failures_list_always_matches_canonical_order_subsequence(self):
        report = _clean_report()
        del report["conclusion"]
        report["methodology_landscape"]["content"] = ""
        report["references"] = [_reference(1, kind="paper", paper_id="ghost", title="Ghost")]
        prediction = rq.predict(_example(report))

        order_index = {name: i for i, name in enumerate(rq.CANONICAL_HARD_FAILURE_ORDER)}
        indices = [order_index[h] for h in prediction["hard_failures"]]
        assert indices == sorted(indices)


# --- Citation-marker parsing precision --------------------------------------

class TestCitationMarkerParsing:
    def test_valid_final_numeric_marker_is_accepted(self):
        report = _clean_report()
        report["thematic_findings"]["content"] = "A claim [1]."
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == []

    def test_grouped_final_markers_are_accepted(self):
        report = _clean_report()
        report["references"].append(_reference(2, kind="paper", paper_id="p2", title="Paper p2"))
        report["thematic_findings"]["content"] = "A claim supported by two sources [1][2]."
        prediction = rq.predict(_example(report, selected_papers=[_paper("p1"), _paper("p2")]))
        assert prediction["hard_failures"] == []

    def test_raw_paper_marker_is_rejected(self):
        report = _clean_report()
        report["thematic_findings"]["content"] += " [Paper 1]"
        prediction = rq.predict(_example(report))
        assert "unresolved_citation_marker" in prediction["hard_failures"]

    def test_raw_web_marker_is_rejected(self):
        report = _clean_report()
        report["thematic_findings"]["content"] += " [Web 2]"
        prediction = rq.predict(_example(report))
        assert "unresolved_citation_marker" in prediction["hard_failures"]

    def test_unmatched_raw_source_id_rejected_when_it_matches_a_known_source(self):
        report = _clean_report()
        report["thematic_findings"]["content"] += " [p1]"  # p1 IS a known selected paper_id
        prediction = rq.predict(_example(report))
        assert "unresolved_citation_marker" in prediction["hard_failures"]
        assert "[p1]" in prediction["structural_integrity"]["checks"]["unresolved_markers"]["thematic_findings"]

    def test_ordinary_bracketed_prose_is_not_falsely_rejected(self):
        report = _clean_report()
        report["thematic_findings"]["content"] += " This is a [note] the model left in passing, not a citation."
        prediction = rq.predict(_example(report))
        assert prediction["hard_failures"] == []


# --- Availability checks -----------------------------------------------------

class TestAvailability:
    def test_valid_paper_and_web_references_resolve(self):
        report = _clean_report()
        report["references"].append(_reference(2, kind="web", url="https://example.com/real", title="Real article"))
        report["thematic_findings"]["content"] += " [2]"
        prediction = rq.predict(_example(
            report, selected_papers=[_paper("p1")], approved_web_articles=[_web("https://example.com/real")],
        ))
        assert prediction["hard_failures"] == []

    def test_missing_abstract_warns_but_does_not_hard_fail(self):
        report = _clean_report()
        prediction = rq.predict(_example(report, selected_papers=[_paper("p1", abstract=None)]))
        assert prediction["hard_failures"] == []
        assert any("no abstract" in w for w in prediction["warnings"])

    def test_empty_snippet_warns_but_does_not_hard_fail(self):
        report = _clean_report()
        report["references"].append(_reference(2, kind="web", url="https://example.com/real", title="Real article"))
        report["thematic_findings"]["content"] += " [2]"
        prediction = rq.predict(_example(
            report, approved_web_articles=[_web("https://example.com/real", snippet="")],
        ))
        assert prediction["hard_failures"] == []
        assert any("empty snippet" in w for w in prediction["warnings"])


# --- Informational signals ---------------------------------------------------

class TestInformationalSignals:
    def test_computed_correctly(self):
        report = _clean_report()
        report["thematic_findings"]["content"] = "One two three [1] four five [1]."
        report["skipped_papers"] = [_paper("p_skipped")]
        prediction = rq.predict(_example(report, selected_papers=[_paper("p1"), _paper("p2")]))

        signals = prediction["informational_signals"]
        assert signals["section_word_counts"]["thematic_findings"] == 7
        assert signals["source_citation_counts"]["1"] == 9  # 7 other clean sections each cite [1] once, plus 2 more here
        assert signals["citation_density_by_section"]["thematic_findings"] == round(2 / 7, 4)
        assert signals["skipped_paper_rate"] == round(1 / 2, 4)
        assert signals["selected_source_coverage"]["papers"] == {"cited": 1, "total": 2}
        assert signals["dominant_source_share"] == 1.0  # only reference #1 is ever cited

    def test_never_affect_fixture_agreement_score_even_when_lopsided(self):
        """A structurally clean report with a maximally 'bad-looking'
        informational signal (one paper never cited, 100% dominant
        source) must still score 1.0 against an expected empty
        hard-failure set -- informational signals are never a gate."""
        report = _clean_report()
        prediction = rq.predict(_example(report, selected_papers=[_paper("p1"), _paper("p2")]))

        assert prediction["informational_signals"]["dominant_source_share"] == 1.0
        assert prediction["informational_signals"]["selected_source_coverage"]["papers"]["cited"] == 1

        result = report_quality_hard_failure_agreement(prediction, {"expected_hard_failures": []})
        assert result["score"] == 1.0

    def test_informational_signals_never_appear_as_a_separate_scored_evaluator(self):
        assert list(ALL_EVALUATORS) == ["report_quality_hard_failure_agreement"]


# --- Fixture-agreement evaluator ---------------------------------------------

class TestFixtureAgreementEvaluator:
    def test_exact_match_scores_1(self):
        prediction = {"hard_failures": ["orphan_reference"]}
        result = report_quality_hard_failure_agreement(prediction, {"expected_hard_failures": ["orphan_reference"]})
        assert result["score"] == 1.0

    def test_mismatch_scores_0_with_missing_and_unexpected_in_comment(self):
        prediction = {"hard_failures": ["orphan_reference"]}
        result = report_quality_hard_failure_agreement(
            prediction, {"expected_hard_failures": ["missing_required_section"]},
        )
        assert result["score"] == 0.0
        assert "missing=['missing_required_section']" in result["comment"]
        assert "unexpected=['orphan_reference']" in result["comment"]

    def test_empty_expected_and_empty_actual_scores_1(self):
        result = report_quality_hard_failure_agreement({"hard_failures": []}, {"expected_hard_failures": []})
        assert result["score"] == 1.0

    def test_predict_error_scores_0(self):
        result = report_quality_hard_failure_agreement({"error": "boom"}, {"expected_hard_failures": []})
        assert result["score"] == 0.0

    def test_never_reads_dimension_labels(self):
        """R6B must never score expected_dimension_labels -- those are
        reserved for R6C's independent live judges."""
        expected = {"expected_hard_failures": [], "expected_dimension_labels": {"groundedness": {"label": "fail"}}}
        result = report_quality_hard_failure_agreement({"hard_failures": []}, expected)
        assert result["score"] == 1.0  # only hard_failures mattered; a "fail" dimension label was ignored


# --- Suite / CLI behavior -----------------------------------------------------

class TestSuiteBehavior:
    def test_all_8_real_fixtures_match_their_expected_hard_failures(self):
        result = rq.run_experiment(mode="mock")
        assert result.total == 8
        assert result.failed == 0
        assert result.passed == 8
        assert result.average_score == 1.0

    def test_deliberately_broken_fixture_counts_as_eval_passed_when_correctly_detected(self):
        result = rq.run_experiment(mode="mock")
        entry = next(pe for pe in result.per_example if pe["example_id"] == "structural_and_metadata_corruption")

        assert entry["prediction"]["structural_integrity"]["status"] == "fail"
        assert entry["prediction"]["hard_failures"] != []
        assert entry["evaluator_results"]["report_quality_hard_failure_agreement"]["score"] == 1.0

    def test_predict_never_imports_openai(self):
        assert not hasattr(rq, "OpenAI")
        assert "openai" not in dir(rq)

    def test_run_experiment_live_mode_raises_live_mode_setup_error(self):
        with pytest.raises(LiveModeSetupError, match="no live mode"):
            rq.run_experiment(mode="live")

    def test_run_experiment_unknown_mode_is_a_clean_error(self):
        with pytest.raises(ValueError, match="mock.*live|live.*mock"):
            rq.run_experiment(mode="banana")


class TestCli:
    def test_list_suites_includes_report_quality(self, capsys):
        exit_code = cli.main(["list-suites"])
        assert exit_code == 0
        assert "report_quality" in capsys.readouterr().out

    def test_run_defaults_to_mock(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_quality"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "mode=mock" in out
        assert "total=8" in out
        assert (tmp_path / "report_quality_history.csv").exists()
        assert (tmp_path / "runs" / "report_quality_run_1.json").exists()

    def test_run_explicit_mock_mode(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_quality", "--mode", "mock"])
        assert exit_code == 0
        assert "mode=mock" in capsys.readouterr().out

    def test_run_with_subset(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_quality", "--mode", "mock", "--subset", "3"])
        assert exit_code == 0
        assert "total=3" in capsys.readouterr().out

    def test_run_with_tags(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_quality", "--mode", "mock", "--tags", "structural_integrity"])
        assert exit_code == 0
        assert "total=1" in capsys.readouterr().out

    def test_run_with_note_is_recorded_in_csv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_quality", "--mode", "mock", "--note", "R6B deterministic baseline"])
        assert exit_code == 0
        with (tmp_path / "report_quality_history.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert "R6B deterministic baseline" in rows[0]["note"]

    def test_run_live_mode_exits_cleanly_with_no_artifacts(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_quality", "--mode", "live"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "no live mode" in err
        assert list(tmp_path.iterdir()) == []  # no CSV, no runs/ dir -- nothing written


def test_no_existing_eval_result_csvs_are_touched(tmp_path):
    import research_agent.evals.runners._base as base

    watched = [
        base.EVAL_RESULTS_DIR / "retrieval_history.csv",
        base.EVAL_RESULTS_DIR / "history.csv",
        base.EVAL_RESULTS_DIR / "chat_relevance_history.csv",
        base.EVAL_RESULTS_DIR / "latency_history.csv",
    ]
    before = {path: path.read_bytes() for path in watched if path.exists()}

    result = rq.run_experiment(mode="mock")
    append_result_csv(result, tmp_path / "report_quality_history.csv")
    write_run_detail_json(result, run_id=1, runs_dir=tmp_path / "runs")

    after = {path: path.read_bytes() for path in watched if path.exists()}
    assert before == after


class TestRunDetailJson:
    def test_detail_json_contains_structural_integrity_and_informational_signals(self, tmp_path):
        result = rq.run_experiment(mode="mock", subset=1)
        path = write_run_detail_json(result, run_id=1, runs_dir=tmp_path)
        data = json.loads(path.read_text())

        entry = data["per_example"][0]
        prediction = entry["prediction"]
        assert prediction["schema_version"] == "r6a-v1"
        assert "structural_integrity" in prediction
        assert "informational_signals" in prediction
        assert prediction["judge_dimensions"] is None
        assert prediction["judge_metadata"] is None
        assert entry["evaluator_results"]["report_quality_hard_failure_agreement"]["score"] == 1.0
