"""R6D.4d Part 3: focused tests for `report_refinement_real_inputs.py`
(the real-capture + adjudication loader/adapter) and `runners/
run_report_refinement_real.py` (the thin suite reusing `run_report_
refinement.py`'s own predict/predict_live and evaluators unchanged).

Every test here either (a) exercises the loader/adapter against
synthetic-but-schema-valid capture/adjudication fixtures built in a
tmp_path (never depending on the real, gitignored `eval_results/
captures/*.json` files existing), or (b) is explicitly guarded to skip
when the real files aren't present locally (`TestGenuineRealFiles`).
No test calls a real OpenAI API -- every `mode="live"` path here mocks
`run_report_quality._build_live_client`/`OpenAI` or patches `run_
report_refinement.predict_live` directly.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from research_agent.evals import cli
from research_agent.evals import report_refinement_real_inputs as rrri
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners import run_report_refinement as rrr
from research_agent.evals.runners import run_report_refinement_real as rrr_real
from research_agent.evals.runners._base import LiveModeSetupError


# --- Synthetic-but-schema-valid capture/adjudication fixture builders -----

def _paper(paper_id="p1"):
    return {
        "title": f"Paper {paper_id}", "authors": ["A. Author"], "year": 2024, "venue": "arXiv preprint",
        "abstract": "Some abstract text.", "url": f"https://papers.example.com/{paper_id}", "doi": None,
        "citation_count": None, "source": "arxiv", "paper_id": paper_id,
        "source_urls": {"arxiv": f"https://papers.example.com/{paper_id}"},
    }


_SECTION_KEYS = (
    "executive_summary", "introduction_scope", "thematic_findings",
    "methodology_landscape", "contradictions_open_debates", "gap_analysis",
    "future_research_directions", "conclusion",
)


def _report(template="foundational", suffix=""):
    report = {"report_template": template, "skipped_papers": [], "references": []}
    for key in _SECTION_KEYS:
        report[key] = {"content": f"Content for {key}{suffix}.", "reference_numbers": []}
    report["sections"] = [
        {"key": k, "title": k, "content": report[k]["content"], "reference_numbers": []}
        for k in _SECTION_KEYS
    ]
    return report


def _capture(pair_id, template, revision_applied):
    draft = _report(template)
    refined = _report(template, suffix=" (revised)") if revision_applied else copy.deepcopy(draft)
    return {
        "schema_version": "r6d4-capture-v1",
        "id": pair_id,
        "topic": "A synthetic real-capture test topic",
        "template": template,
        "tags": [],
        "notes": "",
        "selected_papers": [_paper("p1")],
        "approved_web_articles": [],
        "draft_report": draft,
        "refined_report": refined,
        "refinement_context": {
            "source_origin": "real_r4_generated",
            "revision_applied": revision_applied,
            "capture_timestamp": "2026-08-11T00:00:00+00:00",
            "source_session_ref": f"{pair_id}-ref",
            "generation_model": "test-model",
            "refinement_mode": "single",
            "r4_refinement_metadata": {"rounds": 1 if revision_applied else 0, "issues": []},
            "capture_commit_sha": "deadbeef",
        },
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict) -> str:
    """Writes `data` as JSON and returns the sha256 of the bytes written
    -- callers use the returned hash to build a matching adjudication."""
    payload = json.dumps(data)
    path.write_text(payload)
    return _sha256_bytes(payload.encode("utf-8"))


def _adjudication(pair_id, capture_sha256, directions=None, hard_failure_direction="unchanged"):
    dims = directions if directions is not None else {d: "unchanged" for d in rrri.REQUIRED_DIMENSION_NAMES}
    return {
        "schema_version": "r6d4-adjudication-v1",
        "pair_id": pair_id,
        "capture_sha256": capture_sha256,
        "dimension_directions": dims,
        "hard_failure_direction": hard_failure_direction,
        "reviewer_provenance": {"reviewer_type": "deterministic_exact_equality", "note": "synthetic test fixture"},
        "adjudicated_at": "2026-08-11T00:00:00+00:00",
    }


def _write_valid_trio(tmp_path, monkeypatch):
    """Writes 3 valid, schema-conformant captures + direct-capture_sha256
    adjudications (all revision_applied=False / all-unchanged, for
    simplicity) under tmp_path, and points `rrri.CAPTURES_DIR`/
    `rrri.REAL_REVIEWS_DIR` at them. Returns the captures dir."""
    captures_dir = tmp_path / "captures"
    reviews_dir = tmp_path / "real_reviews"
    captures_dir.mkdir()
    reviews_dir.mkdir()
    monkeypatch.setattr(rrri, "CAPTURES_DIR", captures_dir)
    monkeypatch.setattr(rrri, "REAL_REVIEWS_DIR", reviews_dir)

    for pair_id, template in rrri.PAIR_ID_TEMPLATES.items():
        capture = _capture(pair_id, template, revision_applied=False)
        capture_hash = _write_json(captures_dir / f"{pair_id}.json", capture)
        adjudication = _adjudication(pair_id, capture_hash)
        _write_json(reviews_dir / f"{pair_id}-adjudication.json", adjudication)

    return captures_dir, reviews_dir


# --- Loader: happy path, ordering, contract shape ---------------------------

class TestLoaderHappyPath:
    def test_loads_exactly_three_pairs_in_deterministic_order(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        examples = rrri.load_real_refinement_examples()
        assert [e.id for e in examples] == list(rrri.PAIR_IDS) == [
            "real-foundational-01", "real-analytical-01", "real-expert-01",
        ]

    def test_adapter_output_matches_r6d_example_contract(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        examples = rrri.load_real_refinement_examples()
        example = examples[0]
        assert set(example.inputs) == {
            "topic", "template", "selected_papers", "approved_web_articles",
            "draft_report", "refined_report", "refinement_context",
        }
        assert set(example.outputs) == {"expected_hard_failure_direction", "expected_dimension_directions"}
        assert set(example.outputs["expected_dimension_directions"]) == set(rrri.REQUIRED_DIMENSION_NAMES)
        for entry in example.outputs["expected_dimension_directions"].values():
            assert "direction" in entry

    def test_expected_labels_never_enter_inputs(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        for example in rrri.load_real_refinement_examples():
            assert not any(key.startswith("expected_") for key in example.inputs)
            assert "reviewer_provenance" not in example.inputs
            assert "dimension_directions" not in example.inputs
            assert "hard_failure_direction" not in example.inputs

    def test_reviewer_provenance_preserved_only_as_metadata(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        for example in rrri.load_real_refinement_examples():
            assert example.metadata["reviewer_provenance"]["reviewer_type"] == "deterministic_exact_equality"

    def test_mock_predict_runs_cleanly_against_adapted_examples(self, tmp_path, monkeypatch):
        """Sanity check that the adapted Example shape is actually
        consumable by the existing, unmodified `predict()` -- not just
        shaped correctly on paper."""
        _write_valid_trio(tmp_path, monkeypatch)
        for example in rrri.load_real_refinement_examples():
            prediction = rrr.predict(example)
            assert prediction["dimension_directions"] is None
            assert prediction["semantic_evaluation_status"] == "not_evaluated_in_mock_mode"


# --- Loader: rejections, all before any client could be constructed --------

class TestLoaderRejections:
    def test_missing_capture_file_rejected(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        (tmp_path / "captures" / "real-foundational-01.json").unlink()
        with pytest.raises(rrri.RealRefinementLoadError, match="missing capture file"):
            rrri.load_real_refinement_examples()

    def test_missing_adjudication_file_rejected(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        (tmp_path / "real_reviews" / "real-expert-01-adjudication.json").unlink()
        with pytest.raises(rrri.RealRefinementLoadError, match="missing adjudication file"):
            rrri.load_real_refinement_examples()

    def test_hash_mismatch_rejected(self, tmp_path, monkeypatch):
        captures_dir, _ = _write_valid_trio(tmp_path, monkeypatch)
        capture_path = captures_dir / "real-foundational-01.json"
        tampered = json.loads(capture_path.read_text())
        tampered["notes"] = "tampered after adjudication"
        capture_path.write_text(json.dumps(tampered))
        with pytest.raises(rrri.RealRefinementLoadError, match="hash"):
            rrri.load_real_refinement_examples()

    def test_label_mutation_to_invalid_direction_rejected(self, tmp_path, monkeypatch):
        _, reviews_dir = _write_valid_trio(tmp_path, monkeypatch)
        adjudication_path = reviews_dir / "real-analytical-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        adjudication["dimension_directions"]["citation_correctness"] = "much_worse"
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="not one of"):
            rrri.load_real_refinement_examples()

    def test_unknown_dimension_name_rejected(self, tmp_path, monkeypatch):
        _, reviews_dir = _write_valid_trio(tmp_path, monkeypatch)
        adjudication_path = reviews_dir / "real-expert-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        del adjudication["dimension_directions"]["coherence"]
        adjudication["dimension_directions"]["made_up_dimension"] = "unchanged"
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="dimension_directions"):
            rrri.load_real_refinement_examples()

    def test_invalid_hard_failure_direction_rejected(self, tmp_path, monkeypatch):
        _, reviews_dir = _write_valid_trio(tmp_path, monkeypatch)
        adjudication_path = reviews_dir / "real-foundational-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        adjudication["hard_failure_direction"] = "sideways"
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="hard_failure_direction"):
            rrri.load_real_refinement_examples()

    def test_pair_id_mismatch_rejected(self, tmp_path, monkeypatch):
        _, reviews_dir = _write_valid_trio(tmp_path, monkeypatch)
        adjudication_path = reviews_dir / "real-expert-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        adjudication["pair_id"] = "real-foundational-01"
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="pair_id mismatch"):
            rrri.load_real_refinement_examples()

    def test_template_mismatch_rejected(self, tmp_path, monkeypatch):
        captures_dir, _ = _write_valid_trio(tmp_path, monkeypatch)
        capture_path = captures_dir / "real-analytical-01.json"
        capture = json.loads(capture_path.read_text())
        capture["template"] = "expert"
        capture["draft_report"]["report_template"] = "expert"
        capture["refined_report"]["report_template"] = "expert"
        new_hash = _write_json(capture_path, capture)
        adjudication_path = rrri.REAL_REVIEWS_DIR / "real-analytical-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        adjudication["capture_sha256"] = new_hash
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="template"):
            rrri.load_real_refinement_examples()

    def test_forbidden_embedded_content_in_adjudication_rejected(self, tmp_path, monkeypatch):
        _, reviews_dir = _write_valid_trio(tmp_path, monkeypatch)
        adjudication_path = reviews_dir / "real-foundational-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        adjudication["session_id"] = "leaked-session-id"
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="forbidden"):
            rrri.load_real_refinement_examples()

    def test_capture_failing_r6d4_schema_rejected(self, tmp_path, monkeypatch):
        captures_dir, _ = _write_valid_trio(tmp_path, monkeypatch)
        capture_path = captures_dir / "real-expert-01.json"
        capture = json.loads(capture_path.read_text())
        capture["schema_version"] = "r6d1-v1"
        new_hash = _write_json(capture_path, capture)
        adjudication_path = rrri.REAL_REVIEWS_DIR / "real-expert-01-adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        adjudication["capture_sha256"] = new_hash
        adjudication_path.write_text(json.dumps(adjudication))
        with pytest.raises(rrri.RealRefinementLoadError, match="r6d4-capture-v1 validation"):
            rrri.load_real_refinement_examples()


# --- Loader: the blind-assessment hash-chain path (real-analytical-01) -----

class TestBlindAssessmentChain:
    def _write_chained_pair(self, tmp_path, monkeypatch):
        captures_dir = tmp_path / "captures"
        reviews_dir = tmp_path / "real_reviews"
        captures_dir.mkdir()
        reviews_dir.mkdir()
        monkeypatch.setattr(rrri, "CAPTURES_DIR", captures_dir)
        monkeypatch.setattr(rrri, "REAL_REVIEWS_DIR", reviews_dir)

        capture = _capture("real-analytical-01", "analytical", revision_applied=True)
        capture_hash = _write_json(captures_dir / "real-analytical-01.json", capture)

        blind_assessment = {
            "schema_version": "r6d4-review-v1",
            "pair_id": "real-analytical-01",
            "capture_sha256": capture_hash,
            "reviewer_provenance": {"reviewer_type": "ai_assisted_human_confirmed"},
            "dimension_assessments": {d: {"direction": "unchanged", "confidence": 0.5, "rationale": "r"} for d in rrri.REQUIRED_DIMENSION_NAMES},
            "reviewed_at": "2026-08-11T00:00:00+00:00",
        }
        blind_hash = _write_json(reviews_dir / "real-analytical-01-blind-assessment.json", blind_assessment)

        adjudication = {
            "schema_version": "r6d4-adjudication-v1",
            "pair_id": "real-analytical-01",
            "blind_assessment_sha256": blind_hash,
            "dimension_directions": {d: "unchanged" for d in rrri.REQUIRED_DIMENSION_NAMES},
            "hard_failure_direction": "unchanged",
            "reviewer_provenance": {"reviewer_type": "ai_assisted_human_confirmed"},
            "adjudicated_at": "2026-08-11T00:00:00+00:00",
        }
        _write_json(reviews_dir / "real-analytical-01-adjudication.json", adjudication)
        return captures_dir, reviews_dir

    def test_chain_resolves_correctly(self, tmp_path, monkeypatch):
        self._write_chained_pair(tmp_path, monkeypatch)
        capture_sha = rrri._resolve_expected_capture_sha256(
            json.loads((rrri.REAL_REVIEWS_DIR / "real-analytical-01-adjudication.json").read_text()),
            "real-analytical-01",
        )
        actual = hashlib.sha256((rrri.CAPTURES_DIR / "real-analytical-01.json").read_bytes()).hexdigest()
        assert capture_sha == actual

    def test_tampered_blind_assessment_breaks_the_chain(self, tmp_path, monkeypatch):
        captures_dir, reviews_dir = self._write_chained_pair(tmp_path, monkeypatch)
        blind_path = reviews_dir / "real-analytical-01-blind-assessment.json"
        blind = json.loads(blind_path.read_text())
        blind["reviewer_provenance"]["reviewer_type"] = "tampered"
        blind_path.write_text(json.dumps(blind))
        adjudication = json.loads((reviews_dir / "real-analytical-01-adjudication.json").read_text())
        with pytest.raises(rrri.RealRefinementLoadError, match="blind-assessment"):
            rrri._resolve_expected_capture_sha256(adjudication, "real-analytical-01")


# --- The 3 genuine, gitignored real files (skipped if not present locally) -

_REAL_CAPTURES_PRESENT = all(
    (rrri.CAPTURES_DIR / f"{pair_id}.json").exists() for pair_id in rrri.PAIR_IDS
)


class TestGenuineRealFiles:
    @pytest.mark.skipif(
        not _REAL_CAPTURES_PRESENT,
        reason="eval_results/captures/*.json is gitignored and not present in this environment",
    )
    def test_loader_accepts_the_three_genuine_files(self):
        examples = rrri.load_real_refinement_examples()
        assert [e.id for e in examples] == list(rrri.PAIR_IDS)
        for example in examples:
            assert example.outputs["expected_hard_failure_direction"] in rrri.VALID_DIRECTIONS


# --- run_report_refinement_real.run_experiment ------------------------------

class TestRunExperimentMock:
    def test_mock_mode_zero_network_calls_and_unfilled_semantic_predictions(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        with patch.object(rq, "OpenAI", side_effect=AssertionError("mock mode must never construct a client")):
            result = rrr_real.run_experiment(mode="mock")
        assert result.total == 3
        for pe in result.per_example:
            assert pe["prediction"]["dimension_directions"] is None
            assert pe["prediction"]["semantic_evaluation_status"] == "not_evaluated_in_mock_mode"

    def test_mock_mode_uses_same_evaluators_as_synthetic_suite(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        result = rrr_real.run_experiment(mode="mock")
        keys = set(result.per_example[0]["evaluator_results"])
        assert keys == {
            "report_refinement_hard_failure_direction_agreement",
            "report_refinement_semantic_dimensions_not_evaluated",
        }


class TestRunExperimentLive:
    def test_live_path_delegates_to_existing_predict_live(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        fake_client = MagicMock()
        fake_prediction = {
            "pair_id": "x", "draft": {}, "refined": {}, "hard_failure_direction": "unchanged",
            "dimension_directions": {d: "unchanged" for d in rrri.REQUIRED_DIMENSION_NAMES},
            "semantic_evaluation_status": "evaluated", "identical_input_reused": True,
            "claim_change_inventory": None, "claim_direction_detail": {}, "pairwise_holistic": {},
            "judge_call_count": 1,
        }
        with patch.object(rq, "_build_live_client", return_value=fake_client) as mock_build, \
             patch.object(rrr, "predict_live", return_value=fake_prediction) as mock_predict_live:
            result = rrr_real.run_experiment(mode="live")

        mock_build.assert_called_once()
        assert mock_predict_live.call_count == 3
        for call in mock_predict_live.call_args_list:
            example_arg, client_arg = call.args
            assert client_arg is fake_client
            assert example_arg.id in rrri.PAIR_IDS
        assert result.total == 3

    def test_live_mode_uses_semantic_direction_evaluator(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        fake_client = MagicMock()
        fake_prediction = {
            "pair_id": "x", "hard_failure_direction": "unchanged",
            "dimension_directions": {d: "unchanged" for d in rrri.REQUIRED_DIMENSION_NAMES},
        }
        with patch.object(rq, "_build_live_client", return_value=fake_client), \
             patch.object(rrr, "predict_live", return_value=fake_prediction):
            result = rrr_real.run_experiment(mode="live")
        keys = set(result.per_example[0]["evaluator_results"])
        assert keys == {
            "report_refinement_hard_failure_direction_agreement",
            "report_refinement_semantic_direction_agreement",
        }

    def test_load_failure_prevents_client_construction(self, tmp_path, monkeypatch):
        captures_dir, _ = _write_valid_trio(tmp_path, monkeypatch)
        (captures_dir / "real-foundational-01.json").unlink()
        with patch.object(rq, "_build_live_client") as mock_build:
            with pytest.raises(LiveModeSetupError):
                rrr_real.run_experiment(mode="live")
        mock_build.assert_not_called()


# --- Existing synthetic suite unaffected ------------------------------------

class TestSyntheticSuiteUnaffected:
    def test_real_suite_reuses_the_exact_same_predict_functions(self):
        """Identity, not just equal behavior -- proves run_report_
        refinement_real.py never wraps, copies, or monkeypatches
        run_report_refinement.py's own predict/predict_live."""
        import research_agent.evals.runners.run_report_refinement as rrr_module
        assert rrr_real.rrr.predict is rrr_module.predict
        assert rrr_real.rrr.predict_live is rrr_module.predict_live

    def test_real_suite_reuses_the_exact_same_evaluator_objects(self):
        from research_agent.evals.evaluators.report_refinement import ALL_EVALUATORS
        assert rrr_real.ALL_EVALUATORS is ALL_EVALUATORS

    def test_synthetic_suite_name_and_module_unchanged(self):
        assert rrr.SUITE == "report_refinement"
        assert rrr_real.SUITE == "report_refinement_real"


# --- CLI registration + result isolation ------------------------------------

class TestCliRegistration:
    def test_suite_registered_with_isolated_history_csv_and_cost_warning(self):
        assert cli.SUITES["report_refinement_real"]["results_csv"] == "report_refinement_real_history.csv"
        assert cli.SUITES["report_refinement_real"]["results_csv"] != cli.SUITES["report_refinement"]["results_csv"]
        warning = cli.SUITES["report_refinement_real"]["live_warning"]
        assert "OpenAI" in warning
        assert "5" in warning

    def test_mock_run_writes_only_the_real_history_csv(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "report_refinement_real", "--mode", "mock"])
        assert exit_code in (0, 1)
        assert (tmp_path / "report_refinement_real_history.csv").exists()
        assert not (tmp_path / "report_refinement_history.csv").exists()

    def test_missing_credentials_exits_2_with_no_artifact(self, tmp_path, monkeypatch, capsys):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _write_valid_trio(source_dir, monkeypatch)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", results_dir)
        with patch.object(rq, "OpenAI", side_effect=OpenAIError("no api key")):
            exit_code = cli.main(["run", "--suite", "report_refinement_real", "--mode", "live"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "credentials" in err
        assert list(results_dir.iterdir()) == []

    def test_note_and_run_id_flow_into_isolated_csv(self, tmp_path, monkeypatch):
        _write_valid_trio(tmp_path, monkeypatch)
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        cli.main(["run", "--suite", "report_refinement_real", "--mode", "mock", "--note", "test note"])
        with (tmp_path / "report_refinement_real_history.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "test note" in rows[0]["note"]
        assert rows[0]["suite"] == "report_refinement_real"
