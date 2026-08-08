"""R7D.1/R7D.2/R7E.1: focused tests for the research_agent/evals/
package -- JSONL loading/splitting, tag/subset filtering, the mock and
live chat_relevance runners, the CLI's list-suites/run/mock-default/
live-dispatch/live-credential-failure/unknown-suite behavior, CSV
append-only safety, and (R7E.1) per-example run-detail JSON persistence
plus relevance debug-score capture. Deliberately does not duplicate
research_agent/qa.py's own _filter_relevant_web_articles red-team/debug
coverage (tests/test_qa.py already owns that) -- this file tests the
eval HARNESS built on top of it.

Live mode is never exercised against the real OpenAI API here --
`OpenAI` and `research_agent.qa._embed_with_cache` are always patched,
matching this project's "no real API calls in automated tests"
constraint for eval runners (docs/evaluation.md).
"""

from __future__ import annotations

import csv
import json
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from research_agent.evals import cli
from research_agent.evals.runners import run_chat_relevance
from research_agent.evals.runners._base import (
    Example,
    LiveModeSetupError,
    SuiteResult,
    append_result_csv,
    load_examples,
    write_run_detail_json,
)

DATASET_FILE = run_chat_relevance.DATASET_FILE


def _patch_live_embeddings():
    """Live-mode tests still must not touch the real OpenAI API --
    patches `OpenAI` construction (so no credentials are needed) and
    `_embed_with_cache` (so no network call happens), using the same
    per-candidate `mock_relevance` vector scheme mock mode's own
    `predict` uses, so live-mode assertions can reuse the same expected
    relevant/rejected URLs as the mock-mode tests above."""
    vectors: dict[str, list[float]] = {}
    for example in load_examples(DATASET_FILE):
        query = example.inputs.get("query", "")
        topic = example.inputs.get("topic", "")
        vectors[query] = run_chat_relevance._QUERY_VECTOR
        if topic:
            vectors[topic] = run_chat_relevance._TOPIC_VECTOR
        for candidate in example.inputs.get("candidates") or []:
            key = f"{candidate['title']}\n{candidate.get('snippet', '')}"
            label = candidate.get("mock_relevance", "neither")
            vectors[key] = run_chat_relevance._RELEVANCE_VECTORS[label]

    return (
        patch.object(run_chat_relevance, "OpenAI", return_value=MagicMock()),
        patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]),
    )


def test_load_examples_splits_metadata_inputs_and_expected_outputs():
    examples = load_examples(DATASET_FILE)
    assert len(examples) == 9

    positive = next(e for e in examples if e.id == "chat_relevance_004_genuinely_relevant_ai_governance_source")
    assert isinstance(positive, Example)
    assert positive.metadata["tags"] == ["positive"]
    assert "topic" in positive.inputs and "query" in positive.inputs
    assert "tags" not in positive.inputs and "id" not in positive.inputs
    assert positive.outputs["expected_relevant_urls"] == ["https://eu.example.org/ai-act-high-risk-systems"]
    assert positive.outputs["expected_rejected_urls"] == []


def test_load_examples_tag_filter_keeps_only_matching_examples():
    examples = load_examples(DATASET_FILE, tags=["fail_open"])
    assert [e.id for e in examples] == ["chat_relevance_008_embedding_failure_fail_open"]


def test_load_examples_subset_takes_first_n_after_tag_filtering():
    redteam = load_examples(DATASET_FILE, tags=["redteam"])
    subset = load_examples(DATASET_FILE, tags=["redteam"], subset=3)
    assert len(subset) == 3
    assert [e.id for e in subset] == [e.id for e in redteam[:3]]


def test_load_examples_unknown_dataset_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_examples("does_not_exist.jsonl")


class TestMockChatRelevanceRunner:
    def test_topic_drift_case_rejects_the_off_topic_article(self):
        examples = load_examples(DATASET_FILE, tags=["topic_drift"])
        [example] = examples
        prediction = run_chat_relevance.predict(example)
        assert prediction["relevant_urls"] == []

    def test_genuinely_relevant_source_is_kept(self):
        examples = load_examples(DATASET_FILE)
        example = next(e for e in examples if e.id == "chat_relevance_004_genuinely_relevant_ai_governance_source")
        prediction = run_chat_relevance.predict(example)
        assert prediction["relevant_urls"] == ["https://eu.example.org/ai-act-high-risk-systems"]

    def test_empty_candidate_pool_returns_empty_list_without_error(self):
        examples = load_examples(DATASET_FILE)
        example = next(e for e in examples if e.id == "chat_relevance_007_empty_candidate_pool")
        prediction = run_chat_relevance.predict(example)
        assert prediction == {"relevant_urls": [], "debug_scores": []}

    def test_embedding_failure_fail_open_keeps_the_unfiltered_pool(self):
        examples = load_examples(DATASET_FILE)
        example = next(e for e in examples if e.id == "chat_relevance_008_embedding_failure_fail_open")
        prediction = run_chat_relevance.predict(example)
        assert prediction["relevant_urls"] == ["https://eu.example.org/ai-act-guidance-update"]

    def test_embedding_failure_fail_closed_returns_empty_list(self):
        examples = load_examples(DATASET_FILE)
        example = next(e for e in examples if e.id == "chat_relevance_009_embedding_failure_fail_closed")
        prediction = run_chat_relevance.predict(example)
        assert prediction["relevant_urls"] == []

    def test_run_experiment_scores_every_case_at_1_0_or_none(self):
        result = run_chat_relevance.run_experiment(mode="mock")
        assert result.total == 9
        assert result.failed == 0
        assert result.passed == 9
        assert result.average_score == 1.0

    def test_run_experiment_unknown_mode_is_a_clean_error(self):
        with pytest.raises(ValueError, match="mock.*live|live.*mock"):
            run_chat_relevance.run_experiment(mode="banana")

    def test_predict_debug_scores_do_not_change_relevant_urls(self):
        """R7E.1: debug_scores rides along in the prediction dict but
        must never affect which URLs are judged relevant."""
        examples = load_examples(DATASET_FILE)
        example = next(e for e in examples if e.id == "chat_relevance_004_genuinely_relevant_ai_governance_source")

        prediction = run_chat_relevance.predict(example)

        assert prediction["relevant_urls"] == ["https://eu.example.org/ai-act-high-risk-systems"]
        assert prediction["debug_scores"] == [{
            "url": "https://eu.example.org/ai-act-high-risk-systems",
            "title": "EU AI Act: obligations for high-risk AI systems",
            "query_similarity": pytest.approx(0.7071, abs=1e-3),
            "topic_similarity": pytest.approx(0.7071, abs=1e-3),
            "passed_query_threshold": True,
            "passed_topic_threshold": True,
            "kept": True,
        }]


class TestLiveChatRelevanceRunner:
    """R7D.2. Every test here patches OpenAI construction and
    _embed_with_cache -- see _patch_live_embeddings() -- so nothing in
    this class ever makes a real network/API call."""

    def test_live_mode_dispatches_to_the_live_predict_path_and_uses_the_same_loader(self):
        openai_patch, embed_patch = _patch_live_embeddings()
        with openai_patch, embed_patch:
            result = run_chat_relevance.run_experiment(mode="live")

        assert result.mode == "live"
        assert result.total == 7  # 9 fixture cases minus the 2 mock_only embedding-failure cases
        assert result.skipped == 2
        assert result.failed == 0
        assert result.average_score == 1.0

    def test_live_mode_respects_subset_and_tags(self):
        openai_patch, embed_patch = _patch_live_embeddings()
        with openai_patch, embed_patch:
            result = run_chat_relevance.run_experiment(mode="live", subset=2, tags=["redteam"])

        assert result.total + result.skipped == 2

    def test_live_mode_skips_mock_only_cases_with_a_clear_reason(self):
        openai_patch, embed_patch = _patch_live_embeddings()
        with openai_patch, embed_patch:
            result = run_chat_relevance.run_experiment(mode="live")

        skipped = {pe["example_id"]: pe for pe in result.per_example if pe.get("skipped")}
        assert set(skipped) == {
            "chat_relevance_008_embedding_failure_fail_open",
            "chat_relevance_009_embedding_failure_fail_closed",
        }
        for entry in skipped.values():
            assert "mock_only" in entry["skipped_reason"]
            assert entry["prediction"] is None  # predict_live was never called for it

    def test_live_mode_never_calls_predict_live_for_mock_only_cases(self):
        openai_patch, embed_patch = _patch_live_embeddings()
        seen_example_ids = []
        real_predict_live = run_chat_relevance.predict_live

        def spying_predict_live(example, client):
            seen_example_ids.append(example.id)
            return real_predict_live(example, client)

        with openai_patch, embed_patch, patch.object(run_chat_relevance, "predict_live", spying_predict_live):
            run_chat_relevance.run_experiment(mode="live")

        assert "chat_relevance_008_embedding_failure_fail_open" not in seen_example_ids
        assert "chat_relevance_009_embedding_failure_fail_closed" not in seen_example_ids
        assert len(seen_example_ids) == 7

    def test_live_mode_missing_credentials_raises_live_mode_setup_error(self):
        with patch.object(run_chat_relevance, "OpenAI", side_effect=OpenAIError("no api key")):
            with pytest.raises(LiveModeSetupError, match="credentials"):
                run_chat_relevance.run_experiment(mode="live")

    def test_live_mode_missing_credentials_never_reaches_the_predict_loop(self):
        with patch.object(run_chat_relevance, "OpenAI", side_effect=OpenAIError("no api key")):
            with patch("research_agent.qa._embed_with_cache") as embed_mock:
                with pytest.raises(LiveModeSetupError):
                    run_chat_relevance.run_experiment(mode="live")
                embed_mock.assert_not_called()

    def test_live_mode_prediction_carries_debug_scores_without_changing_kept_urls(self):
        """R7E.1: predict_live's debug_scores must reflect the same
        kept/rejected decision as relevant_urls -- proves the debug
        capture is read-only instrumentation, not a second code path
        that could disagree with the real filtering result."""
        openai_patch, embed_patch = _patch_live_embeddings()
        with openai_patch, embed_patch:
            result = run_chat_relevance.run_experiment(mode="live")

        entry = next(pe for pe in result.per_example if pe["example_id"] == "chat_relevance_004_genuinely_relevant_ai_governance_source")
        debug_scores = entry["prediction"]["debug_scores"]
        assert len(debug_scores) == 1
        assert debug_scores[0]["url"] == entry["prediction"]["relevant_urls"][0]
        assert debug_scores[0]["kept"] is True
        assert debug_scores[0]["query_similarity"] is not None
        assert debug_scores[0]["topic_similarity"] is not None


class TestCli:
    def test_list_suites_exits_zero_and_prints_chat_relevance(self, capsys):
        exit_code = cli.main(["list-suites"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "chat_relevance" in out

    def test_run_mock_defaults_mode_to_mock(self, monkeypatch, tmp_path, capsys):
        csv_path = tmp_path / "chat_relevance_history.csv"
        monkeypatch.setitem(cli.SUITES["chat_relevance"], "results_csv", csv_path.name)
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "chat_relevance"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "mode=mock" in out
        assert csv_path.exists()

        detail_path = tmp_path / "runs" / "chat_relevance_run_1.json"
        assert detail_path.exists()
        assert f"run detail written to {detail_path}" in out
        detail = json.loads(detail_path.read_text())
        assert detail["run_id"] == 1
        assert len(detail["per_example"]) == 9

    def test_run_mock_with_subset_and_tags(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(
            ["run", "--suite", "chat_relevance", "--mode", "mock", "--subset", "3", "--tags", "redteam"],
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "total=3" in out

    def test_run_live_mode_dispatches_and_warns_of_cost(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        openai_patch, embed_patch = _patch_live_embeddings()
        with openai_patch, embed_patch:
            exit_code = cli.main(["run", "--suite", "chat_relevance", "--mode", "live"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "can incur cost" in captured.err
        assert "mode=live" in captured.out
        assert (tmp_path / "chat_relevance_history.csv").exists()

    def test_run_live_mode_missing_credentials_is_a_clean_error_with_no_side_effects(
        self, monkeypatch, tmp_path, capsys,
    ):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        with patch.object(run_chat_relevance, "OpenAI", side_effect=OpenAIError("no api key")):
            exit_code = cli.main(["run", "--suite", "chat_relevance", "--mode", "live"])

        assert exit_code != 0
        err = capsys.readouterr().err
        assert "credentials" in err
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []  # no CSV written -- setup failed before any run happened

    def test_run_unknown_suite_is_a_clean_cli_error(self, capsys):
        exit_code = cli.main(["run", "--suite", "nonexistent", "--mode", "mock"])
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "unknown suite" in err

    def test_run_unknown_mode_is_rejected_by_argparse(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["run", "--suite", "chat_relevance", "--mode", "banana"])


def test_append_result_csv_writes_header_and_increments_run_id(tmp_path):
    csv_path = tmp_path / "chat_relevance_history.csv"
    result = SuiteResult(suite="chat_relevance", mode="mock", total=9, passed=9, failed=0, average_score=1.0)

    first_run_id = append_result_csv(result, csv_path, tags=["redteam"], note="first run")
    second_run_id = append_result_csv(result, csv_path, tags=None, note="second run")

    assert (first_run_id, second_run_id) == (1, 2)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["run_id"] for r in rows] == ["1", "2"]
    assert rows[0]["tags"] == "redteam"
    assert rows[0]["suite"] == "chat_relevance"
    assert rows[1]["tags"] == ""


class TestRunDetailJson:
    """R7E.1: write_run_detail_json persists what append_result_csv's
    aggregate-only row can't -- per-example predictions, evaluator
    results, latency, skip reasons, and errors."""

    def test_write_run_detail_json_creates_the_runs_dir_and_file(self, tmp_path):
        result = run_chat_relevance.run_experiment(mode="mock", subset=2)
        runs_dir = tmp_path / "runs"

        path = write_run_detail_json(result, run_id=1, runs_dir=runs_dir, subset=2, tags=None, note="")

        assert path == runs_dir / "chat_relevance_run_1.json"
        assert path.exists()

    def test_detail_json_contains_prediction_evaluator_and_latency_fields(self, tmp_path):
        result = run_chat_relevance.run_experiment(mode="mock", subset=1)

        path = write_run_detail_json(result, run_id=7, runs_dir=tmp_path, note="unit test")
        data = json.loads(path.read_text())

        assert data["run_id"] == 7
        assert data["suite"] == "chat_relevance"
        assert data["mode"] == "mock"
        assert data["note"] == "unit test"
        assert data["total"] == data["passed"] == 1
        assert len(data["per_example"]) == 1

        entry = data["per_example"][0]
        assert entry["skipped"] is False
        assert "relevant_urls" in entry["prediction"]
        assert "chat_relevance_correctness" in entry["evaluator_results"]
        assert entry["evaluator_results"]["chat_relevance_correctness"]["score"] == 1.0
        assert isinstance(entry["latency_ms"], (int, float))
        assert entry["error"] is None

    def test_detail_json_represents_skipped_mock_only_live_cases(self, tmp_path):
        openai_patch, embed_patch = _patch_live_embeddings()
        with openai_patch, embed_patch:
            result = run_chat_relevance.run_experiment(mode="live", tags=["fail_open"])

        path = write_run_detail_json(result, run_id=1, runs_dir=tmp_path, tags=["fail_open"])
        data = json.loads(path.read_text())

        assert data["skipped"] == 1
        [entry] = data["per_example"]
        assert entry["example_id"] == "chat_relevance_008_embedding_failure_fail_open"
        assert entry["skipped"] is True
        assert "mock_only" in entry["skipped_reason"]
        assert entry["prediction"] is None
        assert entry["evaluator_results"] is None

    def test_detail_json_represents_a_predict_exception(self, tmp_path):
        """A predict() exception is recorded per-example (run_suite's
        existing "catch and keep going" posture, unchanged by R7E.1) --
        the detail file must surface it, not just an aggregate failed
        count."""
        from research_agent.evals.evaluators.relevance import ALL_EVALUATORS
        from research_agent.evals.runners._base import run_suite

        def failing_predict(example: Example) -> dict:
            raise RuntimeError("simulated API failure")

        result = run_suite(
            suite="chat_relevance", dataset_file=DATASET_FILE, predict=failing_predict,
            evaluators=[("chat_relevance_correctness", ALL_EVALUATORS["chat_relevance_correctness"])],
            mode="mock", subset=1,
        )

        path = write_run_detail_json(result, run_id=1, runs_dir=tmp_path)
        data = json.loads(path.read_text())

        [entry] = data["per_example"]
        assert entry["error"] == "simulated API failure"
        assert entry["prediction"] == {"error": "simulated API failure"}
        assert entry["evaluator_results"]["chat_relevance_correctness"]["score"] == 0.0


def test_no_existing_eval_result_csvs_are_touched(tmp_path):
    import research_agent.evals.runners._base as base

    retrieval_csv = base.EVAL_RESULTS_DIR / "retrieval_history.csv"
    history_csv = base.EVAL_RESULTS_DIR / "history.csv"
    before = {
        path: path.read_bytes() for path in (retrieval_csv, history_csv) if path.exists()
    }

    result = run_chat_relevance.run_experiment(mode="mock")
    append_result_csv(result, tmp_path / "chat_relevance_history.csv")

    after = {
        path: path.read_bytes() for path in (retrieval_csv, history_csv) if path.exists()
    }
    assert before == after
