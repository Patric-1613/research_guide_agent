"""R7D.1: focused tests for the new research_agent/evals/ package --
JSONL loading/splitting, tag/subset filtering, the mock chat_relevance
runner, the CLI's list-suites/run/mock-default/live-rejected/unknown-
suite behavior, and CSV append-only safety. Deliberately does not
duplicate research_agent/qa.py's own _filter_relevant_web_articles
red-team coverage (tests/test_qa.py already owns that) -- this file
tests the eval HARNESS built on top of it.
"""

from __future__ import annotations

import csv

import pytest

from research_agent.evals import cli
from research_agent.evals.runners import run_chat_relevance
from research_agent.evals.runners._base import (
    Example,
    append_result_csv,
    load_examples,
)

DATASET_FILE = run_chat_relevance.DATASET_FILE


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
        assert prediction == {"relevant_urls": []}

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

    def test_run_experiment_rejects_live_mode(self):
        with pytest.raises(ValueError, match="mock"):
            run_chat_relevance.run_experiment(mode="live")


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

    def test_run_mock_with_subset_and_tags(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(
            ["run", "--suite", "chat_relevance", "--mode", "mock", "--subset", "3", "--tags", "redteam"],
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "total=3" in out

    def test_run_live_mode_is_rejected_without_any_side_effects(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path)
        exit_code = cli.main(["run", "--suite", "chat_relevance", "--mode", "live"])
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "not implemented" in err
        assert list(tmp_path.iterdir()) == []

    def test_run_unknown_suite_is_a_clean_cli_error(self, capsys):
        exit_code = cli.main(["run", "--suite", "nonexistent", "--mode", "mock"])
        assert exit_code != 0
        err = capsys.readouterr().err
        assert "unknown suite" in err


def test_append_result_csv_writes_header_and_increments_run_id(tmp_path):
    from research_agent.evals.runners._base import SuiteResult

    csv_path = tmp_path / "chat_relevance_history.csv"
    result = SuiteResult(suite="chat_relevance", mode="mock", total=9, passed=9, failed=0, average_score=1.0)

    append_result_csv(result, csv_path, tags=["redteam"], note="first run")
    append_result_csv(result, csv_path, tags=None, note="second run")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["run_id"] for r in rows] == ["1", "2"]
    assert rows[0]["tags"] == "redteam"
    assert rows[0]["suite"] == "chat_relevance"
    assert rows[1]["tags"] == ""


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
