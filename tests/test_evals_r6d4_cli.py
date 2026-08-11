"""R6D.4b: tests for the two developer-only CLI commands built on top
of R6D.4a's capture helper -- `capture-refinement` and `validate-
refinement-capture`. Every test here mocks the OpenAI client, the
curation-session checkpointer/loader, and (where end-to-end coverage
is wanted) `client.chat.completions.parse` directly -- no real network
call, no real report capture, and no real curation session is ever
touched anywhere in this file.
"""

from __future__ import annotations

import contextlib
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAIError

from research_agent.evals import cli
from research_agent.evals.r6d4_capture import R6D4CaptureError
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import REPORT_TEMPLATES, ReportEvaluation
from research_agent import report as report_module
from research_agent.schema import Paper


# --- Shared helpers (same conventions as test_evals_r6d4_capture.py) -----

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


def _no_revision_client(paper_id: str = "1111", template: str = "foundational"):
    schema = report_module._build_report_schema([paper_id], None, REPORT_TEMPLATES[template])
    parsed = _full_analytical_parsed(schema, paper_id)
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(parsed),
        _mock_parsed_response(_evaluation(overall_score=92, needs_revision=False)),
    ]
    return client


def _revision_client(paper_id: str = "1111", template: str = "foundational"):
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
        _mock_parsed_response(_evaluation(overall_score=35, needs_revision=True, issues=["too shallow"], revision_instructions="add depth")),
        _mock_parsed_response(revised_parsed),
    ]
    return client


@contextlib.contextmanager
def _fake_checkpointer():
    yield MagicMock()


def _patched_session_loading(session):
    """Patches both halves of cli.py's own deferred import chain
    (`research_agent.qa.sqlite_checkpointer`, `research_agent.
    curation_session.load_curation_session`) so `cmd_capture_
    refinement`'s own local `from ... import ...` picks up the fakes at
    call time -- returns a context manager suitable for `with`."""
    return (
        patch("research_agent.qa.sqlite_checkpointer", _fake_checkpointer),
        patch("research_agent.curation_session.load_curation_session", return_value=session),
    )


_BOOLEAN_FLAGS = {"--allow-paid-calls"}


def _base_capture_args(tmp_path, **overrides) -> list[str]:
    args = {
        "--session-id": "some-local-session-id",
        "--pair-id": "real-foundational-01",
        "--template": "foundational",
        "--output-dir": str(tmp_path / "captures"),
    }
    args.update(overrides)
    flat: list[str] = ["capture-refinement"]
    for key, value in args.items():
        if value is None:
            continue
        if key in _BOOLEAN_FLAGS:
            flat.append(key)  # store_true -- never takes a value
            continue
        # `--key=value` (one token) rather than `--key value` (two tokens) --
        # avoids argparse mistaking a value that itself starts with "-"
        # (e.g. a deliberately-unsafe --pair-id under test) for a new flag.
        flat.append(f"{key}={value}")
    return flat


# --- Command registration / help -------------------------------------------

class TestCommandRegistration:
    def test_both_commands_registered(self):
        help_text = cli.build_parser().format_help()
        assert "capture-refinement" in help_text
        assert "validate-refinement-capture" in help_text

    def test_capture_refinement_help_does_not_crash(self, capsys):
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["capture-refinement", "--help"])
        assert exc_info.value.code == 0

    def test_validate_refinement_capture_help_does_not_crash(self, capsys):
        parser = cli.build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["validate-refinement-capture", "--help"])
        assert exc_info.value.code == 0


# --- Paid-call guard ---------------------------------------------------------

class TestPaidCallGuard:
    def test_missing_allow_paid_calls_exits_2_before_any_side_effect(self, tmp_path, capsys):
        args = _base_capture_args(tmp_path)  # no --allow-paid-calls
        with patch.object(cli, "OpenAI") as openai_spy, \
             patch("research_agent.curation_session.load_curation_session") as load_spy:
            exit_code = cli.main(args)

        assert exit_code == 2
        openai_spy.assert_not_called()
        load_spy.assert_not_called()
        assert not (tmp_path / "captures").exists()
        err = capsys.readouterr().err
        assert "--allow-paid-calls" in err
        assert "Traceback" not in err

    def test_allow_paid_calls_prints_warning_before_work(self, tmp_path, capsys):
        session = _session(selected_papers=[_paper("1111", "Paper One")])
        cm1, cm2 = _patched_session_loading(session)
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=_no_revision_client()):
            exit_code = cli.main(_base_capture_args(tmp_path, **{"--allow-paid-calls": ""}))
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "REAL R4" in err
        assert exit_code == 0


# --- Existing destination -----------------------------------------------------

class TestExistingDestination:
    def test_existing_destination_exits_before_client_or_session(self, tmp_path, capsys):
        captures_dir = tmp_path / "captures"
        captures_dir.mkdir(parents=True)
        destination = captures_dir / "real-foundational-01.json"
        original_content = '{"already": "here"}'
        destination.write_text(original_content)

        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with patch.object(cli, "OpenAI") as openai_spy, \
             patch("research_agent.curation_session.load_curation_session") as load_spy:
            exit_code = cli.main(args)

        assert exit_code == 2
        openai_spy.assert_not_called()
        load_spy.assert_not_called()
        assert destination.read_text() == original_content
        err = capsys.readouterr().err
        assert "already exists" in err


# --- Missing credentials -------------------------------------------------------

class TestMissingCredentials:
    def test_missing_credentials_exits_2_with_no_artifact(self, tmp_path, capsys):
        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with patch.object(cli, "OpenAI", side_effect=OpenAIError("no api key")), \
             patch("research_agent.curation_session.load_curation_session") as load_spy:
            exit_code = cli.main(args)

        assert exit_code == 2
        load_spy.assert_not_called()
        assert not (tmp_path / "captures").exists()
        err = capsys.readouterr().err
        assert "credentials" in err
        assert "Traceback" not in err


# --- Session preconditions -----------------------------------------------------

class TestSessionPreconditions:
    def test_unknown_session_exits_cleanly(self, tmp_path, capsys):
        cm1, cm2 = _patched_session_loading(None)
        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=MagicMock()):
            exit_code = cli.main(args)
        assert exit_code == 2
        assert not (tmp_path / "captures").exists()
        err = capsys.readouterr().err
        assert "Traceback" not in err

    def test_invalid_session_stage_rejected(self, tmp_path, capsys):
        session = _session(selected_papers=[_paper("1111", "Paper One")], stage="curate")
        cm1, cm2 = _patched_session_loading(session)
        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=MagicMock()) as openai_mock:
            exit_code = cli.main(args)
        assert exit_code == 2
        assert not (tmp_path / "captures").exists()
        openai_mock.return_value.chat.completions.parse.assert_not_called()

    def test_empty_selected_papers_rejected(self, tmp_path, capsys):
        session = _session(selected_papers=[], stage="synthesize")
        cm1, cm2 = _patched_session_loading(session)
        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=MagicMock()) as openai_mock:
            exit_code = cli.main(args)
        assert exit_code == 2
        assert not (tmp_path / "captures").exists()
        openai_mock.return_value.chat.completions.parse.assert_not_called()

    def test_precondition_errors_never_echo_raw_session_id(self, tmp_path, capsys):
        cm1, cm2 = _patched_session_loading(None)
        args = _base_capture_args(tmp_path, **{"--session-id": "totally-secret-session-abc123", "--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=MagicMock()):
            cli.main(args)
        captured = capsys.readouterr()
        assert "totally-secret-session-abc123" not in captured.out
        assert "totally-secret-session-abc123" not in captured.err


# --- pair_id safety ------------------------------------------------------------

class TestPairIdSafety:
    @pytest.mark.parametrize("bad_pair_id", ["../escape", "a/b", ".hidden", "sp ace", "", "a", "-leadinghyphen"])
    def test_unsafe_pair_id_rejected(self, tmp_path, bad_pair_id, capsys):
        args = _base_capture_args(tmp_path, **{"--pair-id": bad_pair_id, "--allow-paid-calls": ""})
        with patch.object(cli, "OpenAI") as openai_spy, \
             patch("research_agent.curation_session.load_curation_session") as load_spy:
            exit_code = cli.main(args)
        assert exit_code == 2
        openai_spy.assert_not_called()
        load_spy.assert_not_called()

    def test_safe_pair_id_accepted(self, tmp_path):
        session = _session(selected_papers=[_paper("1111", "Paper One")])
        cm1, cm2 = _patched_session_loading(session)
        args = _base_capture_args(tmp_path, **{"--pair-id": "real-foundational-01", "--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=_no_revision_client()):
            exit_code = cli.main(args)
        assert exit_code == 0
        assert (tmp_path / "captures" / "real-foundational-01.json").exists()


# --- End-to-end capture (real capture_real_refinement_pair + mocked client) --

class TestEndToEndCapture:
    def _run_capture(self, tmp_path, client, template="foundational", pair_id="real-foundational-01"):
        session = _session(selected_papers=[_paper("1111", "Paper One")])
        cm1, cm2 = _patched_session_loading(session)
        args = _base_capture_args(tmp_path, **{"--template": template, "--pair-id": pair_id, "--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=client):
            exit_code = cli.main(args)
        return exit_code, tmp_path / "captures" / f"{pair_id}.json", session

    def test_valid_capture_writes_one_complete_artifact(self, tmp_path):
        exit_code, path, _session_obj = self._run_capture(tmp_path, _no_revision_client())
        assert exit_code == 0
        assert path.exists()
        artifact = json.loads(path.read_text())
        assert artifact["schema_version"] == "r6d4-capture-v1"
        assert artifact["id"] == "real-foundational-01"

    def test_resulting_artifact_passes_validate_r6d4_capture(self, tmp_path):
        from research_agent.evals.r6d4_capture import validate_r6d4_capture
        _, path, _ = self._run_capture(tmp_path, _no_revision_client())
        artifact = json.loads(path.read_text())
        validate_r6d4_capture(artifact)  # must not raise

    def test_output_path_and_name_deterministic(self, tmp_path):
        _, path, _ = self._run_capture(tmp_path, _no_revision_client(), pair_id="real-expert-02")
        assert path == tmp_path / "captures" / "real-expert-02.json"

    def test_raw_session_id_absent_from_artifact_bytes(self, tmp_path):
        exit_code, path, _ = self._run_capture(tmp_path, _no_revision_client())
        assert exit_code == 0
        raw_bytes = path.read_bytes()
        assert b"some-local-session-id" not in raw_bytes

    def test_raw_session_id_absent_from_success_output(self, tmp_path, capsys):
        session = _session(selected_papers=[_paper("1111", "Paper One")])
        cm1, cm2 = _patched_session_loading(session)
        args = _base_capture_args(tmp_path, **{"--session-id": "some-local-session-id", "--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=_no_revision_client()):
            cli.main(args)
        captured = capsys.readouterr()
        assert "some-local-session-id" not in captured.out
        assert "some-local-session-id" not in captured.err

    def test_chat_history_and_turn_history_absent(self, tmp_path):
        _, path, _ = self._run_capture(tmp_path, _no_revision_client())
        dumped = path.read_text()
        assert "chat_history" not in dumped
        assert "turn_history" not in dumped

    def test_session_remains_unchanged(self, tmp_path):
        import copy
        session_snapshot_holder = {}

        def _tracking_loader(*_args, **_kwargs):
            s = _session(selected_papers=[_paper("1111", "Paper One")])
            session_snapshot_holder["before"] = copy.deepcopy(s)
            session_snapshot_holder["session"] = s
            return s

        cm1 = patch("research_agent.qa.sqlite_checkpointer", _fake_checkpointer)
        cm2 = patch("research_agent.curation_session.load_curation_session", side_effect=_tracking_loader)
        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=_no_revision_client()):
            exit_code = cli.main(args)

        assert exit_code == 0
        assert session_snapshot_holder["session"] == session_snapshot_holder["before"]

    def test_no_append_report_version_or_save_call(self, tmp_path):
        with patch.object(report_module, "append_report_version") as append_spy, \
             patch("research_agent.curation_session.save_curation_session") as save_spy:
            exit_code, _, _ = self._run_capture(tmp_path, _no_revision_client())
        assert exit_code == 0
        append_spy.assert_not_called()
        save_spy.assert_not_called()

    def test_no_r6d_judge_called(self, tmp_path):
        from research_agent.evals.judges import claim_source, refinement_holistic
        with patch.object(claim_source, "judge_claims") as claim_spy, \
             patch.object(refinement_holistic, "judge_refinement_holistic") as pairwise_spy:
            exit_code, _, _ = self._run_capture(tmp_path, _no_revision_client())
        assert exit_code == 0
        claim_spy.assert_not_called()
        pairwise_spy.assert_not_called()

    def test_no_result_csv_modified(self, tmp_path):
        with patch.object(cli, "append_result_csv") as append_csv_spy:
            exit_code, _, _ = self._run_capture(tmp_path, _no_revision_client())
        assert exit_code == 0
        append_csv_spy.assert_not_called()

    def test_no_expected_or_winner_fields_in_written_artifact(self, tmp_path):
        _, path, _ = self._run_capture(tmp_path, _no_revision_client())
        artifact = json.loads(path.read_text())
        for forbidden in ("expected", "expected_dimension_directions", "overall_direction", "overall_score", "winner", "accept_refinement"):
            assert forbidden not in artifact

    def test_capture_makes_two_r4_calls_for_no_revision(self, tmp_path):
        client = _no_revision_client()
        exit_code, path, _ = self._run_capture(tmp_path, client)
        assert exit_code == 0
        assert client.chat.completions.parse.call_count == 2
        artifact = json.loads(path.read_text())
        assert artifact["refinement_context"]["revision_applied"] is False
        assert artifact["draft_report"] == artifact["refined_report"]

    def test_capture_makes_three_r4_calls_for_revision(self, tmp_path):
        client = _revision_client()
        exit_code, path, _ = self._run_capture(tmp_path, client)
        assert exit_code == 0
        assert client.chat.completions.parse.call_count == 3
        artifact = json.loads(path.read_text())
        assert artifact["refinement_context"]["revision_applied"] is True
        assert artifact["draft_report"] != artifact["refined_report"]

    def test_summary_output_omits_report_body_and_credentials(self, tmp_path, capsys):
        exit_code, _, _ = self._run_capture(tmp_path, _no_revision_client())
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "A finding about" not in out  # report body content never printed
        assert "sk-" not in out
        assert "api_key" not in out.lower()


# --- validate-refinement-capture --------------------------------------------

class TestValidateCommand:
    def _valid_artifact_path(self, tmp_path):
        session = _session(selected_papers=[_paper("1111", "Paper One")])
        cm1, cm2 = _patched_session_loading(session)
        args = _base_capture_args(tmp_path, **{"--allow-paid-calls": ""})
        with cm1, cm2, patch.object(cli, "OpenAI", return_value=_no_revision_client()):
            cli.main(args)
        return tmp_path / "captures" / "real-foundational-01.json"

    def test_validate_succeeds_without_network_or_session_loading(self, tmp_path, capsys):
        path = self._valid_artifact_path(tmp_path)
        with patch.object(cli, "OpenAI") as openai_spy, \
             patch("research_agent.curation_session.load_curation_session") as load_spy:
            exit_code = cli.main(["validate-refinement-capture", "--path", str(path)])
        assert exit_code == 0
        openai_spy.assert_not_called()
        load_spy.assert_not_called()
        out = capsys.readouterr().out
        assert "VALID" in out

    def test_validate_rejects_malformed_json(self, tmp_path, capsys):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json")
        exit_code = cli.main(["validate-refinement-capture", "--path", str(path)])
        assert exit_code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err

    def test_validate_rejects_synthetic_r6d1_fixture_as_wrong_schema(self, tmp_path, capsys):
        import glob
        real_fixtures = glob.glob(
            str(__import__("pathlib").Path(__file__).resolve().parent.parent / "eval_data" / "report_refinement" / "fixtures" / "*.json")
        )
        assert real_fixtures, "expected at least one synthetic r6d1-v1 fixture to exist for this test"
        exit_code = cli.main(["validate-refinement-capture", "--path", real_fixtures[0]])
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "schema_version" in err

    def test_validate_does_not_modify_the_artifact_file(self, tmp_path):
        path = self._valid_artifact_path(tmp_path)
        before = path.read_bytes()
        cli.main(["validate-refinement-capture", "--path", str(path)])
        assert path.read_bytes() == before

    def test_validate_missing_file_exits_cleanly(self, tmp_path, capsys):
        exit_code = cli.main(["validate-refinement-capture", "--path", str(tmp_path / "nope.json")])
        assert exit_code == 2
        err = capsys.readouterr().err
        assert "Traceback" not in err


# --- Atomic write behavior ---------------------------------------------------

class TestAtomicWrite:
    def test_no_partial_or_temp_files_on_serialization_failure(self, tmp_path):
        destination = tmp_path / "out" / "p1.json"
        unserializable = {"bad": object()}
        with pytest.raises(TypeError):
            cli._atomic_write_json(unserializable, destination)
        assert not destination.parent.exists()  # never even created

    def test_no_partial_or_temp_files_on_replace_failure(self, tmp_path):
        destination = tmp_path / "out" / "p1.json"
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                cli._atomic_write_json({"a": 1}, destination)
        assert not destination.exists()
        leftover = list(destination.parent.glob("*.tmp"))
        assert leftover == []

    def test_existing_file_never_overwritten_by_write_helper_caller(self, tmp_path):
        """The atomic writer itself doesn't check for an existing
        destination (that's `cmd_capture_refinement`'s own job, tested
        above) -- this test proves the write actually replaces exactly
        once and leaves no stray temp files behind on the happy path."""
        destination = tmp_path / "out" / "p1.json"
        cli._atomic_write_json({"a": 1}, destination)
        assert destination.exists()
        assert json.loads(destination.read_text()) == {"a": 1}
        leftover = [p for p in destination.parent.iterdir() if p.name.startswith(".")]
        assert leftover == []

    def test_write_is_utf8_with_deterministic_indentation(self, tmp_path):
        destination = tmp_path / "out" / "p1.json"
        cli._atomic_write_json({"topic": "café"}, destination)
        raw = destination.read_bytes().decode("utf-8")
        assert "café" in raw
        assert raw.startswith("{\n  ")  # indent=2


class TestNoRealNetworkCalls:
    def test_module_never_constructs_openai_without_going_through_the_module_symbol(self):
        """Every OpenAI() construction in this file goes through
        `cli.OpenAI` (the module-level import), which every test above
        patches -- confirmed here so a future refactor that constructs
        a client some other way (bypassing every mock in this file)
        would be caught."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(cli))
        calls = [
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "OpenAI"
        ]
        assert len(calls) == 1  # exactly the one construction inside cmd_capture_refinement
