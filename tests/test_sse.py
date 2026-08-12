"""Usage Protection M4.1: tests for research_agent/sse.py -- the
generic SSE frame encoder. No network/provider calls anywhere in this
file; this module has no dependency on either.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.sse import format_sse_event


def test_frame_has_event_and_data_lines_and_trailing_blank_line():
    frame = format_sse_event("started", {})
    assert frame == "event: started\ndata: {}\n\n"


def test_frame_ends_with_exactly_one_blank_line():
    frame = format_sse_event("delta", {"text": "hello"})
    assert frame.endswith("\n\n")
    assert not frame.endswith("\n\n\n")


def test_data_payload_is_valid_json_matching_input():
    frame = format_sse_event("completed", {"answer": "ok", "cited_papers": [{"paper_id": "p1", "title": "T"}]})
    data_line = frame.split("\n")[1]
    assert data_line.startswith("data: ")
    parsed = json.loads(data_line[len("data: "):])
    assert parsed == {"answer": "ok", "cited_papers": [{"paper_id": "p1", "title": "T"}]}


def test_event_type_appears_on_its_own_line():
    frame = format_sse_event("error", {"reason_code": "provider_error", "message": "x"})
    lines = frame.split("\n")
    assert lines[0] == "event: error"


def test_produces_real_utf8_non_ascii_content_not_escaped():
    frame = format_sse_event("delta", {"text": "café résumé 论文"})
    assert "café résumé 论文" in frame
    assert "\\u" not in frame  # ensure_ascii=False -- real UTF-8 bytes, not \uXXXX escapes
    # And it round-trips as valid UTF-8 bytes with no encoding errors.
    encoded = frame.encode("utf-8")
    assert encoded.decode("utf-8") == frame


def test_data_json_is_a_single_line_no_embedded_raw_newline():
    frame = format_sse_event("delta", {"text": "line one\nline two"})
    lines = frame.split("\n")
    # Exactly 4 lines: "event: ...", "data: ...", "", "" (from the trailing \n\n)
    assert len(lines) == 4
    assert lines[1].startswith("data: ")
    parsed = json.loads(lines[1][len("data: "):])
    assert parsed["text"] == "line one\nline two"  # preserved, just JSON-escaped on the wire


def test_empty_data_dict_produces_empty_json_object():
    frame = format_sse_event("done", {})
    assert "data: {}" in frame


def test_json_serializable_bounded_payload_only_no_extra_keys_leak():
    payload = {"reason_code": "provider_error", "message": "The model provider returned an error."}
    frame = format_sse_event("error", payload)
    data_line = frame.split("\n")[1]
    parsed = json.loads(data_line[len("data: "):])
    assert set(parsed.keys()) == {"reason_code", "message"}
