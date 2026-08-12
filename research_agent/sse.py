"""Usage Protection M4.1: the generic Server-Sent-Events wire-format
encoder every future streaming endpoint (chat now, report-generation
progress later) shares. This module owns ONLY frame syntax -- an
`event:` line, one `data:` line, the required trailing blank line --
never any domain-specific event type or payload shape (see
`chat_streaming.py` for chat's own event vocabulary).

Foundation only: nothing here is wired into a live FastAPI
`StreamingResponse` route yet (that is M4.2's job).
"""

from __future__ import annotations

import json
from typing import Any


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """One complete SSE frame for `event_type` carrying `data` as its
    JSON payload: `event: <event_type>\\ndata: <json>\\n\\n`.

    `data` must already be the complete, approved, JSON-serializable
    payload for `event_type` -- this function performs no filtering or
    field-allowlisting of its own; it only knows how to frame whatever
    it is given. Callers (chat_streaming.py's own event builders) are
    responsible for ensuring nothing unapproved (prompts, raw exception
    text, secrets, ...) ever reaches this function.

    `ensure_ascii=False`: real UTF-8 bytes on the wire for non-ASCII
    content (an answer fragment, a citation title) rather than escaped
    `\\uXXXX` sequences -- this is deliberate, not an oversight: a
    naive byte-oriented SSE reader must be able to encounter and
    correctly reassemble a real multibyte UTF-8 codepoint that a
    network chunk boundary happens to split, which an ASCII-escaped
    payload would never exercise. `separators=(",", ":")` keeps each
    frame's JSON on one compact line -- SSE's own framing already
    treats a bare newline as significant (it would otherwise be
    misread as a second, empty `data:` field or the frame terminator),
    so the payload itself must never contain a raw, unescaped newline;
    `json.dumps` already guarantees this (a `\\n` inside a JSON string
    value is always escaped to the two-character sequence `\\n`, never
    emitted as a literal newline byte).

    The trailing blank line (`\\n\\n`) is the SSE spec's own frame
    terminator -- required exactly once per frame, never omitted, never
    doubled.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n"
