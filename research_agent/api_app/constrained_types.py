"""Usage Protection M2.2C Part B: reusable constrained Pydantic
annotations for genuinely user-authored request text and user-submitted
ID lists -- centralized here so every request schema field that needs
one of these limits uses the exact same type, rather than a
per-field-copied `Field(max_length=...)` that could drift.

**Only applied to request fields**, never response models: a generated
report/summary/answer, persisted report prose, references/citations,
provider-returned URLs, opaque IDs, and enum values are all explicitly
exempt (see api_app/schemas.py's own field-by-field choices) -- an LLM
answer or a stored report section can legitimately exceed 2,000
characters, and constraining a RESPONSE model would make ordinary,
correct output fail to serialize.

**Read once at import time.** `get_usage_policy()` is deliberately
uncached everywhere else in this project so runtime code sees an env
var change immediately -- but a Pydantic field constraint is baked into
the model's schema at class-definition time regardless, the same way
SearchRequest.top_k's `Field(ge=3, le=30)` already is. Changing
USAGE_MAX_TEXT_LENGTH/USAGE_MAX_PICKED_IDS_PER_MUTATION takes effect on
the next process start, not the next request -- an accepted, documented
tradeoff for schema-level validation, consistent with how every other
Pydantic constraint in this codebase already behaves.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from research_agent.config import get_usage_policy

_policy = get_usage_policy()

# Genuinely user-authored request text: a search/curation topic, a chat
# question/message, free-text refinement steering. Never response
# fields -- see module docstring.
UserText = Annotated[str, StringConstraints(max_length=_policy.max_text_length)]

# Same bound, for the handful of optional free-text request fields
# (e.g. CurationPicksRequest.refinement) where None/omitted is also valid.
OptionalUserText = Annotated[str | None, StringConstraints(max_length=_policy.max_text_length)]

# A request list of user-submitted opaque IDs in one mutation -- paper
# picks, chat exchange ids to delete/add-to-report. Bounds how many IDs
# a single request can submit, not the IDs' own content/shape.
IdList = Annotated[list[str], Field(max_length=_policy.max_picked_ids_per_mutation)]
