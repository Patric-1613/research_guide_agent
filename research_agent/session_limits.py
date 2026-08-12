"""Usage Protection M2.2C Parts C/D: session-capacity caps -- selected
papers per curation session (60, provisional) and chat turns per
curation session (100, provisional). Both are business-capacity
concerns, not usage/rate concerns: `research_agent/usage_guard.py`'s
own AdmissionDecision/LeaseDecision machinery (hourly/daily/global
budgets, same-session leases) is deliberately untouched by this module
-- a full session is a capacity conflict (HTTP 409, no Retry-After),
not a rate-limit rejection (HTTP 429) or a concurrency conflict.

**Chat-turn counting rule.** One counted turn = one `role == "user"`
dict entry in a curation session's `chat_history`. Traced directly
against curation_chat.py's own append sites (not assumed): every branch
that appends to chat_history (`ask_in_session`, `_accept_web_offer`,
`_accept_report_update`) appends exactly one user entry and one
assistant entry together, in that order -- confirmed by
_attach_exchange_metadata's own documented invariant ("exactly ONE user
entry and ONE assistant entry get appended per _chat_turn_impl call").
Counting `role == "user"` entries, rather than grouping by
`exchange_id`, is deliberately chosen because it is correct for BOTH
old entries (which predate exchange_id and have it as None) and new
ones, with no special-casing:
  - deletion (delete_chat_exchanges) only ever removes a MATCHED
    (user, assistant) pair sharing a real exchange_id, so the count
    naturally decreases when a delete succeeds, restoring capacity, per
    this phase's own requirement.
  - edit (edit_chat_exchange) is documented as "truncate-and-regenerate,
    not an in-place edit": it removes the edited exchange and
    everything chronologically after it, then appends exactly one fresh
    (user, assistant) pair. The raw user-entry count after an edit
    already reflects "replaced in place, net at most the same size" --
    no separate edit-specific counting logic is needed; recomputing the
    count from the (already-truncated) chat_history after the edit's
    own truncation step is exact.
"""

from __future__ import annotations

from typing import Literal

from research_agent.config import UsagePolicy

SessionCapacityReasonCode = Literal["selected_paper_limit_reached", "chat_turn_limit_reached"]


class SessionCapacityError(Exception):
    """Raised before any session mutation, refill, provider call, or
    save when a mutation would push a session over one of its capacity
    caps. No Retry-After -- this is a capacity conflict (more capacity
    only appears via deselecting/deleting, not by waiting), not a time
    window. Carries only a reason code and a concise, user-safe
    message -- no session contents, no request text."""

    def __init__(self, reason_code: SessionCapacityReasonCode, message: str):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.message = message


def check_selected_paper_capacity(
    current_selected_ids: list[str], prospective_new_ids: list[str], policy: UsagePolicy,
) -> None:
    """Raises SessionCapacityError if applying `prospective_new_ids` (IDs
    already validated/filtered by the caller -- e.g. curation_loop.py's
    own `valid_picks`, papers actually presented this session) on top of
    `current_selected_ids` would push the session's UNIQUE selected-paper
    count past the cap. An ID already selected consumes no additional
    capacity -- exactly 60 unique selected papers is allowed; a mutation
    that would make it 61 is rejected before anything is mutated.

    **Residual race, deliberately not claimed as fixed here.** Both call
    sites (curation_loop.py's `_present_and_apply_node`, curation_session.
    py's `select_paper_from_history`) read the session, call this check,
    then mutate and save -- a plain read-check-write, not a compare-and-
    swap against the persisted state. Neither this project's LangGraph
    checkpointer (picks, via resume_curation_turn) nor its plain load/
    mutate/save path (select-from-history) provides cross-request mutual
    exclusion on its own; this is a pre-existing property of the session
    persistence design, not introduced by this check. Two genuinely
    concurrent requests for the SAME session (e.g. two browser tabs
    submitting picks at nearly the same instant) can each read a state
    just under the cap, each independently compute "still allowed," and
    both write -- landing a session a small amount over 60 in that rare
    window. Closing this fully would need session-level optimistic
    concurrency (a version/etag check on save) or unconditionally
    acquiring the session's own expensive-action lease around every
    mutating curation endpoint, not just the ones already guarded for
    paid work -- a broader persistence-layer change explicitly out of
    scope for this chunk. This check is still real, narrow protection:
    it closes the gap for the overwhelmingly common case (sequential
    requests, single-tab usage) and makes a single oversized mutation
    impossible even without concurrency."""
    existing = set(current_selected_ids)
    genuinely_new = {pid for pid in prospective_new_ids if pid not in existing}
    prospective_count = len(existing) + len(genuinely_new)
    if prospective_count > policy.max_selected_papers_per_session:
        raise SessionCapacityError(
            "selected_paper_limit_reached",
            "This session has reached the maximum number of selected papers.",
        )


def count_chat_turns(chat_history: list[dict]) -> int:
    """See module docstring for the exact rule and why it was chosen."""
    return sum(1 for turn in chat_history if turn.get("role") == "user")


def check_chat_turn_capacity(chat_history: list[dict], policy: UsagePolicy) -> None:
    """Raises SessionCapacityError if the session is already at its
    chat-turn cap -- called before a NEW turn (one more paid chat/edit
    operation) begins, so the first `max_chat_turns_per_session` counted
    turns are allowed and the next one is rejected before admission,
    the lease, paid_action, or any provider call."""
    if count_chat_turns(chat_history) >= policy.max_chat_turns_per_session:
        raise SessionCapacityError(
            "chat_turn_limit_reached",
            "This session has reached the maximum number of chat turns.",
        )
