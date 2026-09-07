"""Day 3: GET /me -- the authenticated account's own safe profile.

Requires a valid identity (via `identity.get_current_user`) but works
for an unapproved account -- the frontend needs to render an
"awaiting approval" state, which it can't do if /me itself 403s. Approval
is enforced elsewhere (Day 4), not here.

Returns ONLY safe account fields: the internal user id, email, display
name, and the approved/disabled flags. Never a Firebase token, a raw
provider claim, `firebase_uid`, or any infrastructure detail.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from research_agent.identity import RequestIdentity, get_current_user

router = APIRouter()


class MeResponse(BaseModel):
    user_id: str
    email: str | None
    display_name: str | None
    approved: bool
    disabled: bool


@router.get("/me", response_model=MeResponse)
def me(identity: RequestIdentity = Depends(get_current_user)) -> MeResponse:
    return MeResponse(
        user_id=str(identity.user_id),
        email=identity.email,
        display_name=identity.display_name,
        approved=identity.approved,
        disabled=identity.disabled,
    )
