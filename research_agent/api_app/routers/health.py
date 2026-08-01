from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Cheap connectivity check -- no DB or LLM calls, just confirms the
    process is up."""
    return {"status": "ok"}
