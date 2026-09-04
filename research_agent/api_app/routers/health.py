from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Cheap connectivity check -- no DB or LLM calls, just confirms the
    process is up.

    Day 1 (public multi-user deployment foundation, see
    docs/plans/public-multi-user-deployment-review.md): `async def`, not
    `def` -- a sync route handler runs on Starlette's AnyIO worker
    threadpool (see api_app/app.py's own thread-limiter comment), so under
    a burst of long-running synchronous curation/report requests occupying
    every thread, a sync `/health` would queue behind them. This handler
    does no I/O of any kind (no DB, no Chroma, no provider call, no
    `await`), so making it a native coroutine means the event loop answers
    it directly -- it never touches the threadpool at all, and therefore
    can never be queued behind blocking work regardless of the thread
    ceiling. This is a pure signature change: the route path, response
    body, status code, `BasicAuthMiddleware`'s public-GET allowlist (keyed
    on method+path, not handler type), and the Docker HEALTHCHECK (a plain
    HTTP GET) are all unaffected."""
    return {"status": "ok"}
