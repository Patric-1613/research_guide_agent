"""Centralized, typed access to this project's environment-variable-driven
configuration.

`get_settings()` deliberately covers only the always-safe, always-succeeds
env vars our own code reads directly on every request:
`SEMANTIC_SCHOLAR_API_KEY`, `UNPAYWALL_EMAIL`, `TAVILY_API_KEY`,
`OPENALEX_MAILTO` — same names, same defaults, same falsy-value-becomes-
None handling as every call site being replaced.

Two other env-var contracts live in this module but NOT on `Settings` /
`get_settings()`, because each must be validated ONCE at app construction
and be ALLOWED to raise: `AUTH_ENABLED`/`AUTH_USERNAME`/`AUTH_PASSWORD`/
`APP_ENV` (`get_auth_config()`) and `FRONTEND_ORIGIN` (`get_cors_config()`).

**Intentionally NOT covered here, and why:**

- `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/
  `LANGFUSE_BASE_URL` — nothing in this codebase reads these directly.
  The OpenAI SDK's bare `OpenAI()` constructor call
  (`api_app/app.py`'s `lifespan()`, reached as the patch target
  `api.OpenAI`) and the Langfuse SDK's `get_client()` (`tracing.py`)
  both read these from `os.environ` internally. Routing them through
  this module would mean explicitly passing credentials into
  constructors that currently read them implicitly — a real behavior
  touchpoint on sensitive, central code paths, deferred rather than
  folded into this first, low-risk pass.
- Model-name constants (`EMBEDDING_MODEL`, `SUMMARY_MODEL`,
  `AGENT_MODEL`, etc.) and data/cache/Chroma paths (`DATA_DIR`,
  `DB_PATH`, `CHROMA_PERSIST_DIR`, `QA_CHECKPOINT_DB_PATH`) — none of
  these are read from the environment today; they're plain Python
  literals. Centralizing them would mean changing import structure
  across every domain module that defines one, for zero behavior
  difference — a separately-scoped step, not this one.

`get_settings()` re-reads `os.environ` on every call — deliberately
uncached, so existing tests that wrap a single call in
`unittest.mock.patch.dict(os.environ, ...)` keep working exactly as
before, and so a `.env` change takes effect without restarting anything
that imports this module lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    semantic_scholar_api_key: str | None
    unpaywall_email: str | None
    tavily_api_key: str | None
    openalex_mailto: str | None
    # K5D.2: off-by-default product behavior flags belong here, not in
    # UsagePolicy -- UsagePolicy is exclusively safety/threshold
    # configuration (request limits, admission budgets, lease TTLs; see
    # its own module docstring), never a feature on/off switch. Disabling
    # this restores plain YAKE-v2 for every FUTURE served batch
    # immediately (research_agent/curation_loop.py's _serve_batch_node
    # reads it fresh every turn, same uncached-per-call convention as
    # every other value here) -- it does NOT retroactively rewrite
    # keywords already persisted into a session's turn_history/
    # selected_papers from while the flag was on; those stay exactly as
    # served. That's a real, stated pilot limitation, not a bug -- see
    # research_agent/keyword_filter.py's own module docstring and
    # tests/test_keyword_filter.py's rollback test for the same point
    # proven in code rather than only asserted here.
    #
    # K5D.2a fix (Codex MEDIUM finding): this is the ONLY keyword_filter
    # field left on Settings -- a cheap, always-safe os.getenv() + strict
    # bool parse. KEYWORD_FILTER_MAX_CONCURRENT_CALLS deliberately does
    # NOT live here (see get_keyword_filter_max_concurrent_calls below):
    # computing it required get_usage_policy().provider_fan_out_limit,
    # which meant EVERY call to get_settings() -- including every
    # request on the old, disabled curation path, which has nothing to
    # do with keyword filtering -- silently depended on UsagePolicy
    # parsing successfully, and a malformed/out-of-range
    # KEYWORD_FILTER_MAX_CONCURRENT_CALLS could break curation even
    # while this flag was False. get_settings() now stays a lightweight
    # "is the feature even on" check; the full filter-only configuration
    # is loaded lazily, only by a caller that already confirmed this
    # field is True.
    keyword_filter_policy_c_enabled: bool
    # Research Lanes (RL1): off-by-default product feature switch, same
    # placement rationale as keyword_filter_policy_c_enabled above --
    # UsagePolicy is exclusively safety/threshold config, never a feature
    # on/off switch. RL1 adds ONLY the flag: it is parsed here (strict
    # bool, default False) and read NOWHERE yet. RL2+ will gate lane
    # suggestion/creation on it. Flipping it never affects the
    # loadability of a session that already has lanes -- deserialization
    # of persisted lane data is unconditional (see
    # research_agent/curation_session.py's _dict_to_session).
    research_lanes_enabled: bool


def _strict_bool(name: str, default: bool) -> bool:
    """Same fail-loud posture as config/limits.py's `_positive_int`: an
    unset/empty var falls back to `default`, but a SET, unparseable value
    raises rather than silently defaulting -- a typo'd
    KEYWORD_FILTER_POLICY_C_ENABLED=disabled must never be silently read
    as `default` (currently False) and quietly enable/disable production
    behavior differently than the operator intended."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ValueError(f"{name}={raw!r} is not a valid boolean (use true/false or 1/0)")


def _positive_int(name: str, default: int) -> int:
    """Fail-loud like config/limits.py's own `_positive_int`: unset/empty
    falls back to `default`; a SET but non-integer or non-positive value
    always raises. Deliberately has NO upper bound of its own -- clamping
    against provider_fan_out_limit is
    get_keyword_filter_max_concurrent_calls's job below, kept separate so
    parsing this env var never needs UsagePolicy."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from None
    if value <= 0:
        raise ValueError(f"{name}={value} must be a positive integer")
    return value


def get_keyword_filter_max_concurrent_calls(fan_out_limit: int) -> int:
    """K5D.2a fix (Codex MEDIUM finding): the effective, ready-to-use
    concurrency bound for keyword_filter.py's per-paper provider calls --
    deliberately NOT part of Settings/get_settings() (see that
    dataclass's own field comment for why). Callers (research_agent.
    curation_loop's _serve_batch_node) must only call this after already
    confirming Settings.keyword_filter_policy_c_enabled is True, and
    must supply `fan_out_limit` themselves (typically
    get_usage_policy().provider_fan_out_limit) -- fetching UsagePolicy
    is the caller's decision to make on the already-enabled path, never
    something this module reaches for on its own.

    KEYWORD_FILTER_MAX_CONCURRENT_CALLS: malformed or <= 0 always raises
    clearly -- a real misconfiguration must surface, not silently
    default. A valid positive value (explicit, or the provisional
    default of 3, same "conservative given this project's own rate-
    limiting history" judgment call as query_expansion.py's
    _MAX_CONCURRENT_TITLE_PAIRS) that EXCEEDS `fan_out_limit` is CLAMPED
    down to it, never rejected -- fan_out_limit is this project's one
    general cross-feature concurrency ceiling, not a per-feature
    configuration error to reject a perfectly valid request over. The
    result is always >= 1.
    """
    requested = _positive_int("KEYWORD_FILTER_MAX_CONCURRENT_CALLS", 3)
    return max(1, min(requested, fan_out_limit))


def get_settings() -> Settings:
    """Deliberately lightweight and UsagePolicy-free -- called on every
    request regardless of whether the keyword-filter feature is on, so
    it must never be able to fail (or do meaningfully more work) because
    of a keyword-filter-only misconfiguration while the feature is off.
    See get_keyword_filter_max_concurrent_calls above for the
    concurrency setting this deliberately excludes."""
    return Settings(
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
        unpaywall_email=os.getenv("UNPAYWALL_EMAIL") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
        keyword_filter_policy_c_enabled=_strict_bool("KEYWORD_FILTER_POLICY_C_ENABLED", False),
        research_lanes_enabled=_strict_bool("RESEARCH_LANES_ENABLED", False),
    )


# --- PR2B: single-user HTTP Basic Auth gate ---
#
# Deliberately NOT a field on Settings/get_settings() above -- same
# reasoning as get_keyword_filter_max_concurrent_calls's own separation
# (see that function's docstring): Settings must stay cheap and always
# succeed on every request. Auth config is the opposite by design -- it
# must be read exactly ONCE, at app construction (api_app/app.py's
# create_app(), reached via research_agent.api's module-level `app =
# create_app()`), and it must be ALLOWED to raise and abort startup
# entirely. Folding it into get_settings() would mean either every
# request pays for this validation, or a misconfigured/missing-credential
# production deployment silently degrades instead of refusing to boot.
_VALID_APP_ENVS = ("local", "production")

# Provisional minimum for a platform-secret-provisioned shared credential
# (not a memorized human password) -- generous on purpose, since anything
# generated by a password manager or platform secret generator clears it
# trivially. Applies whenever the gate is enabled, in either APP_ENV, so
# there is only one validation path to reason about.
MIN_AUTH_PASSWORD_LENGTH = 16


@dataclass(frozen=True)
class AuthConfig:
    """The outcome of validating this deployment's access-gate
    configuration -- returned only by get_auth_config() below, never
    constructed ad hoc elsewhere. `enabled=False` means
    auth_middleware.BasicAuthMiddleware must pass every request straight
    through unchecked (the current, unauthenticated local-dev/test
    behavior, unchanged); `enabled=True` means username/password are both
    present and already validated -- the middleware never re-validates
    them itself."""

    enabled: bool
    username: str | None
    password: str | None


def _app_env() -> str:
    """APP_ENV: 'local' (default) or 'production'. Same fail-loud
    posture as _strict_bool/_positive_int above -- unset/empty falls back
    to 'local', but a SET, unrecognized value always raises rather than
    silently defaulting, since silently treating a typo'd APP_ENV=prod
    (missing the 'uction') as 'local' would mean a real deployment boots
    with the gate treated as optional."""
    raw = os.getenv("APP_ENV")
    if raw is None or raw == "":
        return "local"
    normalized = raw.strip().lower()
    if normalized not in _VALID_APP_ENVS:
        raise ValueError(f"APP_ENV={raw!r} is not valid (use 'local' or 'production')")
    return normalized


def get_auth_config() -> AuthConfig:
    """Validates and returns this process's access-gate configuration.
    Called exactly once, from api_app/app.py's create_app() -- see this
    module's own note above for why. Raises RuntimeError/ValueError (never
    returns a partially-valid config) on any of:

    - APP_ENV set to something other than 'local'/'production'
    - APP_ENV=production with AUTH_ENABLED not true -- there is
      deliberately NO override for this: disabling the gate is never a
      valid production configuration, and this function does not accept
      one. A production rollback means restoring a previous known-good
      image/commit (or otherwise fixing the credentials), never flipping
      auth off.
    - AUTH_ENABLED=true (in either APP_ENV) with AUTH_USERNAME empty
    - AUTH_ENABLED=true (in either APP_ENV) with AUTH_USERNAME containing
      ':' -- ambiguous under RFC 7617's own username/password delimiter;
      AUTH_PASSWORD may still contain ':' (see the check's own comment)
    - AUTH_ENABLED=true (in either APP_ENV) with AUTH_PASSWORD missing or
      shorter than MIN_AUTH_PASSWORD_LENGTH

    AUTH_ENABLED=false with APP_ENV=local (the default with no env vars
    set at all) returns AuthConfig(enabled=False, ...) -- the exact
    current, unauthenticated local-dev/test behavior, unchanged.

    Never includes the actual configured username/password value in any
    raised message -- only which requirement was unmet.
    """
    app_env = _app_env()
    enabled = _strict_bool("AUTH_ENABLED", False)

    if app_env == "production" and not enabled:
        raise RuntimeError(
            "AUTH_ENABLED must be true when APP_ENV=production -- refusing to "
            "start an unauthenticated production instance. There is no "
            "production auth-disable override; if the gate is broken, restore "
            "a previous known-good image/commit instead of disabling it."
        )

    if not enabled:
        return AuthConfig(enabled=False, username=None, password=None)

    username = os.getenv("AUTH_USERNAME") or ""
    if not username:
        raise RuntimeError("AUTH_USERNAME must be set (non-empty) when AUTH_ENABLED=true.")
    # PR2B.1: ':' is the wire-format delimiter between the username and
    # password fields of a decoded Basic-Auth header (RFC 7617) -- a
    # configured username containing one is not a value any standard
    # Basic-Auth client can address unambiguously (a colon typed into a
    # username field is indistinguishable, on the wire, from the same
    # colon marking the username/password boundary). Rejected here,
    # at startup, rather than left to surface as a confusing runtime
    # auth failure. Passwords may still contain ':' -- auth_middleware.py's
    # _parse_basic_credentials only ever splits on the FIRST colon, so a
    # colon anywhere in the password is unambiguous.
    if ":" in username:
        raise RuntimeError("AUTH_USERNAME must not contain ':' when AUTH_ENABLED=true.")

    password = os.getenv("AUTH_PASSWORD") or ""
    if len(password) < MIN_AUTH_PASSWORD_LENGTH:
        raise RuntimeError(
            f"AUTH_PASSWORD must be set and at least {MIN_AUTH_PASSWORD_LENGTH} "
            "characters when AUTH_ENABLED=true."
        )

    return AuthConfig(enabled=True, username=username, password=password)


# --- FRONTEND_ORIGIN / credentialed CORS contract ---
#
# Same discipline as get_auth_config() above: validated ONCE at app
# construction (api_app/app.py's create_app()), ALLOWED to raise (aborting
# startup on an invalid value), never folded into the lightweight,
# always-succeeds get_settings(). Separate from AuthConfig because CORS is
# not authentication -- it only decides which browser origins the backend
# emits Access-Control-* headers for; it grants no access on its own.

# The two conventional Vite-dev-server origins. Both, not just
# `http://localhost:5173`, because a browser may resolve `localhost` to
# the loopback IP and send `Origin: http://127.0.0.1:5173` instead. This
# is a LOCAL-DEV-ONLY default -- production never gets it implicitly.
_LOCAL_DEV_ORIGINS: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class CorsConfig:
    """The validated set of browser origins CORSMiddleware may emit
    credentialed CORS headers for. Every entry is a canonical RFC 6454
    origin (lowercased `scheme://host[:port]`, no path, no trailing
    slash) -- never `*`, never a reflected arbitrary origin. May be empty:
    a same-origin production deployment (the frontend served by this same
    process) needs no cross-origin entry at all."""

    allowed_origins: tuple[str, ...]


def _validate_frontend_origin(raw: str) -> str:
    """Normalize + strictly validate one configured FRONTEND_ORIGIN value.
    Returns the canonical origin string. Raises RuntimeError naming the
    unmet rule. An origin is not itself a secret, but the message stays
    minimal, matching get_auth_config()'s no-secret-echo discipline."""
    from urllib.parse import urlsplit

    value = raw.strip()
    if "*" in value:
        raise RuntimeError("FRONTEND_ORIGIN must not contain '*' -- no wildcard origins.")
    try:
        parts = urlsplit(value)
        _ = parts.hostname, parts.username, parts.password  # force any lazy parse error now
    except ValueError:
        raise RuntimeError("FRONTEND_ORIGIN is not a parseable URL origin.") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise RuntimeError("FRONTEND_ORIGIN must use the http or https scheme, with an explicit '://'.")
    if parts.username is not None or parts.password is not None:
        raise RuntimeError("FRONTEND_ORIGIN must not contain embedded credentials (user:pass@host).")
    if parts.query or parts.fragment:
        raise RuntimeError("FRONTEND_ORIGIN must be an origin only -- no query string or fragment.")
    if parts.path not in ("", "/"):
        raise RuntimeError("FRONTEND_ORIGIN must be an origin only -- no path (a lone trailing '/' is normalized away).")
    if not parts.hostname:
        raise RuntimeError("FRONTEND_ORIGIN must include a host.")
    # Canonical origin: lowercased scheme + netloc (host[:port], no
    # userinfo -- already rejected above), no path/trailing slash. Lower-
    # casing the whole netloc is safe: ports are digits, and browsers
    # lowercase the host (and any IPv6 literal) in the Origin header too,
    # so this matches what CORSMiddleware will exact-match against.
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _is_loopback_origin(origin: str) -> bool:
    from urllib.parse import urlsplit

    return (urlsplit(origin).hostname or "").lower() in _LOOPBACK_HOSTS


def get_cors_config() -> CorsConfig:
    """The FRONTEND_ORIGIN contract. Read once at app construction; may
    raise (aborting startup) on an invalid value or a
    production/dev-origin mismatch.

    - unset/empty + APP_ENV=local -> the two conventional local dev
      origins (http://localhost:5173, http://127.0.0.1:5173). Exactly the
      current local-dev CORS default, unchanged.
    - unset/empty + APP_ENV=production -> NO cross-origin (allowed_origins
      == ()). A same-origin production deployment (the built frontend
      served by this same FastAPI process) is valid with nothing set; a
      SPLIT-origin production deployment MUST set FRONTEND_ORIGIN.
    - set (either APP_ENV) -> exactly that one validated origin. In
      production a loopback value (localhost / 127.0.0.1 / ::1) is
      REFUSED -- production must never silently allow a development origin.

    One origin, not a comma-separated list: a single-frontend, single-user
    app has exactly one trusted browser origin.

    Never `*`, never a reflected arbitrary origin, never wildcard
    credentials -- the middleware and the auth gate both exact-match
    against this list.
    """
    app_env = _app_env()
    raw = os.getenv("FRONTEND_ORIGIN") or ""
    if not raw.strip():
        if app_env == "production":
            return CorsConfig(allowed_origins=())
        return CorsConfig(allowed_origins=_LOCAL_DEV_ORIGINS)

    origin = _validate_frontend_origin(raw)
    if app_env == "production" and _is_loopback_origin(origin):
        raise RuntimeError(
            "FRONTEND_ORIGIN must not be a local development origin "
            "(localhost / 127.0.0.1 / ::1) when APP_ENV=production."
        )
    return CorsConfig(allowed_origins=(origin,))


# --- Day 1 (public multi-user deployment foundation, see
# docs/plans/public-multi-user-deployment-review.md): explicit AnyIO
# sync-route thread-pool capacity ---
#
# Same "validated once, at startup, allowed to raise" discipline as
# get_auth_config()/get_cors_config() above -- deliberately NOT part of
# Settings/get_settings(), which must stay cheap and always succeed on
# every request. This governs a process-wide resource (the one AnyIO
# CapacityLimiter Starlette uses to dispatch every synchronous `def`
# route handler -- effectively every route in this app except
# GET /health as of this phase), so a malformed value must abort startup
# rather than silently fall back to AnyIO's own implicit default (40) or
# clamp to some other, unstated number.
#
# Identical in both APP_ENV values on purpose: unlike AUTH_ENABLED (a
# security gate that must be strictly enforced in production and may be
# left off in local/test), this is a pure resource-tuning knob with no
# security implication -- there is no reason production should be
# stricter than local, or vice versa, so both read the same env var, the
# same default, and the same validation rule.
DEFAULT_ANYIO_THREAD_LIMIT = 16

# A deliberately conservative ceiling on the ceiling: exceeding this always
# raises (never silently clamps) -- see get_thread_pool_config()'s own
# docstring for why 128, and why an operator who genuinely needs more must
# change this constant deliberately rather than override the check via an
# arbitrarily large env value.
MAX_ANYIO_THREAD_LIMIT = 128


@dataclass(frozen=True)
class ThreadPoolConfig:
    """The validated outcome of ANYIO_THREAD_LIMIT -- returned only by
    get_thread_pool_config() below. `limit` is always a positive integer
    no greater than MAX_ANYIO_THREAD_LIMIT."""

    limit: int


def get_thread_pool_config() -> ThreadPoolConfig:
    """ANYIO_THREAD_LIMIT: the ceiling on concurrent synchronous route
    dispatch. `research_agent/api_app/app.py`'s `lifespan()` applies the
    returned value to `anyio.to_thread.current_default_thread_limiter()`
    exactly once, before the app starts serving requests.

    Unset/empty falls back to `DEFAULT_ANYIO_THREAD_LIMIT` (16). A SET
    value must be a plain positive integer no greater than
    `MAX_ANYIO_THREAD_LIMIT` (128); anything else (non-integer, zero,
    negative, or over the ceiling) raises `ValueError` and aborts startup
    -- same fail-loud posture as `get_auth_config()`'s own env-var
    validation, and for the same reason: an operator's typo (an extra
    digit, a stray minus sign, a float) in a setting that controls a real
    resource ceiling must never be silently reinterpreted as "use the
    default" or silently clamped to a different, unstated number.

    **Why 16, not AnyIO's own implicit default of 40.** Every route in
    this app except `GET /health` is a synchronous `def` handler, so this
    ceiling is effectively "how many requests can simultaneously be
    blocked inside a possibly 60-second OpenAI call" (see
    `UsagePolicy.provider_timeout_seconds`, `config/limits.py`) -- not a
    generic web-server tuning knob. Each such in-flight request holds a
    full deserialized session/prompt/response in memory; this project's
    own deployment-sizing review
    (`docs/plans/public-multi-user-deployment-review.md`) estimates
    roughly 0.15-0.3 GiB per concurrent heavy request, against a measured
    idle footprint of roughly 0.4-0.7 GiB (chromadb + onnxruntime +
    pandas/pyarrow + ragas + the full LangChain/LangGraph stack +
    reportlab/lxml -- this project's actual dependency set) on the plan's
    target e2-standard-2 VM (8 GiB). At the implicit default of 40, a
    genuine burst of 40 simultaneously-blocking requests could demand
    roughly 40 * 0.3 =~ 12 GiB on top of the idle footprint -- enough to
    OOM-kill the container well before any individual request's own
    60-second provider timeout would even fire. 16 keeps the worst case
    (16 * 0.3 + 0.7 =~ 5.5 GiB) comfortably inside 8 GiB while staying well
    above the realistic concurrent-tester count for a small,
    approval-gated beta (see the plan's Tier 1 scope) -- legitimate
    traffic should never queue behind this ceiling; it exists to survive
    an abusive or accidental burst, not to reflect expected steady load.
    This is a conservative STARTING value, explicitly meant to be
    revisited using the plan's own Day 7 load-test evidence, not a final
    number (the same "measure, then decide" posture the plan's Correction
    6 applies to every cost figure, applied here to a technical setting
    instead).

    **Scope note (not a limitation of this setting, a boundary of what it
    is for).** This ceiling governs ONLY Starlette's dispatch of
    synchronous route handlers via AnyIO's worker thread pool. It does
    NOT bound the separate, per-request-nested `asyncio.run()`/
    `asyncio.to_thread()` fan-out already used inside
    `curation_loop.py`'s keyword-filter path and
    `query_expansion.py`'s/`agent.py`'s provider fan-out -- each such call
    creates and tears down its OWN throwaway event loop with its OWN
    default `ThreadPoolExecutor`, entirely distinct from AnyIO's
    `CapacityLimiter` (confirmed directly: nothing in `research_agent/`
    imports `anyio.to_thread`/`run_in_threadpool` directly; every internal
    fan-out path goes through `asyncio.run()` + `asyncio.to_thread()`
    instead). That fan-out is already independently bounded by
    `USAGE_PROVIDER_FAN_OUT_LIMIT` / `KEYWORD_FILTER_MAX_CONCURRENT_CALLS`
    and by the one-expensive-action-per-session lease
    (`research_agent/leases.py`). The two pools never share a token and
    nothing in this codebase calls back into AnyIO's own limiter from
    inside an already-dispatched worker thread, so there is no cross-pool
    deadlock risk from that fan-out -- it is an independent, already-
    governed resource dimension this setting does not additionally
    constrain, not a gap in it.
    """
    raw = os.getenv("ANYIO_THREAD_LIMIT")
    if raw is None or raw == "":
        return ThreadPoolConfig(limit=DEFAULT_ANYIO_THREAD_LIMIT)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"ANYIO_THREAD_LIMIT={raw!r} is not a valid integer") from None
    if value <= 0:
        raise ValueError(f"ANYIO_THREAD_LIMIT={value} must be a positive integer")
    if value > MAX_ANYIO_THREAD_LIMIT:
        raise ValueError(
            f"ANYIO_THREAD_LIMIT={value} exceeds the conservative upper bound of "
            f"{MAX_ANYIO_THREAD_LIMIT} -- if a real, load-tested need to go higher exists, "
            "raise MAX_ANYIO_THREAD_LIMIT deliberately rather than overriding this check."
        )
    return ThreadPoolConfig(limit=value)


# --- Day 2 (public multi-user deployment foundation, see
# docs/plans/public-multi-user-deployment-review.md): the production
# PostgreSQL ownership-store contract ---
#
# Same "validated once, at startup, allowed to raise" discipline as
# get_auth_config()/get_cors_config()/get_thread_pool_config() above.
# This is the ONE place DATABASE_URL is ever read -- research_agent/db/
# and research_agent/curation_ownership.py always go through
# get_database_config(), never os.getenv("DATABASE_URL") directly, so
# there is exactly one validation path and exactly one place a raw
# connection string could ever be logged from (nowhere: see
# DatabaseConfig.redacted_url below, the only representation any caller
# is expected to log).
#
# **No silent production fallback to SQLite.** APP_ENV=production with
# DATABASE_URL unset/invalid always raises -- there is no "just use
# SQLite instead" path in production, mirroring get_auth_config()'s own
# "no production auth-disable override" posture exactly. APP_ENV=local
# (the default) never requires DATABASE_URL at all: local development
# and the existing test suite keep today's zero-setup, SQLite-only
# experience completely unchanged unless a developer explicitly opts in
# by setting DATABASE_URL themselves (e.g. to point at a local Postgres
# for Day 2-style development work).
DEFAULT_DATABASE_POOL_MIN_SIZE = 1

# Conservative on purpose: this app's own routes are entirely
# synchronous `def` handlers dispatched through the AnyIO thread-pool
# ceiling (see get_thread_pool_config() above) -- a small pool is
# sufficient headroom for a controlled beta's realistic concurrent-owner-
# lookup volume, and a small pool is also what keeps this compatible
# with Cloud SQL Auth Proxy later (the proxy itself, plus Cloud SQL's own
# small-tier connection ceilings, both favor a modest, explicit pool size
# over an implicit/unbounded one).
DEFAULT_DATABASE_POOL_MAX_SIZE = 5
MAX_DATABASE_POOL_MAX_SIZE = 20


@dataclass(frozen=True)
class DatabaseConfig:
    """The validated outcome of DATABASE_URL/DATABASE_POOL_MIN_SIZE/
    DATABASE_POOL_MAX_SIZE -- returned only by get_database_config()
    below. `configured=False` means no production Postgres is set up at
    all (the local/test SQLite-only default); every other field is only
    meaningful when `configured=True`.

    `url` is intentionally the only field carrying the real connection
    string (including its embedded password) -- it exists so a real
    connection can actually be opened, not so it can be printed. Never
    log, format into an error message, or otherwise surface `url` itself;
    use `redacted_url` (or nothing at all) for any log/error/debug
    output. This mirrors get_auth_config()'s own "never echo the
    configured username/password value" discipline for AUTH_PASSWORD.

    Defense in depth: `repr()`/`str()` of this object (dataclass's own
    default `__repr__` would otherwise include every field, `url`
    included) are overridden below to show `redacted_url` in `url`'s
    place -- so an accidental `logger.info(config)`/an uncaught
    exception's traceback printing a local variable never leaks the
    password either, without every future caller having to remember to
    reach for `.redacted_url` specifically.
    """

    configured: bool
    url: str | None
    pool_min_size: int
    pool_max_size: int
    redacted_url: str | None

    def __repr__(self) -> str:
        return (
            f"DatabaseConfig(configured={self.configured!r}, url=<redacted, see redacted_url>, "
            f"pool_min_size={self.pool_min_size!r}, pool_max_size={self.pool_max_size!r}, "
            f"redacted_url={self.redacted_url!r})"
        )


def _redact_database_url(url: str) -> str:
    """scheme://user:***@host:port/dbname -- password replaced outright
    (never partially shown, never length-revealing), host/port/dbname and
    the scheme kept exactly as configured, since none of that is a secret
    and all of it is useful for a human reading a log line to confirm
    which database a process is actually pointed at. The query string is
    dropped entirely -- libpq's URL form allows credentials there too
    (`?password=...`, `?sslpassword=...`), so "reveal nothing from the
    query" is the safe choice even though it also hides non-secret
    options like `sslmode`.

    Falls back to a fixed, fully-opaque placeholder for anything this
    simple parse can't confidently handle -- never partially redacts and
    never raises; a redaction helper must fail toward "reveal nothing,"
    not toward an exception that could bypass logging entirely."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        if not parts.hostname:
            raise ValueError("no host")
        # Re-bracket an IPv6 literal -- urlsplit strips the brackets from
        # `parts.hostname`, so `[::1]:5432` would otherwise redact to the
        # ambiguous/malformed `::1:5432`.
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        host_part = f"{host}:{parts.port}" if parts.port else host
        user_part = f"{parts.username}:***@" if parts.username else ""
        return f"{parts.scheme}://{user_part}{host_part}{parts.path}"
    except Exception:
        return "<database url, redacted>"


def get_database_config() -> DatabaseConfig:
    """DATABASE_URL: the production PostgreSQL connection string. Read
    and validated exactly once, at application/script startup -- never
    per-request, and never re-derived from a cached Settings instance
    (see this module's own top-level docstring for why AUTH_ENABLED-style
    settings are deliberately excluded from get_settings()/Settings).

    - unset/empty + APP_ENV=local (the default) -> `configured=False`.
      This is the current, unchanged local-dev/test SQLite-only
      experience -- nothing in this codebase requires Postgres to be
      present at all unless DATABASE_URL is explicitly set.
    - unset/empty + APP_ENV=production -> **raises**. There is no
      production fallback to SQLite -- same "refuse to boot rather than
      silently run in a lesser-safe mode" posture as
      get_auth_config()'s own APP_ENV=production requirement.
    - set (either APP_ENV) -> validated for scheme only (must be
      `postgresql://` or `postgres://` -- the two URI scheme designators
      PostgreSQL/libpq itself documents as equivalent, and both of which
      `psycopg.connect()` accepts). The SQLAlchemy dialect spelling
      `postgresql+psycopg://` is deliberately REJECTED: it is not a libpq
      scheme, `psycopg.connect()` raises `ProgrammingError` on it at
      connection time, and that error echoes the whole connection string
      (password included) -- far better to reject it here, at startup,
      with a message that names no value. A malformed value (unparseable,
      wrong scheme, no host) always raises, in either APP_ENV, so a typo
      is caught at startup in local/dev too, not just in production.

    DATABASE_POOL_MIN_SIZE / DATABASE_POOL_MAX_SIZE: strict positive
    integers (same _positive_int-style parsing as
    get_thread_pool_config()'s ANYIO_THREAD_LIMIT), defaulting to 1/5.
    `min_size` must not exceed `max_size`; `max_size` must not exceed
    MAX_DATABASE_POOL_MAX_SIZE (20) -- exceeding either always raises,
    never silently clamps, matching every other startup-validated
    setting in this module. Both are read (and validated) even when
    `configured=False`, so a malformed pool-size env var is caught at
    startup regardless of whether Postgres is actually in use yet --
    the same "a typo must never be silently ignored" posture as every
    other check in this function.
    """
    app_env = _app_env()
    raw_url = os.getenv("DATABASE_URL") or ""

    pool_min_size = _positive_int("DATABASE_POOL_MIN_SIZE", DEFAULT_DATABASE_POOL_MIN_SIZE)
    pool_max_size = _positive_int("DATABASE_POOL_MAX_SIZE", DEFAULT_DATABASE_POOL_MAX_SIZE)
    if pool_max_size > MAX_DATABASE_POOL_MAX_SIZE:
        raise ValueError(
            f"DATABASE_POOL_MAX_SIZE={pool_max_size} exceeds the conservative upper bound of "
            f"{MAX_DATABASE_POOL_MAX_SIZE}."
        )
    if pool_min_size > pool_max_size:
        raise ValueError(
            f"DATABASE_POOL_MIN_SIZE={pool_min_size} must not exceed "
            f"DATABASE_POOL_MAX_SIZE={pool_max_size}."
        )

    if not raw_url.strip():
        if app_env == "production":
            raise RuntimeError(
                "DATABASE_URL must be set when APP_ENV=production -- refusing to start an "
                "unconfigured production instance. There is no production fallback to "
                "SQLite; if Postgres is genuinely unreachable, fix DATABASE_URL rather than "
                "unsetting it."
            )
        return DatabaseConfig(
            configured=False, url=None, pool_min_size=pool_min_size, pool_max_size=pool_max_size, redacted_url=None,
        )

    from urllib.parse import urlsplit

    url = raw_url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        raise RuntimeError("DATABASE_URL is not a parseable URL.") from None
    if parts.scheme not in ("postgresql", "postgres"):
        raise RuntimeError(
            "DATABASE_URL must use the 'postgresql://' or 'postgres://' scheme "
            "(the SQLAlchemy-style 'postgresql+psycopg://' spelling is not a libpq scheme "
            "and psycopg cannot connect with it)."
        )
    if not parts.hostname:
        raise RuntimeError("DATABASE_URL must include a host.")

    return DatabaseConfig(
        configured=True, url=url, pool_min_size=pool_min_size, pool_max_size=pool_max_size,
        redacted_url=_redact_database_url(url),
    )
