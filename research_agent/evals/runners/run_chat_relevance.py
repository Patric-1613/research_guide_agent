"""Runner for the chat_relevance suite -- mock mode (R7D.1) and opt-in
live mode (R7D.2).

Both modes call the REAL research_agent.qa._filter_relevant_web_articles
unmodified -- only how the embedding call underneath it is satisfied
differs:

- mock mode (predict) patches research_agent.qa._embed_with_cache to
  return small, fixed, hand-picked vectors derived from each fixture
  case's own per-candidate `mock_relevance` label -- the exact same
  "patch _embed_with_cache with a vectors lookup, call the real
  function" pattern tests/test_qa.py's own R7A/R7B/R7C red-team tests
  already use.
- live mode (predict_live) constructs a real OpenAI client and lets
  _filter_relevant_web_articles make real embedding calls through it --
  the exact same client construction (`OpenAI()`, reading
  OPENAI_API_KEY from the environment) qa.ask() itself uses.

Either way this is a genuine regression test of the real threshold/
AND-of-query-and-topic decision logic (imported, not reimplemented),
never a parallel reimplementation of relevance scoring -- and neither
mode duplicates the predict -> evaluate -> aggregate loop itself, which
lives once in runners/_base.py::run_suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from openai import OpenAI, OpenAIError

from research_agent.evals.evaluators.relevance import ALL_EVALUATORS
from research_agent.evals.runners._base import Example, LiveModeSetupError, SuiteResult, run_suite
from research_agent.qa import _filter_relevant_web_articles
from research_agent.schema import WebArticle

SUITE = "chat_relevance"
DATASET_FILE = "chat_web_relevance_redteam.jsonl"

# Fixed, hand-picked 3D vectors -- not tuned per fixture case, just a
# small, readable way to represent all 4 combinations of "clears the
# per-turn query check" x "clears the session-topic check"
# independently, against the real _WEB_ARTICLE_RELEVANCE_THRESHOLD
# (0.25) via plain cosine similarity. Mock mode only -- see module
# docstring.
_QUERY_VECTOR = [1.0, 0.0, 0.0]
_TOPIC_VECTOR = [0.0, 1.0, 0.0]
_RELEVANCE_VECTORS = {
    "query_and_topic": [1.0, 1.0, 0.0],
    "query_only": [1.0, 0.0, 0.0],
    "topic_only": [0.0, 1.0, 0.0],
    "neither": [0.0, 0.0, 1.0],
    # R7E.3: cosine([1,1,3], query)==cosine([1,1,3], topic)==1/sqrt(11)
    # ~= 0.3015 -- clears the general 0.25 threshold on BOTH dimensions
    # (unlike the other 4 labels above, deliberately weak rather than a
    # clean 0/0.71/1.0) but fails _STALE_POOL_QUERY_THRESHOLD (0.50) --
    # for stale-pool fixture cases that need to reproduce a candidate
    # clearing the base AND-check yet still failing the stricter
    # different-source-query re-check, matching the real numbers R7E's
    # live-eval analysis found (query_similarity=0.4756,
    # topic_similarity=0.4291 for the actual stale US-policy source).
    "query_and_topic_weak": [1.0, 1.0, 3.0],
}


def _build_web_article(candidate: dict[str, Any]) -> WebArticle:
    return WebArticle(
        title=candidate["title"],
        url=candidate["url"],
        snippet=candidate.get("snippet", ""),
        published_date=candidate.get("published_date"),
        source_domain=candidate.get("source_domain", "example.com"),
    )


def predict(example: Example) -> dict[str, Any]:
    """Mock-mode prediction -- see module docstring. `mock_relevance`/
    `mock_embedding_error` are mock-only fixture fields; live mode
    ignores them entirely (see predict_live below).

    R7E.1: also captures `_filter_relevant_web_articles`'s new `debug`
    scores under `debug_scores` -- since mock mode drives the same real
    function through mocked (not skipped) `_embed_with_cache` calls, the
    resulting query/topic "similarity" numbers are meaningful relative
    to the fixed 3D vector scheme above, useful for confirming the mock
    scheme itself behaves as designed.

    R7E.3: also passes a fixture's optional `provenance_by_url` straight
    through to the real `_filter_relevant_web_articles` -- no
    reimplementation of the stale-pool comparison here, and no mock
    vector needed for `source_query` text itself (the real function
    never embeds it -- see that function's own R7E.3 docstring section:
    it only does a plain string comparison against `query`, reusing the
    query_similarity already computed for the candidate)."""
    query = example.inputs.get("query", "")
    topic = example.inputs.get("topic", "")
    fail_open = example.inputs.get("fail_open", True)
    mock_embedding_error = bool(example.inputs.get("mock_embedding_error", False))
    candidates = example.inputs.get("candidates") or []
    provenance_by_url = example.inputs.get("provenance_by_url")

    articles = [_build_web_article(c) for c in candidates]
    vectors: dict[str, list[float]] = {query: _QUERY_VECTOR}
    if topic:
        vectors[topic] = _TOPIC_VECTOR
    for candidate, article in zip(candidates, articles):
        label = candidate.get("mock_relevance", "neither")
        vectors[f"{article.title}\n{article.snippet or ''}"] = _RELEVANCE_VECTORS[label]

    if mock_embedding_error:
        embed_patch = patch(
            "research_agent.qa._embed_with_cache", side_effect=RuntimeError("mock embedding failure"),
        )
    else:
        embed_patch = patch(
            "research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text],
        )

    debug_scores: list[dict[str, Any]] = []
    with embed_patch:
        kept = _filter_relevant_web_articles(
            query, articles, MagicMock(), topic=topic, fail_open=fail_open, debug=debug_scores,
            provenance_by_url=provenance_by_url,
        )

    return {"relevant_urls": [a.url for a in kept], "debug_scores": debug_scores}


def predict_live(example: Example, client: OpenAI) -> dict[str, Any]:
    """Live-mode prediction (R7D.2) -- the real embedding pathway, no
    mocking at all. Never called for a `mock_only` example -- see
    `_mock_only_skip_reason` and run_experiment's own skip_if.

    R7E.1: passes `debug=debug_scores` through to
    `_filter_relevant_web_articles` so the real per-candidate query/
    topic cosine-similarity scores ride along in the returned
    prediction (under `debug_scores`) -- persisted by
    write_run_detail_json, this is what closes R7E's own finding that
    the first live run's failures couldn't be diagnosed from the
    aggregate CSV row alone.

    R7E.3: also passes a fixture's optional `provenance_by_url` straight
    through -- same real, unmodified stale-pool logic as mock mode,
    driven by real embeddings instead of mocked ones."""
    query = example.inputs.get("query", "")
    topic = example.inputs.get("topic", "")
    fail_open = example.inputs.get("fail_open", True)
    candidates = example.inputs.get("candidates") or []
    provenance_by_url = example.inputs.get("provenance_by_url")

    articles = [_build_web_article(c) for c in candidates]
    debug_scores: list[dict[str, Any]] = []
    kept = _filter_relevant_web_articles(
        query, articles, client, topic=topic, fail_open=fail_open, debug=debug_scores,
        provenance_by_url=provenance_by_url,
    )
    return {"relevant_urls": [a.url for a in kept], "debug_scores": debug_scores}


def _build_live_client() -> OpenAI:
    """Constructs the real OpenAI client live mode needs -- the same
    bare `OpenAI()` call qa.ask() itself uses, which reads
    OPENAI_API_KEY (or another SDK-recognized credential) from the
    environment and raises OpenAIError immediately, before any network
    call, if none is set. Re-raised as LiveModeSetupError so the CLI can
    print one clean line instead of a raw traceback -- and so this
    happens BEFORE run_suite's per-example loop even starts, not
    partway through a run."""
    try:
        return OpenAI()
    except OpenAIError as exc:
        raise LiveModeSetupError(
            "live mode requires OpenAI credentials (OPENAI_API_KEY or another OpenAI-SDK-recognized "
            f"credential) -- client construction failed: {exc}"
        ) from exc


def _mock_only_skip_reason(example: Example, mode: str) -> str | None:
    if mode == "live" and example.metadata.get("mock_only"):
        return (
            "mock_only fixture case -- skipped in live mode (simulates an embedding-API "
            "exception that can't be forced against the real API)"
        )
    return None


def run_experiment(mode: str = "mock", subset: int | None = None, tags: list[str] | None = None) -> SuiteResult:
    evaluators: list[tuple[str, Any]] = [
        ("chat_relevance_correctness", ALL_EVALUATORS["chat_relevance_correctness"]),
    ]

    if mode == "mock":
        run_predict = predict
    elif mode == "live":
        client = _build_live_client()
        run_predict = lambda example: predict_live(example, client)  # noqa: E731
    else:
        raise ValueError(f"run_chat_relevance only supports mode in {{'mock', 'live'}} (got {mode!r})")

    return run_suite(
        suite=SUITE, dataset_file=DATASET_FILE, predict=run_predict, evaluators=evaluators,
        mode=mode, subset=subset, tags=tags,
        skip_if=lambda example: _mock_only_skip_reason(example, mode),
    )
