"""Mock-only runner for the chat_relevance suite (R7D.1). No live mode
yet -- see cli.py's own explicit "not implemented" rejection.

predict() calls the REAL research_agent.qa._filter_relevant_web_articles
unmodified, with research_agent.qa._embed_with_cache patched to return
small, fixed, hand-picked vectors derived from each fixture case's own
per-candidate `mock_relevance` label -- the exact same "patch
_embed_with_cache with a vectors lookup, call the real function" pattern
tests/test_qa.py's own R7A/R7B/R7C red-team tests already use. This
makes the suite a genuine regression test of the real threshold/AND-of-
query-and-topic decision logic (imported, not reimplemented), driven by
deterministic fixture data instead of real embeddings -- not a parallel
reimplementation of relevance scoring.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from research_agent.evals.evaluators.relevance import ALL_EVALUATORS
from research_agent.evals.runners._base import Example, SuiteResult, run_suite
from research_agent.qa import _filter_relevant_web_articles
from research_agent.schema import WebArticle

SUITE = "chat_relevance"
DATASET_FILE = "chat_web_relevance_redteam.jsonl"

# Fixed, hand-picked 3D vectors -- not tuned per fixture case, just a
# small, readable way to represent all 4 combinations of "clears the
# per-turn query check" x "clears the session-topic check"
# independently, against the real _WEB_ARTICLE_RELEVANCE_THRESHOLD
# (0.25) via plain cosine similarity. See module docstring.
_QUERY_VECTOR = [1.0, 0.0, 0.0]
_TOPIC_VECTOR = [0.0, 1.0, 0.0]
_RELEVANCE_VECTORS = {
    "query_and_topic": [1.0, 1.0, 0.0],
    "query_only": [1.0, 0.0, 0.0],
    "topic_only": [0.0, 1.0, 0.0],
    "neither": [0.0, 0.0, 1.0],
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
    """Runs the real relevance filter for one fixture case, with only
    the embedding call itself mocked -- see module docstring."""
    query = example.inputs.get("query", "")
    topic = example.inputs.get("topic", "")
    fail_open = example.inputs.get("fail_open", True)
    mock_embedding_error = bool(example.inputs.get("mock_embedding_error", False))
    candidates = example.inputs.get("candidates") or []

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

    with embed_patch:
        kept = _filter_relevant_web_articles(
            query, articles, MagicMock(), topic=topic, fail_open=fail_open,
        )

    return {"relevant_urls": [a.url for a in kept]}


def run_experiment(mode: str = "mock", subset: int | None = None, tags: list[str] | None = None) -> SuiteResult:
    if mode != "mock":
        raise ValueError(f"run_chat_relevance only supports mode='mock' as of R7D.1 (got {mode!r})")
    evaluators: list[tuple[str, Any]] = [
        ("chat_relevance_correctness", ALL_EVALUATORS["chat_relevance_correctness"]),
    ]
    return run_suite(
        suite=SUITE, dataset_file=DATASET_FILE, predict=predict, evaluators=evaluators,
        mode=mode, subset=subset, tags=tags,
    )
