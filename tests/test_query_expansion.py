"""Tests for query_expansion.py's Phase 1a change (curation-pool-foundation):
rank_full_pool() returns the FULL ranked candidate list instead of
truncating to top-k, with expanded_search() slicing afterward instead of
truncating inside the ranking step itself.

The one thing this file exists to prove, not assume: that moving the
truncation point from inside semantic_search() (top_k=k) to a plain
Python slice after asking for everything (top_k=len(pool))[:k] produces
the IDENTICAL top-k Chroma would have returned directly. Real (ephemeral,
in-memory) Chroma is used for this — not mocked away — since the risk
being tested is specifically about Chroma's own nearest-neighbor
behavior, which a mock can't stand in for. No live OpenAI calls: query
embeddings are injected via a fake client, same pattern
tests/test_embeddings.py already established.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import pytest

from research_agent.embeddings import _init_cache_db, _serialize_metadata, semantic_search
from research_agent.query_expansion import expanded_search, rank_full_pool, suggest_related_titles
from research_agent.schema import Paper

QUERY_VECTOR = [1.0, 0.0]


@pytest.fixture(autouse=True)
def _isolated_embedding_cache():
    """embed_and_index_papers() (called inside rank_full_pool()) always
    hits the real on-disk cache at data/cache/embeddings.sqlite unless
    redirected — without this, repeated toy abstracts like "abstract for
    p0" across different tests/runs in this file would hit stale cache
    entries from a previous run, undercounting cache_misses and silently
    making the "did we actually re-embed everything" assertions
    meaningless. Same isolation pattern tests/test_embeddings.py already
    established."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "cache.sqlite"
        # side_effect (a fresh connection per call), not return_value (one
        # connection object reused) — embed_and_index_papers() closes its
        # connection after each call, and this fixture's tests call it
        # more than once per test (looping over n/k), so a single reused
        # connection object would be operated on after it's already closed.
        with patch("research_agent.embeddings._init_cache_db", side_effect=lambda *a, **kw: _init_cache_db(cache_path)):
            yield


def _paper(paper_id: str) -> Paper:
    return Paper(
        title=f"Paper {paper_id}", authors=["A"], year=2024, venue="X",
        abstract=f"abstract for {paper_id}", url=None, doi=None, citation_count=None,
        source="arxiv", paper_id=paper_id,
    )


def _fake_client(text_to_vector: dict[str, list[float]]) -> MagicMock:
    """Maps each embedded text to a pre-assigned vector — critically,
    rank_full_pool() calls embed_and_index_papers() BEFORE ranking (to
    ensure the pool is indexed), which re-embeds and re-upserts every
    paper's abstract. A fake client that returns one fixed vector
    regardless of input would overwrite the carefully-seeded, distinct
    per-paper vectors below with garbage, corrupting the very ranking
    this test is trying to verify. Falls back to QUERY_VECTOR for any
    text not in the map (i.e. the actual query string)."""
    client = MagicMock()

    def _create(model, input):
        response = MagicMock()
        vectors = [text_to_vector.get(text, QUERY_VECTOR) for text in input]
        response.usage.total_tokens = 3 * len(input)
        response.data = [MagicMock(embedding=v, index=i) for i, v in enumerate(vectors)]
        return response

    client.embeddings.create.side_effect = _create
    return client


def _seeded_collection(name: str, n: int):
    """n papers on a unit circle at evenly spaced angles, so each has a
    distinct, well-defined cosine similarity to QUERY_VECTOR — a real,
    non-degenerate ranking for Chroma to actually compute, not ties that
    could hide a reordering bug. Returns (collection, papers, client) —
    the client already knows every paper's abstract -> vector mapping, so
    rank_full_pool()'s internal re-embed/upsert is a no-op in effect
    (same vectors go back in), not a corruption.
    """
    collection = chromadb.EphemeralClient().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    papers = [_paper(f"p{i}") for i in range(n)]
    vectors = [[math.cos(i * 0.07), math.sin(i * 0.07)] for i in range(n)]
    text_to_vector = {p.abstract: v for p, v in zip(papers, vectors)}
    client = _fake_client(text_to_vector)

    collection.upsert(
        ids=[p.paper_id for p in papers],
        embeddings=vectors,
        metadatas=[_serialize_metadata(p, used_title_fallback=False) for p in papers],
        documents=[p.abstract for p in papers],
    )
    return collection, papers, client


def test_rank_full_pool_returns_every_candidate_not_just_k():
    collection, papers, client = _seeded_collection("full-pool-1", n=25)
    ranked, stats = rank_full_pool("q", papers, client=client, collection=collection)
    assert len(ranked) == 25
    assert stats["cache_misses"] == 25


def test_rank_full_pool_empty_input_returns_empty_without_touching_chroma():
    ranked, stats = rank_full_pool("q", [], client=MagicMock(), collection=MagicMock())
    assert ranked == []
    assert stats["cache_misses"] == 0


def test_full_pool_sliced_to_k_matches_direct_top_k_query_exactly():
    """The actual regression this phase must not introduce: does asking
    Chroma for everything and slicing in Python give the SAME top-k as
    asking Chroma for top_k directly? Real ephemeral Chroma, not mocked —
    this is specifically testing Chroma's own nearest-neighbor behavior
    at two different n_results values, which a mock can't stand in for."""
    for n in (10, 25, 40):
        collection, papers, client = _seeded_collection(f"equiv-{n}", n=n)
        ids = [p.paper_id for p in papers]

        for k in (1, 5, 10):
            if k > n:
                continue
            direct = semantic_search("q", collection=collection, client=client, top_k=k, where={"paper_id": {"$in": ids}})
            full, _ = rank_full_pool("q", papers, client=client, collection=collection)
            sliced = full[:k]

            direct_ids = [p.paper_id for p, _ in direct]
            sliced_ids = [p.paper_id for p, _ in sliced]
            assert sliced_ids == direct_ids, (
                f"n={n}, k={k}: full-pool-then-slice {sliced_ids} != direct top_k {direct_ids}"
            )
            direct_scores = [round(s, 6) for _, s in direct]
            sliced_scores = [round(s, 6) for _, s in sliced]
            assert sliced_scores == direct_scores, f"n={n}, k={k}: scores differ between the two approaches"


def test_expanded_search_slices_full_ranked_pool_to_k_unchanged_from_before():
    """expanded_search()'s own contract to its callers (api.py, eval
    scripts): given the same candidate pool, it must still return exactly
    k results in the same order as before this refactor — build_candidate_pool
    is mocked (real network calls, irrelevant to what's being tested here),
    everything downstream of it is real."""
    collection, papers, client = _seeded_collection("expanded-search-1", n=12)

    with patch("research_agent.query_expansion.build_candidate_pool", return_value=papers), \
         patch("research_agent.query_expansion.get_chroma_collection", return_value=collection):
        ranked = expanded_search("q", k=4, client=client)

    assert len(ranked) == 4
    # Highest-similarity papers first, same convention semantic_search() itself uses.
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


# --- Phase 1b: PaperPoolSession — seen-set, cursor, serve/refill ---
#
# Fully mocked per the brief's own guidance for this phase: this is testing
# bookkeeping (does a batch overlap with a previous one, does refill
# trigger at the right threshold, does exclusion actually exclude), not
# ranking quality — that's already covered by Phase 1a's real-Chroma tests
# above. rank_full_pool() is mocked here too, with a simple deterministic
# "rank by input order" stand-in, so the combining/exclusion logic itself
# is what's under test, not real embedding similarity.

def _identity_rank(topic, papers, client=None, **kwargs):
    return [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}


def test_serve_next_batch_returns_non_overlapping_batches_and_advances_cursor():
    from research_agent.query_expansion import PaperPoolSession, serve_next_batch

    reserve = [(_paper(f"p{i}"), 1.0 - i * 0.01) for i in range(24)]
    session = PaperPoolSession(topic="q", reserve=reserve)

    batch1 = serve_next_batch(session, batch_size=10)
    batch2 = serve_next_batch(session, batch_size=10)
    batch3 = serve_next_batch(session, batch_size=10)  # only 4 left

    ids1 = {p.paper_id for p, _ in batch1}
    ids2 = {p.paper_id for p, _ in batch2}
    ids3 = {p.paper_id for p, _ in batch3}

    assert len(ids1) == 10 and len(ids2) == 10 and len(ids3) == 4
    assert ids1.isdisjoint(ids2) and ids2.isdisjoint(ids3) and ids1.isdisjoint(ids3)
    assert session.cursor == 24
    assert session.seen_paper_ids == ids1 | ids2 | ids3


def test_needs_refill_triggers_exactly_when_remaining_drops_below_batch_size():
    from research_agent.query_expansion import PaperPoolSession, serve_next_batch

    reserve = [(_paper(f"p{i}"), 1.0) for i in range(20)]
    session = PaperPoolSession(topic="q", reserve=reserve)

    serve_next_batch(session, batch_size=10)  # 10 left
    assert session.needs_refill(batch_size=10) is False  # exactly 10 remaining, not YET below

    serve_next_batch(session, batch_size=5)  # 5 left
    assert session.needs_refill(batch_size=10) is True


def test_refill_pool_excludes_seen_papers_and_merges_with_unserved_tail():
    from research_agent.query_expansion import PaperPoolSession, refill_pool, serve_next_batch

    reserve = [(_paper(f"p{i}"), 1.0 - i * 0.01) for i in range(12)]
    session = PaperPoolSession(topic="q", reserve=reserve)
    serve_next_batch(session, batch_size=10)  # seen = p0..p9, unserved tail = p10, p11

    # Fresh search "finds" 3 papers already seen (p3, p7 — must be excluded),
    # 2 already in the unserved tail (p10 — must not be duplicated), and 4
    # genuinely new ones.
    fresh_papers = [_paper(pid) for pid in ["p3", "p7", "p10", "new1", "new2", "new3", "new4"]]

    with patch("research_agent.query_expansion.build_candidate_pool", return_value=fresh_papers), \
         patch("research_agent.query_expansion.rank_full_pool", side_effect=_identity_rank):
        new_count = refill_pool(session, client=MagicMock())

    assert new_count == 4  # only new1-new4 are genuinely new
    reserve_ids = [p.paper_id for p, _ in session.reserve]
    assert reserve_ids.count("p10") == 1  # unserved tail item not duplicated
    assert "p3" not in reserve_ids and "p7" not in reserve_ids  # already-seen excluded
    assert set(reserve_ids) == {"p10", "p11", "new1", "new2", "new3", "new4"}
    assert session.cursor == 0  # reset — new reserve is guaranteed seen-free


# --- Phase 1c: suggest_related_titles's exclude_titles ---

def _fake_title_suggestion_client(titles: list[str]) -> MagicMock:
    from unittest.mock import MagicMock as _MM
    client = _MM()
    parsed = _MM(titles=titles)
    message = _MM(parsed=parsed)
    response = _MM(usage=_MM(total_tokens=50, prompt_tokens=40, completion_tokens=10))
    response.choices = [_MM(message=message)]
    client.chat.completions.parse.return_value = response
    return client


def test_suggest_related_titles_prompt_includes_exclusion_list_when_provided():
    """The actual Phase 1c fix: the excluded titles must appear in the
    PROMPT sent to the model, not just be used to filter its response
    afterward — this is the difference between "ask it not to repeat
    itself" and "silently throw away whatever it repeats anyway"."""
    client = _fake_title_suggestion_client(["A New Paper"])
    suggest_related_titles("topic", client=client, exclude_titles=["Attention Is All You Need", "BERT"])

    sent_messages = client.chat.completions.parse.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "Attention Is All You Need" in user_content
    assert "BERT" in user_content
    assert "must NOT be suggested again" in user_content or "NOT be suggested" in user_content


def test_suggest_related_titles_no_exclusion_list_prompt_unchanged_from_before():
    """Regression proof: omitting exclude_titles (the default) must send
    the EXACT prompt text this function always sent — no new behavior for
    every existing caller that doesn't pass this new parameter."""
    client = _fake_title_suggestion_client(["A New Paper"])
    suggest_related_titles("my topic", max_titles=5, client=client)

    sent_messages = client.chat.completions.parse.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert user_content == "Topic: my topic\n\nSuggest up to 5 well-known real papers on this topic."


def test_suggest_related_titles_defensively_filters_repeated_titles_despite_prompt():
    """Belt-and-suspenders: even if the model ignores the prompt
    instruction and returns an excluded title anyway, it must not survive
    into the final result."""
    client = _fake_title_suggestion_client(["Attention Is All You Need", "A Genuinely New Paper"])
    titles = suggest_related_titles("topic", client=client, exclude_titles=["Attention Is All You Need"])
    assert titles == ["A Genuinely New Paper"]


# --- Phase 1d: adversarial edge cases ---

def test_needs_refill_is_true_from_the_start_when_initial_pool_is_smaller_than_batch_size():
    """Refill triggered on the very first turn: if the initial search only
    found 6 papers total and batch_size is 10, refill must be signaled
    immediately — not only after some serving has happened."""
    from research_agent.query_expansion import PaperPoolSession

    session = PaperPoolSession(topic="a very obscure topic", reserve=[(_paper(f"p{i}"), 1.0) for i in range(6)])
    assert session.cursor == 0
    assert session.needs_refill(batch_size=10) is True


def test_serve_next_batch_on_empty_reserve_returns_empty_list_without_crashing():
    from research_agent.query_expansion import PaperPoolSession, serve_next_batch

    session = PaperPoolSession(topic="q", reserve=[])
    batch = serve_next_batch(session, batch_size=10)
    assert batch == []
    assert session.cursor == 0
    assert session.seen_paper_ids == set()


def test_refill_pool_when_topic_genuinely_exhausted_returns_zero_without_crashing():
    """Real papers genuinely run out: the fresh search finds nothing at
    all. Must degrade to an empty reserve and a clear 0-new-papers signal,
    never raise."""
    from research_agent.query_expansion import PaperPoolSession, refill_pool, serve_next_batch

    session = PaperPoolSession(topic="an exhaustively obscure topic", reserve=[(_paper(f"p{i}"), 1.0) for i in range(10)])
    serve_next_batch(session, batch_size=10)  # fully served, nothing left in the tail

    with patch("research_agent.query_expansion.build_candidate_pool", return_value=[]), \
         patch("research_agent.query_expansion.rank_full_pool", side_effect=_identity_rank):
        new_count = refill_pool(session, client=MagicMock())

    assert new_count == 0
    assert session.reserve == []
    assert session.cursor == 0


def test_refill_pool_when_search_finds_only_already_seen_papers_returns_zero():
    """Distinct from a literally-empty search: the search DOES find
    papers, but every single one is already seen — still 0 genuinely new,
    not an error, and the already-seen papers must not reappear."""
    from research_agent.query_expansion import PaperPoolSession, refill_pool, serve_next_batch

    session = PaperPoolSession(topic="q", reserve=[(_paper(f"p{i}"), 1.0) for i in range(10)])
    serve_next_batch(session, batch_size=10)

    with patch("research_agent.query_expansion.build_candidate_pool", return_value=[_paper("p3"), _paper("p7")]), \
         patch("research_agent.query_expansion.rank_full_pool", side_effect=_identity_rank):
        new_count = refill_pool(session, client=MagicMock())

    assert new_count == 0
    assert session.reserve == []


def test_repeated_refill_on_exhausted_topic_stays_at_zero_idempotently():
    """Calling refill_pool again on an already-exhausted topic must keep
    returning 0 cleanly (e.g. a UI retry button), not error or grow the
    reserve with duplicates."""
    from research_agent.query_expansion import PaperPoolSession, refill_pool, serve_next_batch

    session = PaperPoolSession(topic="q", reserve=[(_paper(f"p{i}"), 1.0) for i in range(10)])
    serve_next_batch(session, batch_size=10)

    with patch("research_agent.query_expansion.build_candidate_pool", return_value=[]), \
         patch("research_agent.query_expansion.rank_full_pool", side_effect=_identity_rank):
        first_retry = refill_pool(session, client=MagicMock())
        second_retry = refill_pool(session, client=MagicMock())

    assert (first_retry, second_retry) == (0, 0)
    assert session.reserve == []


if __name__ == "__main__":
    test_rank_full_pool_returns_every_candidate_not_just_k()
    test_rank_full_pool_empty_input_returns_empty_without_touching_chroma()
    test_full_pool_sliced_to_k_matches_direct_top_k_query_exactly()
    test_expanded_search_slices_full_ranked_pool_to_k_unchanged_from_before()
    test_serve_next_batch_returns_non_overlapping_batches_and_advances_cursor()
    test_needs_refill_triggers_exactly_when_remaining_drops_below_batch_size()
    test_refill_pool_excludes_seen_papers_and_merges_with_unserved_tail()
    test_suggest_related_titles_prompt_includes_exclusion_list_when_provided()
    test_suggest_related_titles_no_exclusion_list_prompt_unchanged_from_before()
    test_suggest_related_titles_defensively_filters_repeated_titles_despite_prompt()
    test_needs_refill_is_true_from_the_start_when_initial_pool_is_smaller_than_batch_size()
    test_serve_next_batch_on_empty_reserve_returns_empty_list_without_crashing()
    test_refill_pool_when_topic_genuinely_exhausted_returns_zero_without_crashing()
    test_refill_pool_when_search_finds_only_already_seen_papers_returns_zero()
    test_repeated_refill_on_exhausted_topic_stays_at_zero_idempotently()
    print("All query_expansion tests passed.")
