"""Deterministic tests for qa.py's non-LLM logic: schema constraints, the
defensive "no citations if unanswerable" override, empty-pool handling,
condense-question being skipped on the first turn, and the
classify_message non-substantive-message gate (semantic-classify-message:
embedding similarity, not exact-match). The OpenAI calls are mocked so
these run without network access or billing — real embedding similarity
scores for the adversarial cases live in
scripts/test_semantic_classify_live.py instead, since that needs a real
embedding call to mean anything.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch

from research_agent.qa import MAX_HISTORY_TURNS, ChatSession, _build_answer_schema, _classify_non_substantive, condense_question, capped_history, _renumber_citation_markers, _filter_relevant_web_articles, ask
from research_agent.schema import Paper, WebArticle


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet=f"Snippet for {title}.", published_date=None, source_domain="example.com")


def _mock_parse_response(schema_cls, **kwargs):
    parsed = schema_cls(**kwargs)
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def test_answer_schema_rejects_unknown_paper_id():
    schema = _build_answer_schema(["a", "b"])
    schema(answerable=True, answer="fine [1]", cited_paper_ids=["a"])
    try:
        schema(answerable=True, answer="bad", cited_paper_ids=["not-real"])
        assert False, "expected a validation error for an unknown paper_id"
    except Exception:
        pass


def test_ask_with_no_papers_short_circuits_without_calling_client():
    session = ChatSession(papers=[])
    mock_client = MagicMock()
    result = ask(session, "anything?", client=mock_client)
    assert result["answerable"] is False
    assert result["cited_papers"] == []
    mock_client.chat.completions.parse.assert_not_called()
    # still logs the turn so a caller building a transcript sees the refusal
    assert session.history[-2] == {"role": "user", "content": "anything?"}


def testcondense_question_skips_llm_call_on_first_turn():
    mock_client = MagicMock()
    result = condense_question([], "what about it?", mock_client)
    assert result == "what about it?"
    mock_client.chat.completions.create.assert_not_called()


def test_answer_schema_without_web_urls_has_no_cited_web_urls_field():
    schema = _build_answer_schema(["a"])  # no web_urls -> field must not exist at all
    assert "cited_web_urls" not in schema.model_fields


def test_answer_schema_rejects_unknown_web_url():
    schema = _build_answer_schema(["a"], ["https://real.com"])
    schema(answerable=True, answer="fine [Paper 1] [Web 1]", cited_paper_ids=["a"], cited_web_urls=["https://real.com"])
    try:
        schema(answerable=True, answer="bad", cited_paper_ids=[], cited_web_urls=["https://not-retrieved.com"])
        assert False, "expected a validation error for an unretrieved URL"
    except Exception:
        pass


# --- _renumber_citation_markers (chat-ux-fixes bug 5) ---

def test_renumber_citation_markers_closes_a_gap_that_skips_1():
    """The exact real-world symptom reported: web citations starting at
    [Web 2] without [Web 1] ever appearing."""
    answer = "Per [Web 2] and [Web 3], the hornet arrived in 2016."
    assert _renumber_citation_markers(answer) == "Per [Web 1] and [Web 2], the hornet arrived in 2016."


def test_renumber_citation_markers_leaves_already_correct_numbering_unchanged():
    answer = "Per [Paper 1] and [Paper 2], X is true."
    assert _renumber_citation_markers(answer) == answer


def test_renumber_citation_markers_treats_paper_and_web_as_independent_sequences():
    answer = "Per [Paper 3] and [Web 5], X is true."
    assert _renumber_citation_markers(answer) == "Per [Paper 1] and [Web 1], X is true."


def test_renumber_citation_markers_reuses_the_same_new_number_for_repeated_references():
    answer = "[Web 4] says X. Later, [Web 4] also says Y. But [Web 7] disagrees."
    assert _renumber_citation_markers(answer) == "[Web 1] says X. Later, [Web 1] also says Y. But [Web 2] disagrees."


def test_renumber_citation_markers_no_markers_is_a_no_op():
    answer = "There are no citations in this answer at all."
    assert _renumber_citation_markers(answer) == answer


def test_ask_with_only_web_articles_no_papers_still_answers():
    session = ChatSession(papers=[], web_articles=[_web_article("https://x.com/a", "Article A")])
    schema = _build_answer_schema([], ["https://x.com/a"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 1], X is true.", cited_paper_ids=[], cited_web_urls=["https://x.com/a"],
    )

    result = ask(session, "what does the web say?", client=mock_client)

    assert result["answerable"] is True
    assert result["cited_papers"] == []
    assert len(result["cited_web_articles"]) == 1
    assert result["cited_web_articles"][0].url == "https://x.com/a"
    mock_client.chat.completions.parse.assert_called_once()


def test_ask_renumbers_web_citations_the_model_got_wrong():
    """chat-ux-fixes bug 5, through the real ask() path (not just the
    isolated _renumber_citation_markers unit): a model response that
    skips straight to [Web 2]/[Web 3] must come back through ask()
    already fixed to [Web 1]/[Web 2] -- both in the returned result AND
    in what gets persisted to session.history."""
    session = ChatSession(papers=[], web_articles=[
        _web_article("https://x.com/a", "Article A"), _web_article("https://x.com/b", "Article B"),
    ])
    schema = _build_answer_schema([], ["https://x.com/a", "https://x.com/b"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 2] and [Web 3], X is true.",
        cited_paper_ids=[], cited_web_urls=["https://x.com/a", "https://x.com/b"],
    )

    result = ask(session, "what does the web say?", client=mock_client)

    assert result["answer"] == "Per [Web 1] and [Web 2], X is true."
    assert session.history[-1] == {"role": "assistant", "content": "Per [Web 1] and [Web 2], X is true."}


def test_ask_forces_empty_web_citations_when_model_marks_unanswerable():
    session = ChatSession(papers=[], web_articles=[_web_article("https://x.com/a", "Article A")])
    schema = _build_answer_schema([], ["https://x.com/a"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="Can't answer.", cited_paper_ids=[], cited_web_urls=["https://x.com/a"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)):
        result = ask(session, "unanswerable", client=mock_client)

    assert result["answerable"] is False
    assert result["cited_web_articles"] == []  # forced empty despite model returning a url


# --- curation-chat-web-relevance: _filter_relevant_web_articles ---

def test_filter_relevant_web_articles_keeps_relevant_and_removes_stale_articles():
    """Deterministic via patched embeddings, not a live API call --
    real cosine-similarity behavior against actual questions/articles is
    a separate calibration concern (see qa.py's own threshold comment),
    not what this test proves. This test proves the filtering LOGIC:
    given known vectors, the article whose embedding points the same
    direction as the query survives, the orthogonal one doesn't."""
    query = "is jailbreaking covered?"
    relevant = _web_article("https://relevant.com", "Jailbreak coverage")
    stale = _web_article("https://stale.com", "Unrelated roundup")
    vectors = {
        query: [1.0, 0.0],
        f"{relevant.title}\n{relevant.snippet}": [1.0, 0.0],  # same direction -- similarity 1.0
        f"{stale.title}\n{stale.snippet}": [0.0, 1.0],  # orthogonal -- similarity 0.0
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [relevant, stale], MagicMock())

    assert kept == [relevant]


def test_filter_relevant_web_articles_returns_original_articles_on_embedding_exception():
    """Fail OPEN, not closed -- the default (fail_open=True, unpassed
    here, matching _filter_web_relevance_node's own live call site as of
    R7B) -- same defensive posture as this module's existing search_web/
    condense_question try/except guards. Also proves the original list
    itself is returned (not a copy), matching the 'never mutate the
    original list' contract trivially since nothing about it is touched
    on this path."""
    articles = [_web_article("https://a.com", "A"), _web_article("https://b.com", "B")]

    with patch("research_agent.qa._embed_with_cache", side_effect=RuntimeError("embedding API down")):
        result = _filter_relevant_web_articles("some query", articles, MagicMock())

    assert result == articles
    assert result is articles


def test_filter_relevant_web_articles_fail_open_false_returns_empty_list_on_embedding_exception():
    """chat-web-relevance-guardrails R7B: fail_open=False (used by
    curation_chat.py's insertion-time gate) inverts the default posture
    above -- an embedding failure returns [] instead of the unfiltered
    original list, since this is the one gate deciding whether a
    brand-new article ever joins a persistent pool. Still never RAISES
    -- same no-raise contract, just a different failure value."""
    articles = [_web_article("https://a.com", "A"), _web_article("https://b.com", "B")]

    with patch("research_agent.qa._embed_with_cache", side_effect=RuntimeError("embedding API down")):
        result = _filter_relevant_web_articles("some query", articles, MagicMock(), fail_open=False)

    assert result == []


def test_filter_relevant_web_articles_empty_list_is_a_noop_with_no_embedding_calls():
    with patch("research_agent.qa._embed_with_cache") as mock_embed:
        result = _filter_relevant_web_articles("some query", [], MagicMock())

    assert result == []
    mock_embed.assert_not_called()


# --- chat-web-relevance-guardrails R7C: outcome recording ---

def test_filter_relevant_web_articles_outcome_records_false_on_normal_success():
    """A genuine, completed check records fail_open_triggered=False --
    even one that keeps everything it was given (nothing to distinguish
    "ran and passed" from "ran and happened to keep all of it" here;
    both are a real, executed check)."""
    query = "is jailbreaking covered?"
    relevant = _web_article("https://relevant.com", "Jailbreak coverage")
    vectors = {
        query: [1.0, 0.0],
        f"{relevant.title}\n{relevant.snippet}": [1.0, 0.0],
    }
    outcome: dict = {}

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        _filter_relevant_web_articles(query, [relevant], MagicMock(), outcome=outcome)

    assert outcome == {"fail_open_triggered": False}


def test_filter_relevant_web_articles_outcome_records_false_on_empty_input():
    """The trivial empty-input early return also records False -- there
    was nothing to fail on, so it's vacuously "not fail-open," not
    "unknown."""
    outcome: dict = {}

    with patch("research_agent.qa._embed_with_cache") as mock_embed:
        _filter_relevant_web_articles("some query", [], MagicMock(), outcome=outcome)

    assert outcome == {"fail_open_triggered": False}
    mock_embed.assert_not_called()


def test_filter_relevant_web_articles_outcome_records_true_on_fail_open_exception():
    articles = [_web_article("https://a.com", "A")]
    outcome: dict = {}

    with patch("research_agent.qa._embed_with_cache", side_effect=RuntimeError("embedding API down")):
        result = _filter_relevant_web_articles("some query", articles, MagicMock(), outcome=outcome)

    assert outcome == {"fail_open_triggered": True}
    assert result == articles  # fail_open=True default still returns the unfiltered list


def test_filter_relevant_web_articles_outcome_records_true_on_fail_closed_exception():
    articles = [_web_article("https://a.com", "A")]
    outcome: dict = {}

    with patch("research_agent.qa._embed_with_cache", side_effect=RuntimeError("embedding API down")):
        result = _filter_relevant_web_articles("some query", articles, MagicMock(), fail_open=False, outcome=outcome)

    assert outcome == {"fail_open_triggered": True}
    assert result == []


# --- eval-instrumentation R7E.1: debug scores ---

def test_filter_relevant_web_articles_debug_is_additive_and_does_not_change_kept_articles():
    """`debug` is off by default and, when given, must never change
    which articles survive -- proven by running the exact same inputs
    with and without it and asserting the kept list is identical."""
    query = "is jailbreaking covered?"
    topic = "AI safety research"
    relevant = _web_article("https://relevant.com", "Jailbreak coverage")
    stale = _web_article("https://stale.com", "Unrelated roundup")
    vectors = {
        query: [1.0, 0.0],
        topic: [1.0, 0.0],
        f"{relevant.title}\n{relevant.snippet}": [1.0, 0.0],
        f"{stale.title}\n{stale.snippet}": [0.0, 1.0],
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept_without_debug = _filter_relevant_web_articles(query, [relevant, stale], MagicMock(), topic=topic)
        debug: list[dict] = []
        kept_with_debug = _filter_relevant_web_articles(
            query, [relevant, stale], MagicMock(), topic=topic, debug=debug,
        )

    assert kept_with_debug == kept_without_debug == [relevant]


def test_filter_relevant_web_articles_debug_records_per_candidate_scores_with_topic():
    query = "is jailbreaking covered?"
    topic = "AI safety research"
    relevant = _web_article("https://relevant.com", "Jailbreak coverage")
    stale = _web_article("https://stale.com", "Unrelated roundup")
    vectors = {
        query: [1.0, 0.0],
        topic: [1.0, 0.0],
        f"{relevant.title}\n{relevant.snippet}": [1.0, 0.0],  # clears both -- kept
        f"{stale.title}\n{stale.snippet}": [0.0, 1.0],  # orthogonal to both -- dropped on the query check
    }
    debug: list[dict] = []

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        _filter_relevant_web_articles(query, [relevant, stale], MagicMock(), topic=topic, debug=debug)

    assert len(debug) == 2
    kept_entry = next(d for d in debug if d["url"] == "https://relevant.com")
    dropped_entry = next(d for d in debug if d["url"] == "https://stale.com")

    assert kept_entry["query_similarity"] == pytest.approx(1.0)
    assert kept_entry["topic_similarity"] == pytest.approx(1.0)
    assert kept_entry["passed_query_threshold"] is True
    assert kept_entry["passed_topic_threshold"] is True
    assert kept_entry["kept"] is True

    # Dropped on the query check alone -- but the topic score is still
    # surfaced for debugging even though it wasn't needed for the decision.
    assert dropped_entry["query_similarity"] == pytest.approx(0.0)
    assert dropped_entry["topic_similarity"] == pytest.approx(0.0)
    assert dropped_entry["passed_query_threshold"] is False
    assert dropped_entry["passed_topic_threshold"] is False
    assert dropped_entry["kept"] is False


def test_filter_relevant_web_articles_debug_topic_fields_are_none_without_a_topic():
    query = "is jailbreaking covered?"
    article = _web_article("https://relevant.com", "Jailbreak coverage")
    vectors = {query: [1.0, 0.0], f"{article.title}\n{article.snippet}": [1.0, 0.0]}
    debug: list[dict] = []

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        _filter_relevant_web_articles(query, [article], MagicMock(), debug=debug)

    assert debug == [{
        "url": "https://relevant.com", "title": article.title,
        "query_similarity": 1.0, "topic_similarity": None,
        "passed_query_threshold": True, "passed_topic_threshold": None, "kept": True,
    }]


def test_filter_relevant_web_articles_debug_stays_empty_on_embedding_exception():
    """An embedding-call exception is caught before any per-candidate
    scoring happens -- debug should reflect that (empty), not a
    fabricated entry."""
    articles = [_web_article("https://a.com", "A")]
    debug: list[dict] = []

    with patch("research_agent.qa._embed_with_cache", side_effect=RuntimeError("embedding API down")):
        _filter_relevant_web_articles("some query", articles, MagicMock(), debug=debug)

    assert debug == []


def test_ask_stale_web_pool_filtered_out_is_not_passed_into_the_model_prompt_for_unrelated_question():
    """End-to-end through ask()'s real graph (filter_web_relevance patched
    at the seam, not the embedding math -- that's covered by the direct
    unit tests above): proves the WIRING -- once the filter says "nothing
    relevant," the stale article's content never reaches the model's
    prompt, even though it's still sitting in session.web_articles."""
    paper = _paper("p1", "AI Risk Tiering")
    stale = _web_article("https://stale.com", "Stale roundup")
    session = ChatSession(papers=[paper], web_articles=[stale])
    schema = _build_answer_schema(["p1"], None)

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="Not covered by the retrieved papers.", cited_paper_ids=[],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._filter_relevant_web_articles", return_value=[]):
        result = ask(session, "is jailbreaking covered?", client=mock_client)

    assert result["retrieved_web_articles"] == []
    assert result["cited_web_articles"] == []
    generate_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in generate_messages)
    assert "Stale roundup" not in joined
    assert "https://stale.com" not in joined


def test_ask_mixed_web_pool_passes_only_the_relevant_article_to_the_model_prompt():
    paper = _paper("p1", "AI Risk Tiering")
    relevant = _web_article("https://relevant.com", "Fresh relevant article")
    stale = _web_article("https://stale.com", "Stale roundup")
    session = ChatSession(papers=[paper], web_articles=[stale, relevant])
    schema = _build_answer_schema(["p1"], ["https://relevant.com"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 1], the latest development is X.",
        cited_paper_ids=[], cited_web_urls=["https://relevant.com"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._filter_relevant_web_articles", return_value=[relevant]):
        result = ask(session, "what's the latest on this?", client=mock_client)

    generate_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in generate_messages)
    assert "Fresh relevant article" in joined
    assert "Stale roundup" not in joined
    assert len(result["cited_web_articles"]) == 1
    assert result["cited_web_articles"][0].url == "https://relevant.com"


# --- chat-web-relevance-guardrails R7A: topic-aware relevance foundation
# (red-team fixtures) ---
#
# All of these test _filter_relevant_web_articles' new optional `topic`
# parameter directly -- deterministic via patched _embed_with_cache, same
# convention the pre-R7A tests above already use (parallel vectors ->
# similarity 1.0 -> passes; orthogonal -> similarity 0.0 -> fails). None
# of this is wired into the live graph yet (see the dedicated
# not-wired-yet test at the end of this section) -- these prove the
# HELPER's own logic in isolation.

def test_filter_relevant_web_articles_rejects_housing_source_for_ai_governance_topic():
    """The actual reported regression: a chat session about AI governance
    cited a web source about a housing/zoning case study that merely
    shared governance-adjacent vocabulary ("policy", "regulatory
    framework"). Modeled here as a query that has drifted to something
    generic enough to superficially match the housing article (passes
    the query-only check on its own -- see the query-relevant-but-topic-
    irrelevant test below for the same mechanism without the narrative),
    while the session's own stable topic ("AI governance") does not
    match it at all. Topic-aware filtering must reject it."""
    topic = "AI governance frameworks"
    query = "what's the latest on governance frameworks?"  # drifted -- lost "AI"
    housing_article = WebArticle(
        title="Housing Policy Case Study: Zoning Reform",
        url="https://example.com/housing-zoning-case-study",
        snippet="A regulatory framework and governance case study examining local zoning policy reform.",
        published_date=None, source_domain="example.com",
    )
    vectors = {
        query: [1.0, 0.0],
        f"{housing_article.title}\n{housing_article.snippet}": [1.0, 0.0],  # matches the drifted query
        topic: [0.0, 1.0],  # orthogonal to the housing article -- genuinely unrelated to AI governance
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [housing_article], MagicMock(), topic=topic)

    assert kept == []


def test_filter_relevant_web_articles_accepts_a_genuinely_relevant_ai_governance_source():
    topic = "AI governance frameworks"
    query = "what's the latest on AI governance frameworks?"
    relevant_article = _web_article("https://example.com/ai-governance-report", "New AI Governance Report")
    vectors = {
        query: [1.0, 0.0],
        f"{relevant_article.title}\n{relevant_article.snippet}": [1.0, 0.0],
        topic: [1.0, 0.0],
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [relevant_article], MagicMock(), topic=topic)

    assert kept == [relevant_article]


def test_filter_relevant_web_articles_rejects_query_relevant_but_topic_irrelevant_source():
    """The general mechanism behind the housing regression above, without
    the narrative: an article can match the per-turn query closely while
    having nothing to do with the session's actual topic. Query-only
    filtering (pre-R7A behavior) would have kept this; topic-aware
    filtering must not."""
    topic = "quantum computing hardware"
    query = "tell me more about this"
    article = _web_article("https://example.com/unrelated", "Matches The Query, Not The Topic")
    vectors = {
        query: [1.0, 0.0],
        f"{article.title}\n{article.snippet}": [1.0, 0.0],
        topic: [0.0, 1.0],
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [article], MagicMock(), topic=topic)

    assert kept == []


def test_filter_relevant_web_articles_rejects_topic_relevant_but_query_irrelevant_source_when_both_required():
    """The inverse: an article that's squarely on-topic but doesn't
    actually answer THIS question. Both dimensions are required (AND,
    not OR) -- topic relevance alone is not sufficient to keep an
    article that doesn't match what was actually asked."""
    topic = "quantum computing hardware"
    query = "what's the weather like today"
    article = _web_article("https://example.com/on-topic", "Matches The Topic, Not The Query")
    vectors = {
        query: [1.0, 0.0],
        f"{article.title}\n{article.snippet}": [0.0, 1.0],
        topic: [0.0, 1.0],
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [article], MagicMock(), topic=topic)

    assert kept == []


def test_filter_relevant_web_articles_empty_topic_preserves_query_only_behavior():
    """topic="" (the default) must reproduce the exact pre-R7A, query-
    only outcome -- same scenario as test_filter_relevant_web_articles_
    keeps_relevant_and_removes_stale_articles above, run explicitly with
    topic="" this time, proving the new parameter is a true no-op when
    omitted/empty rather than just "usually behaves the same."."""
    query = "is jailbreaking covered?"
    relevant = _web_article("https://relevant.com", "Jailbreak coverage")
    stale = _web_article("https://stale.com", "Unrelated roundup")
    vectors = {
        query: [1.0, 0.0],
        f"{relevant.title}\n{relevant.snippet}": [1.0, 0.0],
        f"{stale.title}\n{stale.snippet}": [0.0, 1.0],
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept_no_topic_arg = _filter_relevant_web_articles(query, [relevant, stale], MagicMock())
        kept_explicit_empty_topic = _filter_relevant_web_articles(query, [relevant, stale], MagicMock(), topic="")

    assert kept_no_topic_arg == [relevant]
    assert kept_explicit_empty_topic == [relevant]


def test_filter_relevant_web_articles_temporal_query_trap_rejects_off_topic_recent_source():
    """chat-ux-fixes bug 2/4's own documented trap pattern (see curation_
    chat.py's _accept_web_offer): a temporal follow-up like "what about
    very recent developments, like in 2026?" condenses to something
    generic enough to match almost any "recent roundup" article on query
    alone, with the actual subject lost. Topic-aware filtering catches
    what a temporal, subject-less query cannot."""
    topic = "AI governance frameworks"
    query = "what about very recent developments, like in 2026?"
    recent_but_unrelated = WebArticle(
        title="2026 Tech Trends Roundup",
        url="https://example.com/2026-trends",
        snippet="The latest developments and recent trends shaping industries in 2026.",
        published_date="2026-01-01", source_domain="example.com",
    )
    vectors = {
        query: [1.0, 0.0],
        f"{recent_but_unrelated.title}\n{recent_but_unrelated.snippet}": [1.0, 0.0],  # matches the generic temporal query
        topic: [0.0, 1.0],  # not actually about AI governance
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [recent_but_unrelated], MagicMock(), topic=topic)

    assert kept == []


def test_filter_relevant_web_articles_rejects_when_title_looks_on_topic_but_combined_snippet_is_not():
    """Proves title+snippet (not title alone) is genuinely what gets
    judged, matching this function's own existing "article text is
    title+snippet" contract: a title reading "AI Governance in Practice"
    would pass a title-only check, but the embedded TEXT here is the
    combined title+snippet string, and it's this combined string's
    vector that's off-topic -- the title alone is never embedded or
    checked in isolation."""
    topic = "AI governance frameworks"
    query = "how is AI governance applied in practice?"
    misleading_title_article = WebArticle(
        title="AI Governance in Practice",
        url="https://example.com/misleading-title",
        snippet="A zoning board case study on housing permit approval timelines in suburban planning.",
        published_date=None, source_domain="example.com",
    )
    combined_text = f"{misleading_title_article.title}\n{misleading_title_article.snippet}"
    vectors = {
        query: [1.0, 0.0],
        topic: [1.0, 0.0],
        combined_text: [0.0, 1.0],  # the SNIPPET's real subject dominates the combined embedding
    }

    with patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        kept = _filter_relevant_web_articles(query, [misleading_title_article], MagicMock(), topic=topic)

    assert kept == []


def test_ask_now_filters_stale_web_pool_by_topic():
    """chat-web-relevance-guardrails R7B: _filter_web_relevance_node now
    passes state["session"].topic through to _filter_relevant_web_
    articles (R7A built the mechanism but deliberately left it
    unwired). Proven here by running the same article/query through
    twice: with no topic, the article survives on query-only relevance
    (matching pre-R7B behavior exactly); with a topic the article is
    genuinely unrelated to, it's filtered out before generate_answer
    ever sees it -- proven both via retrieved/cited_web_articles being
    empty AND via the article's own text being absent from the actual
    prompt sent to the model."""
    paper = _paper("p1", "AI Risk Tiering")
    article = _web_article("https://housing.example.com/zoning", "Housing Case Study")
    query = "governance frameworks"
    rejecting_topic = "a topic this article would fail against"
    vectors = {
        query: [1.0, 0.0],
        rejecting_topic: [0.0, 1.0],
        f"{article.title}\n{article.snippet}": [1.0, 0.0],
    }

    session_no_topic = ChatSession(papers=[paper], web_articles=[article], topic="")
    schema_with_web = _build_answer_schema(["p1"], [article.url])
    mock_client_no_topic = MagicMock()
    mock_client_no_topic.chat.completions.parse.return_value = _mock_parse_response(
        schema_with_web, answerable=True, answer="Per [Web 1], X.", cited_paper_ids=[], cited_web_urls=[article.url],
    )
    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        result_no_topic = ask(session_no_topic, query, client=mock_client_no_topic)

    assert [a.url for a in result_no_topic["retrieved_web_articles"]] == [article.url]
    assert [a.url for a in result_no_topic["cited_web_articles"]] == [article.url]

    session_with_topic = ChatSession(papers=[paper], web_articles=[article], topic=rejecting_topic)
    schema_papers_only = _build_answer_schema(["p1"], None)
    mock_client_with_topic = MagicMock()
    mock_client_with_topic.chat.completions.parse.return_value = _mock_parse_response(
        schema_papers_only, answerable=True, answer="Per papers, X.", cited_paper_ids=[],
    )
    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        result_with_topic = ask(session_with_topic, query, client=mock_client_with_topic)

    assert result_with_topic["retrieved_web_articles"] == []
    assert result_with_topic["cited_web_articles"] == []
    generate_messages = mock_client_with_topic.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in generate_messages)
    assert "Housing Case Study" not in joined
    assert article.url not in joined


def test_ask_housing_vs_ai_governance_article_does_not_reach_generate_answer_prompt():
    """The actual reported regression, now proven live through ask()'s
    real graph (R7A only proved this at the _filter_relevant_web_
    articles helper level, in isolation). A chat session about AI
    governance must not let a housing/zoning article with merely
    governance-adjacent vocabulary reach the model's context, even when
    the per-turn query has drifted generic enough to match it on query
    relevance alone."""
    paper = _paper("p1", "AI Risk Tiering")
    topic = "AI governance frameworks"
    query = "what's the latest on governance frameworks?"  # drifted -- lost "AI"
    housing_article = WebArticle(
        title="Housing Policy Case Study: Zoning Reform",
        url="https://example.com/housing-zoning-case-study",
        snippet="A regulatory framework and governance case study examining local zoning policy reform.",
        published_date=None, source_domain="example.com",
    )
    session = ChatSession(papers=[paper], web_articles=[housing_article], topic=topic)
    schema = _build_answer_schema(["p1"], None)
    vectors = {
        query: [1.0, 0.0],
        f"{housing_article.title}\n{housing_article.snippet}": [1.0, 0.0],  # matches the drifted query
        topic: [0.0, 1.0],  # genuinely unrelated to AI governance
    }
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per papers, X.", cited_paper_ids=[],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        result = ask(session, query, client=mock_client)

    assert result["retrieved_web_articles"] == []
    assert result["cited_web_articles"] == []
    generate_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    joined = " ".join(m["content"] for m in generate_messages)
    assert "Housing Policy Case Study" not in joined
    assert housing_article.url not in joined


def test_ask_result_carries_web_relevance_verified_true_on_genuine_filter_run():
    """chat-web-relevance-guardrails R7C: ask()'s result surfaces
    web_relevance_verified end-to-end -- True when the answer-time
    filter genuinely ran this turn (curation_chat.py stamps this onto
    the chat exchange for report-promotion gating)."""
    paper = _paper("p1", "AI Risk Tiering")
    article = _web_article("https://relevant.com", "Relevant article")
    schema = _build_answer_schema(["p1"], [article.url])
    query = "some query"
    vectors = {
        query: [1.0, 0.0],
        f"{article.title}\n{article.snippet}": [1.0, 0.0],
    }
    session = ChatSession(papers=[paper], web_articles=[article])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 1], X.", cited_paper_ids=[], cited_web_urls=[article.url],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._embed_with_cache", side_effect=lambda client, text: vectors[text]):
        result = ask(session, query, client=mock_client)

    assert result["web_relevance_verified"] is True


def test_ask_result_carries_web_relevance_verified_false_on_fail_open():
    """The answer-time gate's own fail-open default still lets a cited
    article through (unchanged R7B behavior) -- but the turn's citation
    is now correctly flagged as unverified rather than silently
    indistinguishable from a genuine pass."""
    paper = _paper("p1", "AI Risk Tiering")
    article = _web_article("https://relevant.com", "Relevant article")
    schema = _build_answer_schema(["p1"], [article.url])
    session = ChatSession(papers=[paper], web_articles=[article])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 1], X.", cited_paper_ids=[], cited_web_urls=[article.url],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(paper, 0.9)]), \
         patch("research_agent.qa._embed_with_cache", side_effect=RuntimeError("embedding API down")):
        result = ask(session, "some query", client=mock_client)

    assert result["web_relevance_verified"] is False
    assert [a.url for a in result["retrieved_web_articles"]] == [article.url]  # fail-open still let it through
    assert [a.url for a in result["cited_web_articles"]] == [article.url]


def test_ask_ignores_chat_session_topic_end_to_end():
    """ChatSession(..., topic=...) round-trips into the graph state without
    raising or otherwise changing behavior -- a minimal smoke test that
    the new field is wired through ask()'s own state construction
    cleanly (session is held by reference in QAState, per its own
    docstring), independent of the more targeted not-wired-yet proof
    above."""
    session = ChatSession(papers=[], topic="some topic")
    mock_client = MagicMock()

    result = ask(session, "anything?", client=mock_client)

    assert result["answerable"] is False
    mock_client.chat.completions.parse.assert_not_called()


def test_ask_forces_empty_citations_when_model_marks_unanswerable():
    """Defensive check: even if the model violates instructions and returns
    cited_paper_ids alongside answerable=False, ask() must not surface a
    citation on a claim the model itself says it can't support."""
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers)
    schema = _build_answer_schema(["1111"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="I can't answer this.", cited_paper_ids=["1111"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask(session, "unanswerable question", client=mock_client)

    assert result["answerable"] is False
    assert result["cited_papers"] == []  # forced empty despite model returning an id


# --- classify_message: semantic non-substantive-message gate ---
#
# Two layers of tests. The gating tests below (question mark / word count
# / empty) are pure, deterministic, and must short-circuit BEFORE any
# embedding call is made — asserted directly via a MagicMock client whose
# .embeddings.create is never touched. The threshold-decision tests mock
# _get_reference_embeddings/_embed_with_cache with small hand-picked
# vectors so the >=threshold boundary itself is deterministically
# testable without a real embedding call. Real similarity scores for the
# actual adversarial phrases ("hii", "thnak you", "thx", ...) are in
# scripts/test_semantic_classify_live.py — a mocked vector can prove the
# threshold LOGIC is correct, but only a real embedding call can prove
# real typos land above it.

def test_classify_non_substantive_question_mark_veto_short_circuits_before_any_embedding_call():
    """The canonical trap: a real follow-up question that happens to open
    with a closer phrase. The question mark alone must be enough to save
    it — and must never even reach the embedding call."""
    mock_client = MagicMock()
    is_skip, category, score = _classify_non_substantive("Thanks, but what about its limitations?", mock_client)
    assert (is_skip, category, score) == (False, None, 0.0)
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_length_gate_short_circuits_before_any_embedding_call():
    """Isolates the word-count gate: no question mark present, so this
    relies entirely on the message being too long to be a bare closer —
    and, same as the question-mark veto, must never reach the embedding
    call (a real follow-up shouldn't cost an embedding call it can't use)."""
    mock_client = MagicMock()
    for message in ["thanks but can you also compare it to BERT", "ok what does the paper say about this"]:
        is_skip, category, score = _classify_non_substantive(message, mock_client)
        assert (is_skip, category, score) == (False, None, 0.0), f"expected {message!r} to fail the length gate"
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_content_override_word_short_circuits_before_any_embedding_call():
    """Isolates the wh-word/imperative-verb guard specifically: short
    (<=5 words), no question mark — so neither of the other two guards
    would catch these — real measured evidence showed similarity alone
    scores these HIGHER than genuine closers ("thanks so explain" 0.6104
    vs. "sup" 0.5406), so this guard is the only thing protecting them."""
    mock_client = MagicMock()
    for message in ["Thanks explain", "Ok cool but why", "Hi define overfitting", "Great but how"]:
        is_skip, category, score = _classify_non_substantive(message, mock_client)
        assert (is_skip, category, score) == (False, None, 0.0), f"expected {message!r} to be caught by the content-override guard"
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_empty_message_does_not_skip_and_skips_embedding_call():
    mock_client = MagicMock()
    for message in ["", "   "]:
        is_skip, category, score = _classify_non_substantive(message, mock_client)
        assert (is_skip, category, score) == (False, None, 0.0)
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_similarity_at_or_above_threshold_skips_with_matched_category():
    mock_client = MagicMock()
    # [0.45, 0.8930] is unit-length (0.45^2 + 0.8930^2 ~= 1), so its cosine
    # similarity against [1.0, 0.0] is exactly ~0.45 — the real threshold
    # itself (see _NON_SUBSTANTIVE_SIMILARITY_THRESHOLD's own comment for
    # how that value was derived from real scores), deliberately chosen to
    # test the boundary (>=), not just comfortably above it.
    with patch("research_agent.qa._get_reference_embeddings", return_value={"acknowledgment": [("thanks", [0.45, 0.8930])]}), \
         patch("research_agent.qa._embed_with_cache", return_value=[1.0, 0.0]):
        is_skip, category, score = _classify_non_substantive("thnx", mock_client)
    assert score == pytest.approx(0.45, abs=1e-3)
    assert category == "acknowledgment"
    assert is_skip is True


def test_classify_non_substantive_similarity_below_threshold_does_not_skip():
    mock_client = MagicMock()
    with patch("research_agent.qa._get_reference_embeddings", return_value={"greeting": [("hi", [1.0, 0.0])]}), \
         patch("research_agent.qa._embed_with_cache", return_value=[0.0, 1.0]):  # orthogonal -> similarity 0.0
        is_skip, category, score = _classify_non_substantive("what is transfer learning", mock_client)
    assert score == pytest.approx(0.0)
    assert category is None
    assert is_skip is False


def test_classify_non_substantive_picks_the_best_scoring_category_across_multiple():
    mock_client = MagicMock()
    with patch("research_agent.qa._get_reference_embeddings", return_value={
        "greeting": [("hi", [0.0, 1.0])],
        "acknowledgment": [("thanks", [1.0, 0.0])],
    }), patch("research_agent.qa._embed_with_cache", return_value=[1.0, 0.0]):
        is_skip, category, score = _classify_non_substantive("thanks", mock_client)
    assert category == "acknowledgment"  # exact match (similarity 1.0), not greeting (similarity 0.0)
    assert score == pytest.approx(1.0)
    assert is_skip is True


def test_ask_short_circuits_on_non_substantive_message_without_any_llm_or_retrieval_call():
    """Mocks _classify_non_substantive directly rather than the embedding
    call it makes internally — this test's job is proving ask() reacts
    correctly to a "skip" verdict, not re-proving the classifier's own
    similarity math (that's the dedicated _classify_non_substantive tests
    above, plus real scores in scripts/test_semantic_classify_live.py)."""
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers, history=[
        {"role": "user", "content": "what is RoCoFT?"},
        {"role": "assistant", "content": "RoCoFT is a parameter-efficient fine-tuning method."},
    ])
    mock_client = MagicMock()

    with patch("research_agent.qa._classify_non_substantive", return_value=(True, "acknowledgment", 0.91)), \
         patch("research_agent.qa.embed_and_index_papers") as mock_embed, \
         patch("research_agent.qa.get_chroma_collection") as mock_collection, \
         patch("research_agent.qa.semantic_search") as mock_search:
        result = ask(session, "Thanks!", client=mock_client)

    assert result["answerable"] is True
    assert result["answer"] == "You're welcome! Let me know if you have any more questions about these papers."
    mock_client.chat.completions.create.assert_not_called()  # condense_question never ran
    mock_client.chat.completions.parse.assert_not_called()  # generate_answer never ran
    mock_embed.assert_not_called()
    mock_collection.assert_not_called()
    mock_search.assert_not_called()
    assert session.history[-2:] == [
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome! Let me know if you have any more questions about these papers."},
    ]


def test_ask_does_not_short_circuit_the_trap_case_thanks_but_a_real_question():
    """The real regression test: a message opening with a closer phrase but
    containing an actual follow-up question must go through the full
    condense/retrieve/generate pipeline, not the canned response."""
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers, history=[
        {"role": "user", "content": "what is RoCoFT?"},
        {"role": "assistant", "content": "RoCoFT is a parameter-efficient fine-tuning method."},
    ])
    schema = _build_answer_schema(["1111"])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="What are RoCoFT's limitations?"))],
        usage=MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10),
    )
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="RoCoFT's main limitation is X [Paper 1].", cited_paper_ids=["1111"],
    )

    with patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask(session, "Thanks, but what about its limitations?", client=mock_client)

    assert result["answerable"] is True
    assert len(result["cited_papers"]) == 1
    mock_client.chat.completions.create.assert_called_once()  # condense_question DID run
    mock_client.chat.completions.parse.assert_called_once()  # generate_answer DID run


def testcapped_history_keeps_only_last_n_turns():
    history = []
    for i in range(12):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})

    capped = capped_history(history, max_turns=3)
    assert capped == [
        {"role": "user", "content": "question 9"},
        {"role": "assistant", "content": "answer 9"},
        {"role": "user", "content": "question 10"},
        {"role": "assistant", "content": "answer 10"},
        {"role": "user", "content": "question 11"},
        {"role": "assistant", "content": "answer 11"},
    ]


def testcapped_history_is_a_no_op_below_the_cap():
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert capped_history(history, max_turns=8) == history


def testcapped_history_strips_extra_metadata_keys():
    """curation-chat-metadata Phase 1: entries persisted with extra
    keys (exchange_id/used_web_search/cited_web_articles/cited_papers/
    added_to_report) must come back as plain {role, content} -- this is
    what's actually handed to the model, so any extra key here would
    ride along into the LLM-bound messages list (qa.py's _generate_node
    splices this return value directly into `messages`).

    report-quality Phase R3.2 Chunk 1: cited_papers (added alongside the
    already-existing cited_web_articles) is stripped by the exact same
    mechanism, with no code change needed -- capped_history was already
    written to keep only {role, content} regardless of what extra keys
    the input dicts carry."""
    history = [
        {
            "role": "user", "content": "what's new?", "exchange_id": "abc123",
        },
        {
            "role": "assistant", "content": "Per [Web 1], ...", "exchange_id": "abc123",
            "used_web_search": True,
            "cited_web_articles": [{"url": "https://x.com", "title": "X"}],
            "cited_papers": [{"paper_id": "p1", "title": "Paper One"}],
            "added_to_report": False,
        },
    ]

    capped = capped_history(history, max_turns=8)

    assert capped == [
        {"role": "user", "content": "what's new?"},
        {"role": "assistant", "content": "Per [Web 1], ..."},
    ]
    assert list(capped[0].keys()) == ["role", "content"]
    assert list(capped[1].keys()) == ["role", "content"]


def testcapped_history_does_not_mutate_the_original_list_or_dicts():
    """The persisted history (session.history/chat_history) must keep its
    full metadata -- only the returned copy handed to the LLM boundary is
    stripped."""
    original_assistant_turn = {
        "role": "assistant", "content": "Per [Web 1], ...", "exchange_id": "abc123",
        "used_web_search": True, "cited_web_articles": [{"url": "https://x.com", "title": "X"}],
        "cited_papers": [{"paper_id": "p1", "title": "Paper One"}],
        "added_to_report": False,
    }
    history = [{"role": "user", "content": "what's new?", "exchange_id": "abc123"}, original_assistant_turn]
    history_snapshot = [dict(turn) for turn in history]

    capped = capped_history(history, max_turns=8)

    # The original list still has its original two dicts, unchanged...
    assert history == history_snapshot
    # ...and the returned entries are NEW dict objects, not the same ones
    # (so a caller mutating the capped copy can never corrupt the original).
    assert capped[1] is not original_assistant_turn
    assert "used_web_search" in original_assistant_turn  # untouched by capping
    assert "cited_papers" in original_assistant_turn  # untouched by capping


def test_ask_caps_history_to_last_n_turns_in_prompt_sent_to_model():
    # A long simulated conversation (12 prior turns, more than the 8-turn
    # cap) — only the last MAX_HISTORY_TURNS turns should reach the actual
    # prompt sent to the model; older ones must be dropped.
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers)
    for i in range(12):
        session.history.append({"role": "user", "content": f"question {i}"})
        session.history.append({"role": "assistant", "content": f"answer {i}"})

    schema = _build_answer_schema(["1111"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Final answer [Paper 1].", cited_paper_ids=["1111"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        ask(session, "new question", client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    # [0] is the system prompt, [-1] is the new question + context, and
    # everything in between should be exactly the last MAX_HISTORY_TURNS
    # turns (2 messages each) — no more, no less.
    history_in_prompt = sent_messages[1:-1]
    assert len(history_in_prompt) == 2 * MAX_HISTORY_TURNS
    assert history_in_prompt[0] == {"role": "user", "content": "question 4"}  # oldest turn kept
    assert history_in_prompt[-1] == {"role": "assistant", "content": "answer 11"}  # newest prior turn

    sent_contents = [m["content"] for m in sent_messages if isinstance(m.get("content"), str)]
    assert "question 0" not in sent_contents  # dropped: older than the cap
    assert "question 3" not in sent_contents  # dropped: older than the cap

    # The full, uncapped history is still preserved on the session object —
    # only the prompt sent to the model is capped, not what's stored (a
    # caller building a UI transcript still sees every turn).
    assert len(session.history) == 12 * 2 + 2


if __name__ == "__main__":
    test_answer_schema_rejects_unknown_paper_id()
    test_ask_with_no_papers_short_circuits_without_calling_client()
    testcondense_question_skips_llm_call_on_first_turn()
    test_answer_schema_without_web_urls_has_no_cited_web_urls_field()
    test_answer_schema_rejects_unknown_web_url()
    test_renumber_citation_markers_closes_a_gap_that_skips_1()
    test_renumber_citation_markers_leaves_already_correct_numbering_unchanged()
    test_renumber_citation_markers_treats_paper_and_web_as_independent_sequences()
    test_renumber_citation_markers_reuses_the_same_new_number_for_repeated_references()
    test_renumber_citation_markers_no_markers_is_a_no_op()
    test_ask_with_only_web_articles_no_papers_still_answers()
    test_ask_renumbers_web_citations_the_model_got_wrong()
    test_ask_forces_empty_web_citations_when_model_marks_unanswerable()
    test_ask_forces_empty_citations_when_model_marks_unanswerable()
    test_classify_non_substantive_question_mark_veto_short_circuits_before_any_embedding_call()
    test_classify_non_substantive_length_gate_short_circuits_before_any_embedding_call()
    test_classify_non_substantive_content_override_word_short_circuits_before_any_embedding_call()
    test_classify_non_substantive_empty_message_does_not_skip_and_skips_embedding_call()
    test_classify_non_substantive_similarity_at_or_above_threshold_skips_with_matched_category()
    test_classify_non_substantive_similarity_below_threshold_does_not_skip()
    test_classify_non_substantive_picks_the_best_scoring_category_across_multiple()
    test_ask_short_circuits_on_non_substantive_message_without_any_llm_or_retrieval_call()
    test_ask_does_not_short_circuit_the_trap_case_thanks_but_a_real_question()
    testcapped_history_keeps_only_last_n_turns()
    testcapped_history_is_a_no_op_below_the_cap()
    testcapped_history_strips_extra_metadata_keys()
    testcapped_history_does_not_mutate_the_original_list_or_dicts()
    test_ask_caps_history_to_last_n_turns_in_prompt_sent_to_model()
    print("All qa tests passed.")
