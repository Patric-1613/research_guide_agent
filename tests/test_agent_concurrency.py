"""PR2.6B: deterministic concurrency regression tests for research_agent/
agent.py's session.papers/session.web_articles accumulation.

PR2.6A's barrier-controlled probe reproduced a lost update in 20/20
parallel runs of the OLD unprotected `session.papers = deduplicate(
session.papers + papers)` read/merge/write -- LangGraph's ToolNode runs
same-turn tool calls on a real thread pool, so search_arxiv_tool and
search_semantic_scholar_tool could genuinely interleave. PR2.6B fixes
this with a dedicated per-session lock (agent.py's
`_merge_papers_into_session`/`_merge_web_articles_into_session`, guarded
by ResearchSession._papers_lock/._web_articles_lock).

Every test here uses REAL threading.Thread + threading.Barrier/Event to
force deterministic interleavings -- never a bare sleep() race, which
would be flaky by construction. Every provider function is mocked; a
module-level autouse fixture also blocks real socket connections as a
hard backstop, matching tests/test_k5_heldout_llm_metrics.py's own
`socket.create_connection` guard convention.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch

from research_agent.agent import ResearchSession, build_tools
from research_agent.dedup import deduplicate as real_deduplicate
from research_agent.schema import Paper, WebArticle


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    monkeypatch.setattr(
        socket, "create_connection",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("real network call attempted")),
    )


def _paper(title: str, source: str, paper_id: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract="An abstract about " + title, url=f"http://example.com/{paper_id}",
        doi=None, citation_count=None, source=source, paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet="a snippet", published_date=None, source_domain="example.com")


def _run_threads(*targets: threading.Thread, join_timeout: float = 5.0) -> None:
    for t in targets:
        t.start()
    for t in targets:
        t.join(timeout=join_timeout)
        assert not t.is_alive(), "a thread did not finish -- likely deadlocked"


# --- 1 & 2: concurrent arXiv + Semantic Scholar retain both sources, ---
# --- cross-source dedup/merge unchanged. ---

def test_concurrent_arxiv_and_semantic_scholar_retain_both_sources():
    """Reproduces PR2.6A's exact scenario deterministically: both search
    functions block on a shared barrier so neither can finish before the
    other has started -- the old code lost one side's papers 20/20 times
    under this kind of overlap; the fix must retain both every time."""
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)
    release = threading.Barrier(2, timeout=5)

    def fake_arxiv(query, max_results=20):
        release.wait()
        return [_paper("arXiv Only Paper", "arxiv", "1111.1111")]

    def fake_s2(query, api_key=None):
        release.wait()
        return [_paper("Semantic Scholar Only Paper", "semantic_scholar", "2222.2222")]

    with patch("research_agent.agent.search_arxiv", side_effect=fake_arxiv), \
         patch("research_agent.agent.search_semantic_scholar", side_effect=fake_s2):
        t1 = threading.Thread(target=lambda: arxiv_tool.invoke({"query": "test"}))
        t2 = threading.Thread(target=lambda: s2_tool.invoke({"query": "test"}))
        _run_threads(t1, t2)

    titles = {p.title for p in session.papers}
    assert titles == {"arXiv Only Paper", "Semantic Scholar Only Paper"}
    assert len(session.papers) == 2


def test_concurrent_same_paper_from_both_sources_still_merges_not_duplicates():
    """Cross-source merge behavior (title-matched papers combine into one
    record with a joined source string) must be unchanged under
    concurrent execution, not just sequential (test_agent.py's existing
    test_search_tools_accumulate_and_dedup_into_session already proves
    the sequential case)."""
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)
    release = threading.Barrier(2, timeout=5)

    def fake_arxiv(query, max_results=20):
        release.wait()
        return [_paper("Same Paper", "arxiv", "1111.1111")]

    def fake_s2(query, api_key=None):
        release.wait()
        return [_paper("Same Paper", "semantic_scholar", "abc123")]

    with patch("research_agent.agent.search_arxiv", side_effect=fake_arxiv), \
         patch("research_agent.agent.search_semantic_scholar", side_effect=fake_s2):
        t1 = threading.Thread(target=lambda: arxiv_tool.invoke({"query": "test"}))
        t2 = threading.Thread(target=lambda: s2_tool.invoke({"query": "test"}))
        _run_threads(t1, t2)

    assert len(session.papers) == 1
    assert session.papers[0].source == "arxiv+semantic_scholar"


# --- 3: the merge critical section is genuinely serialized. ---

def test_papers_merge_critical_section_is_serialized_not_timing_luck():
    """Instruments deduplicate() itself (the operation inside the lock)
    to record how many threads are executing it AT THE SAME TIME. A
    barrier forces both tool calls to reach the merge step at essentially
    the same instant; a short sleep INSIDE the instrumented deduplicate()
    widens the window an unprotected race would need to actually be
    caught in. If _papers_lock did not serialize this section, max
    concurrent executions would be 2; the lock must keep it at 1 --
    a hard invariant, not a probabilistic one."""
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)

    enter_barrier = threading.Barrier(2, timeout=5)
    count_lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def tracking_deduplicate(papers):
        with count_lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.05)
        with count_lock:
            state["current"] -= 1
        return real_deduplicate(papers)

    def fake_arxiv(query, max_results=20):
        enter_barrier.wait()
        return [_paper("Paper A", "arxiv", "1111.1111")]

    def fake_s2(query, api_key=None):
        enter_barrier.wait()
        return [_paper("Paper B", "semantic_scholar", "2222.2222")]

    with patch("research_agent.agent.search_arxiv", side_effect=fake_arxiv), \
         patch("research_agent.agent.search_semantic_scholar", side_effect=fake_s2), \
         patch("research_agent.agent.deduplicate", side_effect=tracking_deduplicate):
        t1 = threading.Thread(target=lambda: arxiv_tool.invoke({"query": "test"}))
        t2 = threading.Thread(target=lambda: s2_tool.invoke({"query": "test"}))
        _run_threads(t1, t2)

    assert state["max"] == 1, "both threads executed the merge step concurrently -- the lock did not serialize it"
    assert len(session.papers) == 2  # and no data was lost either


# --- 4: concurrent web-result accumulation does not lose either result. ---

def test_concurrent_web_searches_retain_both_results():
    session = ResearchSession()
    _arxiv_tool, _s2_tool, _rerank_tool, web_tool = build_tools(session)
    release = threading.Barrier(2, timeout=5)

    def fake_search_web(query, max_results=4):
        release.wait()
        if query == "query-a":
            return [_web_article("https://x.com/a", "Article A")]
        return [_web_article("https://x.com/b", "Article B")]

    with patch("research_agent.agent.search_web", side_effect=fake_search_web):
        t1 = threading.Thread(target=lambda: web_tool.invoke({"query": "query-a", "max_results": 4}))
        t2 = threading.Thread(target=lambda: web_tool.invoke({"query": "query-b", "max_results": 4}))
        _run_threads(t1, t2)

    urls = {a.url for a in session.web_articles}
    assert urls == {"https://x.com/a", "https://x.com/b"}


# --- 5: provider/search calls still overlap -- the lock must not cover ---
# --- network waiting. ---

def test_provider_calls_still_overlap_lock_does_not_cover_network_wait():
    """Both fake search functions block on a 2-party barrier BEFORE
    returning. If _papers_lock incorrectly wrapped the network call
    itself (not just the merge), thread A would hold the lock while
    blocked inside fake_arxiv waiting on the barrier, and thread B could
    never even START fake_s2 (it would block trying to acquire the same
    lock first) -- the barrier could never release, and this test would
    hang/timeout. A bounded barrier.wait() timeout turns that failure
    mode into a clean, fast assertion instead of an actual test hang."""
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)
    both_entered = threading.Barrier(2, timeout=2)

    def fake_arxiv(query, max_results=20):
        both_entered.wait()  # raises BrokenBarrierError on timeout, not a hang
        return [_paper("arXiv Paper", "arxiv", "1111.1111")]

    def fake_s2(query, api_key=None):
        both_entered.wait()
        return [_paper("S2 Paper", "semantic_scholar", "2222.2222")]

    with patch("research_agent.agent.search_arxiv", side_effect=fake_arxiv), \
         patch("research_agent.agent.search_semantic_scholar", side_effect=fake_s2):
        t1 = threading.Thread(target=lambda: arxiv_tool.invoke({"query": "test"}))
        t2 = threading.Thread(target=lambda: s2_tool.invoke({"query": "test"}))
        # No exception here means both threads genuinely reached the
        # barrier concurrently -- proof the lock never serialized the
        # network calls themselves.
        _run_threads(t1, t2)

    assert len(session.papers) == 2


# --- 6: sequential behavior is unchanged. ---

def test_sequential_arxiv_then_semantic_scholar_still_accumulates_normally():
    """No threads at all here -- confirms the lock adds no observable
    behavior change to the plain, single-threaded call pattern
    tests/test_agent.py's existing tests already exercise."""
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)

    with patch("research_agent.agent.search_arxiv", return_value=[_paper("Paper One", "arxiv", "1111.1111")]):
        arxiv_tool.invoke({"query": "test"})
    with patch("research_agent.agent.search_semantic_scholar", return_value=[_paper("Paper Two", "semantic_scholar", "2222.2222")]):
        s2_tool.invoke({"query": "test"})

    titles = {p.title for p in session.papers}
    assert titles == {"Paper One", "Paper Two"}


# --- 7: a tool failure still leaves the accumulated pool usable, even ---
# --- when it fails concurrently with a sibling tool call. ---

def test_concurrent_failure_on_one_tool_leaves_the_other_sibling_result_usable():
    """One thread's search_arxiv raises while the other thread's
    search_semantic_scholar succeeds, at essentially the same instant --
    proves the lock is never left held/broken by the failing sibling
    (Python's `with` guarantees release on exception) and that a
    concurrent failure doesn't corrupt or discard the successful
    sibling's merge."""
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)
    release = threading.Barrier(2, timeout=5)

    def failing_arxiv(query, max_results=20):
        release.wait()
        raise RuntimeError("arXiv is down")

    def succeeding_s2(query, api_key=None):
        release.wait()
        return [_paper("Recovered Paper", "semantic_scholar", "xyz")]

    with patch("research_agent.agent.search_arxiv", side_effect=failing_arxiv), \
         patch("research_agent.agent.search_semantic_scholar", side_effect=succeeding_s2):
        results = {}

        def call_arxiv():
            results["arxiv"] = arxiv_tool.invoke({"query": "test"})

        def call_s2():
            results["s2"] = s2_tool.invoke({"query": "test"})

        t1 = threading.Thread(target=call_arxiv)
        t2 = threading.Thread(target=call_s2)
        _run_threads(t1, t2)

    assert "failed" in results["arxiv"].lower()
    assert len(session.papers) == 1
    assert session.papers[0].title == "Recovered Paper"

    # The pool is still usable afterward for a normal follow-up call too.
    with patch("research_agent.agent.search_arxiv", return_value=[_paper("Retry Paper", "arxiv", "9999.9999")]):
        arxiv_tool.invoke({"query": "retry"})
    titles = {p.title for p in session.papers}
    assert titles == {"Recovered Paper", "Retry Paper"}


# --- Lock ownership sanity checks: dedicated locks, not the ---
# --- suggested-title lock reused for convenience. ---

def test_papers_lock_and_web_articles_lock_are_distinct_from_each_other_and_from_suggested_titles_lock():
    session = ResearchSession()
    assert session._papers_lock is not session._web_articles_lock
    assert session._papers_lock is not session._suggested_titles_lock
    assert session._web_articles_lock is not session._suggested_titles_lock
    # Every ResearchSession gets its OWN locks -- not a class-level shared
    # lock across sessions/requests (which would be the "process-wide/
    # global lock" this fix explicitly must not add).
    other_session = ResearchSession()
    assert other_session._papers_lock is not session._papers_lock
    assert other_session._web_articles_lock is not session._web_articles_lock


if __name__ == "__main__":
    test_concurrent_arxiv_and_semantic_scholar_retain_both_sources()
    test_concurrent_same_paper_from_both_sources_still_merges_not_duplicates()
    test_papers_merge_critical_section_is_serialized_not_timing_luck()
    test_concurrent_web_searches_retain_both_results()
    test_provider_calls_still_overlap_lock_does_not_cover_network_wait()
    test_sequential_arxiv_then_semantic_scholar_still_accumulates_normally()
    test_concurrent_failure_on_one_tool_leaves_the_other_sibling_result_usable()
    test_papers_lock_and_web_articles_lock_are_distinct_from_each_other_and_from_suggested_titles_lock()
    print("All agent concurrency tests passed.")
