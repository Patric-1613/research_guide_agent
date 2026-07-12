"""Round 3, phase 1: tests for the interactive multi-round triage session
data model. Deterministic — no network/LLM calls, mirrors tests/test_dedup.py's
approach of hand-built Paper/WebArticle fixtures.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.schema import Paper, WebArticle
from research_agent.session import (
    TriageSession,
    add_round,
    paper_state,
    web_article_state,
)


def _paper(paper_id: str, title: str, doi: str | None = None) -> Paper:
    return Paper(
        title=title,
        authors=["A. Author"],
        year=2023,
        venue="arXiv preprint",
        abstract=f"Abstract for {title}.",
        url=f"http://example.com/{paper_id}",
        doi=doi,
        citation_count=None,
        source="arxiv",
        paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(
        title=title, url=url, snippet="A snippet.", published_date=None, source_domain="example.com",
    )


def test_first_round_marks_everything_new():
    session = TriageSession(topic="PEFT")
    round1 = add_round(session, ["PEFT"], [_paper("a1", "Paper A"), _paper("a2", "Paper B")])

    assert round1.round_number == 1
    assert set(round1.paper_ids_found) == {"a1", "a2"}
    assert set(round1.new_paper_ids) == {"a1", "a2"}
    assert len(session.all_papers) == 2


def test_resurfacing_paper_under_new_keyword_merges_to_one_record_and_is_seen_not_new():
    session = TriageSession(topic="parameter-efficient fine-tuning")
    round1 = add_round(session, ["PEFT"], [_paper("arxiv:123", "LoRA: Low-Rank Adaptation")])

    # Round 2 finds the same underlying paper via a different source/keyword —
    # same title (fuzzy match), different paper_id, exactly like a genuine
    # cross-source duplicate in tests/test_dedup.py.
    dup = _paper("s2:999", "LoRA: Low-Rank Adaptation")
    round2 = add_round(session, ["low-rank adaptation"], [dup, _paper("arxiv:456", "A New Paper")])

    # Exactly one merged record for the resurfaced paper.
    assert len(session.all_papers) == 2  # merged LoRA record + the genuinely new paper
    merged_id = next(pid for pid in session.all_papers if "arxiv:123" in pid.split("+"))
    assert "s2:999" in merged_id.split("+")

    # Tagged "already seen" in round 2, not "new".
    assert merged_id in round2.paper_ids_found
    assert merged_id not in round2.new_paper_ids
    assert paper_state(session, merged_id, round2) == "seen"

    # The genuinely new paper in round 2 is new.
    assert "arxiv:456" in round2.new_paper_ids

    # Round 1's own record is rewritten to the new canonical id too — so
    # scrolling back to round 1 still shows the same (merged) identity.
    assert round1.paper_ids_found == [merged_id]
    assert round1.new_paper_ids == [merged_id]


def test_basket_status_is_paper_property_not_round_property():
    session = TriageSession(topic="PEFT")
    round1 = add_round(session, ["PEFT"], [_paper("arxiv:123", "LoRA: Low-Rank Adaptation")])
    session.basket_paper_ids.add("arxiv:123")

    dup = _paper("s2:999", "LoRA: Low-Rank Adaptation")
    round2 = add_round(session, ["low-rank adaptation"], [dup])

    merged_id = next(iter(session.all_papers))
    # Basket membership followed the rename.
    assert session.basket_paper_ids == {merged_id}
    # Whichever round's group this paper is displayed under, it reads "basket".
    assert paper_state(session, merged_id, round1) == "basket"
    assert paper_state(session, merged_id, round2) == "basket"


def test_distinct_papers_across_rounds_are_not_merged():
    session = TriageSession(topic="topic")
    add_round(session, ["kw1"], [_paper("a1", "Paper About Cats")])
    round2 = add_round(session, ["kw2"], [_paper("a2", "A Completely Unrelated Paper About Dogs")])

    assert len(session.all_papers) == 2
    assert set(round2.new_paper_ids) == {"a2"}


def test_web_articles_track_new_seen_basket_independently_of_papers():
    session = TriageSession(topic="topic")
    round1 = add_round(
        session, ["kw1"], [], found_web_articles=[_web_article("http://a.com/1", "Article A")]
    )
    session.basket_web_urls.add("http://a.com/1")

    round2 = add_round(
        session,
        ["kw2"],
        [],
        found_web_articles=[_web_article("http://a.com/1", "Article A"), _web_article("http://b.com/2", "Article B")],
    )

    assert web_article_state(session, "http://a.com/1", round1) == "basket"
    assert web_article_state(session, "http://a.com/1", round2) == "basket"
    assert web_article_state(session, "http://b.com/2", round2) == "new"
    assert round2.web_urls_found == ["http://a.com/1", "http://b.com/2"]
    assert round2.new_web_urls == ["http://b.com/2"]
    assert len(session.all_web_articles) == 2


def test_to_dict_from_dict_roundtrip_preserves_state_across_renames():
    session = TriageSession(topic="PEFT")
    add_round(session, ["PEFT"], [_paper("arxiv:123", "LoRA: Low-Rank Adaptation")])
    session.basket_paper_ids.add("arxiv:123")

    blob = session.to_dict()
    restored = TriageSession.from_dict(blob)

    assert restored.topic == "PEFT"
    assert restored.basket_paper_ids == {"arxiv:123"}
    assert set(restored.all_papers) == {"arxiv:123"}
    assert restored.rounds[0].round_number == 1

    # Continuing to add rounds against the restored session still renames
    # ids correctly across the (deserialized) history.
    dup = _paper("s2:999", "LoRA: Low-Rank Adaptation")
    round2 = add_round(restored, ["low-rank adaptation"], [dup])
    merged_id = next(iter(restored.all_papers))
    assert restored.basket_paper_ids == {merged_id}
    assert restored.rounds[0].paper_ids_found == [merged_id]
    assert merged_id in round2.paper_ids_found


def test_third_round_end_to_end_scenario_matches_phase1_success_criteria():
    session = TriageSession(topic="parameter-efficient fine-tuning")
    r1 = add_round(session, ["PEFT"], [_paper("arxiv:1", "LoRA: Low-Rank Adaptation"), _paper("arxiv:2", "Prefix Tuning")])
    r2 = add_round(session, ["low-rank adaptation"], [_paper("s2:1", "LoRA: Low-Rank Adaptation"), _paper("arxiv:3", "Adapter Layers")])
    r3 = add_round(session, ["adapters"], [_paper("s2:3", "Adapter Layers")])

    assert len(session.rounds) == 3

    merged_lora_id = next(pid for pid in session.all_papers if "arxiv:1" in pid.split("+"))
    assert "s2:1" in merged_lora_id.split("+")
    assert merged_lora_id in r2.paper_ids_found
    assert merged_lora_id not in r2.new_paper_ids

    merged_adapter_id = next(pid for pid in session.all_papers if "arxiv:3" in pid.split("+"))
    assert "s2:3" in merged_adapter_id.split("+")
    assert merged_adapter_id in r3.paper_ids_found
    assert merged_adapter_id not in r3.new_paper_ids
    # r1 never touched the adapter paper, so it's untouched there.
    assert merged_adapter_id not in r1.paper_ids_found

    # Every round's ids still resolve against the live pool.
    for round_ in session.rounds:
        for pid in round_.paper_ids_found:
            assert pid in session.all_papers
