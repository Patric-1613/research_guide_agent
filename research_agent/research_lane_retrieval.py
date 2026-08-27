"""Research Lanes (RL3/RL4): multi-lane retrieval, cross-lane
deduplication + discovery provenance, ONE global embedding, per-lane
semantic ranking, and deterministic round-robin interleaving into a
single reserve.

PURE DOMAIN LAYER. No API, service, checkpoint, telemetry-action,
usage-guard, cache, or frontend wiring.

- ``retrieve_across_lanes`` (RL3): the first-search path -- never mutates
  a ``PaperPoolSession`` or the input ``ResearchLane`` list. RL4 calls it
  once from the curation-start service.
- ``refill_lane_session`` (RL4): the refill path -- the multi-lane
  counterpart of ``query_expansion.refill_pool``. Mutates the given
  session (reserve / cursor / cumulative paper_lane_ids /
  lane_result_counts) in place and returns the genuinely-new count, the
  same contract ``refill_pool`` has. Called from
  ``curation_loop._refill_node`` for a lane session, INSIDE that node's
  existing ``curation_refill`` paid-action guard -- no new action type,
  no nested guard.

Both share the same three internal steps (retrieve+tag -> cross-lane
dedup+provenance+keyword-repair -> embed-once + one semantic pass per
enabled lane + round-robin), so refill does NOT rank fresh results and
then rank a second time.

Pipeline -- one pass, lanes run SEQUENTIALLY in v1 (no whole-lane
parallelism; the existing per-query arXiv/S2 and title-search concurrency
inside ``build_candidate_pool`` is untouched). No latency claim is made.

  validate: EVERY list item must be a ResearchLane (checked before any
  field -- including ``.enabled`` -- is read on any of them; a malformed
  entry raises a controlled TypeError regardless of enabled/disabled),
  then: >= 1 enabled lane, <= MAX_LANES_PER_REVIEW enabled, each enabled
  lane through the RL1 construction contract, unique enabled lane_ids
   -> enabled lanes (frozen order; disabled lanes ignored entirely)
   -> per lane: ONE build_candidate_pool(lane.query, k_for_widening, ...)
      call -- reuses the existing arXiv / Semantic Scholar / OpenAlex +
      LLM-title widening, per-lane dedup, and YAKE-v2 keyword computation.
      NO extra lane-generation or query-expansion call is added.
   -> record, per returned Paper OBJECT, the enabled lane_ids that
      returned it -- accumulated + de-duplicated, so the SAME instance
      handed back for two lanes keeps both
   -> ONE cross-lane dedup.deduplicate_with_clusters() over all tagged
      papers (shares production dedup's single _same_paper path -- no
      second identity approximation; the sole paper-identity authority)
   -> provenance: paper_lane_ids[<final merged paper_id>] = every enabled
      lane that discovered ANY member of the dedup cluster, de-duplicated
      and ordered by enabled-lane order
   -> keyword repair: _merge_cluster() builds a fresh Paper (keywords=[])
      for any >1-member cluster; recompute deterministic YAKE-v2 keywords
      exactly ONCE for exactly those clusters (a singleton -- including a
      legitimate keywords=[] -- is left untouched; production YAKE-v2 is
      not changed)
   -> embed_and_index_papers(global_pool) ONCE
   -> per enabled lane: semantic_search(anchored ranking query,
      where=paper_id in that lane's own merged ids, top_k=full subset) --
      never re-embeds the global pool
   -> round-robin interleave the lane rankings into one reserve: each
      paper once, with the (Paper, score) from the FIRST lane to emit it;
      the reserve is NOT re-sorted (lane scores are not comparable).

Unexpected retrieval/ranking failures PROPAGATE -- a failed lane is never
silently dropped and there is no fall-back to single-query mode here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from research_agent.dedup import deduplicate_with_clusters
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.keywords import extract_keywords
from research_agent.query_expansion import BATCH_SIZE, _EMPTY_EMBED_STATS, PaperPoolSession, build_candidate_pool
from research_agent.research_lanes import MAX_LANES_PER_REVIEW, ResearchLane, validate_lane_for_construction
from research_agent.schema import Paper


@dataclass
class MultiLaneRetrievalResult:
    """The complete output of one multi-lane retrieval operation.

    - ``ranked``: the complete globally-deduplicated, interleaved reserve
      -- ``list[(Paper, score)]``, NOT truncated, NOT re-sorted after
      interleaving. ``score`` is from the lane through which the paper
      first entered the reserve.
    - ``paper_lane_ids``: FRESH discovery provenance from THIS operation
      only (RL4 merges it into cumulative session provenance). Keyed by
      the final merged ``paper_id``; each value is the discovering
      lane_ids, de-duplicated and in enabled-lane order.
    - ``lane_result_counts``: for each enabled lane, how many final
      globally-deduplicated papers name that lane in their provenance --
      a paper shared by N lanes counts for all N. Zero for a lane that
      found nothing.
    - ``embed_stats``: the single global embed_and_index_papers() call's
      stats dict (or the existing zero shape when the pool is empty).
    """

    ranked: list[tuple[Paper, float]] = field(default_factory=list)
    paper_lane_ids: dict[str, list[str]] = field(default_factory=dict)
    lane_result_counts: dict[str, int] = field(default_factory=dict)
    embed_stats: dict = field(default_factory=lambda: dict(_EMPTY_EMBED_STATS))


def build_lane_ranking_query(topic: str, lane_query: str, refinement_notes: list[str] | None = None) -> str:
    """The deterministic text a lane's semantic ranking is anchored to.

    Frozen format::

        <original topic>
        Research facet: <lane query>

    A lane's DIRECT retrieval query is ``lane_query`` alone; its RANKING
    is anchored to BOTH the original review topic and the facet, so a
    productive lane cannot drift into an adjacent field. Refinement notes,
    when supplied, are appended exactly as ``query_expansion.refill_pool``
    already does for the single-query path (``". "`` + space-joined).
    Never anchored to an LLM-suggested paper title.
    """
    text = f"{topic}\nResearch facet: {lane_query}"
    if refinement_notes:
        text = f"{text}. {' '.join(refinement_notes)}"
    return text


def _round_robin(lane_rankings: list[list[tuple[Paper, float]]]) -> list[tuple[Paper, float]]:
    """Deterministic round-robin merge. Iterate lanes in the given
    (enabled) order; each round take every lane's next not-yet-emitted
    paper, skipping any already emitted via another lane; stop once a full
    round emits nothing (every ranking exhausted). Each paper appears
    exactly once, carrying the ``(Paper, score)`` from the FIRST lane to
    reach it. The result is returned as built -- never re-sorted.
    """
    cursors = [0] * len(lane_rankings)
    emitted: set[str] = set()
    reserve: list[tuple[Paper, float]] = []
    progressed = True
    while progressed:
        progressed = False
        for i, ranking in enumerate(lane_rankings):
            while cursors[i] < len(ranking) and ranking[cursors[i]][0].paper_id in emitted:
                cursors[i] += 1
            if cursors[i] < len(ranking):
                paper, score = ranking[cursors[i]]
                cursors[i] += 1
                emitted.add(paper.paper_id)
                reserve.append((paper, score))
                progressed = True
    return reserve


# --- shared internal steps (used by both retrieve_across_lanes and
#     refill_lane_session, so refill never double-ranks) --------------------


def _validate_enabled_lanes(lanes: list[ResearchLane]) -> list[ResearchLane]:
    """EVERY list item is checked to be a ResearchLane FIRST -- before
    ``.enabled`` (or any field) is read on any of them -- so a malformed
    entry raises a controlled TypeError regardless of whether it "would
    have been" enabled or disabled. Then the enabled subset is validated:
    >= 1 enabled, <= MAX_LANES_PER_REVIEW enabled, each through the RL1
    construction contract, unique enabled lane_ids. Returns the enabled
    lanes in input order."""
    for lane in lanes:
        if not isinstance(lane, ResearchLane):
            raise TypeError(f"every research lane must be a ResearchLane instance, got {type(lane).__name__}")
    enabled = [lane for lane in lanes if lane.enabled]
    if not enabled:
        raise ValueError("multi-lane retrieval requires at least one enabled research lane")
    if len(enabled) > MAX_LANES_PER_REVIEW:
        raise ValueError(
            f"multi-lane retrieval accepts at most {MAX_LANES_PER_REVIEW} enabled research lanes, got {len(enabled)}"
        )
    for lane in enabled:
        validate_lane_for_construction(lane)
    enabled_lane_ids = [lane.lane_id for lane in enabled]
    if len(set(enabled_lane_ids)) != len(enabled_lane_ids):
        raise ValueError("enabled research lanes must have unique lane_ids")
    return enabled


def _retrieve_and_tag(
    enabled: list[ResearchLane], *, k_for_widening: int, s2_api_key: str | None, client: OpenAI | None,
    use_openalex_fallback: bool, openalex_mailto: str | None,
    exclude_titles: list[str] | None, refinement_notes: list[str] | None,
) -> list[tuple[Paper, str]]:
    """ONE build_candidate_pool call per enabled lane, in frozen order --
    lane.query is the direct retrieval query; k_for_widening is identical
    for every lane. Every returned paper is tagged with its discovering
    lane_id. No extra lane-generation / query-expansion call."""
    tagged: list[tuple[Paper, str]] = []
    for lane in enabled:
        for paper in build_candidate_pool(
            lane.query, k_for_widening, s2_api_key=s2_api_key, client=client,
            use_openalex_fallback=use_openalex_fallback, openalex_mailto=openalex_mailto,
            exclude_titles=exclude_titles, refinement_notes=refinement_notes,
        ):
            tagged.append((paper, lane.lane_id))
    return tagged


def _cross_lane_dedup(
    tagged: list[tuple[Paper, str]], enabled_lane_ids: list[str],
) -> tuple[list[Paper], dict[str, list[str]]]:
    """ONE cross-lane dedup.deduplicate_with_clusters() (the sole
    paper-identity authority -- no second _same_paper approximation).
    Returns (merged pool, provenance keyed by final merged paper_id).
    Provenance = every enabled lane that discovered ANY cluster member,
    de-duplicated and in enabled-lane order. Object identity is only the
    *discovery* signal, accumulated per instance so the SAME object handed
    back for two lanes keeps both. _merge_cluster() drops keywords for a
    >1-member cluster; deterministic YAKE-v2 is recomputed exactly once
    for exactly those clusters (a singleton -- keywords=[] or not -- is
    left untouched)."""
    lane_ids_by_obj: dict[int, list[str]] = {}
    for paper, lane_id in tagged:
        lst = lane_ids_by_obj.setdefault(id(paper), [])
        if lane_id not in lst:
            lst.append(lane_id)

    pool: list[Paper] = []
    provenance: dict[str, list[str]] = {}
    for merged, members in deduplicate_with_clusters([p for p, _ in tagged]):
        member_lane_ids: set[str] = set()
        for m in members:
            member_lane_ids.update(lane_ids_by_obj.get(id(m), ()))
        provenance[merged.paper_id] = [lid for lid in enabled_lane_ids if lid in member_lane_ids]
        if len(members) > 1:
            merged.keywords = extract_keywords(merged.title, merged.abstract)
        pool.append(merged)
    return pool, provenance


def _embed_once_and_rank_lanes(
    topic: str, enabled: list[ResearchLane], pool: list[Paper], pool_provenance: dict[str, list[str]],
    *, collection, client: OpenAI | None, refinement_notes: list[str] | None,
) -> tuple[list[tuple[Paper, float]], dict]:
    """Embed/index ``pool`` ONCE, then ONE semantic_search pass per
    enabled lane over that lane's own subset of ``pool`` (per
    ``pool_provenance``), then deterministic round-robin interleave.
    ``pool`` must be non-empty. Never re-embeds per lane; never fully
    ranks then ranks again."""
    resolved_collection = collection if collection is not None else get_chroma_collection()
    embed_stats = embed_and_index_papers(pool, collection=resolved_collection, client=client)

    lane_rankings: list[list[tuple[Paper, float]]] = []
    for lane in enabled:
        lane_ids = [pid for pid, lids in pool_provenance.items() if lane.lane_id in lids]
        if not lane_ids:
            lane_rankings.append([])
            continue
        lane_rankings.append(
            semantic_search(
                build_lane_ranking_query(topic, lane.query, refinement_notes),
                collection=resolved_collection, client=client,
                top_k=len(lane_ids),  # complete ranking, no early truncation
                where={"paper_id": {"$in": lane_ids}},
            )
        )
    return _round_robin(lane_rankings), embed_stats


def _recount_lanes(all_lane_ids: list[str], provenance: dict[str, list[str]]) -> dict[str, int]:
    """lane_result_counts recomputed from scratch off cumulative
    provenance -- a paper shared by N lanes counts for all N; a lane with
    no papers (incl. a disabled, never-searched lane) is 0. Recomputing
    (never incrementing) is what makes duplicate/refill discoveries unable
    to inflate a count."""
    return {lid: sum(1 for lids in provenance.values() if lid in lids) for lid in all_lane_ids}


# --- RL3: the first-search path ----------------------------------------


def retrieve_across_lanes(
    topic: str,
    lanes: list[ResearchLane],
    *,
    k_for_widening: int = BATCH_SIZE,
    s2_api_key: str | None = None,
    client: OpenAI | None = None,
    collection=None,
    use_openalex_fallback: bool = False,
    openalex_mailto: str | None = None,
    exclude_titles: list[str] | None = None,
    refinement_notes: list[str] | None = None,
) -> MultiLaneRetrievalResult:
    """Retrieve, dedup, rank, and interleave across the enabled lanes of
    ``lanes`` (a frozen list -- not mutated here). See the module
    docstring for the pipeline.
    """
    enabled = _validate_enabled_lanes(lanes)
    enabled_lane_ids = [lane.lane_id for lane in enabled]

    tagged = _retrieve_and_tag(
        enabled, k_for_widening=k_for_widening, s2_api_key=s2_api_key, client=client,
        use_openalex_fallback=use_openalex_fallback, openalex_mailto=openalex_mailto,
        exclude_titles=exclude_titles, refinement_notes=refinement_notes,
    )
    pool, paper_lane_ids = _cross_lane_dedup(tagged, enabled_lane_ids)
    lane_result_counts = _recount_lanes(enabled_lane_ids, paper_lane_ids)

    if not pool:
        return MultiLaneRetrievalResult(
            ranked=[], paper_lane_ids={}, lane_result_counts=lane_result_counts,
            embed_stats=dict(_EMPTY_EMBED_STATS),
        )

    ranked, embed_stats = _embed_once_and_rank_lanes(
        topic, enabled, pool, paper_lane_ids, collection=collection, client=client,
        refinement_notes=refinement_notes,
    )
    return MultiLaneRetrievalResult(
        ranked=ranked, paper_lane_ids=paper_lane_ids,
        lane_result_counts=lane_result_counts, embed_stats=embed_stats,
    )


# --- RL4: the refill path (multi-lane counterpart of refill_pool) ------


def refill_lane_session(
    session: PaperPoolSession,
    *,
    k_for_widening: int = BATCH_SIZE,
    s2_api_key: str | None = None,
    client: OpenAI | None = None,
    collection=None,
    use_openalex_fallback: bool = False,
    openalex_mailto: str | None = None,
) -> int:
    """Multi-lane refill -- the counterpart of
    ``query_expansion.refill_pool`` for a session with frozen lanes.
    MUTATES ``session`` in place and returns the genuinely-new count (0 is
    a valid "exhausted" signal, same contract as ``refill_pool``).

    Feature-flag-independent: the caller (``curation_loop._refill_node``)
    dispatches here purely on ``session.lanes`` being non-empty, so an
    existing lane session keeps refilling even after
    RESEARCH_LANES_ENABLED is turned off.

    - re-searches EVERY enabled persisted lane (disabled lanes are never
      searched), excluding already-seen titles and folding in refinement
      notes -- exactly as ``refill_pool`` does for the single query;
    - preserves the unserved reserve tail (never re-searched, carried
      forward) -- and guarantees it is carried forward even if a tail
      paper has no surviving lane provenance to be ranked under;
    - deduplicates the fresh results AGAINST the carry-forward tail in ONE
      pass (``deduplicate_with_clusters`` is the sole identity authority):
      a fresh record that is the SAME real paper as a tail paper -- matched
      by DOI or fuzzy title even when its paper_id differs -- lands in that
      tail paper's cluster, so the tail paper survives unchanged and the
      fresh lane_ids flow into ITS cumulative provenance; it never becomes
      a second reserve entry;
    - merges the genuinely-new results with the tail into ONE combined
      pool, embeds it once, runs ONE semantic pass per enabled lane, and
      round-robin interleaves -- it does NOT fully rank fresh results and
      then rank a second time;
    - UNIONs this refill's fresh discovery provenance into
      ``session.paper_lane_ids`` (never removes an existing lane_id) and
      RECOMPUTES ``session.lane_result_counts`` from the cumulative
      provenance, so a re-discovery cannot inflate either.
    """
    enabled = _validate_enabled_lanes(session.lanes)
    enabled_lane_ids = [lane.lane_id for lane in enabled]
    all_lane_ids = [lane.lane_id for lane in session.lanes]

    unserved_tail = session.reserve[session.cursor:]
    tail_papers = [p for p, _ in unserved_tail]
    tail_id_set = {p.paper_id for p in tail_papers}

    fresh_tagged = _retrieve_and_tag(
        enabled, k_for_widening=k_for_widening, s2_api_key=s2_api_key, client=client,
        use_openalex_fallback=use_openalex_fallback, openalex_mailto=openalex_mailto,
        exclude_titles=list(session.seen_titles) or None,
        refinement_notes=list(session.refinement_notes) or None,
    )
    # object identity -> the enabled lane_ids that returned THIS fresh
    # instance (accumulated + de-duped) -- the same discovery signal
    # _cross_lane_dedup uses; tail papers carry no fresh tag.
    fresh_lane_by_obj: dict[int, list[str]] = {}
    for paper, lane_id in fresh_tagged:
        lst = fresh_lane_by_obj.setdefault(id(paper), [])
        if lane_id not in lst:
            lst.append(lane_id)
    fresh_papers = [p for p, _ in fresh_tagged]

    # ONE dedup pass over the carry-forward tail AND every fresh result.
    genuinely_new: list[Paper] = []
    for merged, members in deduplicate_with_clusters(tail_papers + fresh_papers):
        fresh_lanes: set[str] = set()
        for m in members:
            fresh_lanes.update(fresh_lane_by_obj.get(id(m), ()))
        cluster_tail_ids = [m.paper_id for m in members if m.paper_id in tail_id_set]

        if cluster_tail_ids:
            # a fresh record merged into a tail paper (or the tail paper
            # alone) -> the tail paper survives; only genuinely-new fresh
            # lane_ids are added to its provenance, in enabled-lane order.
            if fresh_lanes:
                for tp_id in cluster_tail_ids:
                    combined = set(session.paper_lane_ids.get(tp_id, [])) | fresh_lanes
                    session.paper_lane_ids[tp_id] = [lid for lid in enabled_lane_ids if lid in combined]
            continue

        # fresh-only cluster: enrich provenance either way (cumulative,
        # never removes), then decide if it is genuinely new.
        existing = set(session.paper_lane_ids.get(merged.paper_id, []))
        session.paper_lane_ids[merged.paper_id] = [lid for lid in enabled_lane_ids if lid in (existing | fresh_lanes)]
        if merged.paper_id in session.seen_paper_ids:
            continue  # already served this session -- provenance enriched, not re-served
        if len(members) > 1:
            merged.keywords = extract_keywords(merged.title, merged.abstract)
        genuinely_new.append(merged)

    combined_pool = tail_papers + genuinely_new
    if combined_pool:
        combined_provenance = {
            p.paper_id: list(session.paper_lane_ids.get(p.paper_id, [])) for p in combined_pool
        }
        ranked, _embed_stats = _embed_once_and_rank_lanes(
            session.topic, enabled, combined_pool, combined_provenance,
            collection=collection, client=client,
            refinement_notes=list(session.refinement_notes) or None,
        )
        # Defensive tail preservation: a tail paper with no (surviving)
        # lane provenance -- or one Chroma could not return -- is appended
        # (its prior reserve score) rather than dropped. In the normal
        # case every reserve paper has provenance and this adds nothing.
        ranked_ids = {p.paper_id for p, _ in ranked}
        tail_score = {p.paper_id: s for p, s in unserved_tail}
        ranked = ranked + [
            (p, tail_score.get(p.paper_id, 0.0)) for p in tail_papers if p.paper_id not in ranked_ids
        ]
        session.reserve = ranked
    else:
        session.reserve = []
    session.cursor = 0

    session.lane_result_counts = _recount_lanes(all_lane_ids, session.paper_lane_ids)
    return len(genuinely_new)
