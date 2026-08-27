"""Research Lanes (RL3): multi-lane retrieval, cross-lane deduplication +
discovery provenance, ONE global embedding, per-lane semantic ranking,
and deterministic round-robin interleaving into a single reserve.

PURE DOMAIN LAYER. No API, service, checkpoint, curation-loop,
telemetry-action, usage-guard, cache, or frontend wiring -- RL4 owns
integrating this into ``/curation/start`` and refill. ``retrieve_across_
lanes`` never mutates a ``PaperPoolSession`` or the input
``ResearchLane`` list.

Pipeline -- one pass, lanes run SEQUENTIALLY in v1 (no whole-lane
parallelism; the existing per-query arXiv/S2 and title-search concurrency
inside ``build_candidate_pool`` is untouched). No latency claim is made.

  enabled lanes (frozen order; disabled lanes ignored entirely)
   -> per lane: ONE build_candidate_pool(lane.query, k_for_widening, ...)
      call -- reuses the existing arXiv / Semantic Scholar / OpenAlex +
      LLM-title widening, per-lane dedup, and YAKE-v2 keyword computation.
      NO extra lane-generation or query-expansion call is added.
   -> tag every returned paper with its discovering lane_id
   -> ONE cross-lane dedup.deduplicate_with_clusters() over all tagged
      papers (shares production dedup's single _same_paper path -- no
      second identity approximation)
   -> provenance: paper_lane_ids[<final merged paper_id>] = every lane
      represented in the dedup cluster, de-duplicated and ordered by
      enabled-lane order
   -> keyword repair: _merge_cluster() builds a fresh Paper (no keywords)
      for any multi-member cluster; recompute deterministic YAKE-v2
      keywords for exactly those (production YAKE-v2 is not changed)
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
from research_agent.query_expansion import BATCH_SIZE, _EMPTY_EMBED_STATS, build_candidate_pool
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
    docstring for the pipeline. ``exclude_titles`` / ``refinement_notes``
    / ``use_openalex_fallback`` / ``openalex_mailto`` / ``s2_api_key`` /
    ``client`` / ``collection`` are forwarded to the existing retrieval /
    embedding primitives unchanged; RL3 does not wire refill.
    """
    # --- validation, BEFORE any provider / search / embedding / Chroma work ---
    enabled = [lane for lane in lanes if lane.enabled]
    if not enabled:
        raise ValueError("retrieve_across_lanes requires at least one enabled research lane")
    if len(enabled) > MAX_LANES_PER_REVIEW:
        raise ValueError(
            f"retrieve_across_lanes accepts at most {MAX_LANES_PER_REVIEW} enabled research lanes, got {len(enabled)}"
        )
    for lane in enabled:
        if not isinstance(lane, ResearchLane):
            raise TypeError("every enabled lane must be a ResearchLane")
        validate_lane_for_construction(lane)  # RL1 construction contract
    enabled_lane_ids = [lane.lane_id for lane in enabled]
    if len(set(enabled_lane_ids)) != len(enabled_lane_ids):
        raise ValueError("enabled research lanes must have unique lane_ids")

    # --- one build_candidate_pool call per enabled lane, in frozen order ---
    tagged: list[tuple[Paper, str]] = []
    for lane in enabled:
        lane_pool = build_candidate_pool(
            lane.query,
            k_for_widening,
            s2_api_key=s2_api_key,
            client=client,
            use_openalex_fallback=use_openalex_fallback,
            openalex_mailto=openalex_mailto,
            exclude_titles=exclude_titles,
            refinement_notes=refinement_notes,
        )
        for paper in lane_pool:
            tagged.append((paper, lane.lane_id))

    # --- one cross-lane dedup + provenance merge + keyword repair ---
    lane_by_obj: dict[int, str] = {id(paper): lane_id for paper, lane_id in tagged}
    clustered = deduplicate_with_clusters([paper for paper, _ in tagged])

    global_pool: list[Paper] = []
    paper_lane_ids: dict[str, list[str]] = {}
    for merged, members in clustered:
        member_lane_ids = {lane_by_obj[id(m)] for m in members if id(m) in lane_by_obj}
        paper_lane_ids[merged.paper_id] = [lid for lid in enabled_lane_ids if lid in member_lane_ids]
        if not merged.keywords:
            # _merge_cluster() builds a fresh Paper with no keywords for a
            # multi-member cluster -- recompute the same deterministic
            # YAKE-v2 keywords build_candidate_pool() computes once per
            # deduped paper. A single-member cluster's merged paper IS its
            # sole member (keywords intact) -> already truthy -> skipped.
            merged.keywords = extract_keywords(merged.title, merged.abstract)
        global_pool.append(merged)

    lane_result_counts = {
        lane_id: sum(1 for lids in paper_lane_ids.values() if lane_id in lids)
        for lane_id in enabled_lane_ids
    }

    # --- empty: no embedding, no ranking ---
    if not global_pool:
        return MultiLaneRetrievalResult(
            ranked=[],
            paper_lane_ids={},
            lane_result_counts=lane_result_counts,  # all zero
            embed_stats=dict(_EMPTY_EMBED_STATS),
        )

    # --- ONE global embed / index ---
    resolved_collection = collection if collection is not None else get_chroma_collection()
    embed_stats = embed_and_index_papers(global_pool, collection=resolved_collection, client=client)

    # --- one semantic ranking per enabled lane, over ITS OWN candidate subset ---
    lane_rankings: list[list[tuple[Paper, float]]] = []
    for lane in enabled:
        lane_ids = [pid for pid, lids in paper_lane_ids.items() if lane.lane_id in lids]
        if not lane_ids:
            lane_rankings.append([])
            continue
        lane_rankings.append(
            semantic_search(
                build_lane_ranking_query(topic, lane.query, refinement_notes),
                collection=resolved_collection,
                client=client,
                top_k=len(lane_ids),  # complete ranking, no early truncation
                where={"paper_id": {"$in": lane_ids}},
            )
        )

    return MultiLaneRetrievalResult(
        ranked=_round_robin(lane_rankings),
        paper_lane_ids=paper_lane_ids,
        lane_result_counts=lane_result_counts,
        embed_stats=embed_stats,
    )
