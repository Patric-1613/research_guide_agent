from __future__ import annotations

import os
import sqlite3

import research_agent.api as api
from research_agent.api_app.schemas import SearchRequest, SearchResponse
from research_agent.api_app.serializers import _paper_to_out, _web_article_to_out
from research_agent.schema import WebArticle
from research_agent.services.errors import ServiceError
from research_agent.services.search_helpers import _filtered_candidate_count, _merge_web_articles, _server_side_rerank
from research_agent.storage import save_search


def run_search(db: sqlite3.Connection, req: SearchRequest) -> SearchResponse:
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    if req.use_query_expansion:
        # LLM-assisted query expansion (query_expansion.py) — a separate,
        # non-agent code path for direct comparison against the agent's own
        # search/rerank. See SearchRequest.use_query_expansion: web-article
        # search is agent-only right now, so web_articles is empty here, a
        # known and deliberate gap for this phase, not an oversight.
        ranked = api.expanded_search(
            req.topic, k=req.top_k, s2_api_key=s2_key,
            doi_required=req.doi_required, min_citation_count=req.min_citation_count,
        )
        web_articles: list[WebArticle] = []
        if not ranked:
            raise ServiceError(404, "No papers found for this topic.")
    else:
        session = api.run_research_agent(
            req.topic, s2_api_key=s2_key, top_k=req.top_k,
            doi_required=req.doi_required, min_citation_count=req.min_citation_count,
            web_max_results=req.web_max_results,
        )

        ranked = session.ranked
        expected_count = min(req.top_k, _filtered_candidate_count(session.papers, req.doi_required, req.min_citation_count))
        if session.papers and len(ranked) != expected_count:
            # The agent is prompted to rerank with exactly top_k results before
            # finishing, and its rerank tool applies the doi/citation filters
            # itself — but that's still a prompted/tool-execution behavior, not
            # a guarantee (the model might skip reranking entirely). Re-rank
            # server-side whenever what came back doesn't match what the user
            # asked for, so the returned count and filters are always
            # code-enforced rather than dependent on the model's behavior.
            ranked = _server_side_rerank(session, req.topic, req.top_k, req.doi_required, req.min_citation_count)

        if not ranked:
            if session.papers and (req.doi_required or req.min_citation_count):
                raise ServiceError(
                    404,
                    f"Found {len(session.papers)} paper(s) for this topic, but none matched the "
                    f"active filters (DOI required: {req.doi_required}, "
                    f"min citations: {req.min_citation_count}). Try relaxing the filters.",
                )
            raise ServiceError(404, "No papers found for this topic.")

        if len(session.web_articles) < req.web_max_results:
            # Whether to call search_web_tool at all is the agent's judgment
            # call (agent.py's system prompt) — it may skip it entirely for a
            # topic it judges purely historical/theoretical, or just not call it
            # this run. But web_max_results is a user-set request parameter like
            # top_k, so the user gets that many results whenever they're
            # available, not only when the model happened to decide to look.
            # Same code-enforced-count guarantee as top_k's server-side rerank
            # fallback above.
            fallback_articles = api.search_web(req.topic, max_results=req.web_max_results)
            session.web_articles = _merge_web_articles(session.web_articles, fallback_articles)
        # The agent may have accumulated more than web_max_results across
        # multiple search_web_tool calls (deduped by URL, not by count) —
        # truncate here so the returned count is never silently uncontrolled,
        # same reasoning as top_k in enhancement 1.
        web_articles = session.web_articles[: req.web_max_results]

    paper_ids = [p.paper_id for p, _ in ranked]
    scores = [score for _, score in ranked]
    search_id, created_at = save_search(
        db, req.topic, paper_ids, scores,
        web_articles=[a.to_dict() for a in web_articles],
    )

    return SearchResponse(
        search_id=search_id, topic=req.topic, created_at=created_at,
        papers=[_paper_to_out(p, score) for p, score in ranked],
        web_articles=[_web_article_to_out(a) for a in web_articles],
    )
