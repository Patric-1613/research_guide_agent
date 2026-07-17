"""Phase 8: Streamlit frontend. A thin HTTP client over the phase-7 FastAPI
backend — no research_agent internals are imported directly, matching the
brief's separation of "FastAPI (backend API)" from "Streamlit (frontend)".
Run both processes separately:

    uvicorn research_agent.api:app --reload
    streamlit run research_agent/app.py

Every call into the backend is gated behind an explicit button/chat-input
action, never made unconditionally at module level. Streamlit reruns this
whole script top-to-bottom on *every* widget interaction (typing, clicking
anything), so an eager, unguarded API call here would silently re-trigger
on each rerun — for the summarize/chat/export calls specifically, that
would mean re-running billed LLM calls just because the user, say, expanded
an unrelated paper's abstract. /library is the one exception: it's a cheap
SQLite read with no LLM cost, so refreshing it every rerun is fine and
keeps the sidebar current.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Research Paper Summarizer", page_icon="📚", layout="wide")

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# /search is the long pole: it runs the phase-4 agent through several
# tool-calling round trips (arXiv, Semantic Scholar, embeddings, rerank),
# each its own network call, so 120s was cutting it too close under normal
# latency, not just when something's actually stuck.
REQUEST_TIMEOUT = 240


def _api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)


def _request_error_detail(exc: requests.exceptions.RequestException) -> str:
    if isinstance(exc, requests.exceptions.Timeout):
        return (f"The backend at {_api_base_url()} didn't respond within {REQUEST_TIMEOUT}s. "
                f"It may still be working — check the uvicorn terminal for progress logs.")
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (f"Could not reach the API backend at {_api_base_url()}. "
                f"Is it running? (uvicorn research_agent.api:app --reload)")
    return f"Request to the API backend failed: {exc}"


def _api_post(path: str, json_body: dict) -> tuple[bool, dict]:
    try:
        resp = requests.post(f"{_api_base_url()}{path}", json=json_body, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return False, {"detail": _request_error_detail(exc)}
    if resp.status_code >= 400:
        return False, {"detail": _error_detail(resp)}
    return True, resp.json()


def _api_get(path: str) -> tuple[bool, dict | list]:
    try:
        resp = requests.get(f"{_api_base_url()}{path}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return False, {"detail": _request_error_detail(exc)}
    if resp.status_code >= 400:
        return False, {"detail": _error_detail(resp)}
    return True, resp.json()


def _api_get_text(path: str) -> tuple[bool, str]:
    try:
        resp = requests.get(f"{_api_base_url()}{path}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return False, _request_error_detail(exc)
    if resp.status_code >= 400:
        return False, _error_detail(resp)
    return True, resp.text


def _error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


def _set_search_result(search_id: int, topic: str, papers: list[dict], web_articles: list[dict] | None = None) -> None:
    st.session_state.search_id = search_id
    st.session_state.topic = topic
    st.session_state.papers = papers
    st.session_state.web_articles = web_articles or []
    st.session_state.summary = None
    st.session_state.chat_history = []
    st.session_state.export_md = None


for key, default in [
    ("api_base_url", DEFAULT_API_BASE_URL),
    ("search_id", None),
    ("topic", ""),
    ("papers", []),
    ("web_articles", []),
    ("summary", None),
    ("chat_history", []),
    ("export_md", None),
]:
    st.session_state.setdefault(key, default)


# ---- sidebar: backend config + saved searches ---------------------------------

with st.sidebar:
    st.header("Settings")
    st.session_state.api_base_url = st.text_input("API base URL", value=st.session_state.api_base_url)

    st.divider()
    st.header("📖 Saved Searches")
    ok, library = _api_get("/library")
    if not ok:
        st.caption(library["detail"])
    elif not library:
        st.caption("No saved searches yet — run one to see it here.")
    else:
        for item in library:
            summary_tag = " ✓" if item["has_summary"] else ""
            web_tag = f" · 🌐{item['web_article_count']}" if item.get("web_article_count") else ""
            label = f"{item['topic']} · {item['paper_count']} papers{web_tag}{summary_tag}"
            if st.button(label, key=f"lib_{item['search_id']}", use_container_width=True):
                ok2, detail = _api_get(f"/library/{item['search_id']}")
                if not ok2:
                    st.error(detail["detail"])
                else:
                    _set_search_result(detail["search_id"], detail["topic"], detail["papers"], detail.get("web_articles"))
                    if item["has_summary"]:
                        # Free: the backend reuses the already-persisted
                        # summary for this search_id rather than re-billing.
                        ok3, summary = _api_post("/summarize", {"search_id": item["search_id"]})
                        if ok3:
                            st.session_state.summary = summary
                    st.rerun()


# ---- main: topic input ---------------------------------------------------------

st.title("📚 Research Paper Summarizer")
st.caption("Searches arXiv + Semantic Scholar, ranks by relevance, and grounds every summary and answer in the retrieved abstracts.")

topic_input = st.text_input(
    "Research topic",
    placeholder="e.g. parameter-efficient fine-tuning for large language models",
)
top_k_input = st.number_input(
    "Number of papers to return",
    min_value=3, max_value=30, value=10, step=1,
    help="Exact number of ranked results the search will return (after dedup across sources).",
)

with st.expander("Filters"):
    doi_required_input = st.checkbox("Only show papers with a DOI")
    min_citations_input = st.number_input(
        "Minimum citation count", min_value=0, value=0, step=1,
        help="0 = no filter. Papers with an unknown citation count never pass a nonzero minimum.",
    )

web_max_results_input = st.number_input(
    "Web articles to include (current context)",
    min_value=1, max_value=10, value=4, step=1,
    help="A separate, independently-sized set of current web articles (news, tooling, docs) — "
         "never counted toward the paper total above. The agent decides per-topic whether web "
         "context is actually relevant; this only caps how many it pulls in if it does.",
)

use_query_expansion_input = st.checkbox(
    "Query expansion (experimental)",
    help="Widens the search net with a few LLM-suggested well-known paper titles before "
         "reranking, to fix cases where a generic topic phrase misses foundational papers "
         "that a literal keyword search alone can't find. Bypasses the agent for this search "
         "(no query reformulation, no web articles) — for direct comparison against the "
         "default search above, not yet the default itself.",
)

if st.button("Search", type="primary") and topic_input.strip():
    with st.spinner("Searching arXiv + Semantic Scholar and ranking results — this can take up to a minute..."):
        ok, data = _api_post("/search", {
            "topic": topic_input.strip(),
            "top_k": int(top_k_input),
            "doi_required": doi_required_input,
            "min_citation_count": int(min_citations_input),
            "web_max_results": int(web_max_results_input),
            "use_query_expansion": use_query_expansion_input,
        })
    if ok:
        _set_search_result(data["search_id"], data["topic"], data["papers"], data.get("web_articles"))
        st.rerun()
    else:
        st.error(data["detail"])


# ---- results list ---------------------------------------------------------------

if st.session_state.papers:
    st.divider()
    st.subheader(f"Results for: {st.session_state.topic}")

    for i, p in enumerate(st.session_state.papers, 1):
        score = p.get("score")
        score_label = f"{score:.3f}" if score is not None else "n/a"
        with st.expander(f"{i}. {p['title']}  —  relevance {score_label}"):
            if score is not None:
                st.progress(min(max(score, 0.0), 1.0))
            st.markdown(f"**Authors:** {', '.join(p['authors']) or 'Unknown'}")
            citations = p["citation_count"] if p["citation_count"] is not None else "n/a"
            st.markdown(f"**Year:** {p['year'] or 'n/a'}　|　**Venue:** {p['venue'] or 'n/a'}　|　**Citations:** {citations}")
            for src, url in (p.get("source_urls") or {}).items():
                st.markdown(f"- [{src}]({url})")
            st.write(p["abstract"] or "_No abstract available._")

    # ---- web context ----------------------------------------------------------------
    # Deliberately a separate section, not interleaved with the paper list above and
    # never counted toward the paper total — a genuinely different corpus (news,
    # tooling, docs), not peer-reviewed literature. Absent entirely if the agent judged
    # this topic didn't call for it, or if no web search provider is configured — the
    # rest of the page works identically either way.
    if st.session_state.web_articles:
        st.divider()
        st.subheader("🌐 Current Web Context")
        st.caption("Supplementary web results (news, tooling, docs) — not peer-reviewed, shown separately from the papers above.")
        for a in st.session_state.web_articles:
            with st.expander(f"{a['title']}  —  {a['source_domain']}"):
                if a.get("published_date"):
                    st.caption(f"Published: {a['published_date']}")
                st.write(a["snippet"] or "_No snippet available._")
                st.markdown(f"[{a['url']}]({a['url']})")

    # ---- summary ---------------------------------------------------------------

    st.divider()
    st.subheader("Literature Summary")

    citation_style = st.selectbox(
        "Citation style",
        options=["apa", "harvard", "bibtex"],
        format_func=lambda s: {"apa": "APA", "harvard": "Harvard", "bibtex": "BibTeX"}[s],
        key="citation_style",
        help="Applies to the citation shown under each paper below, and to the Export References section.",
    )

    if st.button("Generate Summary"):
        with st.spinner("Clustering papers into themes and writing grounded summaries..."):
            ok, data = _api_post("/summarize", {"search_id": st.session_state.search_id, "style": citation_style})
        if ok:
            st.session_state.summary = data
        else:
            st.error(data["detail"])

    if st.session_state.summary:
        summary = st.session_state.summary
        for theme in summary["themes"]:
            st.markdown(f"#### {theme['theme_name']}")
            for entry in theme["papers"]:
                st.markdown(f"**{entry['title']}**")
                st.write(entry["summary"])
                st.caption(entry.get("citation", entry["apa_citation"]))
        st.markdown("#### Gaps & Disagreements")
        st.write(summary["gaps_and_disagreements"])
        if summary.get("skipped_paper_ids"):
            st.caption(f"{len(summary['skipped_paper_ids'])} retrieved paper(s) weren't referenced in the summary above.")

        # Its own block below the paper-themes summary, never merged into it —
        # None when this search found no web articles to summarize.
        if summary.get("web_summary"):
            st.markdown("#### 🌐 Web Context Summary")
            st.write(summary["web_summary"]["synthesis"])
            for a in summary["web_summary"]["cited_articles"]:
                st.caption(f"[{a['title']}]({a['url']}) — {a['source_domain']}")

    # ---- chat --------------------------------------------------------------------

    st.divider()
    st.subheader("Ask a follow-up question")

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    question = st.chat_input("Ask about these papers...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Thinking..."):
            ok, data = _api_post("/chat", {
                "search_id": st.session_state.search_id,
                "question": question,
                "history": st.session_state.chat_history,
            })
        if ok:
            st.session_state.chat_history = data["history"]
            with st.chat_message("assistant"):
                st.write(data["answer"])
                # Kept as two separate captions (not one merged list) so a
                # peer-reviewed paper citation and a web source stay visually
                # distinguishable, matching the answer text's [Paper N]/[Web N]
                # marker distinction all the way to the chat UI.
                if data["cited_papers"]:
                    st.caption("📄 Papers: " + ", ".join(p["title"] for p in data["cited_papers"]))
                if data.get("cited_web_articles"):
                    st.caption("🌐 Web: " + ", ".join(a["title"] for a in data["cited_web_articles"]))
        else:
            st.error(data["detail"])

    # ---- export --------------------------------------------------------------------

    st.divider()
    st.subheader("Export")

    if st.button("Prepare Markdown Export"):
        with st.spinner("Preparing export..."):
            ok, md_text = _api_get_text(f"/export/{st.session_state.search_id}?style={citation_style}")
        if ok:
            st.session_state.export_md = md_text
        else:
            st.error(md_text)

    if st.session_state.export_md:
        st.download_button(
            "Download summary as Markdown",
            data=st.session_state.export_md,
            file_name=f"summary_{st.session_state.search_id}.md",
            mime="text/markdown",
        )
        with st.expander("Preview Markdown"):
            st.code(st.session_state.export_md, language="markdown")
else:
    st.info("Search a topic above to get started.")
