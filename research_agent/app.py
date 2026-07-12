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

Round 3, phase 2 adds a second tab: an interactive, multi-round triage
flow (search a keyword, browse the round's results alongside every prior
round's, pick papers into a basket, repeat) alongside the original
single-shot flow below, per the round-3 brief's constraint 4 — both flows
stay available; nothing from round 1/2 was removed. The triage tab's
session state (rounds, accumulated pool, basket) is plain JSON held in
st.session_state and round-tripped through the stateless /round_search
endpoint on every search action; basket add/remove is local set mutation
with no network call, since toggling a pick needs none of the
dedup/embedding logic that actually lives server-side.
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
    # Round-3 interactive triage tab state — a plain-JSON mirror of
    # TriageSession.to_dict() (research_agent/session.py), or None before
    # the first round of a session. Kept entirely separate from the
    # classic-flow keys above.
    ("triage_session", None),
    ("triage_latest_round", None),
    ("triage_summary", None),
    ("viewed_bag", None),
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

    st.divider()
    st.header("🧺 Saved Bags")
    ok, bags = _api_get("/bags")
    if not ok:
        st.caption(bags["detail"])
    elif not bags:
        st.caption("No saved bags yet — save a basket from the Interactive Triage tab to see it here.")
    else:
        # Phase-4 requirement: the flat, undifferentiated list problem the
        # brief calls out for /library above shouldn't repeat here — filter
        # by name/keyword, then group by year, so a long history stays scannable.
        name_filter = st.text_input("Filter by name", key="bag_name_filter", placeholder="e.g. PEFT")
        all_keywords = sorted({kw for b in bags for kw in b["keywords"]})
        keyword_filter = st.selectbox("Filter by keyword", options=["(all)"] + all_keywords, key="bag_keyword_filter")
        all_years = sorted({b["year"] for b in bags}, reverse=True)
        year_filter = st.selectbox("Filter by year", options=["(all)"] + [str(y) for y in all_years], key="bag_year_filter")

        filtered = [
            b for b in bags
            if (not name_filter.strip() or name_filter.strip().lower() in b["name"].lower())
            and (keyword_filter == "(all)" or keyword_filter in b["keywords"])
            and (year_filter == "(all)" or str(b["year"]) == year_filter)
        ]
        if not filtered:
            st.caption("No bags match these filters.")

        for year in sorted({b["year"] for b in filtered}, reverse=True):
            st.caption(f"**{year}**")
            for item in [b for b in filtered if b["year"] == year]:
                web_tag = f" · 🌐{item['web_article_count']}" if item["web_article_count"] else ""
                kw_tag = f" · {', '.join(item['keywords'][:2])}" if item["keywords"] else ""
                label = f"{item['name']} · {item['paper_count']} papers{web_tag}{kw_tag}"
                row_col, delete_col = st.columns([4, 1])
                with row_col:
                    if st.button(label, key=f"bag_{item['bag_id']}", use_container_width=True):
                        ok2, detail = _api_get(f"/bags/{item['bag_id']}")
                        if not ok2:
                            st.error(detail["detail"])
                        else:
                            st.session_state.viewed_bag = detail
                            st.rerun()
                with delete_col:
                    if st.button("🗑️", key=f"bag_delete_{item['bag_id']}"):
                        requests.delete(f"{_api_base_url()}/bags/{item['bag_id']}", timeout=REQUEST_TIMEOUT)
                        if st.session_state.viewed_bag and st.session_state.viewed_bag["bag_id"] == item["bag_id"]:
                            st.session_state.viewed_bag = None
                        st.rerun()


# ---- main: shared title, then classic vs. interactive-triage tabs --------------

st.title("📚 Research Paper Summarizer")
st.caption("Searches arXiv + Semantic Scholar, ranks by relevance, and grounds every summary and answer in the retrieved abstracts.")

tab_classic, tab_triage = st.tabs(["🔍 Classic Search", "🧭 Interactive Triage (new)"])

with tab_classic:
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

    if st.button("Search", type="primary") and topic_input.strip():
        with st.spinner("Searching arXiv + Semantic Scholar and ranking results — this can take up to a minute..."):
            ok, data = _api_post("/search", {
                "topic": topic_input.strip(),
                "top_k": int(top_k_input),
                "doi_required": doi_required_input,
                "min_citation_count": int(min_citations_input),
                "web_max_results": int(web_max_results_input),
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


# ---- interactive triage tab (round 3) -------------------------------------------

def _paper_badge(paper_id: str, round_: dict, basket_ids: set[str]) -> str:
    if paper_id in basket_ids:
        return "✅ In basket"
    if paper_id in round_["new_paper_ids"]:
        return "🟢 New"
    return "👀 Already seen"


def _web_badge(url: str, round_: dict, basket_urls: set[str]) -> str:
    if url in basket_urls:
        return "✅ In basket"
    if url in round_["new_web_urls"]:
        return "🟢 New"
    return "👀 Already seen"


with tab_triage:
    st.caption(
        "Search across as many keywords as you like — every round's results stay browsable, papers found again "
        "under a new keyword merge into one entry instead of duplicating. ⚠️ Nothing here is saved as a reusable "
        "bag until you summarize and explicitly save it; refreshing the page any time before that loses your "
        "in-progress session."
    )

    if st.session_state.viewed_bag:
        bag = st.session_state.viewed_bag
        st.subheader(f"📦 Saved bag: {bag['name']}")
        st.caption(f"Topic: {bag['topic']} · saved {bag['created_at']}")
        if st.button("Close saved bag", key="close_viewed_bag"):
            st.session_state.viewed_bag = None
            st.rerun()
        for r in bag["rounds"]:
            st.caption(f"🔑 Round {r['round_number']}: {', '.join(r['keywords_used'])}")
        for theme in bag["themes"]:
            st.markdown(f"#### {theme['theme_name']}")
            for entry in theme["papers"]:
                st.markdown(f"**{entry['title']}**")
                st.write(entry["summary"])
                st.caption(entry.get("citation", entry["apa_citation"]))
        st.markdown("#### Gaps & Disagreements")
        st.write(bag["gaps_and_disagreements"])
        if bag.get("web_summary"):
            st.markdown("#### 🌐 Web Context Summary")
            st.write(bag["web_summary"]["synthesis"])
            for a in bag["web_summary"]["cited_articles"]:
                st.caption(f"[{a['title']}]({a['url']}) — {a['source_domain']}")
        st.divider()

    triage_keyword = st.text_input(
        "Add a keyword and search",
        key="triage_keyword_input",
        placeholder="e.g. parameter-efficient fine-tuning",
    )
    triage_include_web = st.checkbox("Include web context for this round", value=True, key="triage_include_web")

    if st.button("Search this keyword", type="primary", key="triage_search_button") and triage_keyword.strip():
        keyword = triage_keyword.strip()
        existing = st.session_state.triage_session
        with st.spinner(f"Searching arXiv + Semantic Scholar for {keyword!r}..."):
            ok, data = _api_post("/round_search", {
                "topic": existing["topic"] if existing else keyword,
                "keyword": keyword,
                "session_state": existing,
                "include_web": triage_include_web,
            })
        if ok:
            st.session_state.triage_session = data["session_state"]
            st.session_state.triage_latest_round = data["round"]["round_number"]
            st.rerun()
        else:
            st.error(data["detail"])

    session_state = st.session_state.triage_session
    if not session_state:
        st.info("Search a keyword above to start an interactive triage session.")
    else:
        st.divider()
        all_papers = session_state["all_papers"]
        all_web_articles = session_state["all_web_articles"]
        basket_paper_ids = set(session_state["basket_paper_ids"])
        basket_web_urls = set(session_state["basket_web_urls"])

        results_col, basket_col = st.columns([3, 1])

        with results_col:
            st.subheader(f"Session: {session_state['topic']}")

            for round_ in session_state["rounds"]:
                is_latest = round_["round_number"] == st.session_state.triage_latest_round
                header = (
                    f"Round {round_['round_number']} · {', '.join(round_['keywords_used'])!r} · "
                    f"{len(round_['paper_ids_found'])} paper(s) found"
                    + (f" · 🌐 {len(round_['web_urls_found'])} web result(s)" if round_["web_urls_found"] else "")
                )
                with st.expander(header, expanded=is_latest):
                    for pid in round_["paper_ids_found"]:
                        paper = all_papers.get(pid)
                        if not paper:
                            continue
                        in_basket = pid in basket_paper_ids
                        with st.container(border=True):
                            st.markdown(f"{_paper_badge(pid, round_, basket_paper_ids)} — **{paper['title']}**")
                            st.markdown(f"**Authors:** {', '.join(paper['authors']) or 'Unknown'}")
                            citations = paper["citation_count"] if paper["citation_count"] is not None else "n/a"
                            st.markdown(f"**Year:** {paper['year'] or 'n/a'}　|　**Venue:** {paper['venue'] or 'n/a'}　|　**Citations:** {citations}")
                            st.caption(paper["abstract"] or "_No abstract available._")
                            button_label = "Remove from basket" if in_basket else "Add to basket"
                            if st.button(button_label, key=f"basket_toggle_r{round_['round_number']}_{pid}"):
                                if in_basket:
                                    basket_paper_ids.discard(pid)
                                else:
                                    basket_paper_ids.add(pid)
                                session_state["basket_paper_ids"] = sorted(basket_paper_ids)
                                st.rerun()

                    if round_["web_urls_found"]:
                        st.markdown("**🌐 Web context**")
                        for url in round_["web_urls_found"]:
                            article = all_web_articles.get(url)
                            if not article:
                                continue
                            in_basket_web = url in basket_web_urls
                            with st.container(border=True):
                                st.markdown(f"{_web_badge(url, round_, basket_web_urls)} — 🌐 **{article['title']}**")
                                st.caption(article["source_domain"])
                                st.caption(article["snippet"] or "_No snippet available._")
                                button_label = "Remove from basket" if in_basket_web else "Add to basket"
                                if st.button(button_label, key=f"basket_toggle_web_r{round_['round_number']}_{url}"):
                                    if in_basket_web:
                                        basket_web_urls.discard(url)
                                    else:
                                        basket_web_urls.add(url)
                                    session_state["basket_web_urls"] = sorted(basket_web_urls)
                                    st.rerun()

                    st.divider()
                    st.caption(f"🔑 Papers found via: {', '.join(round_['keywords_used'])}")
                    if round_["web_urls_found"]:
                        st.caption(f"🔑 Web results found via: {', '.join(round_['keywords_used'])}")

        with basket_col:
            st.subheader(f"🧺 Basket ({len(basket_paper_ids) + len(basket_web_urls)})")
            if not basket_paper_ids and not basket_web_urls:
                st.caption("Nothing picked yet — add papers or web results from the results on the left.")
            for pid in sorted(basket_paper_ids):
                paper = all_papers.get(pid)
                if not paper:
                    continue
                st.markdown(f"📄 {paper['title']}")
                if st.button("Remove", key=f"basket_remove_{pid}"):
                    basket_paper_ids.discard(pid)
                    session_state["basket_paper_ids"] = sorted(basket_paper_ids)
                    st.rerun()
            for url in sorted(basket_web_urls):
                article = all_web_articles.get(url)
                if not article:
                    continue
                st.markdown(f"🌐 {article['title']}")
                if st.button("Remove", key=f"basket_remove_web_{url}"):
                    basket_web_urls.discard(url)
                    session_state["basket_web_urls"] = sorted(basket_web_urls)
                    st.rerun()

        # ---- summarize (round 3, phase 3) ------------------------------------------
        # Deliberately the only place in this whole tab that hits an
        # expensive endpoint: /triage/summarize is the sole call that embeds
        # anything, and only for whatever is in the basket right now — never
        # session_state["all_papers"], the full accumulated pool across every
        # round. See api.py's triage_summarize for the actual guarantee.

        st.divider()
        st.subheader("Summarize your basket")
        if not basket_paper_ids and not basket_web_urls:
            st.caption("Add at least one paper or web result to your basket to summarize.")
        else:
            triage_citation_style = st.selectbox(
                "Citation style",
                options=["apa", "harvard", "bibtex"],
                format_func=lambda s: {"apa": "APA", "harvard": "Harvard", "bibtex": "BibTeX"}[s],
                key="triage_citation_style",
            )
            if st.button("Summarize basket", type="primary", key="triage_summarize_button"):
                with st.spinner(f"Embedding {len(basket_paper_ids)} paper(s) in your basket and writing grounded summaries..."):
                    ok, data = _api_post("/triage/summarize", {
                        "session_state": session_state,
                        "style": triage_citation_style,
                    })
                if ok:
                    st.session_state.triage_summary = data
                else:
                    st.error(data["detail"])

        if st.session_state.get("triage_summary"):
            summary = st.session_state.triage_summary
            stats = summary["embed_stats"]
            st.caption(
                f"Embedded exactly the basket: {summary['basket_paper_count']} paper(s), "
                f"{stats['cache_hits']} cache hit(s), {stats['cache_misses']} newly embedded "
                f"(~${stats['estimated_cost_usd']:.6f})."
            )
            for theme in summary["themes"]:
                st.markdown(f"#### {theme['theme_name']}")
                for entry in theme["papers"]:
                    st.markdown(f"**{entry['title']}**")
                    st.write(entry["summary"])
                    st.caption(entry.get("citation", entry["apa_citation"]))
            st.markdown("#### Gaps & Disagreements")
            st.write(summary["gaps_and_disagreements"])
            if summary.get("web_summary"):
                st.markdown("#### 🌐 Web Context Summary")
                st.write(summary["web_summary"]["synthesis"])
                for a in summary["web_summary"]["cited_articles"]:
                    st.caption(f"[{a['title']}]({a['url']}) — {a['source_domain']}")

            # ---- save or discard (round 3, phase 4) --------------------------------
            # The embeddings above are already sitting in Chroma the moment
            # Summarize completes (phase 3) — nothing here re-embeds anything.
            # Save just adds the SQLite bag record on top; Discard removes
            # those Chroma vectors again (unless another saved bag still
            # needs them) so an abandoned session leaves no trace at all.
            st.divider()
            st.warning(
                "This basket isn't a saved bag yet — save it below to keep it, or discard it. "
                "Refreshing the page before doing either loses it."
            )
            save_col, discard_col = st.columns([2, 1])
            with save_col:
                bag_name_input = st.text_input(
                    "Bag name", value=session_state["topic"], key="triage_bag_name_input",
                )
                if st.button("Save as bag", type="primary", key="triage_save_bag_button") and bag_name_input.strip():
                    ok, data = _api_post("/triage/save_bag", {
                        "name": bag_name_input.strip(),
                        "session_state": session_state,
                        "summary": summary,
                    })
                    if ok:
                        st.session_state.triage_session = None
                        st.session_state.triage_summary = None
                        st.session_state.triage_latest_round = None
                        st.success(f"Saved as bag {data['bag_id']!r} — {data['name']!r}.")
                        st.rerun()
                    else:
                        st.error(data["detail"])
            with discard_col:
                if st.button("Discard this session", key="triage_discard_button"):
                    ok, data = _api_post("/triage/discard", {"paper_ids": list(basket_paper_ids)})
                    if ok:
                        st.session_state.triage_session = None
                        st.session_state.triage_summary = None
                        st.session_state.triage_latest_round = None
                        st.success("Session discarded — no trace kept in SQLite or Chroma.")
                        st.rerun()
                    else:
                        st.error(data["detail"])
