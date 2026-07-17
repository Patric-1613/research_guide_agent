"""Phase 5: cluster retrieved papers by theme and generate a grounded,
structured literature summary.

Clustering design: theme grouping is done by the same LLM call that writes
the per-paper summaries (prompt-based grouping), not a separate embedding
clustering step (e.g. KMeans over the phase-3 vectors). At the scale this
agent operates at (a handful to ~20 ranked papers per topic), a clustering
algorithm would still need an LLM call afterward to turn "cluster 3" into a
readable theme name — so folding grouping into the summary call is strictly
fewer LLM calls, not more, and produces themes named in the user's terms
instead of an arbitrary cluster index. Embedding-based clustering would earn
its keep at much larger N (hundreds of papers, too many for one prompt) —
out of scope at this project's scale.

Grounding is enforced two ways:
  1. Structural: each paper reference is constrained to a dynamic Literal
     type built from the exact paper_ids passed in, so the model cannot
     reference a paper that wasn't retrieved — fabricated citations are
     impossible by construction, not just discouraged by the prompt.
  2. Content: the model sees only paper_id/title/abstract per paper — no
     external knowledge is fed in — and is instructed to ground each
     summary strictly in the given abstract. This doesn't structurally
     prevent a hallucinated *claim* about a correctly-cited paper (an
     inherent limitation of any free-text LLM generation), but it removes
     the two commonest failure modes: citing a paper that doesn't exist,
     and drawing on the model's background knowledge instead of the
     abstract actually on hand.

Round-2 enhancement 5 adds generate_web_summary() below, for the separate
web-article corpus (web_search.py). It lives here rather than in a sibling
module: it reuses the exact same dynamic-Literal grounding technique and the
same OpenAI-call conventions (model constant, usage logging, "refusal"
handling) as generate_summary() above, just keyed on URL instead of
paper_id — splitting it into its own file would either duplicate that
technique or force an awkward shared-helper import back into this module
anyway, for a function a fraction of the size of generate_summary(). It does
NOT cluster into themes the way generate_summary() does: at the scale a web
context pool actually runs at (3-4 articles, not tens of papers), a single
short synthesis paragraph is the natural unit, not multiple themes each
needing their own LLM-derived name.
"""

from __future__ import annotations

import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.citations import (
    CitationStyle,
    format_apa_citation,
    format_bibtex_citation,
    format_harvard_citation,
    select_citation,
)
from research_agent.schema import Paper, WebArticle

logger = logging.getLogger(__name__)

# gpt-4.1: this call is infrequent (once per finalized search, not looped
# per tool call like the phase-4 agent), and faithfulness/instruction-
# following matters more here than raw cost — worth paying more per call
# for a call this rare. gpt-4.1-mini remains the right choice for the
# agent's tool loop, where the same call type runs many times per session
# and cost compounds.
SUMMARY_MODEL = "gpt-4.1"

GAPS_FIELD_DESCRIPTION = (
    "Explicit callout of gaps in coverage relative to the topic, or disagreements/"
    "contradictions across papers' claims. State 'No notable gaps or disagreements "
    "observed among the retrieved papers.' if you don't see any — don't invent one."
)

SYSTEM_PROMPT = """You are a research literature summarizer. You will be given a research topic and a list of papers (id, title, abstract only).

Group the papers into 3-6 themes by shared topic or methodology. Name each theme descriptively (not "Theme 1"). Each paper belongs in exactly one theme — whichever it fits best.

For each paper, write a 2-3 sentence summary grounded STRICTLY in its given abstract. Do not use outside knowledge about the paper, its authors, or its topic beyond what the abstract states. Do not speculate about results, methods, or claims the abstract doesn't mention.

Finally, write a short explicit callout of any gaps in what these papers cover relative to the topic, or disagreements/contradictions between papers' claims. If you don't see any, say so explicitly rather than inventing one.
"""


def _build_response_schema(paper_ids: list[str]) -> type[BaseModel]:
    """Per-call schema whose paper_id field is a Literal restricted to the
    exact ids passed in, so the model structurally cannot reference a paper
    that wasn't retrieved.
    """
    paper_id_literal = Literal[tuple(paper_ids)]

    constrained_paper_summary = create_model(
        "ConstrainedPaperSummary",
        paper_id=(paper_id_literal, ...),
        summary=(str, Field(description="2-3 sentences grounded strictly in this paper's abstract")),
    )
    constrained_theme = create_model(
        "ConstrainedTheme",
        theme_name=(str, Field(description="Short, descriptive theme/methodology label")),
        papers=(list[constrained_paper_summary], ...),
    )
    constrained_summary = create_model(
        "ConstrainedResearchSummary",
        themes=(list[constrained_theme], ...),
        gaps_and_disagreements=(str, Field(description=GAPS_FIELD_DESCRIPTION)),
    )
    return constrained_summary


def generate_summary(
    topic: str,
    papers: list[Paper],
    client: OpenAI | None = None,
    model: str = SUMMARY_MODEL,
    style: CitationStyle = "apa",
) -> dict:
    """Cluster papers into themes and write grounded per-paper summaries.

    Returns {"themes": [{"theme_name", "papers": [{"paper", "summary",
    "apa_citation", "harvard_citation", "bibtex", "citation"}]}],
    "gaps_and_disagreements": str, "skipped_papers": [Paper]} —
    skipped_papers are input papers the model didn't reference in any theme
    (logged as a warning, not an error: the model may judge a paper
    redundant or off-topic within the retrieved set).

    All three citation formats are always computed — they're pure/cheap
    string formatting, not an LLM call, so there's no cost reason to skip
    the two the caller didn't ask for. `citation` is whichever one matches
    `style` (round-2 enhancement 3), included for callers that just want
    "the" citation without knowing the style themselves; `apa_citation` and
    `bibtex` are kept as their own fields for backward compatibility with
    callers that predate `style`.
    """
    if not papers:
        return {"themes": [], "gaps_and_disagreements": "", "skipped_papers": []}

    papers_by_id = {p.paper_id: p for p in papers}
    schema = _build_response_schema(list(papers_by_id))

    paper_listing = "\n\n".join(
        f"paper_id: {p.paper_id}\ntitle: {p.title}\nabstract: {p.abstract or '(no abstract available)'}"
        for p in papers
    )
    user_message = f"Research topic: {topic}\n\nPapers:\n\n{paper_listing}"

    client = client or OpenAI()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=schema,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Model refused to produce a structured summary: {response.choices[0].message.refusal}")

    usage = response.usage
    logger.info(
        "generate_summary: %d tokens billed (prompt=%d, completion=%d)",
        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
    )

    referenced_ids: set[str] = set()
    themes_out = []
    for theme in parsed.themes:
        entries = []
        for ps in theme.papers:
            if ps.paper_id in referenced_ids:
                # The Literal grounding only guarantees a referenced paper_id
                # was actually retrieved — it doesn't stop the model placing
                # the SAME paper_id in more than one theme. Keep the first
                # occurrence (whichever theme listed it first) and drop the
                # duplicate rather than silently letting a paper appear twice.
                logger.warning(
                    "generate_summary: model placed paper_id=%r in more than one theme; "
                    "dropping the duplicate from theme %r, keeping its first occurrence",
                    ps.paper_id, theme.theme_name,
                )
                continue
            paper = papers_by_id[ps.paper_id]  # guaranteed present: Literal enforced it structurally
            referenced_ids.add(ps.paper_id)
            apa_citation = format_apa_citation(paper)
            harvard_citation = format_harvard_citation(paper)
            bibtex = format_bibtex_citation(paper)
            entries.append({
                "paper": paper,
                "summary": ps.summary,
                "apa_citation": apa_citation,
                "harvard_citation": harvard_citation,
                "bibtex": bibtex,
                "citation": select_citation(apa_citation, harvard_citation, bibtex, style),
            })
        themes_out.append({"theme_name": theme.theme_name, "papers": entries})

    skipped = [p for pid, p in papers_by_id.items() if pid not in referenced_ids]
    if skipped:
        logger.warning(
            "generate_summary: model did not reference %d retrieved paper(s): %s",
            len(skipped), [p.title for p in skipped],
        )

    return {
        "themes": themes_out,
        "gaps_and_disagreements": parsed.gaps_and_disagreements,
        "skipped_papers": skipped,
    }


WEB_SYSTEM_PROMPT = """You are summarizing current web context (news, tooling, docs, benchmarks, industry adoption) gathered alongside academic papers for a research topic. This is a supplementary, practical-context corpus — not a replacement for the peer-reviewed literature, and you should not imply otherwise.

You will be given a research topic and a list of web articles (url, title, snippet only).

Write a short synthesis (2-4 sentences) of what these articles collectively indicate about the topic's current or practical state. Ground every claim STRICTLY in the given snippets — do not use outside knowledge about these specific sources, or about the topic, beyond what the snippets state.

List which article URLs actually support your synthesis in cited_urls, in the order you'd reference them.
"""


def _build_web_response_schema(urls: list[str]) -> type[BaseModel]:
    """Same technique as _build_response_schema above, keyed on URL instead
    of paper_id — the model is structurally unable to cite a web article
    that wasn't actually retrieved this run, the same guarantee level as
    the paper citations above, not a weaker one."""
    url_literal = Literal[tuple(urls)]
    return create_model(
        "WebContextSummary",
        synthesis=(str, Field(description="2-4 sentences grounded strictly in the given article snippets")),
        cited_urls=(list[url_literal], Field(description="URLs of articles that actually support the synthesis, in reference order")),
    )


def generate_web_summary(
    topic: str,
    articles: list[WebArticle],
    client: OpenAI | None = None,
    model: str = SUMMARY_MODEL,
) -> dict:
    """Synthesize a short, grounded summary of the web context pool.

    Returns {"synthesis": str, "cited_articles": [WebArticle...]} —
    cited_articles is the subset of `articles` the model actually
    referenced, in the order it cited them (mirrors generate_summary()'s
    skipped_papers idea but inverted: here the useful signal is which
    articles WERE used, since the pool is small enough that "all of them"
    is the common case, not the exception).
    """
    if not articles:
        return {"synthesis": "", "cited_articles": []}

    articles_by_url = {a.url: a for a in articles}
    schema = _build_web_response_schema(list(articles_by_url))

    listing = "\n\n".join(
        f"url: {a.url}\ntitle: {a.title}\nsnippet: {a.snippet or '(no snippet available)'}"
        for a in articles
    )
    user_message = f"Research topic: {topic}\n\nWeb articles:\n\n{listing}"

    client = client or OpenAI()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": WEB_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=schema,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Model refused to produce a web summary: {response.choices[0].message.refusal}")

    usage = response.usage
    logger.info(
        "generate_web_summary: %d tokens billed (prompt=%d, completion=%d)",
        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
    )

    cited_articles = [articles_by_url[url] for url in parsed.cited_urls]  # guaranteed present: Literal enforced it structurally

    return {"synthesis": parsed.synthesis, "cited_articles": cited_articles}
