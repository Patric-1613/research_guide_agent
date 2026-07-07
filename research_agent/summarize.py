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
"""

from __future__ import annotations

import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.citations import format_apa_citation, format_bibtex_citation
from research_agent.schema import Paper

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
) -> dict:
    """Cluster papers into themes and write grounded per-paper summaries.

    Returns {"themes": [{"theme_name", "papers": [{"paper", "summary",
    "apa_citation", "bibtex"}]}], "gaps_and_disagreements": str,
    "skipped_papers": [Paper]} — skipped_papers are input papers the model
    didn't reference in any theme (logged as a warning, not an error: the
    model may judge a paper redundant or off-topic within the retrieved set).
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
            paper = papers_by_id[ps.paper_id]  # guaranteed present: Literal enforced it structurally
            referenced_ids.add(ps.paper_id)
            entries.append({
                "paper": paper,
                "summary": ps.summary,
                "apa_citation": format_apa_citation(paper),
                "bibtex": format_bibtex_citation(paper),
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
