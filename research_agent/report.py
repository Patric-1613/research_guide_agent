"""curation-report-synthesis Phase 4: generates the literature-review
report over a curation session's exact hand-picked paper set — a
different, 3-part structure (findings / limitations / future scope)
from summarize.py's theme-clustering, but reusing the identical
grounding TECHNIQUE: a dynamic Literal built from the exact paper_ids
in play, so the model is structurally unable to cite a paper outside
that set. summarize.py itself is untouched — this is a new, separate
schema/function, not a rewrite of it (see this module's own tests for
the adversarial proof that a report never cites a seen-but-rejected
candidate).

Deliberately a plain function, not a graph node — matching
summarize.py's own generate_summary()/generate_web_summary() pattern
exactly. By the time session.stage == "synthesize", the curation
DECISION is already finalized (that's what Phase 3's interactive loop
was for); there's nothing left here to pause/resume/branch on. One
topic + one fixed paper set -> one LLM call -> one structured result,
the same shape as summarize.py's own one-shot generation. A future
graph (e.g. Phase 5's chat) can call this plain function directly from
one of its own nodes, the same way qa.py's _generate_node already
calls the plain function _generate_answer internally.
"""

from __future__ import annotations

import logging
from typing import Literal

from langfuse import get_client, observe
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper
from research_agent.tracing import paper_metadata

logger = logging.getLogger(__name__)

# Same model choice/reasoning as summarize.py's SUMMARY_MODEL: infrequent
# (once per finished curation session, not looped), faithfulness matters
# more than cost for a call this rare.
REPORT_MODEL = "gpt-4.1"

FINDINGS_DESCRIPTION = "Synthesized findings and key contributions across the selected papers, grounded strictly in their abstracts."
LIMITATIONS_DESCRIPTION = "Limitations or shortcomings noted across the selected papers, and/or gaps or disagreements between them. State explicitly if none are apparent -- don't invent one."
FUTURE_SCOPE_DESCRIPTION = "Future research directions plausibly implied by the selected papers -- may be more speculative than the other two sections, but must still stay grounded in what the papers actually cover, not outside knowledge."

SYSTEM_PROMPT = f"""You are writing a literature review report over a specific, deliberately hand-picked set of papers for a research topic. You will be given the topic and exactly this set of papers (id, title, abstract only) -- this is the FINAL set a user has already chosen; do not second-guess or add papers beyond it.

Write exactly three sections:
1. Findings: {FINDINGS_DESCRIPTION}
2. Limitations: {LIMITATIONS_DESCRIPTION}
3. Future scope: {FUTURE_SCOPE_DESCRIPTION}

For EACH section, ground every claim STRICTLY in the given abstracts -- do not use outside knowledge about these papers, their authors, or the topic beyond what the abstracts state. List which paper_ids actually support that section's content in its own cited_paper_ids, in the order you'd reference them. A section's citations are independent of the other two sections' -- a paper cited in Findings does not need to also appear in Limitations or Future scope, and vice versa.
"""


def _build_report_schema(paper_ids: list[str]) -> type[BaseModel]:
    """Same technique as summarize.py's _build_response_schema/
    _build_web_response_schema (a per-call dynamic Literal restricted to
    the exact ids passed in) -- reimplemented here, not imported, since
    the schema SHAPE is genuinely different (3 fixed sections vs. N
    theme clusters), not because the grounding mechanism differs."""
    paper_id_literal = Literal[tuple(paper_ids)]

    section_model = create_model(
        "ReportSection",
        content=(str, Field(description="The section's write-up text")),
        cited_paper_ids=(
            list[paper_id_literal],
            Field(description="Selected paper_ids that support this section's content, in citation order. May be empty if none specifically support it."),
        ),
    )

    return create_model(
        "LiteratureReviewReport",
        findings=(section_model, Field(description=FINDINGS_DESCRIPTION)),
        limitations=(section_model, Field(description=LIMITATIONS_DESCRIPTION)),
        future_scope=(section_model, Field(description=FUTURE_SCOPE_DESCRIPTION)),
    )


@observe(name="generate_report_sections", as_type="generation", capture_input=False, capture_output=False)
def _generate_report_sections(
    topic: str, papers: list[Paper], schema: type[BaseModel], client: OpenAI, model: str = REPORT_MODEL,
) -> BaseModel:
    paper_listing = "\n\n".join(
        f"paper_id: {p.paper_id}\ntitle: {p.title}\nabstract: {p.abstract or '(no abstract available)'}"
        for p in papers
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Research topic: {topic}\n\nSelected papers:\n\n{paper_listing}"},
    ]

    langfuse = get_client()
    langfuse.update_current_generation(input=messages, model=model)

    response = client.chat.completions.parse(model=model, messages=messages, response_format=schema)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        langfuse.update_current_generation(output=None, level="WARNING", status_message="Model refused")
        raise RuntimeError(f"Model refused to produce a report: {response.choices[0].message.refusal}")

    usage = response.usage
    if usage is not None:
        logger.info(
            "generate_report: %d tokens billed (prompt=%d, completion=%d)",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
        )
        langfuse.update_current_generation(
            usage_details={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )
    langfuse.update_current_generation(output=parsed.model_dump())
    return parsed


@observe(name="generate_report", capture_input=False, capture_output=False)
def generate_report(
    topic: str, selected_papers: list[Paper], client: OpenAI | None = None, model: str = REPORT_MODEL,
) -> dict:
    """Generates the 3-part report over EXACTLY selected_papers -- the
    model is structurally unable to cite any paper_id outside this set
    (Literal-constrained, same guarantee level as summarize.py's own
    citations). Returns {"findings": {"content", "cited_papers"},
    "limitations": {...}, "future_scope": {...}, "skipped_papers"} --
    skipped_papers (mirroring summarize.py's own convention) is which
    selected papers were never cited in ANY of the 3 sections, logged as
    a warning, not an error.
    """
    if not selected_papers:
        empty_section = {"content": "", "cited_papers": []}
        return {"findings": empty_section, "limitations": empty_section, "future_scope": empty_section, "skipped_papers": []}

    papers_by_id = {p.paper_id: p for p in selected_papers}
    schema = _build_report_schema(list(papers_by_id))

    client = client or OpenAI()
    parsed = _generate_report_sections(topic, selected_papers, schema, client, model)

    referenced_ids: set[str] = set()
    sections_out = {}
    for section_name in ("findings", "limitations", "future_scope"):
        section = getattr(parsed, section_name)
        cited_papers = [papers_by_id[pid] for pid in section.cited_paper_ids]  # guaranteed present: Literal enforced it structurally
        referenced_ids.update(section.cited_paper_ids)
        sections_out[section_name] = {"content": section.content, "cited_papers": cited_papers}

    skipped = [p for pid, p in papers_by_id.items() if pid not in referenced_ids]
    if skipped:
        logger.warning(
            "generate_report: %d selected paper(s) never cited in any section: %s",
            len(skipped), [p.title for p in skipped],
        )

    get_client().update_current_span(
        input={"topic": topic, "num_selected_papers": len(selected_papers)},
        output={
            "findings_citation_count": len(sections_out["findings"]["cited_papers"]),
            "limitations_citation_count": len(sections_out["limitations"]["cited_papers"]),
            "future_scope_citation_count": len(sections_out["future_scope"]["cited_papers"]),
            "skipped_papers": paper_metadata(skipped),
        },
    )

    return {**sections_out, "skipped_papers": skipped}


def generate_report_for_session(session: PaperPoolSession, client: OpenAI | None = None, model: str = REPORT_MODEL) -> dict:
    """Session-aware wrapper: refuses cleanly if the session isn't
    actually ready for synthesis yet (stage != "synthesize") rather than
    generating prematurely over a still-in-progress pick set, then
    resolves selected_papers directly from session.selected_papers
    (already full Paper data, populated at pick-time in
    curation_loop.py -- NOT reconstructed from session.reserve, which
    may no longer contain already-served papers after a refill).
    """
    if session.stage != "synthesize":
        raise ValueError(
            f"Session is not ready for report synthesis (stage={session.stage!r}, expected 'synthesize') -- "
            "curation must finish (target met, user stopped, or topic exhausted) before generating a report."
        )
    return generate_report(session.topic, session.selected_papers, client=client, model=model)
