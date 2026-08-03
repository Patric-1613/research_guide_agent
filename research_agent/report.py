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
import re
from typing import Literal

from langfuse import get_client, observe
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.citations import format_apa_citation, format_web_citation
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper, WebArticle
from research_agent.tracing import paper_metadata

logger = logging.getLogger(__name__)

# Same model choice/reasoning as summarize.py's SUMMARY_MODEL: infrequent
# (once per finished curation session, not looped), faithfulness matters
# more than cost for a call this rare.
REPORT_MODEL = "gpt-4.1"

FINDINGS_DESCRIPTION = "Synthesized findings and key contributions across the selected papers, grounded strictly in their abstracts."
LIMITATIONS_DESCRIPTION = "Limitations or shortcomings noted across the selected papers, and/or gaps or disagreements between them. State explicitly if none are apparent -- don't invent one."
FUTURE_SCOPE_DESCRIPTION = "Future research directions plausibly implied by the selected papers -- may be more speculative than the other two sections, but must still stay grounded in what the papers actually cover, not outside knowledge."

SECTION_NAMES = ("findings", "limitations", "future_scope")

SYSTEM_PROMPT = f"""You are writing a literature review report over a specific, deliberately hand-picked set of papers for a research topic. You will be given the topic and exactly this set of papers (id, title, abstract only) -- this is the FINAL set a user has already chosen; do not second-guess or add papers beyond it.

Write exactly three sections:
1. Findings: {FINDINGS_DESCRIPTION}
2. Limitations: {LIMITATIONS_DESCRIPTION}
3. Future scope: {FUTURE_SCOPE_DESCRIPTION}

For EACH section, ground every claim STRICTLY in the given abstracts -- do not use outside knowledge about these papers, their authors, or the topic beyond what the abstracts state. List which paper_ids actually support that section's content in its own cited_paper_ids, in the order you'd reference them. A section's citations are independent of the other two sections' -- a paper cited in Findings does not need to also appear in Limitations or Future scope, and vice versa.

If web articles are also provided below, you may additionally cite them (news, tooling, docs -- current/practical context, not peer-reviewed) via each section's cited_web_urls, but never in place of a paper citation -- keep the two kinds of sources clearly distinguished, the same way qa.py's chat answers do.

Use inline bracket markers in each section's content to mark which source supports each claim: [Paper 1], [Paper 2], ... for papers, in the order you list them in that section's own cited_paper_ids; [Web 1], [Web 2], ... for web articles (if any were provided), in the order you list them in that section's own cited_web_urls. These are two separate numbering sequences, never merged into one, and independent PER SECTION -- start over at [Paper 1]/[Web 1] in each of the three sections, never carried over from a previous section. These are temporary, section-local markers only -- they are automatically converted into one shared, report-wide [1], [2], [3]... numbering afterward, so you do not need to (and should not try to) coordinate numbers across sections yourself.
"""


def _build_report_schema(paper_ids: list[str], web_urls: list[str] | None = None) -> type[BaseModel]:
    """Same technique as summarize.py's _build_response_schema/
    _build_web_response_schema (a per-call dynamic Literal restricted to
    the exact ids passed in) -- reimplemented here, not imported, since
    the schema SHAPE is genuinely different (3 fixed sections vs. N
    theme clusters), not because the grounding mechanism differs.

    web_urls is optional-guarded exactly like qa.py's
    _build_answer_schema's own web_urls parameter -- a Literal can't be
    built from an empty tuple, and most reports still have no web
    sources at all (Phase 4 predates web escalation entirely), so
    cited_web_urls simply doesn't exist as a field unless there's
    something it could legitimately reference.
    """
    paper_id_literal = Literal[tuple(paper_ids)]

    section_fields: dict = {
        "content": (str, Field(description="The section's write-up text")),
        "cited_paper_ids": (
            list[paper_id_literal],
            Field(description="Selected paper_ids that support this section's content, in citation order. May be empty if none specifically support it."),
        ),
    }
    if web_urls:
        web_url_literal = Literal[tuple(web_urls)]
        section_fields["cited_web_urls"] = (
            list[web_url_literal],
            Field(description="Web article urls that support this section's content, in citation order. May be empty."),
        )

    section_model = create_model("ReportSection", **section_fields)

    return create_model(
        "LiteratureReviewReport",
        findings=(section_model, Field(description=FINDINGS_DESCRIPTION)),
        limitations=(section_model, Field(description=LIMITATIONS_DESCRIPTION)),
        future_scope=(section_model, Field(description=FUTURE_SCOPE_DESCRIPTION)),
    )


# --- report-quality Phase R1: inline numbered citations + References ---

# Matches a section-local, temporary marker as prompted above -- e.g.
# "[Paper 2]" or "[Web 1]". Deliberately reimplemented here rather than
# imported from qa.py's own _CITATION_MARKER_RE: the pattern itself is
# identical, but this module already reimplements _build_report_schema
# for the same reason (see that function's own docstring) -- avoiding a
# coupling on qa.py's private internals for what's a genuinely
# self-contained, three-line regex.
_SECTION_CITATION_MARKER_RE = re.compile(r"\[(Paper|Web) (\d+)\]")


def _densify_section_markers(content: str) -> str:
    """Renumbers ONE section's own [Paper N]/[Web N] markers to be dense
    and sequential starting at 1, per kind, in the order they first
    appear in the text -- same defensive technique qa.py's own
    _renumber_citation_markers uses, and for the same reason: the
    model's raw marker numbering isn't guaranteed to already align with
    its cited_paper_ids/cited_web_urls list order (observed skipping
    numbers in practice there; not assumed reliable here either). This
    is a prerequisite step for _build_references_and_renumber below --
    it does NOT resolve markers to actual paper_ids/urls or produce the
    final report-wide numbering, only guarantees each section's own
    markers are clean (1, 2, 3, ... with no gaps) before that mapping is
    attempted.
    """
    next_number = {"Paper": 1, "Web": 1}
    seen: dict[tuple[str, str], int] = {}

    def _replace(match: re.Match[str]) -> str:
        kind, old_number = match.group(1), match.group(2)
        key = (kind, old_number)
        if key not in seen:
            seen[key] = next_number[kind]
            next_number[kind] += 1
        return f"[{kind} {seen[key]}]"

    return _SECTION_CITATION_MARKER_RE.sub(_replace, content)


class _ReferenceAssigner:
    """Assigns (or reuses) a report-wide reference number for a given
    source, appending a new ReferenceEntry-shaped dict to `references`
    the first time each source is seen, and exposing the (kind, key) ->
    number map so a caller can look up an already-assigned number
    without re-deriving it. Shared by _build_references_and_renumber and
    derive_legacy_references below so the actual reference-entry-
    building logic (citation formatting, link_url resolution) lives in
    exactly one place, not duplicated between the "fresh generation" and
    "old persisted report" paths.
    """

    def __init__(self, references: list[dict]) -> None:
        self.references = references
        self.number_by_key: dict[tuple[str, str], int] = {}

    def get_or_assign(self, kind: str, key: str, title: str, *, paper: Paper | None = None, article: WebArticle | None = None) -> int:
        existing = self.number_by_key.get((kind, key))
        if existing is not None:
            return existing
        number = len(self.references) + 1
        self.number_by_key[(kind, key)] = number
        if kind == "paper":
            assert paper is not None
            link_url = f"https://doi.org/{paper.doi}" if paper.doi else paper.url
            self.references.append({
                "number": number, "kind": "paper", "paper_id": key, "url": None,
                "title": title, "formatted": format_apa_citation(paper), "link_url": link_url,
            })
        else:
            assert article is not None
            self.references.append({
                "number": number, "kind": "web", "paper_id": None, "url": key,
                "title": title, "formatted": format_web_citation(article), "link_url": key,
            })
        return number


def _build_references_and_renumber(sections_out: dict) -> dict:
    """The R1 post-processing pass: converts each section's temporary,
    section-local [Paper N]/[Web N] markers into ONE global, bare-number
    [N] sequence shared across the whole report, and builds the
    top-level `references` list those numbers point to. Deterministic,
    no LLM call -- operates purely on the already-attached
    cited_papers/cited_web_articles per section (real Paper/WebArticle
    objects, already in citation order), so it's directly testable
    without going through model-calling machinery at all.

    MUST run after _restore_dropped_citations on the regeneration path
    (i.e. after cited_papers/cited_web_articles are already final) --
    this function only reads citations and rewrites content/reference_
    numbers, it never changes WHICH papers/web articles are cited, so
    it's safe to call exactly once, as the very last step, regardless of
    caller.

    Algorithm:
      1. Per section, in FIXED order (findings, limitations,
         future_scope): densify that section's own markers (see
         _densify_section_markers), then resolve each densified marker
         to the underlying paper_id/url via that section's own
         cited_papers[n-1]/cited_web_articles[n-1] (0-indexed from the
         1-based densified marker number), assigning/reusing a GLOBAL
         reference number for it. The first time a source is seen
         anywhere in the report, it gets the next global number; every
         later occurrence (same section or a different one) reuses that
         same number -- "the same source keeps the same number across
         the whole report."
      2. A densified marker number beyond that section's own cited list
         length (e.g. [Paper 3] when only 2 papers were cited there) is
         INVALID -- stripped from the text entirely (never guessed at),
         and logged.
      3. Only AFTER every section's markers are resolved (so marker-
         derived citations always get the lowest/earliest numbers): any
         source a section structurally cited but never actually marked
         inline still gets a trailing global number and References
         entry, appended in that section's own citation-list order --
         a citation the model forgot to bracket is never silently
         dropped from References.
    """
    references: list[dict] = []
    assigner = _ReferenceAssigner(references)
    section_marked_keys: dict[str, set[tuple[str, str]]] = {name: set() for name in SECTION_NAMES}
    rewritten_content: dict[str, str] = {}

    for section_name in SECTION_NAMES:
        section = sections_out[section_name]
        cited_papers = section["cited_papers"]
        cited_web_articles = section.get("cited_web_articles", [])
        densified = _densify_section_markers(section["content"])

        def _resolve(
            match: re.Match[str], section_name: str = section_name,
            cited_papers: list[Paper] = cited_papers, cited_web_articles: list[WebArticle] = cited_web_articles,
        ) -> str:
            kind, densified_number = match.group(1), int(match.group(2))
            index = densified_number - 1
            if kind == "Paper":
                if index >= len(cited_papers):
                    logger.warning(
                        "_build_references_and_renumber: dropping out-of-range marker [Paper %d] in %r section "
                        "(only %d paper(s) cited there)", densified_number, section_name, len(cited_papers),
                    )
                    return ""
                paper = cited_papers[index]
                section_marked_keys[section_name].add(("paper", paper.paper_id))
                number = assigner.get_or_assign("paper", paper.paper_id, paper.title, paper=paper)
            else:
                if index >= len(cited_web_articles):
                    logger.warning(
                        "_build_references_and_renumber: dropping out-of-range marker [Web %d] in %r section "
                        "(only %d web article(s) cited there)", densified_number, section_name, len(cited_web_articles),
                    )
                    return ""
                article = cited_web_articles[index]
                section_marked_keys[section_name].add(("web", article.url))
                number = assigner.get_or_assign("web", article.url, article.title, article=article)
            return f"[{number}]"

        rewritten_content[section_name] = _SECTION_CITATION_MARKER_RE.sub(_resolve, densified)

    # Trailing pass: structurally cited but never marked -- still counts.
    for section_name in SECTION_NAMES:
        section = sections_out[section_name]
        for paper in section["cited_papers"]:
            if ("paper", paper.paper_id) not in section_marked_keys[section_name]:
                assigner.get_or_assign("paper", paper.paper_id, paper.title, paper=paper)
        for article in section.get("cited_web_articles", []):
            if ("web", article.url) not in section_marked_keys[section_name]:
                assigner.get_or_assign("web", article.url, article.title, article=article)

    for section_name in SECTION_NAMES:
        section = sections_out[section_name]
        section["content"] = rewritten_content[section_name]
        all_keys = (
            section_marked_keys[section_name]
            | {("paper", p.paper_id) for p in section["cited_papers"]}
            | {("web", a.url) for a in section.get("cited_web_articles", [])}
        )
        section["reference_numbers"] = sorted(assigner.number_by_key[key] for key in all_keys)

    return {**sections_out, "references": references}


def derive_legacy_references(report: dict) -> dict:
    """report-quality Phase R1 backward compatibility: a report persisted
    before this phase has no `references`/`reference_numbers` at all
    (see curation_session.py's _deserialize_report) and its `content`
    was never generated with any inline markers to rewrite. Called by
    the API serializer (research_agent/api_app/serializers.py's
    _report_to_out), not the storage layer, so old persisted dicts stay
    an untouched, byte-identical round-trip and this derivation is
    recomputed fresh on every read instead of being silently migrated
    into storage.

    Same numbering policy as _build_references_and_renumber's own
    trailing "structurally cited but unmarked" pass (Pass 3 there),
    applied to EVERY citation in an old report, since none of them were
    ever inline-marked in the first place: global numbers assigned in
    section order (findings, limitations, future_scope), then citation-
    list order within each section. `content` is returned unchanged --
    an old report's prose is never retroactively rewritten with markers
    it was never generated with.
    """
    references: list[dict] = []
    assigner = _ReferenceAssigner(references)

    updated_sections = {}
    for section_name in SECTION_NAMES:
        section = report[section_name]
        numbers: list[int] = []
        for paper in section["cited_papers"]:
            numbers.append(assigner.get_or_assign("paper", paper.paper_id, paper.title, paper=paper))
        for article in section.get("cited_web_articles", []):
            numbers.append(assigner.get_or_assign("web", article.url, article.title, article=article))
        updated_sections[section_name] = {**section, "reference_numbers": sorted(set(numbers))}

    return {**report, **updated_sections, "references": references}


@observe(name="generate_report_sections", as_type="generation", capture_input=False, capture_output=False)
def _generate_report_sections(
    topic: str, papers: list[Paper], web_articles: list[WebArticle], schema: type[BaseModel],
    client: OpenAI, model: str = REPORT_MODEL, system_prompt: str = SYSTEM_PROMPT,
) -> BaseModel:
    paper_listing = "\n\n".join(
        f"paper_id: {p.paper_id}\ntitle: {p.title}\nabstract: {p.abstract or '(no abstract available)'}"
        for p in papers
    )
    context_sections = [f"Selected papers:\n\n{paper_listing}"]
    if web_articles:
        web_listing = "\n\n".join(
            f"url: {a.url}\ntitle: {a.title}\nsnippet: {a.snippet or '(no snippet available)'}"
            for a in web_articles
        )
        context_sections.append(f"Additional web sources:\n\n{web_listing}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Research topic: {topic}\n\n" + "\n\n".join(context_sections)},
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
    topic: str, selected_papers: list[Paper], web_articles: list[WebArticle] | None = None,
    client: OpenAI | None = None, model: str = REPORT_MODEL,
) -> dict:
    """Generates the 3-part report over EXACTLY selected_papers -- the
    model is structurally unable to cite any paper_id outside this set
    (Literal-constrained, same guarantee level as summarize.py's own
    citations). Returns {"findings": {"content", "cited_papers"},
    "limitations": {...}, "future_scope": {...}, "skipped_papers"} --
    skipped_papers (mirroring summarize.py's own convention) is which
    selected papers were never cited in ANY of the 3 sections, logged as
    a warning, not an error.

    web_articles (curation-chat-web-escalation Phase 5d) is optional and
    backward-compatible: every existing caller that doesn't pass it gets
    byte-identical behavior to before -- each section dict simply has no
    "cited_web_articles" key at all unless web_articles is non-empty,
    the same convention _build_report_schema() uses for the schema
    field itself.
    """
    web_articles = web_articles or []
    if not selected_papers:
        empty_section = {"content": "", "cited_papers": [], "reference_numbers": []}
        if web_articles:
            empty_section["cited_web_articles"] = []
        return {
            "findings": dict(empty_section), "limitations": dict(empty_section), "future_scope": dict(empty_section),
            "skipped_papers": [], "references": [],
        }

    papers_by_id = {p.paper_id: p for p in selected_papers}
    web_by_url = {a.url: a for a in web_articles}
    schema = _build_report_schema(list(papers_by_id), list(web_by_url) or None)

    client = client or OpenAI()
    parsed = _generate_report_sections(topic, selected_papers, web_articles, schema, client, model)

    referenced_ids: set[str] = set()
    sections_out = {}
    for section_name in SECTION_NAMES:
        section = getattr(parsed, section_name)
        cited_papers = [papers_by_id[pid] for pid in section.cited_paper_ids]  # guaranteed present: Literal enforced it structurally
        referenced_ids.update(section.cited_paper_ids)
        section_out = {"content": section.content, "cited_papers": cited_papers}
        if web_by_url:
            cited_web_urls = list(getattr(section, "cited_web_urls", []))
            section_out["cited_web_articles"] = [web_by_url[url] for url in cited_web_urls]
        sections_out[section_name] = section_out

    skipped = [p for pid, p in papers_by_id.items() if pid not in referenced_ids]
    if skipped:
        logger.warning(
            "generate_report: %d selected paper(s) never cited in any section: %s",
            len(skipped), [p.title for p in skipped],
        )

    get_client().update_current_span(
        input={"topic": topic, "num_selected_papers": len(selected_papers), "num_web_articles": len(web_articles)},
        output={
            "findings_citation_count": len(sections_out["findings"]["cited_papers"]),
            "limitations_citation_count": len(sections_out["limitations"]["cited_papers"]),
            "future_scope_citation_count": len(sections_out["future_scope"]["cited_papers"]),
            "skipped_papers": paper_metadata(skipped),
        },
    )

    return _build_references_and_renumber({**sections_out, "skipped_papers": skipped})


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


def _build_regeneration_system_prompt(existing_report: dict) -> str:
    per_section_citations = "\n".join(
        f"- {name}: {[p.paper_id for p in existing_report[name]['cited_papers']]}"
        for name in SECTION_NAMES
    )
    return SYSTEM_PROMPT + f"""
This is a REGENERATION of an existing report -- additional web sources have been approved since it was first written, and you should incorporate them where genuinely relevant. You MUST preserve every paper_id already cited in a section below: never drop an existing paper citation from a section just because new web sources are now available. You may add new web citations, cite additional papers, and refine the prose, but an already-cited paper must remain cited in any section that previously cited it.

Previously cited paper_ids per section:
{per_section_citations}
"""


def _restore_dropped_citations(existing_report: dict, section_name: str, cited_paper_ids: list[str]) -> list[str]:
    """Layer 2 of the two-layer citation-preservation guarantee (same
    two-layer pattern as Phase 1c: prompt instruction first, structural
    enforcement as the backstop that doesn't depend on the model
    actually following it). Appends back, in their original order, any
    paper_id this exact section cited in the PRIOR report but the
    regeneration's own output dropped -- restoring is unconditional, not
    a best-effort nudge, so it holds even if the prompt instruction
    above is ignored entirely (see tests/test_report.py's isolation test,
    which defeats the prompt layer on purpose to prove this layer alone
    is what's actually load-bearing)."""
    restored = list(cited_paper_ids)
    for pid in [p.paper_id for p in existing_report[section_name]["cited_papers"]]:
        if pid not in restored:
            restored.append(pid)
    return restored


def _regenerate_report_sections_with_sources(
    session: PaperPoolSession, web_articles: list[WebArticle], client: OpenAI | None, model: str, caller_name: str,
) -> dict:
    """Shared body for regenerate_report_with_new_sources (whole-pool) and
    regenerate_report_with_approved_web_sources (curation-chat-add-to-
    report Phase 4's selective subset) -- identical schema-building,
    citation-restoration, and skipped-paper logic either way; the only
    difference between the two public callers is WHICH web_articles list
    gets passed in here. Neither public function duplicates this body, so
    citation handling can never drift between the two paths. caller_name
    is only for the skipped-papers log line, so it stays attributable.

    Same preconditions as before this refactor (session.report must
    already exist; stage must be "synthesize") -- unchanged, just moved
    here from regenerate_report_with_new_sources's own body.
    """
    if session.report is None:
        raise ValueError(
            "Session has no existing report to regenerate -- call generate_report_for_session() first."
        )
    if session.stage != "synthesize":
        raise ValueError(
            f"Session is not ready for report synthesis (stage={session.stage!r}, expected 'synthesize') -- "
            "curation must finish (target met, user stopped, or topic exhausted) before generating a report."
        )

    existing_report = session.report
    selected_papers = session.selected_papers

    papers_by_id = {p.paper_id: p for p in selected_papers}
    web_by_url = {a.url: a for a in web_articles}
    schema = _build_report_schema(list(papers_by_id), list(web_by_url) or None)
    system_prompt = _build_regeneration_system_prompt(existing_report)

    client = client or OpenAI()
    parsed = _generate_report_sections(
        session.topic, selected_papers, web_articles, schema, client, model, system_prompt=system_prompt,
    )

    referenced_ids: set[str] = set()
    sections_out = {}
    for section_name in SECTION_NAMES:
        section = getattr(parsed, section_name)
        cited_paper_ids = _restore_dropped_citations(existing_report, section_name, list(section.cited_paper_ids))

        referenced_ids.update(cited_paper_ids)
        cited_papers = [papers_by_id[pid] for pid in cited_paper_ids]
        section_out = {"content": section.content, "cited_papers": cited_papers}
        if web_by_url:
            cited_web_urls = list(getattr(section, "cited_web_urls", []))
            section_out["cited_web_articles"] = [web_by_url[url] for url in cited_web_urls]
        sections_out[section_name] = section_out

    skipped = [p for pid, p in papers_by_id.items() if pid not in referenced_ids]
    if skipped:
        logger.warning(
            "%s: %d selected paper(s) never cited in any section: %s",
            caller_name, len(skipped), [p.title for p in skipped],
        )

    get_client().update_current_span(
        input={
            "topic": session.topic, "num_selected_papers": len(selected_papers),
            "num_web_articles": len(web_articles),
        },
        output={
            "findings_citation_count": len(sections_out["findings"]["cited_papers"]),
            "limitations_citation_count": len(sections_out["limitations"]["cited_papers"]),
            "future_scope_citation_count": len(sections_out["future_scope"]["cited_papers"]),
            "skipped_papers": paper_metadata(skipped),
        },
    )

    # report-quality Phase R1: renumbering/reference-building runs LAST,
    # after _restore_dropped_citations above has already finalized which
    # papers/web articles each section cites -- a restored citation the
    # model's own regenerated content never bracketed correctly falls
    # into this pass's own "structurally cited but never marked" handling
    # rather than being silently missing from References.
    return _build_references_and_renumber({**sections_out, "skipped_papers": skipped})


def regenerate_report_with_new_sources(
    session: PaperPoolSession, client: OpenAI | None = None, model: str = REPORT_MODEL,
) -> dict:
    """curation-chat-web-escalation Phase 5d: regenerates session.report
    over the SAME session.selected_papers plus ALL of
    session.web_articles_added approved so far via Phase 5c's
    offer-and-decide mechanism -- typically called after a new web
    source is approved into an already-synthesized session, not a
    from-scratch generation (that's generate_report_for_session()).

    Requires session.report to already exist -- refuses cleanly
    otherwise, since "regenerate" implies something to regenerate FROM;
    generate_report_for_session() is the right call for a session's
    first report.

    Never mutates session.report itself -- returns the new report dict,
    same convention as generate_report()/generate_report_for_session();
    the caller decides when to actually replace session.report with it.

    curation-chat-add-to-report Phase 4: this function's behavior is
    UNCHANGED by that phase -- still whole-pool, still what POST
    /curation/{id}/report/regenerate uses. The selective, approved-
    subset-only path added in Phase 4 is the separate
    regenerate_report_with_approved_web_sources() below; the two are
    intentionally independent (see that function's own docstring for
    why using both on the same session has a real interaction worth
    knowing about).
    """
    return _regenerate_report_sections_with_sources(
        session, session.web_articles_added, client, model, "regenerate_report_with_new_sources",
    )


def regenerate_report_with_approved_web_sources(
    session: PaperPoolSession, approved_web_articles: list[WebArticle],
    client: OpenAI | None = None, model: str = REPORT_MODEL,
) -> dict:
    """curation-chat-add-to-report Phase 4: regenerates session.report over
    session.selected_papers plus ONLY approved_web_articles -- deliberately
    does NOT read session.web_articles_added (the raw, unfiltered pool of
    every web article ever discovered during chat) at all. The caller
    (curation_chat.py's resolve_approved_web_articles_for_regeneration)
    is responsible for filtering the raw pool down to the approved subset
    BEFORE calling this; an unapproved web article can never reach the
    model through this function no matter what else is sitting in the
    session's raw pool.

    Same preconditions/return convention as regenerate_report_with_new_
    sources (session.report must already exist, stage must be
    "synthesize", never mutates session.report itself).

    Interaction worth knowing: if a session ever also uses the OLD
    whole-pool regenerate_report_with_new_sources (e.g. via the Report
    tab's existing "Regenerate" button), that call will overwrite
    session.report with one reflecting the ENTIRE raw pool, including any
    web articles never approved through this selective path -- the two
    mechanisms don't defer to each other. Not resolved in Phase 4 (kept
    as an explicit, named follow-up), since /report/regenerate's own
    behavior must stay whole-pool and unchanged this phase.
    """
    return _regenerate_report_sections_with_sources(
        session, approved_web_articles, client, model, "regenerate_report_with_approved_web_sources",
    )
