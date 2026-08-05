"""Pure output/serialization/rendering helpers for research_agent/api.py's
endpoints — domain objects (Paper, WebArticle, raw dicts) in, response
schema objects or plain strings out. No _state access, no DB/network
calls, no LLM calls.

Moved out of api.py (Phase 4) so these have a real, independent home —
api.py re-exports every name here so `research_agent.api.<name>` and
`patch.object(api, "<name>", ...)` keep working unchanged for anything
still reaching them that way.
"""

from __future__ import annotations

from research_agent.api_app.schemas import (
    CitedPaperOut,
    CitedWebArticleOut,
    CurationTurnResponse,
    PaperOut,
    ReferenceEntry,
    ReportOut,
    ReportRefinementOut,
    ReportSection,
    ReportSectionOut,
    ReportVersionSummary,
    TurnHistoryEntryOut,
    WebArticleOut,
)
from research_agent.citations import CitationStyle, select_citation
from research_agent.report import derive_legacy_references, derive_sections_from_legacy_report
from research_agent.schema import Paper, WebArticle


def _paper_to_out(paper: Paper, score: float | None = None) -> PaperOut:
    return PaperOut(
        paper_id=paper.paper_id, title=paper.title, authors=paper.authors,
        year=paper.year, venue=paper.venue, abstract=paper.abstract,
        url=paper.url, doi=paper.doi, citation_count=paper.citation_count,
        source=paper.source, source_urls=paper.source_urls, score=score,
    )


def _web_article_to_out(article: WebArticle) -> WebArticleOut:
    return WebArticleOut(
        title=article.title, url=article.url, snippet=article.snippet,
        published_date=article.published_date, source_domain=article.source_domain,
    )


def _web_articles_from_saved(saved) -> list[WebArticle]:
    return [WebArticle(**a) for a in saved.web_articles]


def _summary_to_json(result: dict, style: CitationStyle = "apa") -> dict:
    """Adapt summarize.generate_summary()'s return value (which embeds Paper
    objects) into a plain-JSON dict safe to store in SQLite and return over
    HTTP.

    Uses .get() defensively rather than direct key access for the round-2
    citation-style fields: this also runs against hand-built dicts in tests
    that mock generate_summary() and predate harvard_citation/citation, and
    must degrade to an APA-based default rather than KeyError on those.
    """
    themes_out = []
    for theme in result["themes"]:
        papers_out = []
        for entry in theme["papers"]:
            apa_citation = entry.get("apa_citation", "")
            harvard_citation = entry.get("harvard_citation") or apa_citation
            bibtex = entry.get("bibtex", "")
            citation = entry.get("citation") or select_citation(apa_citation, harvard_citation, bibtex, style)
            papers_out.append({
                "paper_id": entry["paper"].paper_id,
                "title": entry["paper"].title,
                "summary": entry["summary"],
                "apa_citation": apa_citation,
                "harvard_citation": harvard_citation,
                "bibtex": bibtex,
                "citation": citation,
            })
        themes_out.append({"theme_name": theme["theme_name"], "papers": papers_out})

    return {
        "themes": themes_out,
        "gaps_and_disagreements": result["gaps_and_disagreements"],
        "skipped_paper_ids": [p.paper_id for p in result["skipped_papers"]],
    }


def _web_summary_to_json(result: dict) -> dict:
    """Adapt summarize.generate_web_summary()'s return value (which embeds
    WebArticle objects) into a plain-JSON dict safe to store in SQLite and
    return over HTTP — same purpose as _summary_to_json above, kept
    separate since it has its own cache column (web_summary) and its own
    shape (no themes, just a synthesis + the cited subset)."""
    return {
        "synthesis": result["synthesis"],
        "cited_articles": [a.to_dict() for a in result["cited_articles"]],
    }


def _paper_out_from_batch_entry(entry) -> PaperOut:
    paper_dict, score = entry
    return _paper_to_out(Paper(**paper_dict), score)


def _turn_history_out(turn_history: list[dict]) -> list[TurnHistoryEntryOut]:
    return [
        TurnHistoryEntryOut(
            turn_number=entry["turn_number"],
            batch=[_paper_out_from_batch_entry(e) for e in entry["batch"]],
            refilled=entry["refilled"],
        )
        for entry in turn_history
    ]


def _report_to_out(report: dict, version: dict | None = None) -> ReportOut:
    # report-quality Phase R1: a report persisted before this phase has
    # no top-level "references" key at all (see curation_session.py's
    # _deserialize_report) -- absence, not an empty list, is the "this
    # is a pre-R1 report" signal, since a genuinely reference-less fresh
    # report is also a valid (if rare) real case. Derived fresh on every
    # read, never persisted back -- the old report's own prose is never
    # retroactively rewritten with markers it was never generated with.
    if "references" not in report:
        report = derive_legacy_references(report)
    # report-quality Phase R2A: same absence-is-the-signal convention as
    # references above -- every report right now (old persisted, or
    # freshly generated by the still-unchanged 3-section pipeline) has
    # no "sections" key of its own yet, so this always derives it from
    # findings/limitations/future_scope. Once real multi-section
    # generation ships (a later phase), a fresh report will carry its
    # own genuine `sections` and this branch simply won't run for it.
    if "sections" not in report:
        report = {**report, "sections": derive_sections_from_legacy_report(report)}

    def _section(name: str) -> ReportSectionOut:
        s = report[name]
        return ReportSectionOut(
            content=s["content"],
            cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in s["cited_papers"]],
            cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in s.get("cited_web_articles", [])],
            reference_numbers=s.get("reference_numbers", []),
        )

    return ReportOut(
        findings=_section("findings"), limitations=_section("limitations"), future_scope=_section("future_scope"),
        skipped_paper_ids=[p.paper_id for p in report["skipped_papers"]],
        references=[ReferenceEntry(**r) for r in report["references"]],
        sections=[ReportSection(**s) for s in report["sections"]],
        # report-quality Phase R2C: absence (a report persisted before
        # this phase, or a fresh report that was somehow never stamped)
        # defaults to "analytical" -- same absence-is-the-signal
        # convention as references/sections above, resolved here rather
        # than assumed anywhere upstream.
        report_template=report.get("report_template", "analytical"),
        # report-quality Phase R3: version metadata is layered on from a
        # SEPARATE ReportVersion dict (report.py's own get_active_report_
        # version/activate_report_version), not from `report` itself --
        # the report body and its version envelope are two different
        # things. `version` is None for any call site that doesn't have
        # (or care about) version context, in which case every field
        # below stays None -- never guessed at from `report` alone,
        # which has no version_id/version_number/created_at/generation_
        # reason of its own.
        version_id=version["version_id"] if version else None,
        version_number=version["version_number"] if version else None,
        created_at=version.get("created_at") if version else None,
        generation_reason=version["generation_reason"] if version else None,
        # report-quality Phase R4.1: refinement metadata lives on the
        # report BODY itself (report.refine_report_if_requested stamps
        # it there), not the version envelope -- None whenever
        # refinement was never requested for this report, same
        # "absence is the signal" convention every other optional field
        # here already uses.
        refinement=ReportRefinementOut(**report["refinement"]) if report.get("refinement") else None,
    )


def _report_version_to_summary(version: dict, active_version_id: str | None) -> ReportVersionSummary:
    """report-quality Phase R3: the lightweight counterpart to
    _report_to_out above -- never touches `version["report"]`'s own
    (potentially large) body at all, only the envelope fields, so
    building CurationStateResponse.report_versions never re-serializes
    every version's full content just to list them."""
    return ReportVersionSummary(
        version_id=version["version_id"],
        version_number=version["version_number"],
        created_at=version.get("created_at"),
        report_template=version.get("report_template", "analytical"),
        generation_reason=version["generation_reason"],
        is_active=version["version_id"] == active_version_id,
    )


def _turn_result_to_response(session_id: str, target_count: int, result: dict) -> CurationTurnResponse:
    session_dict = result["session"]
    batch = result["__interrupt__"][0].value["batch"] if "__interrupt__" in result else []
    return CurationTurnResponse(
        session_id=session_id, stage=session_dict["stage"], target_count=target_count,
        selected_paper_ids=session_dict["selected_paper_ids"],
        batch=[_paper_out_from_batch_entry(e) for e in batch],
        stop_reason=result.get("stop_reason"),
        refilled=result.get("refilled", False),
        # Computed straight off the raw serialized dict (reserve/cursor),
        # not via _dict_to_session -- no need to reconstruct every
        # reserve Paper's full data just to subtract two lengths.
        reserve_remaining=max(0, len(session_dict["reserve"]) - session_dict["cursor"]),
        refinement_notes=list(session_dict.get("refinement_notes", [])),
    )


_STYLE_LABELS: dict[str, str] = {"apa": "APA", "harvard": "Harvard", "bibtex": "BibTeX"}


def _render_markdown(topic: str, summary_json: dict, style: CitationStyle = "apa", web_summary_json: dict | None = None) -> str:
    lines = [f"# Literature Summary: {topic}", ""]
    for theme in summary_json["themes"]:
        lines.append(f"## {theme['theme_name']}")
        lines.append("")
        for p in theme["papers"]:
            lines.append(f"- **{p['title']}**")
            lines.append(f"  {p['summary']}")
            lines.append("")

    lines.append("## Gaps and Disagreements")
    lines.append("")
    lines.append(summary_json["gaps_and_disagreements"])
    lines.append("")

    if web_summary_json is not None:
        # Its own section, positioned after the paper-themes summary but
        # clearly separate from it — never folded into the themes above.
        lines.append("## Web Context")
        lines.append("")
        lines.append(web_summary_json["synthesis"])
        lines.append("")
        for a in web_summary_json["cited_articles"]:
            lines.append(f"- [{a['title']}]({a['url']}) — {a['source_domain']}")
        lines.append("")

    if style == "bibtex":
        # BibTeX is already a structured export format, not prose — a
        # "References (BibTeX)" section duplicating the BibTeX block below
        # would just repeat it, so this is the one style that skips the
        # separate References section entirely.
        lines.append("## References (BibTeX)")
        lines.append("")
        lines.append("```bibtex")
        for theme in summary_json["themes"]:
            for p in theme["papers"]:
                lines.append(p.get("bibtex", ""))
                lines.append("")
        lines.append("```")
    else:
        citation_key = "harvard_citation" if style == "harvard" else "apa_citation"
        lines.append(f"## References ({_STYLE_LABELS.get(style, 'APA')})")
        lines.append("")
        for theme in summary_json["themes"]:
            for p in theme["papers"]:
                lines.append(f"- {p.get(citation_key) or p.get('apa_citation', '')}")
        lines.append("")

        lines.append("## BibTeX")
        lines.append("")
        lines.append("```bibtex")
        for theme in summary_json["themes"]:
            for p in theme["papers"]:
                lines.append(p.get("bibtex", ""))
                lines.append("")
        lines.append("```")

    return "\n".join(lines)
