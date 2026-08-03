import { Globe } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ReportOut, ReportSection, CurationStateResponse, ReferenceEntry } from '../../types'

interface ReportModePanelProps {
  state: CurationStateResponse
  disabled: boolean
  onGenerateReport: () => void
  onRegenerateReport: () => void
}

// report-quality Phase R1: report prose now carries inline, report-wide
// numbered markers like [1], [2], [3] (see research_agent/report.py's
// _build_references_and_renumber) -- this splits on them and renders each
// one as a clickable anchor jumping to its entry in the References
// section below, rather than as plain, inert text. A content string with
// no markers at all (e.g. an old, pre-R1 report -- see
// derive_legacy_references, which deliberately never retrofits markers
// into old prose) just renders as one plain segment, unchanged.
const MARKER_RE = /(\[\d+\])/g

function renderContentWithMarkers(content: string): ReactNode[] {
  return content.split(MARKER_RE).map((part, i) => {
    const match = /^\[(\d+)\]$/.exec(part)
    if (!match) return <span key={i}>{part}</span>
    const number = match[1]
    return (
      <a
        key={i}
        href={`#ref-${number}`}
        data-testid={`citation-marker-${number}`}
        className="font-medium text-accent hover:underline"
      >
        [{number}]
      </a>
    )
  })
}

// Point 1 of the redesign brief: there was previously NO way to see a
// report anywhere in the UI -- state.report was fetched from the
// backend but never rendered. This panel is the first place it's
// actually shown.
export function ReportModePanel({ state, disabled, onGenerateReport, onRegenerateReport }: ReportModePanelProps) {
  if (!state.report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
        <p className="text-sm text-text-secondary">No report yet for this review.</p>
        <button
          type="button"
          data-testid="generate-report"
          onClick={onGenerateReport}
          disabled={disabled}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-40"
        >
          Generate report
        </button>
      </div>
    )
  }

  const { skipped_paper_ids, references } = state.report
  const sections = sectionsFromReport(state.report)

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-text">Literature review report</h2>
        <button
          type="button"
          data-testid="regenerate-report"
          onClick={onRegenerateReport}
          disabled={disabled}
          className="rounded-md border border-border px-3 py-1.5 text-xs text-text-secondary hover:text-text disabled:opacity-40"
        >
          Regenerate
        </button>
      </div>
      {sections.length > 1 && <SectionNav sections={sections} />}
      {sections.map((section) => (
        <ReportSectionBlock key={section.key} section={section} />
      ))}
      {skipped_paper_ids.length > 0 && (
        <p className="text-xs text-text-muted">
          {skipped_paper_ids.length} selected paper{skipped_paper_ids.length === 1 ? '' : 's'} skipped from synthesis.
        </p>
      )}
      <ReferencesSection references={references ?? []} />
    </div>
  )
}

// report-quality Phase R2A: prefers the backend's own dynamic `sections`
// list (already backend-ordered, and already derived from the legacy
// fields when a report has none of its own -- see the backend's
// derive_sections_from_legacy_report) so this is the ONLY place the
// component reads findings/limitations/future_scope directly, purely as
// a defense-in-depth fallback independent of trusting that derivation --
// not a second rendering path the rest of the component needs to know
// about.
function sectionsFromReport(report: ReportOut): ReportSection[] {
  if (report.sections && report.sections.length > 0) return report.sections

  const legacy: { key: string; title: string; section: typeof report.findings }[] = [
    { key: 'findings', title: 'Findings', section: report.findings },
    { key: 'limitations', title: 'Limitations', section: report.limitations },
    { key: 'future_scope', title: 'Future Scope', section: report.future_scope },
  ]
  return legacy
    .filter((s) => s.section)
    .map((s) => ({
      key: s.key, title: s.title, content: s.section.content, reference_numbers: s.section.reference_numbers,
    }))
}

// A compact "jump to section" list -- only worth showing once there's
// more than one section to jump between (a single-section report, e.g.
// today's own derived fallback shape used pre-generation, gains nothing
// from a table of contents with one entry in it).
function SectionNav({ sections }: { sections: ReportSection[] }) {
  return (
    <nav data-testid="report-section-nav" aria-label="Report sections" className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
      {sections.map((section) => (
        <a
          key={section.key}
          href={`#section-${section.key}`}
          data-testid={`section-nav-link-${section.key}`}
          className="text-text-secondary underline decoration-dotted underline-offset-2 hover:text-accent hover:decoration-solid"
        >
          {section.title}
        </a>
      ))}
    </nav>
  )
}

function ReportSectionBlock({ section }: { section: ReportSection }) {
  return (
    <section id={`section-${section.key}`} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">{section.title}</h3>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-text">
        {renderContentWithMarkers(section.content)}
      </p>
    </section>
  )
}

// report-quality Phase R1: the report's actual References list -- inline
// markers above link into this, by number. Renders nothing at all when
// there are no references (rather than an empty "References" heading),
// which safely covers a genuinely reference-less report and an old
// report where `references` itself is absent (see ReportOut.references'
// own optionality) the same way.
function ReferencesSection({ references }: { references: ReferenceEntry[] }) {
  if (references.length === 0) return null

  return (
    <section data-testid="report-references" className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">References</h3>
      <ol className="flex flex-col gap-1.5 text-sm text-text">
        {references.map((ref) => (
          <li
            key={ref.number} id={`ref-${ref.number}`} data-testid={`reference-${ref.number}`}
            className="flex items-start gap-2"
          >
            <span className="shrink-0 text-text-muted">[{ref.number}]</span>
            {ref.kind === 'web' && (
              // Same shared numbering as papers -- this is purely a
              // subtle, secondary visual cue (not a second numbering
              // scheme, not a heavy pill) so a web source is
              // distinguishable from a peer-reviewed paper at a glance.
              <span
                data-testid={`reference-web-icon-${ref.number}`}
                role="img"
                aria-label="Web source"
                title="Web source"
                className="mt-0.5 shrink-0 text-text-muted"
              >
                <Globe className="h-3.5 w-3.5" aria-hidden="true" />
              </span>
            )}
            {ref.link_url ? (
              <a
                href={ref.link_url}
                target="_blank"
                rel="noreferrer"
                // Underlined by default (dotted, same convention as this
                // app's other secondary links, e.g. the stale-report
                // "Dismiss" control) -- a link that only changes color on
                // hover is easy to miss at rest, especially against this
                // dark theme's already-muted body text.
                className="rounded-sm text-accent underline decoration-dotted underline-offset-2 hover:text-accent-hover hover:decoration-solid focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                {ref.formatted}
              </a>
            ) : (
              <span>{ref.formatted}</span>
            )}
          </li>
        ))}
      </ol>
    </section>
  )
}
