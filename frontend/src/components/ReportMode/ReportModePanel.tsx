import { useEffect, useState } from 'react'
import type { ReportOut, ReportSection, ReportTemplate, ReportVersionSummary, CurationStateResponse } from '../../types'
import { renderContentWithMarkers } from '../../lib/citationMarkers'
import { ReferencesList } from '../shared/ReferencesList'

interface ReportModePanelProps {
  state: CurationStateResponse
  disabled: boolean
  // report-quality Phase R2C: both now receive the panel's own currently
  // SELECTED template (never omitted from this panel's own call sites --
  // the selector always has a concrete value, defaulted to "analytical").
  onGenerateReport: (reportTemplate: ReportTemplate) => void
  onRegenerateReport: (reportTemplate: ReportTemplate) => void
  // report-quality Phase R3: switches the active report version.
  onActivateReportVersion: (versionId: string) => void
}

const TEMPLATE_OPTIONS: { value: ReportTemplate; label: string }[] = [
  { value: 'foundational', label: 'Foundational' },
  { value: 'analytical', label: 'Analytical' },
  { value: 'expert', label: 'Expert' },
]

const TEMPLATE_LABELS: Record<ReportTemplate, string> = {
  foundational: 'Foundational', analytical: 'Analytical', expert: 'Expert',
}

// report-quality Phase R3: short, title-cased labels for the version
// selector's own generation_reason column -- a reason this frontend
// doesn't recognize (a future backend-only addition) still renders
// safely via the raw string fallback in versionLabel() below, rather
// than an empty/undefined label.
const GENERATION_REASON_LABELS: Record<string, string> = {
  initial: 'Initial',
  regenerate: 'Regenerate',
  chat_add_to_report: 'Chat add',
  chat_auto_update: 'Chat update',
}

// Point 1 of the redesign brief: there was previously NO way to see a
// report anywhere in the UI -- state.report was fetched from the
// backend but never rendered. This panel is the first place it's
// actually shown.
export function ReportModePanel({
  state, disabled, onGenerateReport, onRegenerateReport, onActivateReportVersion,
}: ReportModePanelProps) {
  // report-quality Phase R2C: initialized from the current report's own
  // template (defaulting to analytical before a first generation), kept
  // in sync whenever the ACTIVE report's template changes underneath
  // this panel -- a session switch, a completed generate/regenerate, or
  // any other report-replacing action -- without clobbering an in-
  // progress selector choice the user made but hasn't acted on yet
  // (this effect only fires when report_template's own VALUE changes).
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate>(state.report?.report_template ?? 'analytical')
  useEffect(() => {
    setSelectedTemplate(state.report?.report_template ?? 'analytical')
  }, [state.report?.report_template])

  if (!state.report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
        <p className="text-sm text-text-secondary">No report yet for this review.</p>
        <TemplateSelector selected={selectedTemplate} onChange={setSelectedTemplate} disabled={disabled} />
        <button
          type="button"
          data-testid="generate-report"
          onClick={() => onGenerateReport(selectedTemplate)}
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
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-text">Literature review report</h2>
          <TemplateBadge template={state.report.report_template ?? 'analytical'} />
        </div>
        <div className="flex items-center gap-2">
          <VersionSelector
            versions={state.report_versions}
            activeVersionId={state.active_report_version_id}
            onChange={onActivateReportVersion}
            disabled={disabled}
          />
          <TemplateSelector selected={selectedTemplate} onChange={setSelectedTemplate} disabled={disabled} />
          <button
            type="button"
            data-testid="regenerate-report"
            onClick={() => onRegenerateReport(selectedTemplate)}
            disabled={disabled}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-text-secondary hover:text-text disabled:opacity-40"
          >
            Regenerate
          </button>
        </div>
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
      <ReferencesList references={references ?? []} heading="References" sectionTestId="report-references" />
    </div>
  )
}

// report-quality Phase R2C: a compact segmented control, not a select --
// only 3 options, all worth seeing at a glance. No confirmation on
// switching (Regenerate already overwrites the report immediately with
// no confirmation today; adding one only for a template switch would be
// a new, inconsistent one-off pattern).
function TemplateSelector({
  selected, onChange, disabled,
}: {
  selected: ReportTemplate
  onChange: (template: ReportTemplate) => void
  disabled: boolean
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Report template"
      data-testid="report-template-selector"
      className="inline-flex shrink-0 rounded-md border border-border p-0.5 text-xs"
    >
      {TEMPLATE_OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={selected === option.value}
          data-testid={`report-template-option-${option.value}`}
          onClick={() => onChange(option.value)}
          disabled={disabled}
          className={
            selected === option.value
              ? 'rounded bg-accent px-2.5 py-1 font-medium text-accent-fg'
              : 'rounded px-2.5 py-1 text-text-secondary hover:text-text disabled:opacity-40'
          }
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

// report-quality Phase R3: a dropdown, not a segmented control -- unlike
// the fixed 3-option template selector above, version count grows
// without bound as a session accumulates more report generations/
// regenerations, so a segmented control would eventually overflow.
// Hidden entirely (not just disabled) when there are no versions yet --
// nothing to switch between before a first report exists. Switching is
// a real API call (POST .../activate), not a frontend-local view swap,
// so the newly active version's content only appears once the parent's
// onActivateReportVersion handler has refreshed state.
function VersionSelector({
  versions, activeVersionId, onChange, disabled,
}: {
  versions: ReportVersionSummary[]
  activeVersionId: string | null
  onChange: (versionId: string) => void
  disabled: boolean
}) {
  if (versions.length === 0) return null

  return (
    <select
      aria-label="Report version"
      data-testid="report-version-selector"
      value={activeVersionId ?? ''}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="rounded-md border border-border bg-transparent px-2 py-1.5 text-xs text-text-secondary disabled:opacity-40"
    >
      {versions.map((version) => (
        <option key={version.version_id} value={version.version_id}>
          {versionOptionLabel(version)}
        </option>
      ))}
    </select>
  )
}

function versionOptionLabel(version: ReportVersionSummary): string {
  const reason = GENERATION_REASON_LABELS[version.generation_reason] ?? version.generation_reason
  return `Version ${version.version_number} — ${TEMPLATE_LABELS[version.report_template]} — ${reason}`
}

// The current report's own template, so a user opening an already-
// generated report knows which mode produced it without having to
// remember or re-derive it from the selector's (independently
// changeable) current value.
function TemplateBadge({ template }: { template: ReportTemplate }) {
  return (
    <span
      data-testid="report-template-badge"
      className="rounded-full border border-border px-2 py-0.5 text-[11px] text-text-secondary"
    >
      {TEMPLATE_LABELS[template]}
    </span>
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
