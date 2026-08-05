import { useEffect, useState } from 'react'
import type {
  ReportOut, ReportRefinementOut, ReportSection, RefinementMode, ReportTemplate, ReportVersionSummary,
  CurationStateResponse,
} from '../../types'
import { renderContentWithMarkers } from '../../lib/citationMarkers'
import { ReferencesList } from '../shared/ReferencesList'
import { ProgressBar } from '../shared/ProgressBar'

interface ReportModePanelProps {
  state: CurationStateResponse
  disabled: boolean
  // report-quality Phase R2C: both now receive the panel's own currently
  // SELECTED template (never omitted from this panel's own call sites --
  // the selector always has a concrete value, defaulted to "analytical").
  // report-quality Phase R4.1: both also receive the panel's own
  // currently selected refinement mode -- same "never omitted, always
  // concrete" convention, defaulted to "off".
  onGenerateReport: (reportTemplate: ReportTemplate, refinementMode: RefinementMode) => void
  onRegenerateReport: (reportTemplate: ReportTemplate, refinementMode: RefinementMode) => void
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

  // report-quality Phase R4.1: local, session-local toggle -- resets
  // to off on a fresh mount (same as never persisting across reviews),
  // never synced FROM the report the way selectedTemplate is (a report's
  // own past refinement history says nothing about what the user wants
  // the NEXT generate/regenerate to do).
  const [refineOnce, setRefineOnce] = useState(false)
  const refinementMode: RefinementMode = refineOnce ? 'single' : 'off'

  if (!state.report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 text-center">
        <p className="text-sm text-text-secondary">No report yet for this review.</p>
        <TemplateSelector selected={selectedTemplate} onChange={setSelectedTemplate} disabled={disabled} />
        <RefineOnceToggle checked={refineOnce} onChange={setRefineOnce} disabled={disabled} />
        <button
          type="button"
          data-testid="generate-report"
          onClick={() => onGenerateReport(selectedTemplate, refinementMode)}
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
          {state.report.refinement && <RefinementBadge refinement={state.report.refinement} />}
        </div>
        <div className="flex items-center gap-2">
          <VersionSelector
            versions={state.report_versions}
            activeVersionId={state.active_report_version_id}
            onChange={onActivateReportVersion}
            disabled={disabled}
          />
          <TemplateSelector selected={selectedTemplate} onChange={setSelectedTemplate} disabled={disabled} />
          <RefineOnceToggle checked={refineOnce} onChange={setRefineOnce} disabled={disabled} />
          <button
            type="button"
            data-testid="regenerate-report"
            onClick={() => onRegenerateReport(selectedTemplate, refinementMode)}
            disabled={disabled}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-text-secondary hover:text-text disabled:opacity-40"
          >
            Regenerate
          </button>
        </div>
      </div>
      {state.report.refinement && hasEvaluationDetails(state.report.refinement) && (
        <EvaluationDetails refinement={state.report.refinement} sections={sections} />
      )}
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

// report-quality Phase R4.1: a single checkbox, no long explanatory
// text -- opting in to the bounded draft -> evaluate -> revise ->
// finalize loop for the NEXT Generate/Regenerate click. Off by
// default. Shared between the empty (Generate) and main (Regenerate)
// views via the same lifted refineOnce state, same pattern
// TemplateSelector already uses across both.
function RefineOnceToggle({
  checked, onChange, disabled,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  disabled: boolean
}) {
  return (
    <label className="flex shrink-0 items-center gap-1.5 text-xs text-text-secondary">
      <input
        type="checkbox"
        data-testid="refine-once-toggle"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="h-3.5 w-3.5 rounded border-border disabled:opacity-40"
      />
      Refine once
    </label>
  )
}

// report-quality Phase R4.1: compact, score-only summary of whatever
// refinement ran for THIS report -- kept deliberately minimal even
// after R4.2 added the fuller EvaluationDetails disclosure below;
// this badge never grows beyond a one-line label + score. "Refined
// once" when a revision actually happened (rounds===1); "Evaluated"
// when the draft passed evaluation as-is (rounds===0) -- both cases
// mean refinement.enabled is true, so this is only ever rendered when
// there's something real to report.
function RefinementBadge({ refinement }: { refinement: ReportRefinementOut }) {
  const label = refinement.rounds > 0 ? 'Refined once' : 'Evaluated'
  const score = refinement.final_score ?? refinement.initial_score
  return (
    <span
      data-testid="report-refinement-badge"
      className="rounded-full border border-border px-2 py-0.5 text-[11px] text-text-secondary"
    >
      {label}{score !== null ? ` · score ${score}` : ''}
    </span>
  )
}

// report-quality Phase R4.2: only worth showing a disclosure at all
// when there's real evaluator detail to reveal -- a refined report
// whose evaluator found nothing to say (empty issues, no section
// scores) gets no toggle at all, not an empty panel.
function hasEvaluationDetails(refinement: ReportRefinementOut): boolean {
  return refinement.issues.length > 0 || !!refinement.section_scores
}

const MAX_VISIBLE_ISSUES = 5

// report-quality Phase R4.2: a compact, collapsed-by-default
// disclosure -- button+state toggle, matching this app's existing
// expand/collapse convention (e.g. CurationWorkspacePage's "Browse
// past turns") rather than introducing a native <details> element for
// the first time here. Deliberately never renders raw
// revision_instructions -- that text is written FOR THE MODEL, not
// for a human reader, and showing it verbatim would read like a
// leftover prompt fragment. The panel is explicit that issues describe
// the DRAFT the one evaluation ran against, not necessarily what's
// still true of the (possibly revised) report currently on screen --
// see refine_report_if_requested's own docstring for why that's the
// case structurally, not just a UI copy choice.
// report-quality Phase R4.2 polish: three distinct, deliberately non-
// symmetric shapes -- rendering "Initial score N · Final score N" for
// the no-revision case reads like a before/after comparison that never
// happened, which is exactly the confusion this fixes. rounds===0
// means the draft WAS the final report (nothing to compare); rounds>0
// with a null final_score means a revision happened but was never
// re-scored (still nothing to compare, just for a different reason);
// only rounds>0 with a real final_score is an actual N -> M
// transition.
function ScoreSummary({ refinement }: { refinement: ReportRefinementOut }) {
  const score = refinement.initial_score ?? '—'
  if (refinement.rounds === 0) {
    return (
      <div data-testid="evaluation-details-score-summary" className="flex flex-col gap-0.5">
        <p className="text-sm font-semibold text-text">Score {score}</p>
        <p>No revision needed</p>
      </div>
    )
  }
  if (refinement.final_score === null) {
    return (
      <div data-testid="evaluation-details-score-summary" className="flex flex-col gap-0.5">
        <p className="text-sm font-semibold text-text">Initial score {score}</p>
        <p>Revised once</p>
        <p>Final score not re-evaluated</p>
      </div>
    )
  }
  return (
    <div data-testid="evaluation-details-score-summary" className="flex flex-col gap-0.5">
      <p className="text-sm font-semibold text-text">
        Score {score} → {refinement.final_score}
      </p>
      <p>Revised once</p>
    </div>
  )
}

// report-quality Phase R4.2 polish: reuses the existing shared
// ProgressBar (components/shared/ProgressBar.tsx) rather than
// introducing a chart library or a second bar implementation --
// ProgressBar already clamps/rounds defensively, so a section score
// outside 0-100 (shouldn't happen given the evaluator's own schema,
// but not assumed) can't overflow the bar.
function SectionScoreRow({ sectionKey, label, score }: { sectionKey: string; label: string; score: number }) {
  return (
    <div data-testid={`evaluation-details-section-score-${sectionKey}`} className="flex items-center gap-2">
      <span className="w-28 shrink-0 truncate text-text-secondary" title={label}>
        {label}
      </span>
      <div className="flex-1">
        <ProgressBar value={score} max={100} />
      </div>
      <span className="w-6 shrink-0 text-right tabular-nums text-text-secondary">{score}</span>
    </div>
  )
}

function EvaluationDetails({ refinement, sections }: { refinement: ReportRefinementOut; sections: ReportSection[] }) {
  const [open, setOpen] = useState(false)
  const visibleIssues = refinement.issues.slice(0, MAX_VISIBLE_ISSUES)
  const hiddenIssueCount = refinement.issues.length - visibleIssues.length
  const sectionScoreEntries = refinement.section_scores ? Object.entries(refinement.section_scores) : []
  // Prefer the report's own current section titles when the evaluator's
  // section_scores key matches a real section key -- falls back to the
  // raw key itself (still readable) for anything that doesn't match,
  // rather than hiding an otherwise-valid score.
  const sectionTitleByKey = Object.fromEntries(sections.map((s) => [s.key, s.title]))

  return (
    <div className="rounded-lg border border-border bg-card p-3 text-xs">
      <button
        type="button"
        data-testid="evaluation-details-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left font-medium text-text-secondary hover:text-text"
      >
        <span>Evaluation details</span>
        <span aria-hidden="true">{open ? '−' : '+'}</span>
      </button>
      {open && (
        <div data-testid="evaluation-details-panel" className="mt-3 flex flex-col gap-3 text-text-secondary">
          <p className="italic">
            Evaluator findings describe the draft before revision, not necessarily the final report.
          </p>
          <ScoreSummary refinement={refinement} />
          {visibleIssues.length > 0 && (
            <div>
              <h4 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
                Draft issues
              </h4>
              <ul data-testid="evaluation-details-issues" className="list-disc space-y-0.5 pl-4">
                {visibleIssues.map((issue, i) => (
                  <li key={i}>{issue}</li>
                ))}
                {hiddenIssueCount > 0 && (
                  <li data-testid="evaluation-details-more-issues" className="text-text-muted">
                    +{hiddenIssueCount} more
                  </li>
                )}
              </ul>
            </div>
          )}
          {sectionScoreEntries.length > 0 && (
            <div data-testid="evaluation-details-section-scores">
              <h4 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
                Section scores
              </h4>
              <div className="flex flex-col gap-1">
                {sectionScoreEntries.map(([key, score]) => (
                  <SectionScoreRow key={key} sectionKey={key} label={sectionTitleByKey[key] ?? key} score={score} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
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
