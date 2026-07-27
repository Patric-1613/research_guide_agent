import type { ReportSectionOut, CurationStateResponse } from '../../api/types'

interface ReportModePanelProps {
  state: CurationStateResponse
  disabled: boolean
  onGenerateReport: () => void
  onRegenerateReport: () => void
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

  const { findings, limitations, future_scope, skipped_paper_ids } = state.report

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
      <ReportSection title="Findings" section={findings} />
      <ReportSection title="Limitations" section={limitations} />
      <ReportSection title="Future scope" section={future_scope} />
      {skipped_paper_ids.length > 0 && (
        <p className="text-xs text-text-muted">
          {skipped_paper_ids.length} selected paper{skipped_paper_ids.length === 1 ? '' : 's'} skipped from synthesis.
        </p>
      )}
    </div>
  )
}

function ReportSection({ title, section }: { title: string; section: ReportSectionOut }) {
  return (
    <section className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">{title}</h3>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-text">{section.content}</p>
      {(section.cited_papers.length > 0 || section.cited_web_articles.length > 0) && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {section.cited_papers.map((p) => (
            <span key={p.paper_id} className="rounded-full bg-panel-alt px-2 py-0.5 text-[11px] text-text-secondary">
              {p.title}
            </span>
          ))}
          {section.cited_web_articles.map((a) => (
            <a
              key={a.url}
              href={a.url}
              target="_blank"
              rel="noreferrer"
              className="rounded-full bg-accent-soft px-2 py-0.5 text-[11px] text-accent hover:underline"
            >
              {a.title}
            </a>
          ))}
        </div>
      )}
    </section>
  )
}
