import type { PaperOut } from '../../api/types'

interface PaperCardProps {
  paper: PaperOut
  isNew?: boolean
  action:
    | { kind: 'add'; onAdd: () => void }
    | { kind: 'remove'; onRemove: () => void }
    | { kind: 'none' }
}

export function PaperCard({ paper, isNew, action }: PaperCardProps) {
  const metaParts = [paper.venue, paper.year != null ? String(paper.year) : null].filter(Boolean)
  return (
    <div data-testid={`paper-card-${paper.paper_id}`} className="relative flex flex-col gap-1.5 rounded-lg border border-border bg-card p-2.5">
      <div className="flex items-start justify-between gap-2">
        <p className="pr-4 text-sm leading-snug text-text">{paper.title}</p>
        {isNew && (
          <span className="absolute right-2 top-2 shrink-0 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-semibold text-accent">
            NEW
          </span>
        )}
        {action.kind === 'remove' && (
          <button
            type="button"
            onClick={action.onRemove}
            aria-label={`Remove ${paper.title}`}
            className="absolute right-2 top-2 flex h-5 w-5 items-center justify-center rounded-full text-text-muted hover:bg-danger-soft hover:text-danger"
          >
            ×
          </button>
        )}
      </div>
      <p className="text-xs text-text-secondary">
        {metaParts.join(' · ')}
        {paper.citation_count != null && <span className="text-text-muted"> · {paper.citation_count} citations</span>}
      </p>
      {action.kind === 'add' && (
        <button
          type="button"
          data-testid={`add-paper-${paper.paper_id}`}
          onClick={action.onAdd}
          className="mt-0.5 self-start rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-fg"
        >
          + Add to review
        </button>
      )}
    </div>
  )
}
