import type { PaperOut } from '../../types'

interface PaperCardProps {
  paper: PaperOut
  isNew?: boolean
  // Phase 7: shown in the Review-mode candidate browser (the whole point
  // of that panel is picking papers with enough information to actually
  // judge them), omitted in compact contexts like the pool summary's
  // Selected list, where just the title is enough.
  showAbstract?: boolean
  action:
    | { kind: 'add'; onAdd: () => void }
    | { kind: 'remove'; onRemove: () => void }
    | { kind: 'none' }
}

export function PaperCard({ paper, isNew, showAbstract, action }: PaperCardProps) {
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
      {/* Paper Keywords and Filtering, K4.2: pure display of whatever
          PaperOut.keywords already contains -- never derived/computed
          here. Gated on showAbstract (same surfaces that show the
          abstract) AND a non-empty list -- no placeholder, no container
          at all, for a paper with none (a missing/too-short abstract, or
          a legacy paper from before K1 existed). Static <span>s, not
          buttons -- this card has no notion of filtering; that lives one
          level up, in ReviewModePanel. Moved ahead of metadata/abstract
          (was the last thing in the card) so keywords read as part of
          "what is this paper about," alongside the title, rather than
          trailing after the abstract as an afterthought. Deliberately
          more visually present than K2's muted `text-[10px]` chips (still
          subordinate to the title -- accent-soft fill + a thin border,
          not the Add button's solid accent fill) since K2's own
          treatment tested as too easy to miss. whitespace-normal +
          break-words (not truncate) lets a genuinely long keyword phrase
          wrap across lines instead of being cut off -- max-w-full still
          keeps any single chip from forcing the row wider than the card,
          and flex-wrap lets the whole chip row grow to as many lines as
          six keywords need. */}
      {showAbstract && paper.keywords.length > 0 && (
        <div data-testid={`paper-keywords-${paper.paper_id}`} className="flex flex-wrap gap-1.5">
          {paper.keywords.map((keyword) => (
            <span
              key={keyword}
              className="max-w-full whitespace-normal break-words rounded-md border border-accent/30 bg-accent-soft px-1.5 py-0.5 text-xs font-medium text-accent"
            >
              {keyword}
            </span>
          ))}
        </div>
      )}
      <p className="text-xs text-text-secondary">
        {metaParts.join(' · ')}
        {paper.citation_count != null && <span className="text-text-muted"> · {paper.citation_count} citations</span>}
      </p>
      {showAbstract && (
        <p className="text-xs leading-relaxed text-text-secondary">
          {paper.abstract || 'No abstract available.'}
        </p>
      )}
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
