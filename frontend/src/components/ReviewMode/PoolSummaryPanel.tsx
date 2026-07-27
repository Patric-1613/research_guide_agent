import type { CurationStateResponse } from '../../api/types'
import { PoolHeader } from '../PaperPool/PoolHeader'

interface PoolSummaryPanelProps {
  state: CurationStateResponse
  stagedPickIds: string[]
}

// Review mode's right panel is deliberately a lightweight summary, not a
// second copy of the candidate browser -- full cards + abstracts now
// live in the center (ReviewModePanel), matching the reference mockup's
// big-number counters + empty-state text instead of duplicating cards
// in two panels.
export function PoolSummaryPanel({ state, stagedPickIds }: PoolSummaryPanelProps) {
  const pendingBatch = state.pending_batch ?? []
  const stagedSet = new Set(stagedPickIds)
  const newThisTurn = pendingBatch.filter((p) => !stagedSet.has(p.paper_id)).length
  const stagedTitles = pendingBatch.filter((p) => stagedSet.has(p.paper_id))
  const selectedTotal = state.selected_paper_ids.length + stagedPickIds.length

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-l border-border bg-panel">
      <PoolHeader
        selectedCount={state.selected_paper_ids.length}
        targetCount={state.target_count}
        reserveRemaining={state.reserve_remaining}
      />

      <div className="grid grid-cols-2 gap-3 p-3">
        <StatTile testId="stat-new-this-turn" label="New this turn" value={newThisTurn} />
        <StatTile testId="stat-selected" label="Selected" value={selectedTotal} />
      </div>

      <div className="flex flex-1 flex-col gap-2 overflow-y-auto px-3 pb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">Selected papers</h3>
        {selectedTotal === 0 && <p className="text-xs text-text-muted">Nothing selected yet.</p>}
        {stagedTitles.map((paper) => (
          <p key={paper.paper_id} className="truncate text-xs text-accent">
            {paper.title}
          </p>
        ))}
        {state.selected_papers.map((paper) => (
          <p key={paper.paper_id} className="truncate text-xs text-text-secondary">
            {paper.title}
          </p>
        ))}
      </div>
    </aside>
  )
}

function StatTile({ testId, label, value }: { testId: string; label: string; value: number }) {
  return (
    <div data-testid={testId} className="rounded-lg border border-border bg-card p-3 text-center">
      <p className="text-2xl font-semibold tabular-nums text-text">{value}</p>
      <p className="text-[11px] uppercase tracking-wide text-text-muted">{label}</p>
    </div>
  )
}
