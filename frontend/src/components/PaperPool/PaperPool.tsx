import type { CurationStateResponse, PaperOut } from '../../api/types'
import { PaperCard } from './PaperCard'
import { PoolHeader } from './PoolHeader'

interface PaperPoolProps {
  state: CurationStateResponse
  stagedPickIds: string[]
  onAdd: (paperId: string) => void
  onRemoveStaged: (paperId: string) => void
}

export function PaperPool({ state, stagedPickIds, onAdd, onRemoveStaged }: PaperPoolProps) {
  const pendingBatch = state.pending_batch ?? []
  const stagedSet = new Set(stagedPickIds)
  const newThisTurn = pendingBatch.filter((p) => !stagedSet.has(p.paper_id))
  const stagedPapers = pendingBatch.filter((p) => stagedSet.has(p.paper_id))
  const confirmedSelected = state.selected_papers

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-border bg-panel">
      <PoolHeader
        selectedCount={state.selected_paper_ids.length}
        targetCount={state.target_count}
        reserveRemaining={state.reserve_remaining}
      />

      <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-3">
        <section className="flex flex-col gap-2">
          <SectionHeading label="New this turn" count={newThisTurn.length} />
          {newThisTurn.length === 0 && (
            <p className="text-xs text-text-muted">
              {pendingBatch.length === 0 ? 'No batch pending right now.' : 'All of this turn\'s papers have been added.'}
            </p>
          )}
          {newThisTurn.map((paper: PaperOut) => (
            <PaperCard key={paper.paper_id} paper={paper} isNew action={{ kind: 'add', onAdd: () => onAdd(paper.paper_id) }} />
          ))}
        </section>

        <section className="flex flex-col gap-2">
          <SectionHeading label="Selected" count={confirmedSelected.length + stagedPapers.length} />
          {confirmedSelected.length === 0 && stagedPapers.length === 0 && (
            <p className="text-xs text-text-muted">Nothing selected yet.</p>
          )}
          {stagedPapers.map((paper) => (
            <PaperCard
              key={paper.paper_id}
              paper={paper}
              action={{ kind: 'remove', onRemove: () => onRemoveStaged(paper.paper_id) }}
            />
          ))}
          {confirmedSelected.map((paper) => (
            <PaperCard key={paper.paper_id} paper={paper} action={{ kind: 'none' }} />
          ))}
        </section>
      </div>
    </aside>
  )
}

function SectionHeading({ label, count }: { label: string; count: number }) {
  return (
    <div className="flex items-center gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">{label}</h3>
      <span className="rounded-full bg-panel-alt px-1.5 py-0.5 text-[10px] font-medium text-text-secondary">{count}</span>
    </div>
  )
}
