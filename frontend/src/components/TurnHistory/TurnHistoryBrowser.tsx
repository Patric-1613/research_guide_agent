import type { CurationStateResponse } from '../../api/types'
import { PaperCard } from '../PaperPool/PaperCard'

interface TurnHistoryBrowserProps {
  state: CurationStateResponse
  stagedPickIds: string[]
  disabled: boolean
  onAdd: (paperId: string) => void
  onRemoveStaged: (paperId: string) => void
  onSelectFromHistory: (paperId: string) => Promise<void>
  onClose: () => void
}

// curation-turn-history Phase 9f: reachable from ANY workspace mode
// (rendered by App.tsx above the mode-specific panels, not inside one of
// them), addressing the original request directly -- being "stuck" in
// Report/Chat mode no longer means the earlier turns are gone.
//
// Which action a paper gets depends on stage, matching the two
// backend-enforced channels (see select_paper_from_history's own
// docstring in curation_session.py for exactly why there are two):
//   - stage=="synthesize": select-from-history is safe (no pending
//     interrupt) -- picking is immediate.
//   - stage=="curate": a real interrupt IS pending -- picking here just
//     stages the paper via the SAME onAdd/onRemoveStaged mechanism
//     Review mode's current batch already uses, applied on the next
//     /picks submission (which now validates against all of
//     turn_history, not just the current batch -- Phase 9d).
export function TurnHistoryBrowser({
  state,
  stagedPickIds,
  disabled,
  onAdd,
  onRemoveStaged,
  onSelectFromHistory,
  onClose,
}: TurnHistoryBrowserProps) {
  const stagedSet = new Set(stagedPickIds)
  const selectedSet = new Set(state.selected_paper_ids)
  const canPickImmediately = state.stage === 'synthesize'
  const turns = [...state.turn_history].reverse() // most recent turn first

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border bg-panel px-4 py-3">
        <h2 className="text-sm font-semibold text-text">Past turns ({state.turn_history.length})</h2>
        <button
          type="button"
          data-testid="close-turn-history"
          onClick={onClose}
          className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent"
        >
          Close
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3">
        {turns.length === 0 && <p className="text-sm text-text-muted">No turns yet.</p>}
        {turns.map((entry) => (
          <div key={entry.turn_number} className="mb-5">
            <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
              Turn {entry.turn_number} — {entry.refilled ? 'new search' : 'from existing pool'}
            </h3>
            <div className="flex flex-col gap-2">
              {entry.batch.map((paper) => {
                if (selectedSet.has(paper.paper_id)) {
                  return <PaperCard key={paper.paper_id} paper={paper} showAbstract action={{ kind: 'none' }} />
                }
                if (canPickImmediately) {
                  return (
                    <PaperCard
                      key={paper.paper_id}
                      paper={paper}
                      showAbstract
                      action={{
                        kind: 'add',
                        onAdd: () => {
                          if (!disabled) void onSelectFromHistory(paper.paper_id)
                        },
                      }}
                    />
                  )
                }
                return stagedSet.has(paper.paper_id) ? (
                  <PaperCard
                    key={paper.paper_id}
                    paper={paper}
                    showAbstract
                    action={{ kind: 'remove', onRemove: () => onRemoveStaged(paper.paper_id) }}
                  />
                ) : (
                  <PaperCard
                    key={paper.paper_id}
                    paper={paper}
                    showAbstract
                    action={{ kind: 'add', onAdd: () => onAdd(paper.paper_id) }}
                  />
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {!canPickImmediately && (
        <p className="border-t border-border bg-panel px-4 py-3 text-center text-xs text-text-muted">
          Curation is still in progress — papers picked here are added the next time you hit Continue in Review mode.
        </p>
      )}
    </div>
  )
}
