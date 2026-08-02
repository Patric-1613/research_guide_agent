import { useState } from 'react'
import type { CurationStateResponse } from '../../types'
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
// Which action a paper gets depends on stage AND lock status, matching
// the backend-enforced gates (see select_paper_from_history's own
// docstring in curation_session.py for exactly why):
//   - locked (report generated and/or chat started): view-only, no
//     action at all -- reported bug fix. This used to be gated on
//     stage=="synthesize" alone, which stopped being a sufficient
//     "safe to add" proxy once curation-editable-until-locked (Phase
//     10) redefined "locked" as has_report/has_chat specifically --
//     the backend gate was patched to match (select_paper_from_history
//     now refuses once locked, not just while stage=="curate"), and
//     this render logic has to mirror that same rule, or a stale "+
//     Add to review" button would call an endpoint that now correctly
//     rejects it, which is confusing on its own even though it can no
//     longer silently corrupt anything.
//   - stage=="synthesize" and NOT locked: select-from-history is safe
//     (no pending interrupt) -- picking is immediate.
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
  const isLocked = state.report !== null || state.chat_history.length > 0
  const canPickImmediately = state.stage === 'synthesize' && !isLocked
  // Ascending turn_number order (how the backend appends them) -- entry
  // at index i always has turn_number === i + 1, so index and turn_number
  // are interchangeable below.
  const turns = state.turn_history
  // Local, remounts-fresh-on-open state (this component is only ever
  // mounted while showHistory is true -- see CurationWorkspacePage's
  // conditional render -- so this naturally defaults to the latest turn
  // every time the browser is opened, with no extra reset effect needed).
  const [selectedIndex, setSelectedIndex] = useState(() => turns.length - 1)
  const clampedIndex = Math.min(Math.max(selectedIndex, 0), turns.length - 1)
  const entry = turns.length > 0 ? turns[clampedIndex] : null

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

      {entry && (
        <div className="flex items-center justify-center gap-3 border-b border-border-soft bg-panel px-4 py-2">
          <button
            type="button"
            data-testid="turn-history-prev"
            onClick={() => setSelectedIndex((i) => Math.max(i - 1, 0))}
            disabled={clampedIndex === 0}
            className="rounded-md border border-border px-2 py-1 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            Previous
          </button>
          <span data-testid="turn-history-position" className="text-xs text-text-muted">
            Turn {entry.turn_number} of {turns.length}
          </span>
          <button
            type="button"
            data-testid="turn-history-next"
            onClick={() => setSelectedIndex((i) => Math.min(i + 1, turns.length - 1))}
            disabled={clampedIndex === turns.length - 1}
            className="rounded-md border border-border px-2 py-1 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-4 py-3">
        {!entry && <p className="text-sm text-text-muted">No turns yet.</p>}
        {entry && (
          <div className="mb-5">
            <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
              Turn {entry.turn_number} — {entry.refilled ? 'new search' : 'from existing pool'}
            </h3>
            <div className="flex flex-col gap-2">
              {entry.batch.map((paper) => {
                if (selectedSet.has(paper.paper_id) || isLocked) {
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
        )}
      </div>

      {isLocked ? (
        <p className="border-t border-border bg-panel px-4 py-3 text-center text-xs text-text-muted">
          This review is locked (a report has been generated and/or chat has started) — past turns are view-only.
        </p>
      ) : !canPickImmediately ? (
        <p className="border-t border-border bg-panel px-4 py-3 text-center text-xs text-text-muted">
          Curation is still in progress — papers picked here are added the next time you hit Continue in Review mode.
        </p>
      ) : null}
    </div>
  )
}
