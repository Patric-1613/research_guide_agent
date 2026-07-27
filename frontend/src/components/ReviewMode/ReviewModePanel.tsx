import { useState } from 'react'
import type { CurationStateResponse } from '../../api/types'
import type { TurnEvent } from '../../hooks/useCurationSession'
import { TurnBlock, TurnDivider } from '../TurnFeed/TurnBlock'
import { PaperCard } from '../PaperPool/PaperCard'

interface ReviewModePanelProps {
  state: CurationStateResponse
  turnEvents: TurnEvent[]
  stagedPickIds: string[]
  disabled: boolean
  onAdd: (paperId: string) => void
  onRemoveStaged: (paperId: string) => void
  // stop=true submits whatever's staged and ends curation right away --
  // the backend (submitPicks) already supports this; the old UI never
  // exposed it, forcing users to keep hitting target_count exactly.
  onSubmitPicks: (refinement: string | undefined, stop: boolean) => Promise<void>
}

export function ReviewModePanel({
  state,
  turnEvents,
  stagedPickIds,
  disabled,
  onAdd,
  onRemoveStaged,
  onSubmitPicks,
}: ReviewModePanelProps) {
  const [refinement, setRefinement] = useState('')
  const pendingBatch = state.pending_batch
  const stagedSet = new Set(stagedPickIds)
  const totalSelected = state.selected_paper_ids.length + stagedPickIds.length

  if (!pendingBatch) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
        <p className="text-sm text-text-secondary">
          Curation complete — {state.selected_papers.length} papers selected.
        </p>
        <p className="text-sm text-text-muted">Switch to the Report tab on the left to generate your literature review.</p>
      </div>
    )
  }

  async function handleContinue() {
    await onSubmitPicks(refinement.trim() || undefined, false)
    setRefinement('')
  }

  async function handleStop() {
    await onSubmitPicks(refinement.trim() || undefined, true)
    setRefinement('')
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {turnEvents.map((event) => (
          <TurnBlock key={event.turnNumber} event={event} />
        ))}
        <TurnDivider turnNumber={turnEvents.length + 1} refilled={state.refilled} />
        <p className="mb-3 text-sm text-text-secondary">
          {state.refilled
            ? `Searched for more candidates and found ${pendingBatch.length} to show you this turn.`
            : `Showing ${pendingBatch.length} candidates from the pool already fetched — no new search needed.`}
          {' '}Select the ones you want, then continue.
        </p>
        <div className="flex flex-col gap-2">
          {pendingBatch.map((paper) =>
            stagedSet.has(paper.paper_id) ? (
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
                isNew
                showAbstract
                action={{ kind: 'add', onAdd: () => onAdd(paper.paper_id) }}
              />
            ),
          )}
        </div>
      </div>

      <div className="flex flex-col gap-2 border-t border-border bg-panel p-3">
        <input
          data-testid="review-refinement-input"
          value={refinement}
          onChange={(e) => setRefinement(e.target.value)}
          disabled={disabled}
          placeholder="Refine what you're looking for (optional)..."
          className="rounded-md border border-border bg-panel-alt px-3 py-2 text-sm text-text outline-none focus:border-accent disabled:opacity-60"
        />
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            data-testid="review-stop"
            onClick={handleStop}
            disabled={disabled || totalSelected === 0}
            className="text-xs text-text-secondary underline decoration-dotted hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          >
            I&apos;m done — finish with {totalSelected} selected
          </button>
          <button
            type="button"
            data-testid="review-continue"
            onClick={handleContinue}
            disabled={disabled}
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-40"
          >
            {stagedPickIds.length > 0 ? `Continue with ${stagedPickIds.length} added` : 'Get next batch'}
          </button>
        </div>
      </div>
    </div>
  )
}
