import { useState } from 'react'
import type { CurationStateResponse } from '../../api/types'
import type { TurnEvent } from '../../hooks/useCurationSession'
import { TurnBlock, TurnDivider } from '../TurnFeed/TurnBlock'
import { PaperCard } from '../PaperPool/PaperCard'

// Mirrors research_agent/query_expansion.py's BATCH_SIZE -- not exposed
// by the API (nothing currently needs it to be), so kept here as the
// one place the frontend needs to know "is this a full batch or a
// partial one" for the turn-divider messaging below.
const BATCH_SIZE = 10

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
  // requestRefill (Phase 9d): the explicit "search for more now" action --
  // forces a fresh search even when the pool isn't truly exhausted yet.
  onSubmitPicks: (refinement: string | undefined, stop: boolean, requestRefill?: boolean) => Promise<void>
}

function stopReasonMessage(state: CurationStateResponse): string {
  const count = state.selected_papers.length
  if (state.stop_reason === 'exhausted' && count < state.target_count) {
    // Phase 9d, the actual reported bug: this used to look identical to a
    // clean finish, with no way to tell "the search ran dry" apart from
    // "you reached your target" -- and no recourse either way.
    return `The search ran out of new candidates at ${count} of ${state.target_count} selected — nothing more to fetch right now.`
  }
  return `Curation complete — ${count} papers selected.`
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
    // Phase 8, item 3: true both right after curation just finished AND
    // every time an already-completed review is reopened -- state.selected_papers
    // is already in the API response either way, it just wasn't being
    // rendered here before.
    return (
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-3">
          <p className="mb-3 text-center text-sm text-text-secondary">{stopReasonMessage(state)}</p>
          <div className="flex flex-col gap-2">
            {state.selected_papers.map((paper) => (
              <PaperCard key={paper.paper_id} paper={paper} showAbstract action={{ kind: 'none' }} />
            ))}
          </div>
        </div>
        <p className="border-t border-border bg-panel px-4 py-3 text-center text-sm text-text-muted">
          Switch to the Report tab on the left to generate your literature review.
        </p>
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

  async function handleRequestRefill() {
    await onSubmitPicks(refinement.trim() || undefined, false, true)
    setRefinement('')
  }

  // curation-editable-until-locked Phase 10b/10d: an exhausted search or
  // a target-reached batch no longer ends curation -- the backend just
  // presents an empty (or ordinary) batch and leaves the review open, so
  // the frontend needs to spell out clearly what happened instead of
  // silently rendering zero PaperCards.
  const isEmptyBatch = pendingBatch.length === 0
  const targetReached = totalSelected >= state.target_count

  // Phase 9d/9e, the actual reported bug: a partial batch (fewer than
  // BATCH_SIZE, not refilled this turn) used to be indistinguishable
  // from "the pool ran fully dry" -- and the old auto-refill-below-
  // BATCH_SIZE behavior meant this message rarely even had a chance to
  // show. Now that a partial batch serves as-is, say so plainly, and
  // point at the explicit action instead of implying one already happened.
  const isPartialBatch = !state.refilled && !isEmptyBatch && pendingBatch.length < BATCH_SIZE
  const turnMessage = isEmptyBatch
    ? "No new candidates found for this search. Try refining below, or click \"I'm done\" if you're satisfied with what you have."
    : state.refilled
      ? `Searched for more candidates and found ${pendingBatch.length} to show you this turn.`
      : isPartialBatch
        ? `Only ${pendingBatch.length} candidate${pendingBatch.length === 1 ? '' : 's'} left in the already-fetched pool.`
        : `Showing ${pendingBatch.length} candidates from the pool already fetched — no new search needed.`

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {turnEvents.map((event) => (
          <TurnBlock key={event.turnNumber} event={event} />
        ))}
        <TurnDivider turnNumber={turnEvents.length + 1} refilled={state.refilled} />
        {targetReached && (
          <p
            data-testid="review-target-reached-banner"
            className="mb-3 rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent"
          >
            You&apos;ve reached your target of {state.target_count} papers ({totalSelected} selected). Keep
            curating if you&apos;d like more, or click &quot;I&apos;m done&quot; below to finish.
          </p>
        )}
        <p data-testid="review-turn-message" className="mb-3 text-sm text-text-secondary">
          {isEmptyBatch ? (
            turnMessage
          ) : (
            <>
              {turnMessage} Select the ones you want, then continue{isPartialBatch ? ', or search for more below' : ''}.
            </>
          )}
        </p>
        {isEmptyBatch ? (
          <p className="text-center text-sm italic text-text-muted">No candidates to show for this search.</p>
        ) : (
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
        )}
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
          <div className="flex items-center gap-2">
            <button
              type="button"
              data-testid="review-request-refill"
              onClick={handleRequestRefill}
              disabled={disabled}
              title="Search for more candidates now, even though the current pool isn't empty"
              className="rounded-md border border-border px-3 py-2 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              Search for more candidates
            </button>
            <button
              type="button"
              data-testid="review-continue"
              onClick={handleContinue}
              disabled={disabled}
              className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-40"
            >
              {stagedPickIds.length > 0
                ? `Continue with ${stagedPickIds.length} added`
                : isEmptyBatch
                  ? 'Try again'
                  : 'Get next batch'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
