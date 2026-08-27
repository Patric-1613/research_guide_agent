import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { curationApi } from '../../lib/api/client'
import type { CurationReviewSummary } from '../../types'
import { ReviewCard, statusLabel } from './ReviewCard'
import { NewReviewForm } from './NewReviewForm'
import { WorkspaceModeSwitcher, type WorkspaceMode } from '../WorkspaceMode/WorkspaceModeSwitcher'
import type { ResearchLaneOut, SubmittedLane } from '../../types'

interface ReviewsListProps {
  activeSessionId: string | null
  onSelectReview: (sessionId: string) => void
  // RL5b: resolves with the new session id on a genuinely successful start
  // (canonical server state loaded), or undefined on failure -- the form
  // is closed/reset here only on the former.
  onStartReview: (
    topic: string,
    targetCount: number,
    lanes?: SubmittedLane[],
  ) => void | Promise<string | undefined>
  onDeleteReview: (sessionId: string) => void
  // Bumping this triggers a refetch -- the caller increments it after any
  // action that could change a review's summary (a pick, a report, a chat
  // turn), so the list reflects real backend state, not a client guess.
  refreshToken: number
  // The mode switcher lives here, at the bottom of the left panel, per
  // the reference mockup -- it's only meaningful (and only rendered)
  // once a review is open.
  workspaceMode: WorkspaceMode
  workspaceUnlocked: boolean
  onWorkspaceModeChange: (mode: WorkspaceMode) => void
  // UXH.2: true for the whole window between the New Review form's
  // submit and the new session either landing (via the existing
  // successful load path elsewhere) or the action failing -- optional,
  // defaulting to false, so every pre-existing render call in this
  // component's own tests keeps working unchanged. Drives the one
  // "Starting new review…" status location and disables the trigger
  // that would otherwise let a second start be submitted mid-request.
  startingReview?: boolean
  // Research Lanes (RL5): all optional -- forwarded straight to
  // NewReviewForm. When researchLanesAvailable is false/absent, the lane
  // affordances never render and this list behaves exactly as before.
  researchLanesAvailable?: boolean
  laneSuggestions?: ResearchLaneOut[] | null
  laneSuggestionLoading?: boolean
  laneSuggestionError?: string | null
  onSuggestLanes?: (topic: string) => void
  onResetLaneSuggestions?: () => void
}

// Fixed section order, most-active-first -- matches the natural
// curate -> report -> chat progression, not alphabetical or insertion
// order. "Chatted" sits between "Ready for report" and "Report" -- it's
// a real, reachable state (chat_turn() only guards on stage, never on
// report existence) where the user engaged via chat before ever
// generating a report.
const SECTION_ORDER = ['Curating', 'Ready for report', 'Chatted', 'Report', 'Report + Chat'] as const

export function ReviewsList({
  activeSessionId,
  onSelectReview,
  onStartReview,
  onDeleteReview,
  refreshToken,
  workspaceMode,
  workspaceUnlocked,
  onWorkspaceModeChange,
  startingReview = false,
  researchLanesAvailable = false,
  laneSuggestions = null,
  laneSuggestionLoading = false,
  laneSuggestionError = null,
  onSuggestLanes,
  onResetLaneSuggestions,
}: ReviewsListProps) {
  const [reviews, setReviews] = useState<CurationReviewSummary[]>([])
  const [showForm, setShowForm] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    curationApi
      .listReviews()
      .then((data) => {
        if (!cancelled) setReviews(data)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [refreshToken])

  const sections = SECTION_ORDER.map((label) => ({
    label,
    reviews: reviews.filter((r) => statusLabel(r).text === label),
  })).filter((section) => section.reviews.length > 0)

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col gap-3 border-r border-border bg-panel p-3">
      {/* UXH.2 / RL5b: the one status location for "starting a new review".
          While a start is in flight the form STAYS MOUNTED (so a failed
          start can restore every field) but renders only the status via
          its own `submitting` prop -- there is no editable field and no
          second Start action to click. It is closed/reset here ONLY after
          a genuinely successful start (onStartReview resolves with the new
          session id), never merely because the async handler settled. The
          bare `startingReview` status below covers the edge case where a
          start is somehow in flight with the form already closed. */}
      {showForm ? (
        <NewReviewForm
          submitting={startingReview}
          onSubmit={async (...args) => {
            const startedSessionId = await onStartReview(...args)
            if (startedSessionId) {
              setShowForm(false)
              onResetLaneSuggestions?.()
            }
          }}
          onCancel={() => {
            setShowForm(false)
            onResetLaneSuggestions?.()
          }}
          researchLanesAvailable={researchLanesAvailable}
          laneSuggestions={laneSuggestions}
          laneSuggestionLoading={laneSuggestionLoading}
          laneSuggestionError={laneSuggestionError}
          onSuggestLanes={onSuggestLanes}
          onResetLaneSuggestions={onResetLaneSuggestions}
        />
      ) : startingReview ? (
        <p
          role="status"
          aria-live="polite"
          data-testid="starting-review-status"
          className="flex items-center justify-center gap-2 rounded-lg border border-dashed border-border py-2 text-sm font-medium text-text-secondary"
        >
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          Starting new review…
        </p>
      ) : (
        <button
          type="button"
          data-testid="new-review-trigger"
          onClick={() => setShowForm(true)}
          className="w-full rounded-lg border border-dashed border-border py-2 text-sm font-medium text-text-secondary hover:border-accent hover:text-accent"
        >
          + New review
        </button>
      )}

      <div className="flex flex-1 flex-col gap-4 overflow-y-auto">
        {loading && reviews.length === 0 && <p className="px-1 text-xs text-text-muted">Loading reviews…</p>}
        {!loading && reviews.length === 0 && (
          <p className="px-1 text-xs text-text-muted">No reviews yet — start one above.</p>
        )}
        {sections.map((section) => (
          <div key={section.label} className="flex flex-col gap-2">
            <h3 className="px-1 text-[11px] font-semibold uppercase tracking-wide text-text-muted">
              {section.label} <span className="text-text-muted">({section.reviews.length})</span>
            </h3>
            {section.reviews.map((review) => (
              <ReviewCard
                key={review.session_id}
                review={review}
                active={review.session_id === activeSessionId}
                onSelect={() => onSelectReview(review.session_id)}
                onDelete={onDeleteReview}
              />
            ))}
          </div>
        ))}
      </div>

      {activeSessionId && (
        <WorkspaceModeSwitcher mode={workspaceMode} unlocked={workspaceUnlocked} onChange={onWorkspaceModeChange} />
      )}
    </aside>
  )
}
