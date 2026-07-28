import { useEffect, useState } from 'react'
import { curationApi } from '../../api/client'
import type { CurationReviewSummary } from '../../api/types'
import { ReviewCard, statusLabel } from './ReviewCard'
import { NewReviewForm } from './NewReviewForm'
import { WorkspaceModeSwitcher, type WorkspaceMode } from '../WorkspaceMode/WorkspaceModeSwitcher'

interface ReviewsListProps {
  activeSessionId: string | null
  onSelectReview: (sessionId: string) => void
  onStartReview: (topic: string, targetCount: number) => void
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
      {showForm ? (
        <NewReviewForm
          onSubmit={(topic, targetCount) => {
            setShowForm(false)
            onStartReview(topic, targetCount)
          }}
          onCancel={() => setShowForm(false)}
        />
      ) : (
        <button
          type="button"
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
