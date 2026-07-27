import { useEffect, useState } from 'react'
import { curationApi } from '../../api/client'
import type { CurationReviewSummary } from '../../api/types'
import { ReviewCard } from './ReviewCard'
import { NewReviewForm } from './NewReviewForm'

interface ReviewsListProps {
  activeSessionId: string | null
  onSelectReview: (sessionId: string) => void
  onStartReview: (topic: string, targetCount: number) => void
  // Bumping this triggers a refetch -- the caller increments it after any
  // action that could change a review's summary (a pick, a report, a chat
  // turn), so the list reflects real backend state, not a client guess.
  refreshToken: number
}

export function ReviewsList({ activeSessionId, onSelectReview, onStartReview, refreshToken }: ReviewsListProps) {
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

      <div className="flex flex-col gap-2 overflow-y-auto">
        {loading && reviews.length === 0 && <p className="px-1 text-xs text-text-muted">Loading reviews…</p>}
        {!loading && reviews.length === 0 && (
          <p className="px-1 text-xs text-text-muted">No reviews yet — start one above.</p>
        )}
        {reviews.map((review) => (
          <ReviewCard
            key={review.session_id}
            review={review}
            active={review.session_id === activeSessionId}
            onSelect={() => onSelectReview(review.session_id)}
          />
        ))}
      </div>
    </aside>
  )
}
