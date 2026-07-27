import type { CurationReviewSummary } from '../../api/types'

interface ReviewCardProps {
  review: CurationReviewSummary
  active: boolean
  onSelect: () => void
}

function statusLabel(review: CurationReviewSummary): { text: string; tone: 'muted' | 'accent' | 'success' } {
  if (review.has_report && review.has_chat) return { text: 'Report + Chat', tone: 'success' }
  if (review.has_report) return { text: 'Report', tone: 'accent' }
  if (review.stage === 'curate') return { text: 'Curating', tone: 'muted' }
  return { text: 'Ready for report', tone: 'accent' }
}

const toneClasses: Record<string, string> = {
  muted: 'bg-panel-alt text-text-secondary',
  accent: 'bg-accent-soft text-accent',
  success: 'bg-success-soft text-success',
}

export function ReviewCard({ review, active, onSelect }: ReviewCardProps) {
  const status = statusLabel(review)
  return (
    <button
      type="button"
      data-testid={`review-card-${review.session_id}`}
      onClick={onSelect}
      className={`w-full rounded-lg border p-3 text-left transition-colors ${
        active ? 'border-accent bg-accent-soft' : 'border-border bg-card hover:border-text-muted'
      }`}
    >
      <p className="truncate text-sm font-medium text-text">{review.topic}</p>
      <div className="mt-1.5 flex items-center justify-between">
        <span className="text-xs text-text-secondary">
          {review.selected_count} of {review.target_count} papers
        </span>
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${toneClasses[status.tone]}`}>
          {status.text}
        </span>
      </div>
    </button>
  )
}
