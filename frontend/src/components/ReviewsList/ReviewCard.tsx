import { useState } from 'react'
import type { MouseEvent } from 'react'
import type { CurationReviewSummary } from '../../types'
import { ConfirmDialog } from '../shared/ConfirmDialog'

interface ReviewCardProps {
  review: CurationReviewSummary
  active: boolean
  onSelect: () => void
  onDelete: (sessionId: string) => void
}

// Phase 8: stage only ever reaches "curate" or "synthesize" -- chat_turn()
// guards on stage alone, never on report existence, so a session CAN be
// chatted-in before a report is ever generated. Matched explicitly against
// every (has_report, has_chat) combination rather than an if/else chain
// with a catch-all, so a real state can never silently fall through to
// the wrong label the way "Ready for report" previously swallowed
// "chatted, no report yet".
export function statusLabel(review: CurationReviewSummary): { text: string; tone: 'muted' | 'accent' | 'success' } {
  if (review.stage === 'curate') return { text: 'Curating', tone: 'muted' }
  if (!review.has_report && !review.has_chat) return { text: 'Ready for report', tone: 'accent' }
  if (!review.has_report && review.has_chat) return { text: 'Chatted', tone: 'accent' }
  if (review.has_report && !review.has_chat) return { text: 'Report', tone: 'accent' }
  return { text: 'Report + Chat', tone: 'success' }
}

const toneClasses: Record<string, string> = {
  muted: 'bg-panel-alt text-text-secondary',
  accent: 'bg-accent-soft text-accent',
  success: 'bg-success-soft text-success',
}

export function ReviewCard({ review, active, onSelect, onDelete }: ReviewCardProps) {
  const status = statusLabel(review)
  // chat-ux-polish Phase A parity: review deletion used window.confirm()
  // directly here, never migrated when chat delete moved onto the shared
  // in-app ConfirmDialog -- same triggers, same cancel-does-nothing
  // behavior, just a styled overlay instead of the native browser dialog.
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  function handleDeleteClick(e: MouseEvent) {
    // Stops the click from also bubbling into the card's own onSelect --
    // deleting a review must never simultaneously open it.
    e.stopPropagation()
    setShowDeleteConfirm(true)
  }

  return (
    // display: contents -- an invisible-to-layout wrapper, purely so
    // ConfirmDialog (a fixed-position overlay) can be rendered as a
    // SIBLING of the card button rather than nested inside it. Nesting it
    // inside would put its Confirm/Cancel <button>s inside this card's own
    // <button onClick={onSelect}>, which is invalid HTML and would need
    // fragile stopPropagation gymnastics on every dialog interaction to
    // stop a click from also selecting the review. `contents` keeps this
    // wrapper completely out of ReviewsList's flex layout -- the card
    // button behaves exactly as if it were still a direct flex child.
    <div className="contents">
      <button
        type="button"
        data-testid={`review-card-${review.session_id}`}
        onClick={onSelect}
        className={`group relative w-full rounded-lg border p-3 text-left transition-colors ${
          active ? 'border-accent bg-accent-soft' : 'border-border bg-card hover:border-text-muted'
        }`}
      >
        <button
          type="button"
          data-testid={`delete-review-${review.session_id}`}
          onClick={handleDeleteClick}
          aria-label={`Delete ${review.display_title}`}
          className="absolute right-2 top-2 flex h-5 w-5 items-center justify-center rounded-full text-text-muted opacity-0 hover:bg-danger-soft hover:text-danger group-hover:opacity-100"
        >
          ×
        </button>
        <p className="truncate pr-5 text-sm font-medium text-text">{review.display_title}</p>
        <div className="mt-1.5 flex items-center justify-between">
          <span className="text-xs text-text-secondary">
            {review.selected_count} of {review.target_count} papers
          </span>
          <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${toneClasses[status.tone]}`}>
            {status.text}
          </span>
        </div>
      </button>
      {showDeleteConfirm && (
        <ConfirmDialog
          title="Delete review"
          message={`Delete "${review.display_title}"? This permanently removes the saved review/session and cannot be undone.`}
          confirmLabel="Delete"
          danger
          onConfirm={() => {
            setShowDeleteConfirm(false)
            onDelete(review.session_id)
          }}
          onCancel={() => setShowDeleteConfirm(false)}
        />
      )}
    </div>
  )
}
