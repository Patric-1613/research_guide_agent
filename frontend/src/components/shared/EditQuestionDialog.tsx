import { useEffect, useState } from 'react'

// Usage Protection M2.3 Part D: mirrors research_agent/config/limits.py's
// max_text_length -- preventative UX only (CurationChatEditRequest.
// question remains backend-authoritative).
const MAX_QUESTION_LENGTH = 2000

interface EditQuestionDialogProps {
  initialQuestion: string
  onSave: (question: string) => void
  onCancel: () => void
}

// chat-ux-polish Phase A: replaces window.prompt('Edit your question:',
// turn.content) -- same net behavior (starts pre-filled with the existing
// question, Cancel does nothing, a blank submission never reaches
// onSave), just styled to match the app instead of a native browser
// prompt.
export function EditQuestionDialog({ initialQuestion, onSave, onCancel }: EditQuestionDialogProps) {
  const [value, setValue] = useState(initialQuestion)
  const trimmed = value.trim()
  const overLimit = trimmed.length > MAX_QUESTION_LENGTH

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onCancel])

  function handleSave() {
    if (!trimmed || overLimit) return // blank/oversized submit -- no API call
    onSave(trimmed)
  }

  return (
    <div
      data-testid="edit-dialog-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onCancel}
    >
      <div
        data-testid="edit-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="edit-dialog-title"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md rounded-lg border border-border bg-panel p-4 shadow-xl"
      >
        <h2 id="edit-dialog-title" className="text-sm font-semibold text-text">
          Edit your question
        </h2>
        <textarea
          data-testid="edit-dialog-textarea"
          autoFocus
          rows={3}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          maxLength={MAX_QUESTION_LENGTH}
          className="mt-3 w-full resize-none rounded-md border border-border bg-panel-alt px-3 py-2 text-sm text-text outline-none focus:border-accent"
        />
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            data-testid="edit-dialog-cancel"
            onClick={onCancel}
            className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent"
          >
            Cancel
          </button>
          <button
            type="button"
            data-testid="edit-dialog-save"
            onClick={handleSave}
            disabled={!trimmed || overLimit}
            className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-accent-fg disabled:cursor-not-allowed disabled:opacity-40"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}
