import { useEffect } from 'react'

interface ConfirmDialogProps {
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  // Destructive styling (red confirm button) -- used for delete, not for
  // additive/non-destructive confirmations.
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}

// chat-ux-polish Phase A: replaces window.confirm() -- same net behavior
// (asks confirmation, Cancel does nothing, Confirm runs the action), just
// styled to match the app instead of a native, unstyled browser dialog.
export function ConfirmDialog({
  title, message, confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false, onConfirm, onCancel,
}: ConfirmDialogProps) {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onCancel])

  return (
    <div
      data-testid="confirm-dialog-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onCancel}
    >
      <div
        data-testid="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-sm rounded-lg border border-border bg-panel p-4 shadow-xl"
      >
        <h2 id="confirm-dialog-title" className="text-sm font-semibold text-text">
          {title}
        </h2>
        <p className="mt-2 text-sm text-text-secondary">{message}</p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            data-testid="confirm-dialog-cancel"
            onClick={onCancel}
            className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            data-testid="confirm-dialog-confirm"
            onClick={onConfirm}
            className={
              danger
                ? 'rounded-md border border-danger px-3 py-1.5 text-xs font-medium text-danger hover:bg-danger-soft'
                : 'rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-accent-fg hover:opacity-90'
            }
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
