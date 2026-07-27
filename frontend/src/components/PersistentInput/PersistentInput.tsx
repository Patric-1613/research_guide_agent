import { useState, type KeyboardEvent } from 'react'

export type InputMode = 'curate' | 'generate-report' | 'chat'

export function modeForStage(stage: string, hasReport: boolean): InputMode {
  if (stage === 'curate') return 'curate'
  return hasReport ? 'chat' : 'generate-report'
}

interface PersistentInputProps {
  stage: string
  hasReport: boolean
  pendingWebOffer: { question: string } | null
  disabled: boolean
  stagedPickCount: number
  onSubmitPicks: () => void
  onGenerateReport: () => void
  onSendMessage: (message: string) => void
}

const PLACEHOLDERS: Record<InputMode, string> = {
  curate: "Refine what you're looking for, or just pick from the pool on the right...",
  'generate-report': 'Hit Send to generate your report...',
  chat: 'Ask a question about the selected papers...',
}

export function PersistentInput({
  stage,
  hasReport,
  pendingWebOffer,
  disabled,
  stagedPickCount,
  onSubmitPicks,
  onGenerateReport,
  onSendMessage,
}: PersistentInputProps) {
  const [text, setText] = useState('')
  const mode = modeForStage(stage, hasReport)

  // The single dispatch point every send in this component funnels
  // through, regardless of whether it was triggered by typing + Enter/
  // Send, or one of the Yes/No web-offer buttons below -- there is no
  // second, offer-specific code path calling chat_turn() differently.
  function dispatch(message: string) {
    if (mode === 'curate') {
      onSubmitPicks()
    } else if (mode === 'generate-report') {
      onGenerateReport()
    } else {
      onSendMessage(message)
    }
  }

  function handleSend() {
    const trimmed = text.trim()
    if (mode === 'chat' && !trimmed) return
    dispatch(trimmed)
    setText('')
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') handleSend()
  }

  return (
    <div className="border-t border-border bg-panel p-3">
      {pendingWebOffer && mode === 'chat' && (
        <div className="mb-2 flex items-center gap-2">
          <span className="text-xs text-text-secondary">{pendingWebOffer.question ? 'Search the web for more on this?' : ''}</span>
          <button
            type="button"
            data-testid="web-offer-yes"
            onClick={() => dispatch('yes')}
            disabled={disabled}
            className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-fg disabled:opacity-40"
          >
            Yes
          </button>
          <button
            type="button"
            data-testid="web-offer-no"
            onClick={() => dispatch('no')}
            disabled={disabled}
            className="rounded-md border border-border px-2.5 py-1 text-xs text-text-secondary disabled:opacity-40"
          >
            No
          </button>
        </div>
      )}
      <div className="flex items-center gap-2">
        {mode === 'curate' && stagedPickCount > 0 && (
          <span className="shrink-0 rounded-full bg-accent-soft px-2 py-1 text-xs font-medium text-accent">
            {stagedPickCount} added
          </span>
        )}
        <input
          data-testid="persistent-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || mode !== 'chat'}
          placeholder={PLACEHOLDERS[mode]}
          className="flex-1 rounded-md border border-border bg-panel-alt px-3 py-2 text-sm text-text outline-none focus:border-accent disabled:opacity-60"
        />
        <button
          type="button"
          data-testid="persistent-input-send"
          onClick={handleSend}
          disabled={disabled}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  )
}
