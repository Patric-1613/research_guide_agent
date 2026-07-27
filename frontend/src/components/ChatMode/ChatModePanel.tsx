import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import type { CurationStateResponse } from '../../api/types'
import { ChatMessage } from '../TurnFeed/ChatMessage'

interface ChatModePanelProps {
  state: CurationStateResponse
  disabled: boolean
  onSendMessage: (message: string) => void
}

// Chat mode's center panel shows ONLY the conversation -- no paper pool
// alongside it -- per the explicit ask that once in chat mode, the
// candidate/paper-pool UI should disappear entirely.
export function ChatModePanel({ state, disabled, onSendMessage }: ChatModePanelProps) {
  const [text, setText] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [state.chat_history.length])

  const offerLabel = state.pending_web_offer
    ? 'Search the web for more on this?'
    : state.pending_report_update
      ? 'Update the report to include the newly approved source(s)?'
      : null

  function handleSend() {
    const trimmed = text.trim()
    if (!trimmed) return
    onSendMessage(trimmed)
    setText('')
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') handleSend()
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex flex-1 flex-col gap-2 overflow-y-auto px-4 py-3">
        {state.chat_history.length === 0 && (
          <p className="text-center text-sm text-text-muted">Report ready. Ask a question about the selected papers below.</p>
        )}
        {state.chat_history.map((turn, i) => (
          <ChatMessage key={i} turn={turn} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border bg-panel p-3">
        {offerLabel && (
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs text-text-secondary">{offerLabel}</span>
            <button
              type="button"
              data-testid="web-offer-yes"
              onClick={() => onSendMessage('yes')}
              disabled={disabled}
              className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-fg disabled:opacity-40"
            >
              Yes
            </button>
            <button
              type="button"
              data-testid="web-offer-no"
              onClick={() => onSendMessage('no')}
              disabled={disabled}
              className="rounded-md border border-border px-2.5 py-1 text-xs text-text-secondary disabled:opacity-40"
            >
              No
            </button>
          </div>
        )}
        <div className="flex items-center gap-2">
          <input
            data-testid="persistent-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Ask a question about the selected papers..."
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
    </div>
  )
}
