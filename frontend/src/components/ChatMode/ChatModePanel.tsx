import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import type { CurationStateResponse } from '../../types'
import type { ChatSearchMeta } from '../../hooks/useCurationSession'
import { ChatMessage } from '../TurnFeed/ChatMessage'
import { ChatMessageRow } from '../TurnFeed/ChatMessageRow'

interface ChatModePanelProps {
  state: CurationStateResponse
  disabled: boolean
  onSendMessage: (message: string) => Promise<void>
  // chat-ux-fixes bug 2: makes the outcome of accepting a web-search
  // offer visible -- previously the response carried this back but
  // nothing rendered it, so a search that ran and genuinely found
  // nothing useful for the question was indistinguishable from the
  // button having done nothing at all.
  lastSearchMeta: ChatSearchMeta | null
}

// Chat mode's center panel shows ONLY the conversation -- no paper pool
// alongside it -- per the explicit ask that once in chat mode, the
// candidate/paper-pool UI should disappear entirely.
export function ChatModePanel({ state, disabled, onSendMessage, lastSearchMeta }: ChatModePanelProps) {
  const [text, setText] = useState('')
  // chat-ux-fixes bug 3: onSendMessage awaits the FULL round trip
  // (chat_turn() plus a separate state reload) before state.chat_history
  // ever reflects the new turn -- without this, the user's OWN message
  // sat invisible for however long that took. Rendered as one extra
  // bubble past whatever state.chat_history currently holds; cleared the
  // instant onSendMessage resolves, by which point the real reload has
  // already landed the persisted version in state.chat_history itself
  // (same await chain), so there's no flicker/duplicate/gap between the
  // two. The input is disabled for the whole round trip regardless (see
  // `disabled` below), so only one message can ever be in flight at once
  // -- no need to handle overlapping optimistic messages.
  const [pendingMessage, setPendingMessage] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // curation-chat-select Phase 2: UI-foundation-only select mode -- no
  // delete/add-to-report behavior wired up yet (later phases), just the
  // selection mechanics themselves. Selected by exchange_id (Phase 1's
  // shared id linking a question+answer pair), not array index, so it
  // stays correct regardless of how the underlying list is later
  // re-rendered. Session-local like everything else in this panel --
  // resets on refresh.
  const [selectMode, setSelectMode] = useState(false)
  const [selectedExchangeIds, setSelectedExchangeIds] = useState<Set<string>>(new Set())

  function handleEnterSelectMode(exchangeId: string) {
    setSelectMode(true)
    setSelectedExchangeIds(new Set([exchangeId]))
  }

  function handleToggleSelect(exchangeId: string) {
    setSelectedExchangeIds((prev) => {
      const next = new Set(prev)
      if (next.has(exchangeId)) next.delete(exchangeId)
      else next.add(exchangeId)
      return next
    })
  }

  // No separate "exit select mode, keep mode on" control exists in this
  // phase's design -- Clear selection is the only way out, so it does
  // both: empties the selection AND turns select mode off, rather than
  // leaving checkboxes stuck on-screen with nothing left to act on.
  function handleClearSelection() {
    setSelectedExchangeIds(new Set())
    setSelectMode(false)
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [state.chat_history.length, pendingMessage])

  // curation-chat-metadata Phase 1: shown once, next to the FIRST
  // web-backed assistant answer only -- purely derived from chat_history
  // (the first matching index), not separate dismiss-tracking state, so
  // it can never reappear at a later web-backed message. "Session-local"
  // per spec: nothing here persists across a refresh, matching every
  // other client-only piece of state in this app (turnEvents,
  // lastChatSearchMeta) -- a refresh may show it again if chat_history
  // already had a web-backed answer before the refresh, which is
  // explicitly acceptable for this phase.
  const firstWebBackedIndex = state.chat_history.findIndex((t) => t.role === 'assistant' && t.used_web_search)

  const offerLabel = state.pending_web_offer
    ? 'Search the web for more on this?'
    : state.pending_report_update
      ? 'Update the report to include the newly approved source(s)?'
      : null

  async function dispatchMessage(message: string) {
    setPendingMessage(message)
    try {
      await onSendMessage(message)
    } catch {
      // onSendMessage's real implementation (useCurationSession's
      // runAction) already catches failures internally and surfaces them
      // via the shared `error` banner -- nothing further to do here, just
      // don't leave this fire-and-forget dispatch (see handleSend/the
      // offer buttons below, none of which await it) as an unhandled
      // rejection if it ever did throw.
    } finally {
      setPendingMessage(null)
    }
  }

  function handleSend() {
    const trimmed = text.trim()
    if (!trimmed) return
    setText('')
    void dispatchMessage(trimmed)
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
          <div key={i}>
            <ChatMessageRow
              turn={turn}
              selectMode={selectMode}
              selected={!!turn.exchange_id && selectedExchangeIds.has(turn.exchange_id)}
              onEnterSelectMode={handleEnterSelectMode}
              onToggleSelect={handleToggleSelect}
            />
            {i === firstWebBackedIndex && (
              <p data-testid="web-metadata-hint" className="mt-1 text-center text-xs italic text-text-muted">
                This answer used web sources — report-inclusion controls will be added to the message menu in a
                future update.
              </p>
            )}
          </div>
        ))}
        {lastSearchMeta && (
          <p data-testid="web-search-meta-note" className="text-center text-xs italic text-text-muted">
            {lastSearchMeta.newWebArticlesFound
              ? `Searched the web and found ${lastSearchMeta.newWebArticlesFound} new source${lastSearchMeta.newWebArticlesFound === 1 ? '' : 's'}.`
              : "Searched the web, but didn't find anything new."}
          </p>
        )}
        {pendingMessage !== null && (
          <div data-testid="pending-message">
            <ChatMessage turn={{ role: 'user', content: pendingMessage }} />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border bg-panel p-3">
        {selectedExchangeIds.size > 0 && (
          <div
            data-testid="bulk-action-bar"
            className="mb-2 flex items-center gap-2 rounded-md border border-border bg-panel-alt px-2.5 py-1.5"
          >
            <span data-testid="bulk-selected-count" className="text-xs text-text-secondary">
              {selectedExchangeIds.size} selected
            </span>
            <button
              type="button"
              data-testid="bulk-delete"
              disabled
              title="Coming soon"
              className="rounded-md border border-border px-2.5 py-1 text-xs text-text-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              Delete selected (Coming soon)
            </button>
            <button
              type="button"
              data-testid="bulk-add-to-report"
              disabled
              title="Coming soon"
              className="rounded-md border border-border px-2.5 py-1 text-xs text-text-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              Add selected to report (Coming soon)
            </button>
            <button
              type="button"
              data-testid="bulk-clear-selection"
              onClick={handleClearSelection}
              className="ml-auto rounded-md px-2.5 py-1 text-xs text-text-secondary underline decoration-dotted hover:text-accent"
            >
              Clear selection
            </button>
          </div>
        )}
        {offerLabel && (
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs text-text-secondary">{offerLabel}</span>
            <button
              type="button"
              data-testid="web-offer-yes"
              onClick={() => void dispatchMessage('yes')}
              disabled={disabled}
              className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-fg disabled:opacity-40"
            >
              Yes
            </button>
            <button
              type="button"
              data-testid="web-offer-no"
              onClick={() => void dispatchMessage('no')}
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
