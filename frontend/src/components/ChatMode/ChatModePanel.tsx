import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import type { CurationStateResponse } from '../../types'
import type { AddToReportResult, ChatSearchMeta } from '../../hooks/useCurationSession'
import { ChatMessage } from '../TurnFeed/ChatMessage'
import { ChatMessageRow, isEligibleForAddToReport } from '../TurnFeed/ChatMessageRow'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { ReferencesList } from '../shared/ReferencesList'

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
  // curation-chat-delete Phase 3
  onDeleteExchanges: (exchangeIds: string[]) => Promise<void>
  reportPossiblyStale: boolean
  // curation-chat-add-to-report Phase 4
  onAddExchangesToReport: (exchangeIds: string[]) => Promise<void>
  lastAddToReportResult: AddToReportResult | null
  // curation-chat-edit Phase 5
  onEditExchange: (exchangeId: string, question: string) => Promise<void>
  // chat-ux-polish Phase A: lets the user dismiss the stale-report
  // warning explicitly, instead of it only ever going away when another
  // delete/edit response happens to override it.
  onDismissReportStaleWarning: () => void
}

// Chat mode's center panel shows ONLY the conversation -- no paper pool
// alongside it -- per the explicit ask that once in chat mode, the
// candidate/paper-pool UI should disappear entirely.
export function ChatModePanel({
  state, disabled, onSendMessage, lastSearchMeta, onDeleteExchanges, reportPossiblyStale,
  onAddExchangesToReport, lastAddToReportResult, onEditExchange, onDismissReportStaleWarning,
}: ChatModePanelProps) {
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

  // curation-chat-select Phase 2 / curation-chat-delete Phase 3 /
  // curation-chat-add-to-report Phase 4 / curation-chat-edit Phase 5:
  // select mode, delete, add-to-report, and edit are all wired up now.
  // Selected by exchange_id (Phase 1's shared id linking a question+
  // answer pair), not array index, so it stays correct regardless of how
  // the underlying list is later re-rendered. Session-local like
  // everything else in this panel -- resets on refresh.
  const [selectMode, setSelectMode] = useState(false)
  const [selectedExchangeIds, setSelectedExchangeIds] = useState<Set<string>>(new Set())
  // chat-ux-polish Phase A: in-app dialog replaces window.confirm() for
  // bulk delete -- same trigger/cancel-does-nothing behavior, just
  // styled. There's no single row to anchor this one to, so it's owned
  // here rather than in ChatMessageRow.
  const [showBulkDeleteConfirm, setShowBulkDeleteConfirm] = useState(false)

  // curation-chat-add-to-report Phase 4: which exchange_ids are currently
  // eligible, derived fresh from state.chat_history every render (same
  // rule ChatMessageRow's own menu item uses, via the shared
  // isEligibleForAddToReport helper, so the two can never disagree).
  const eligibleForReportExchangeIds = new Set(
    state.chat_history.filter(isEligibleForAddToReport).map((t) => t.exchange_id as string),
  )
  const hasEligibleSelection = [...selectedExchangeIds].some((id) => eligibleForReportExchangeIds.has(id))

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

  // curation-chat-delete Phase 3: confirmation lives at the click site --
  // ChatMessageRow's own Delete menu item already confirmed before
  // calling this, so this is single-exchange delete unconditionally.
  // Bulk delete confirms here instead, since there's no single row to
  // anchor the prompt to. Either way, onDeleteExchanges (useCurationSession's
  // runAction) swallows its own errors into the shared error banner and
  // never rejects -- selection is cleared/select mode exited afterward
  // regardless of success or failure, matching this app's existing
  // delete-review pattern (ReviewsList) of not needing a separate
  // success/failure branch here.
  async function handleDeleteExchange(exchangeId: string) {
    await onDeleteExchanges([exchangeId])
    setSelectedExchangeIds((prev) => {
      if (!prev.has(exchangeId)) return prev
      const next = new Set(prev)
      next.delete(exchangeId)
      return next
    })
  }

  function handleBulkDeleteClick() {
    if (selectedExchangeIds.size === 0) return
    setShowBulkDeleteConfirm(true)
  }

  async function confirmBulkDelete() {
    setShowBulkDeleteConfirm(false)
    const ids = Array.from(selectedExchangeIds)
    await onDeleteExchanges(ids)
    handleClearSelection()
  }

  // curation-chat-add-to-report Phase 4: no confirmation prompt (additive,
  // not destructive). onAddExchangesToReport (useCurationSession's
  // runAction) only reaches loadState() -- and therefore only updates
  // state.chat_history's added_to_report/badges -- on a CONFIRMED backend
  // success; a thrown error is caught by runAction itself, surfaced via
  // the shared error banner, and state is left completely untouched, so
  // badges are never greyed optimistically.
  async function handleAddToReport(exchangeId: string) {
    await onAddExchangesToReport([exchangeId])
    setSelectedExchangeIds((prev) => {
      if (!prev.has(exchangeId)) return prev
      const next = new Set(prev)
      next.delete(exchangeId)
      return next
    })
  }

  async function handleBulkAddToReport() {
    const ids = Array.from(selectedExchangeIds)
    if (ids.length === 0) return
    await onAddExchangesToReport(ids)
    handleClearSelection()
  }

  // curation-chat-edit Phase 5: the prompt/cancel/blank handling all
  // happens at the click site in ChatMessageRow -- this only ever
  // receives a real, non-blank question. Selection is cleared
  // unconditionally on completion (not just the edited id): truncation
  // can invalidate an arbitrary number of previously-selected exchange
  // ids (everything after the edited one), not just the edited exchange
  // itself, so there's no single id to surgically deselect here.
  async function handleEditExchange(exchangeId: string, question: string) {
    await onEditExchange(exchangeId, question)
    handleClearSelection()
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
              onDelete={(exchangeId) => void handleDeleteExchange(exchangeId)}
              onAddToReport={(exchangeId) => void handleAddToReport(exchangeId)}
              onEdit={(exchangeId, question) => void handleEditExchange(exchangeId, question)}
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
        {/* report-quality Phase R3.2 Chunk 3: chat's own compact
            References panel, just above the chat input -- renders
            nothing at all when state.chat_references is empty (see
            ReferencesList), so a chat with no cited papers/web sources
            yet looks exactly like it did before this phase. Numbering is
            independent from the report's own References -- see
            derive_chat_references' own docstring -- so a marker here
            resolving to [2] carries no relationship to a [2] in the
            report. */}
        {state.chat_references.length > 0 && (
          <div className="mb-2">
            <ReferencesList
              references={state.chat_references}
              heading="Chat references"
              sectionTestId="chat-references"
              idPrefix="chat-ref"
              entryTestIdPrefix="chat-reference"
            />
          </div>
        )}
        {reportPossiblyStale && (
          <p data-testid="report-possibly-stale-warning" className="mb-2 flex items-center justify-center gap-2 text-center text-xs text-danger">
            <span>A deleted or edited exchange had been added to the report — the report may now be stale.</span>
            <button
              type="button"
              data-testid="dismiss-stale-warning"
              onClick={onDismissReportStaleWarning}
              className="shrink-0 underline decoration-dotted hover:text-danger/80"
            >
              Dismiss
            </button>
          </p>
        )}
        {lastAddToReportResult && (
          <p data-testid="add-to-report-success-note" className="mb-2 text-center text-xs italic text-text-muted">
            Added {lastAddToReportResult.addedCount} exchange{lastAddToReportResult.addedCount === 1 ? '' : 's'} (
            {lastAddToReportResult.sourceCount} source{lastAddToReportResult.sourceCount === 1 ? '' : 's'}) to the
            report.
          </p>
        )}
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
              onClick={handleBulkDeleteClick}
              disabled={disabled}
              className="rounded-md border border-border px-2.5 py-1 text-xs text-danger hover:border-danger disabled:cursor-not-allowed disabled:opacity-40"
            >
              Delete selected
            </button>
            <button
              type="button"
              data-testid="bulk-add-to-report"
              onClick={() => void handleBulkAddToReport()}
              disabled={disabled || !hasEligibleSelection}
              className="rounded-md border border-border px-2.5 py-1 text-xs text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              Add selected to report
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
      {showBulkDeleteConfirm && (
        <ConfirmDialog
          title="Delete selected exchanges"
          message={`Delete ${selectedExchangeIds.size} selected exchange${selectedExchangeIds.size === 1 ? '' : 's'}? This cannot be undone.`}
          confirmLabel="Delete"
          danger
          onConfirm={() => void confirmBulkDelete()}
          onCancel={() => setShowBulkDeleteConfirm(false)}
        />
      )}
    </div>
  )
}
