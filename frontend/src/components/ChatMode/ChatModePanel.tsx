import { useEffect, useRef, useState, type KeyboardEvent, type UIEvent } from 'react'
import { ChevronDown, ChevronUp, CircleStop } from 'lucide-react'
import type { ChatStreamPhase, CurationStateResponse } from '../../types'
import type { AddToReportResult, ChatSearchMeta } from '../../hooks/useCurationSession'
import { ChatMessage } from '../TurnFeed/ChatMessage'
import { ChatMessageRow, isEligibleForAddToReport } from '../TurnFeed/ChatMessageRow'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { ReferencesList } from '../shared/ReferencesList'

// Usage Protection M2.3 Part D: mirrors research_agent/config/limits.py's
// max_text_length -- preventative UX only.
const MAX_MESSAGE_LENGTH = 2000

// Usage Protection M4.2B: concise, user-facing labels for the backend's
// own internal phase vocabulary (research_agent/chat_streaming.py's
// ChatStreamPhase) -- never the raw identifier itself. Only ever shown
// for a phase this component actually receives; no entry here implies
// every turn shows all six (most show a small subset -- see
// research_agent/curation_chat_streaming.py's own docstring for exactly
// when each one fires).
const CHAT_STREAM_PHASE_LABELS: Record<ChatStreamPhase, string> = {
  preparing_context: 'Preparing context',
  summarizing_history: 'Summarizing conversation',
  checking_relevance: 'Checking sources',
  searching_web: 'Searching the web',
  generating: 'Writing answer',
  saving: 'Saving',
}

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
  // Usage Protection M4.2B: streaming lifecycle -- owned by
  // useCurationSession, passed straight through. onSendMessage above is
  // unchanged (CurationWorkspacePage now wires it to the streaming
  // action; this panel doesn't need to know that).
  chatStreamActive: boolean
  chatStreamPhase: ChatStreamPhase | null
  chatStreamText: string
  chatStreamSyncFailed: boolean
  onCancelChatStream: () => void
  onRetrySync: () => void
}

// Chat mode's center panel shows ONLY the conversation -- no paper pool
// alongside it -- per the explicit ask that once in chat mode, the
// candidate/paper-pool UI should disappear entirely.
export function ChatModePanel({
  state, disabled, onSendMessage, lastSearchMeta, onDeleteExchanges, reportPossiblyStale,
  onAddExchangesToReport, lastAddToReportResult, onEditExchange, onDismissReportStaleWarning,
  chatStreamActive, chatStreamPhase, chatStreamText, chatStreamSyncFailed, onCancelChatStream, onRetrySync,
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
  // UXH.3: the persistent text input -- also the focus-restoration target
  // once a stream settles (see the effect below). Send and Stop are both
  // plain <button> elements at the same ternary slot, so React reuses the
  // one underlying DOM node across the swap (same host type) -- a click
  // that focused Send keeps that focus straight through it becoming
  // Stop, no loss there. The real gap is the persistent input itself:
  // the common keyboard flow (type a message, press Enter) leaves focus
  // ON the input, which then goes from enabled to `disabled` for the
  // whole in-flight window -- real browsers blur a focused control the
  // instant it becomes disabled (HTML spec), dropping focus to
  // document.body, and nothing previously restored it once the input
  // re-enabled. Nothing here acts on the ACTIVE-start transition itself:
  // the input is legitimately disabled then, so there's nothing useful to
  // focus back to until the round trip actually finishes.
  const inputRef = useRef<HTMLInputElement>(null)
  // UXH.1b: the scrollable transcript itself -- read on every scroll event
  // (handleScroll below) to track whether the user is currently near its
  // bottom edge, so the auto-scroll effect further below can tell "the
  // user is following along" apart from "the user deliberately scrolled up
  // to read earlier turns" and only force-scroll in the former case.
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  // UXH.1b: updated synchronously on every real scroll event (including
  // the ones our own scrollIntoView calls below produce), NOT on a timer
  // and NOT recomputed inside the auto-scroll effect itself -- computing
  // it there would already see the just-appended content's height and
  // wrongly conclude "not near bottom" even when the user was exactly at
  // the bottom a moment ago. Starts true so the first render (and a fresh
  // session) auto-scrolls same as before this phase.
  const pinnedToBottomRef = useRef(true)
  const NEAR_BOTTOM_THRESHOLD_PX = 80

  function handleScroll(e: UIEvent<HTMLDivElement>) {
    const el = e.currentTarget
    pinnedToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM_THRESHOLD_PX
  }

  // curation-chat-references disclosure (UXH.1b): collapsed by default,
  // and reset to collapsed whenever the open session changes -- see the
  // session-keyed effect below. Otherwise persists across re-renders
  // (including new references arriving mid-session) since it's plain
  // state, satisfying "preserve the user's current open/closed choice
  // within the same session" without any extra code.
  const [referencesOpen, setReferencesOpen] = useState(false)
  const referencesPanelId = 'chat-references-panel'

  useEffect(() => {
    pinnedToBottomRef.current = true
    setReferencesOpen(false)
  }, [state.session_id])

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
      if (next.has(exchangeId)) {
        next.delete(exchangeId)
        return next
      }
      // Usage Protection M2.3 Part D: bulk delete/add-to-report both
      // submit exchange_ids as one IdList-constrained request field
      // (research_agent/api_app/schemas.py, cap 30) -- a session can
      // have up to 100 chat turns, so selecting past 30 here is a real,
      // reachable case, unlike the current single-batch pick UI. Purely
      // preventative: the backend re-validates the same cap regardless.
      if (next.size >= 30) return next
      next.add(exchangeId)
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

  // UXH.1b: reacts to every visible change in the streaming turn -- the
  // pending user bubble, the stream becoming active at all, each phase
  // transition, each accumulated delta, AND the canonical reload that
  // replaces the temporary row with real chat_history -- not just
  // chat_history.length/pendingMessage as before. That gap was the root
  // cause of the temporary status/answer row rendering invisibly below
  // the fold on any transcript that already needed scrolling: nothing
  // ever told the viewport to follow it into view until the round trip
  // had already finished. Only actually scrolls when the user was already
  // near the bottom (pinnedToBottomRef, tracked by handleScroll above) --
  // never fights a deliberate scroll up to re-read earlier turns.
  useEffect(() => {
    if (pinnedToBottomRef.current) {
      bottomRef.current?.scrollIntoView({ block: 'end' })
    }
  }, [state.chat_history.length, pendingMessage, chatStreamActive, chatStreamPhase, chatStreamText])

  // UXH.3: restores keyboard focus to the text input once a stream
  // settles (completes OR is cancelled) -- chatStreamActive going
  // true -> false is the one signal common to both, since cancellation
  // still routes through the same clearChatStreamPreview() that a normal
  // completion does (see useCurationSession's sendChatMessageStreaming).
  // Only acts when focus actually fell to document.body -- the same
  // "operation's own control was what held focus, and it just got
  // removed/disabled out from under it" signal the Send/Stop button swap
  // and the input's own `disabled` toggle both produce natively. If the
  // user had deliberately moved focus somewhere else first (e.g. the
  // references toggle, which stays enabled during a stream), activeElement
  // is that element, not body, and this intentionally does nothing --
  // never steals focus from a control the user chose. wasStreamActiveRef
  // resets with the component (a session switch away from Chat mode
  // unmounts this panel entirely -- see CurationWorkspacePage's own
  // conditional render), so a stale prior session can never trigger a
  // focus change in a freshly mounted instance.
  const wasStreamActiveRef = useRef(false)
  useEffect(() => {
    const wasActive = wasStreamActiveRef.current
    wasStreamActiveRef.current = chatStreamActive
    if (wasActive && !chatStreamActive && document.activeElement === document.body) {
      inputRef.current?.focus()
    }
  }, [chatStreamActive])

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
    // Usage Protection M2.3 Part D: mirrors research_agent/config/
    // limits.py's max_text_length -- preventative UX only, the backend
    // (CurationChatRequest.message) remains authoritative. The
    // maxLength attribute on the input below already prevents typing/
    // pasting past this in practice; this guard is a defensive backstop
    // for the submit action itself, matching the task's own wording.
    if (!trimmed || trimmed.length > MAX_MESSAGE_LENGTH) return
    setText('')
    void dispatchMessage(trimmed)
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') handleSend()
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        data-testid="chat-scroll-container"
        className="flex flex-1 flex-col gap-2 overflow-y-auto px-4 py-3"
      >
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
        {/* Usage Protection M4.2B: the one temporary assistant response
            area for an in-flight stream. Renders plain text ONLY, never
            through renderContentWithMarkers -- citation rendering is
            deliberately withheld until the canonical reload after
            completed -> done lands the real ChatTurn (see
            useCurationSession's own sendChatMessageStreaming docstring).
            min-h keeps the bubble's height stable as the content swaps
            between a short phase label and the (possibly longer)
            streamed answer text, so the surrounding layout doesn't jump. */}
        {chatStreamActive && (
          <div data-testid="chat-stream-response" className="flex items-start justify-start gap-1.5">
            <div
              role="status"
              aria-live="polite"
              className="min-h-[2.25rem] max-w-[80%] rounded-lg border border-border bg-panel-alt px-3 py-2 text-sm text-text"
            >
              {chatStreamText ? (
                <span data-testid="chat-stream-text" className="whitespace-pre-wrap break-words">
                  {chatStreamText}
                </span>
              ) : (
                <span data-testid="chat-stream-phase" className="text-text-muted">
                  {chatStreamPhase ? CHAT_STREAM_PHASE_LABELS[chatStreamPhase] : 'Thinking…'}
                </span>
              )}
            </div>
          </div>
        )}
        {chatStreamSyncFailed && (
          <p
            data-testid="chat-stream-sync-failed"
            className="flex items-center justify-center gap-2 text-center text-xs text-text-muted"
          >
            <span>Got the reply, but couldn't sync the conversation afterward.</span>
            <button
              type="button"
              data-testid="chat-stream-sync-retry"
              onClick={() => void onRetrySync()}
              className="shrink-0 underline decoration-dotted hover:text-accent"
            >
              Retry
            </button>
          </p>
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
        {/* UXH.1b: collapsed by default -- the previously always-expanded
            panel routinely occupied a large share of the viewport above
            the composer. Toggle state is plain component state (see
            referencesOpen above): it persists as new references arrive
            within the same session, and is reset to collapsed only when
            the open session itself changes. */}
        {state.chat_references.length > 0 && (
          <div className="mb-2">
            <button
              type="button"
              data-testid="chat-references-toggle"
              onClick={() => setReferencesOpen((open) => !open)}
              aria-expanded={referencesOpen}
              aria-controls={referencesPanelId}
              className="flex w-full items-center justify-between rounded-md border border-border bg-panel-alt px-2.5 py-1.5 text-xs text-text-secondary hover:border-accent hover:text-accent"
            >
              <span>References ({state.chat_references.length})</span>
              {referencesOpen ? (
                <ChevronUp className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              )}
            </button>
            {referencesOpen && (
              <div id={referencesPanelId} className="mt-1 max-h-48 overflow-y-auto">
                <ReferencesList
                  references={state.chat_references}
                  heading="Chat references"
                  sectionTestId="chat-references"
                  idPrefix="chat-ref"
                  entryTestIdPrefix="chat-reference"
                />
              </div>
            )}
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
            ref={inputRef}
            data-testid="persistent-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder="Ask a question about the selected papers..."
            maxLength={MAX_MESSAGE_LENGTH}
            className="flex-1 rounded-md border border-border bg-panel-alt px-3 py-2 text-sm text-text outline-none focus:border-accent disabled:opacity-60"
          />
          {/* Usage Protection M4.2B: Send and Stop share this one slot --
              never both at once -- so the input row's own width/layout
              never shifts as a stream starts or ends. */}
          {chatStreamActive ? (
            <button
              type="button"
              data-testid="chat-stream-stop"
              onClick={onCancelChatStream}
              aria-label="Stop generating"
              title="Stop generating"
              className="flex items-center justify-center rounded-md border border-border px-4 py-2 text-sm font-medium text-text-secondary hover:border-danger hover:text-danger"
            >
              <CircleStop className="h-4 w-4" aria-hidden="true" />
            </button>
          ) : (
            <button
              type="button"
              data-testid="persistent-input-send"
              onClick={handleSend}
              disabled={disabled || !text.trim() || text.trim().length > MAX_MESSAGE_LENGTH}
              className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-40"
            >
              Send
            </button>
          )}
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
