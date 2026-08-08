import { useEffect, useRef, useState } from 'react'
import type { ChatTurn } from '../../types'
import { ChatMessage } from './ChatMessage'
import { ConfirmDialog } from '../shared/ConfirmDialog'
import { EditQuestionDialog } from '../shared/EditQuestionDialog'

// curation-chat-add-to-report Phase 4: eligibility rule shared between
// this row's own menu item and ChatModePanel's bulk-button gate, so the
// two can never disagree. Assistant-only, real exchange_id, actually
// used web sources, not already added.
//
// chat-web-relevance-guardrails R7C: additionally requires
// web_relevance_verified !== false -- undefined/null (no web citation
// to judge, or a turn predating R7C) stays eligible for backward
// compatibility; only an explicit false (the citation came from a
// fail-open, unverified relevance check) disables it. Mirrors the
// backend's own select_eligible_exchanges_for_report gate exactly, so
// this pre-check and the server's real enforcement never disagree.
export function isEligibleForAddToReport(turn: ChatTurn): boolean {
  return (
    turn.role === 'assistant'
    && !!turn.exchange_id
    && !!turn.used_web_search
    && (turn.cited_web_articles?.length ?? 0) > 0
    && !turn.added_to_report
    && turn.web_relevance_verified !== false
  )
}

interface ChatMessageRowProps {
  turn: ChatTurn
  selectMode: boolean
  selected: boolean
  // curation-chat-select Phase 2: entering select mode from a message's own
  // "Select" menu item both turns select mode on AND pre-selects that
  // message's exchange -- there's no separate "turn select mode on with
  // nothing checked" action anywhere in this UI.
  onEnterSelectMode: (exchangeId: string) => void
  onToggleSelect: (exchangeId: string) => void
  // curation-chat-delete Phase 3: the confirm() prompt lives right here at
  // the click site (same convention as ReviewCard's own delete button),
  // not in the parent -- onDelete is only ever called after the user has
  // already confirmed.
  onDelete: (exchangeId: string) => void
  // curation-chat-add-to-report Phase 4: no confirmation prompt -- this is
  // additive, not destructive, unlike delete above.
  onAddToReport: (exchangeId: string) => void
  // curation-chat-edit Phase 5: called only with a non-blank, trimmed
  // question -- window.prompt's own cancel (null) and blank-submit cases
  // are both handled at the click site below, never reaching this prop.
  onEdit: (exchangeId: string, question: string) => void
}

// curation-chat-select Phase 2: wraps ChatMessage (unchanged since Phase 1
// -- the web badge/hint keep working exactly as before) with the
// per-message "..." action menu and, in select mode, a checkbox.
//
// Selection is tracked by exchange_id, the id Phase 1 already stamps on
// both entries of a question+answer pair. Entries that predate Phase 1
// have exchange_id === null -- per this phase's explicit preference, those
// are non-selectable (not given a client-side fallback id), shown as a
// disabled checkbox once select mode is on rather than hidden entirely, so
// it's clear WHY they can't be picked rather than just missing.
export function ChatMessageRow({
  turn, selectMode, selected, onEnterSelectMode, onToggleSelect, onDelete, onAddToReport, onEdit,
}: ChatMessageRowProps) {
  const [menuOpen, setMenuOpen] = useState(false)
  // chat-ux-polish Phase A: in-app dialogs replace window.confirm/
  // window.prompt -- same triggers, same cancel/blank-submit no-ops,
  // just rendered as styled overlays instead of native browser dialogs.
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)
  // curation-chat-menu-dismiss: each row's menu used to only close via its
  // own button (a toggle) -- opening a DIFFERENT row's menu, or clicking a
  // checkbox, or clicking blank space, never closed a still-open menu from
  // an earlier row, so two menus (or a menu and a checkbox) could visually
  // overlap. Listeners are only attached while this row's OWN menu is open
  // (not unconditionally on every row, every render) and torn down the
  // moment it closes.
  const menuContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menuOpen) return
    function handlePointerDown(e: MouseEvent) {
      if (menuContainerRef.current && !menuContainerRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    // mousedown, not click -- fires before the target's own click handler,
    // so a click on a DIFFERENT row's "..." button closes this row's menu
    // first, then that other button's own onClick still opens its menu
    // normally (no race, no double-toggle).
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [menuOpen])
  const exchangeId = turn.exchange_id ?? null
  const isSelectable = exchangeId !== null
  const isEligibleForReport = isEligibleForAddToReport(turn)
  // curation-chat-edit Phase 5: user messages only, and only ones with a
  // real exchange_id -- old pre-Phase-1 entries stay ineligible, same
  // rule as Select/Delete/Add to report all already apply.
  const isEligibleForEdit = turn.role === 'user' && isSelectable

  return (
    <div className="flex items-start gap-2">
      {selectMode && (
        <input
          type="checkbox"
          data-testid="exchange-select-checkbox"
          aria-label={isSelectable ? 'Select this exchange' : 'Not selectable (older message, no exchange id)'}
          disabled={!isSelectable}
          checked={isSelectable && selected}
          onChange={() => {
            if (exchangeId) onToggleSelect(exchangeId)
          }}
          className="mt-3 shrink-0 disabled:cursor-not-allowed disabled:opacity-40"
        />
      )}
      <div className="min-w-0 flex-1">
        <ChatMessage turn={turn} />
      </div>
      <div className="relative shrink-0" ref={menuContainerRef}>
        <button
          type="button"
          data-testid="message-menu-button"
          aria-label="Message actions"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((v) => !v)}
          className="mt-1 flex h-7 w-7 items-center justify-center rounded-full border border-border text-base font-bold leading-none text-text-secondary hover:border-accent hover:bg-panel-alt hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          ⋯
        </button>
        {menuOpen && (
          <div
            data-testid="message-menu"
            role="menu"
            className="absolute right-0 z-10 mt-1 w-48 rounded-md border border-border bg-panel py-1 text-xs shadow-lg"
          >
            <button
              type="button"
              role="menuitem"
              data-testid="message-menu-select"
              disabled={!isSelectable}
              onClick={() => {
                if (!exchangeId) return
                onEnterSelectMode(exchangeId)
                setMenuOpen(false)
              }}
              className="block w-full px-3 py-1.5 text-left text-text-secondary hover:bg-panel-alt disabled:cursor-not-allowed disabled:opacity-40"
            >
              Select
            </button>
            {/* curation-chat-select Phase 2, requirement 3: Edit only ever
                makes sense on the user-question side of an exchange -- the
                UI can distinguish that cleanly via turn.role, so it's
                simply not rendered at all on assistant messages, rather
                than shown-and-disabled everywhere. curation-chat-edit
                Phase 5: still requires a real exchange_id (old entries
                stay disabled). chat-ux-polish Phase A: opens an in-app
                EditQuestionDialog -- no inline editor yet. */}
            {turn.role === 'user' && (
              <button
                type="button"
                role="menuitem"
                data-testid="message-menu-edit"
                disabled={!isEligibleForEdit}
                onClick={() => {
                  setMenuOpen(false)
                  setShowEditDialog(true)
                }}
                className="block w-full px-3 py-1.5 text-left text-text-secondary hover:bg-panel-alt disabled:cursor-not-allowed disabled:text-text-muted disabled:opacity-50"
              >
                Edit
              </button>
            )}
            <button
              type="button"
              role="menuitem"
              data-testid="message-menu-delete"
              disabled={!isSelectable}
              onClick={() => {
                setMenuOpen(false)
                setShowDeleteConfirm(true)
              }}
              className="block w-full px-3 py-1.5 text-left text-danger hover:bg-panel-alt disabled:cursor-not-allowed disabled:text-text-muted disabled:opacity-50"
            >
              Delete
            </button>
            <button
              type="button"
              role="menuitem"
              data-testid="message-menu-add-to-report"
              disabled={!isEligibleForReport}
              onClick={() => {
                if (!exchangeId) return
                setMenuOpen(false)
                onAddToReport(exchangeId)
              }}
              className="block w-full px-3 py-1.5 text-left text-text-secondary hover:bg-panel-alt disabled:cursor-not-allowed disabled:text-text-muted disabled:opacity-50"
            >
              Add to report
            </button>
          </div>
        )}
      </div>
      {showDeleteConfirm && exchangeId && (
        <ConfirmDialog
          title="Delete exchange"
          message="Delete this exchange? This cannot be undone."
          confirmLabel="Delete"
          danger
          onConfirm={() => {
            setShowDeleteConfirm(false)
            onDelete(exchangeId)
          }}
          onCancel={() => setShowDeleteConfirm(false)}
        />
      )}
      {showEditDialog && exchangeId && (
        <EditQuestionDialog
          initialQuestion={turn.content}
          onSave={(question) => {
            setShowEditDialog(false)
            onEdit(exchangeId, question)
          }}
          onCancel={() => setShowEditDialog(false)}
        />
      )}
    </div>
  )
}
