import { useState } from 'react'
import type { ChatTurn } from '../../types'
import { ChatMessage } from './ChatMessage'

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
export function ChatMessageRow({ turn, selectMode, selected, onEnterSelectMode, onToggleSelect, onDelete }: ChatMessageRowProps) {
  const [menuOpen, setMenuOpen] = useState(false)
  const exchangeId = turn.exchange_id ?? null
  const isSelectable = exchangeId !== null

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
      <div className="relative shrink-0">
        <button
          type="button"
          data-testid="message-menu-button"
          aria-label="Message actions"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((v) => !v)}
          className="mt-2 rounded-md px-1.5 py-0.5 text-xs text-text-muted hover:bg-panel-alt hover:text-text"
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
                than shown-and-disabled everywhere. */}
            {turn.role === 'user' && (
              <button
                type="button"
                role="menuitem"
                data-testid="message-menu-edit"
                disabled
                title="Coming soon"
                className="block w-full px-3 py-1.5 text-left text-text-muted disabled:cursor-not-allowed disabled:opacity-50"
              >
                Edit (Coming soon)
              </button>
            )}
            <button
              type="button"
              role="menuitem"
              data-testid="message-menu-delete"
              disabled={!isSelectable}
              onClick={() => {
                if (!exchangeId) return
                setMenuOpen(false)
                if (window.confirm('Delete this exchange?')) onDelete(exchangeId)
              }}
              className="block w-full px-3 py-1.5 text-left text-danger hover:bg-panel-alt disabled:cursor-not-allowed disabled:text-text-muted disabled:opacity-50"
            >
              Delete
            </button>
            <button
              type="button"
              role="menuitem"
              data-testid="message-menu-add-to-report"
              disabled
              title="Coming soon"
              className="block w-full px-3 py-1.5 text-left text-text-muted disabled:cursor-not-allowed disabled:opacity-50"
            >
              Add to report (Coming soon)
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
