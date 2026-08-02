import type { ChatTurn } from '../../types'

// curation-chat-metadata Phase 1: blue while used_web_search is true and
// added_to_report is still false; grey once added_to_report flips true
// (no code path sets that yet -- reachable only via fixture/test data
// until a later phase's "Add to report" action exists). No badge at all
// for user messages or paper-only assistant answers.
export function ChatMessage({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === 'user'
  const showWebBadge = !isUser && turn.used_web_search
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start gap-1.5`}>
      {showWebBadge && (
        <span
          data-testid="chat-web-badge"
          title={turn.added_to_report ? 'Used web sources — included in the report' : 'Used web sources'}
          aria-label={turn.added_to_report ? 'Used web sources, included in the report' : 'Used web sources'}
          className={`mt-2 shrink-0 text-sm ${turn.added_to_report ? 'text-text-muted' : 'text-blue-400'}`}
        >
          🌐
        </span>
      )}
      <div
        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
          isUser ? 'bg-accent text-accent-fg' : 'border border-border bg-panel-alt text-text'
        }`}
      >
        {turn.content}
      </div>
    </div>
  )
}
