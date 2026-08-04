import { Globe } from 'lucide-react'
import type { ChatTurn } from '../../types'
import { renderContentWithMarkers } from '../../lib/citationMarkers'

// curation-chat-metadata Phase 1 / chat-ux-polish Phase A: blue while
// used_web_search is true and added_to_report is still false; grey once
// added_to_report flips true. No badge at all for user messages or
// paper-only assistant answers.
//
// Root-cause note (Phase A): this used to be a literal 🌐 emoji glyph
// with a CSS text-color class on it. Color emoji are rendered from a
// fixed, embedded glyph table (COLR/CPAL or CBDT/CBLC) in every major
// browser -- the CSS `color` property has NO effect on them at all, so
// the blue/grey classes were silently doing nothing; the icon always
// rendered in its native emoji colors regardless of state. Using a real
// SVG icon (lucide-react's Globe, which draws with `currentColor`) is
// what actually makes the color change visible.
export function ChatMessage({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === 'user'
  const showWebBadge = !isUser && turn.used_web_search
  const addedToReport = !!turn.added_to_report
  const badgeLabel = addedToReport ? 'Web sources added to report' : 'Web sources used'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start gap-1.5`}>
      {showWebBadge && (
        <span
          data-testid="chat-web-badge"
          role="img"
          aria-label={badgeLabel}
          title={badgeLabel}
          className={`mt-2.5 h-4 w-4 shrink-0 ${addedToReport ? 'text-text-muted' : 'text-blue-400'}`}
        >
          <Globe className="h-4 w-4" aria-hidden="true" />
        </span>
      )}
      <div
        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
          isUser ? 'bg-accent text-accent-fg' : 'border border-border bg-panel-alt text-text'
        }`}
      >
        {/* report-quality Phase R3.2 Chunk 3: only assistant content ever
            carries [N] markers (derive_chat_references only rewrites
            assistant turns) -- a user's own typed message is rendered as
            plain text unconditionally, so literal bracketed numbers they
            type are never mistaken for citation links. */}
        {isUser ? turn.content : renderContentWithMarkers(turn.content, 'chat-ref')}
      </div>
    </div>
  )
}
