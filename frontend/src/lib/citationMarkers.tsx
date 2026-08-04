import type { ReactNode } from 'react'

// report-quality Phase R1: report prose (and, since R3.2 Chunk 3, chat
// content too) carries inline numbered markers like [1], [2], [3] (see
// research_agent/report.py's build_references_and_renumber /
// research_agent/curation_chat.py's derive_chat_references) -- this
// splits on them and renders each one as a clickable anchor jumping to
// its entry in the relevant references list, rather than as plain,
// inert text. Content with no markers at all just renders as one plain
// segment, unchanged.
const MARKER_RE = /(\[\d+\])/g

// report-quality Phase R3.2 Chunk 3: extracted from ReportModePanel so
// ChatMessage can reuse it. refIdPrefix lets a caller point markers at
// its own references list's anchors ("ref" for report sections --
// report-references' own #ref-N anchors -- "chat-ref" for chat
// messages) without changing anything else; report's own call sites
// omit it and get byte-identical output to before this extraction.
export function renderContentWithMarkers(content: string, refIdPrefix: string = 'ref'): ReactNode[] {
  return content.split(MARKER_RE).map((part, i) => {
    const match = /^\[(\d+)\]$/.exec(part)
    if (!match) return <span key={i}>{part}</span>
    const number = match[1]
    return (
      <a
        key={i}
        href={`#${refIdPrefix}-${number}`}
        data-testid={`citation-marker-${number}`}
        className="font-medium text-accent hover:underline"
      >
        [{number}]
      </a>
    )
  })
}
