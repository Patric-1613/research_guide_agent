import { Globe } from 'lucide-react'
import type { ReferenceEntry } from '../../types'

interface ReferencesListProps {
  references: ReferenceEntry[]
  heading: string
  sectionTestId: string
  // report-quality Phase R3.2 Chunk 3: defaults reproduce report's own
  // pre-extraction anchor id / testid convention exactly (#ref-N /
  // reference-N / reference-web-icon-N). Chat passes its own distinct
  // prefixes (chat-ref / chat-reference) so its own anchors and testids
  // never collide with report's, even though report mode and chat mode
  // are never mounted at the same time today (CurationWorkspacePage
  // renders exactly one of them per workspaceMode).
  idPrefix?: string
  entryTestIdPrefix?: string
}

// report-quality Phase R1: the References list -- inline markers above
// (see lib/citationMarkers) link into this, by number. Renders nothing
// at all when there are no references (rather than an empty heading),
// which safely covers a genuinely reference-less report/chat and an old
// report where `references` is absent the same way.
//
// report-quality Phase R3.2 Chunk 3: extracted from ReportModePanel's
// own ReferencesSection so ChatModePanel's "Chat references" panel can
// reuse the exact same rendering (numbered [N], Globe icon for a web
// source, linked vs. plain citation text) instead of a third
// reimplementation.
export function ReferencesList({
  references, heading, sectionTestId, idPrefix = 'ref', entryTestIdPrefix = 'reference',
}: ReferencesListProps) {
  if (references.length === 0) return null

  return (
    <section data-testid={sectionTestId} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">{heading}</h3>
      <ol className="flex flex-col gap-1.5 text-sm text-text">
        {references.map((ref) => (
          <li
            key={ref.number} id={`${idPrefix}-${ref.number}`} data-testid={`${entryTestIdPrefix}-${ref.number}`}
            className="flex items-start gap-2"
          >
            <span className="shrink-0 text-text-muted">[{ref.number}]</span>
            {ref.kind === 'web' && (
              // Same shared numbering as papers -- this is purely a
              // subtle, secondary visual cue (not a second numbering
              // scheme, not a heavy pill) so a web source is
              // distinguishable from a peer-reviewed paper at a glance.
              <span
                data-testid={`${entryTestIdPrefix}-web-icon-${ref.number}`}
                role="img"
                aria-label="Web source"
                title="Web source"
                className="mt-0.5 shrink-0 text-text-muted"
              >
                <Globe className="h-3.5 w-3.5" aria-hidden="true" />
              </span>
            )}
            {ref.link_url ? (
              <a
                href={ref.link_url}
                target="_blank"
                rel="noreferrer"
                // Underlined by default (dotted, same convention as this
                // app's other secondary links, e.g. the stale-report
                // "Dismiss" control) -- a link that only changes color on
                // hover is easy to miss at rest, especially against this
                // dark theme's already-muted body text.
                className="rounded-sm text-accent underline decoration-dotted underline-offset-2 hover:text-accent-hover hover:decoration-solid focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                {ref.formatted}
              </a>
            ) : (
              <span>{ref.formatted}</span>
            )}
          </li>
        ))}
      </ol>
    </section>
  )
}
