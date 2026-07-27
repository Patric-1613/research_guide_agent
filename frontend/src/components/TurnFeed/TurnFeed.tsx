import { useEffect, useRef } from 'react'
import type { CurationStateResponse } from '../../api/types'
import type { TurnEvent } from '../../hooks/useCurationSession'
import { TurnBlock, TurnDivider } from './TurnBlock'
import { ChatMessage } from './ChatMessage'

interface TurnFeedProps {
  state: CurationStateResponse
  turnEvents: TurnEvent[]
}

export function TurnFeed({ state, turnEvents }: TurnFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [turnEvents.length, state.chat_history.length])

  if (state.stage === 'curate') {
    return (
      <div className="flex flex-1 flex-col gap-1 overflow-y-auto px-4 py-3">
        {turnEvents.map((event) => (
          <TurnBlock key={event.turnNumber} event={event} />
        ))}
        {state.pending_batch && (
          <div>
            <TurnDivider turnNumber={turnEvents.length + 1} refilled={state.refilled} />
            <div className="flex items-start gap-2 rounded-lg border border-border bg-panel-alt p-3">
              <span className="shrink-0 rounded bg-card px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-text-secondary">
                AGENT
              </span>
              <p className="text-sm text-text-secondary">
                {state.refilled
                  ? `Searched for more candidates and found ${state.pending_batch.length} to show you this turn.`
                  : `Serving ${state.pending_batch.length} candidates from the pool already fetched — no new search needed.`}
                {' '}Add papers from the pool on the right, then hit Send to continue.
              </p>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    )
  }

  if (state.stage === 'synthesize' && !state.report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 px-4 text-center">
        <p className="text-sm text-text-secondary">
          Curation complete — {state.selected_papers.length} papers selected.
        </p>
        <p className="text-sm text-text-muted">Hit Send below to generate the literature review report.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col gap-2 overflow-y-auto px-4 py-3">
      {state.chat_history.length === 0 && (
        <p className="text-center text-sm text-text-muted">
          Report ready. Ask a question about the selected papers below.
        </p>
      )}
      {state.chat_history.map((turn, i) => (
        <ChatMessage key={i} turn={turn} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
