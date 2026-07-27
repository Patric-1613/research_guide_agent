import type { TurnEvent } from '../../hooks/useCurationSession'

function agentMessage(event: Pick<TurnEvent, 'refilled' | 'batchSize'>): string {
  return event.refilled
    ? `Searched for more candidates and found ${event.batchSize} to show you this turn.`
    : `Served ${event.batchSize} candidates from the pool already fetched — no new search needed.`
}

interface TurnDividerProps {
  turnNumber: number
  refilled: boolean
}

export function TurnDivider({ turnNumber, refilled }: TurnDividerProps) {
  return (
    <div className="my-3 flex items-center gap-2 text-[11px] uppercase tracking-wide text-text-muted">
      <div className="h-px flex-1 bg-border-soft" />
      <span>
        Turn {turnNumber} — {refilled ? 'new search' : 'from existing pool (no new search)'}
      </span>
      <div className="h-px flex-1 bg-border-soft" />
    </div>
  )
}

interface TurnBlockProps {
  event: TurnEvent
}

export function TurnBlock({ event }: TurnBlockProps) {
  return (
    <div>
      <TurnDivider turnNumber={event.turnNumber} refilled={event.refilled} />
      <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-panel-alt p-3">
        <div className="flex items-start gap-2">
          <span className="shrink-0 rounded bg-card px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-text-secondary">
            AGENT
          </span>
          <p className="text-sm text-text-secondary">{agentMessage(event)}</p>
        </div>
        <p className="text-sm text-text">
          You selected <span className="font-medium text-accent">{event.pickedPaperIds.length}</span> of{' '}
          {event.batchSize}
        </p>
      </div>
    </div>
  )
}
