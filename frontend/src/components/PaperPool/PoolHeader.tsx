import { ProgressBar } from '../shared/ProgressBar'

interface PoolHeaderProps {
  selectedCount: number
  targetCount: number
  reserveRemaining: number
}

export function PoolHeader({ selectedCount, targetCount, reserveRemaining }: PoolHeaderProps) {
  return (
    <div className="flex flex-col gap-2 border-b border-border p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
          Paper pool — {selectedCount}/{targetCount} target
        </h2>
      </div>
      <ProgressBar value={selectedCount} max={targetCount} />
      <p className="text-xs text-text-muted">
        {reserveRemaining > 0
          ? `${reserveRemaining} more candidate${reserveRemaining === 1 ? '' : 's'} already fetched — next turn won't need to search again.`
          : 'Pool is empty — the next turn will search again.'}
      </p>
    </div>
  )
}
