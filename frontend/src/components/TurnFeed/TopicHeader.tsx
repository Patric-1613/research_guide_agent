import { ProgressBar } from '../shared/ProgressBar'

interface TopicHeaderProps {
  topic: string
  selectedCount: number
  targetCount: number
}

export function TopicHeader({ topic, selectedCount, targetCount }: TopicHeaderProps) {
  return (
    <div className="flex flex-col gap-2 border-b border-border bg-panel px-4 py-3">
      <h1 className="truncate text-base font-semibold text-text">{topic}</h1>
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <ProgressBar value={selectedCount} max={targetCount} />
        </div>
        <span data-testid="progress-count" className="shrink-0 font-mono text-xs text-text-secondary">
          {selectedCount} of {targetCount} selected
        </span>
      </div>
    </div>
  )
}
