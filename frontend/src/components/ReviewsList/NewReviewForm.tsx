import { useState, type FormEvent } from 'react'

interface NewReviewFormProps {
  onSubmit: (topic: string, targetCount: number) => void
  onCancel: () => void
}

export function NewReviewForm({ onSubmit, onCancel }: NewReviewFormProps) {
  const [topic, setTopic] = useState('')
  const [targetCount, setTargetCount] = useState(10)

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmed = topic.trim()
    if (!trimmed) return
    onSubmit(trimmed, targetCount)
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
      <label className="flex flex-col gap-1 text-xs text-text-secondary">
        Topic
        <input
          autoFocus
          data-testid="new-review-topic"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. parameter-efficient fine-tuning"
          className="rounded-md border border-border bg-panel-alt px-2 py-1.5 text-sm text-text outline-none focus:border-accent"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-text-secondary">
        Target paper count
        <input
          type="number"
          min={1}
          max={30}
          data-testid="new-review-target-count"
          value={targetCount}
          onChange={(e) => setTargetCount(Number(e.target.value))}
          className="rounded-md border border-border bg-panel-alt px-2 py-1.5 text-sm text-text outline-none focus:border-accent"
        />
      </label>
      <div className="mt-1 flex gap-2">
        <button
          type="submit"
          data-testid="new-review-start"
          disabled={!topic.trim()}
          className="flex-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-fg disabled:opacity-40"
        >
          Start
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md border border-border px-3 py-1.5 text-sm text-text-secondary hover:text-text"
        >
          Cancel
        </button>
      </div>
    </form>
  )
}
