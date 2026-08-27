import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Loader2, Plus, Trash2 } from 'lucide-react'
import type { ResearchLaneOut, SubmittedLane } from '../../types'

// Usage Protection M2.3 Part D: mirrors research_agent/config/limits.py's
// max_text_length (2,000 chars) -- preventative UX only, the backend
// remains authoritative (SearchRequest/CurationStartRequest.topic).
const MAX_TOPIC_LENGTH = 2000

// Research Lanes (RL5): client mirrors of the RL1 construction contract
// (research_agent/research_lanes.py) -- preventative UX only; the server
// re-validates every value before any admission/provider/persistence work.
const MAX_LANES = 4
const LANE_LABEL_MAX = 80
const LANE_QUESTION_MAX = 300
const LANE_QUERY_MAX = 2000

type ReviewMode = 'single' | 'lanes'

// A client-only editable row. `key` is a React list key, NOT a persisted
// lane identity -- the server mints the real lane_id on start.
interface DraftLane {
  key: string
  label: string
  question: string
  query: string
  enabled: boolean
}

interface NewReviewFormProps {
  // Single search calls this with exactly (topic, targetCount); lane mode
  // adds the third argument. Kept positional/optional so the existing
  // single-search call sites and tests are byte-identical.
  onSubmit: (topic: string, targetCount: number, lanes?: SubmittedLane[]) => void
  onCancel: () => void
  // Research Lanes (RL5): all optional -- absent/false means the segmented
  // control never renders and this form behaves exactly as it did before.
  researchLanesAvailable?: boolean
  laneSuggestions?: ResearchLaneOut[] | null
  laneSuggestionLoading?: boolean
  laneSuggestionError?: string | null
  onSuggestLanes?: (topic: string) => void
  onResetLaneSuggestions?: () => void
}

let draftLaneKeySeq = 0
function nextDraftLaneKey(): string {
  draftLaneKeySeq += 1
  return `draft-lane-${draftLaneKeySeq}`
}

function toDraftLane(lane: ResearchLaneOut): DraftLane {
  return { key: nextDraftLaneKey(), label: lane.label, question: lane.question, query: lane.query, enabled: lane.enabled }
}

function laneRowValid(lane: DraftLane): boolean {
  const label = lane.label.trim()
  const query = lane.query.trim()
  return (
    label.length > 0 && label.length <= LANE_LABEL_MAX &&
    query.length > 0 && query.length <= LANE_QUERY_MAX &&
    lane.question.trim().length <= LANE_QUESTION_MAX
  )
}

export function NewReviewForm({
  onSubmit,
  onCancel,
  researchLanesAvailable = false,
  laneSuggestions = null,
  laneSuggestionLoading = false,
  laneSuggestionError = null,
  onSuggestLanes,
  onResetLaneSuggestions,
}: NewReviewFormProps) {
  const [topic, setTopic] = useState('')
  const [targetCount, setTargetCount] = useState(10)
  const [mode, setMode] = useState<ReviewMode>('single')
  const [draftLanes, setDraftLanes] = useState<DraftLane[]>([])
  const trimmedTopic = topic.trim()
  const overLimit = trimmedTopic.length > MAX_TOPIC_LENGTH

  // Server suggestions are the canonical starting point for the editable
  // draft: whenever a fresh set arrives, replace the rows with it.
  const lastAppliedSuggestionsRef = useRef<ResearchLaneOut[] | null>(null)
  useEffect(() => {
    if (laneSuggestions && laneSuggestions !== lastAppliedSuggestionsRef.current) {
      lastAppliedSuggestionsRef.current = laneSuggestions
      setDraftLanes(laneSuggestions.slice(0, MAX_LANES).map(toDraftLane))
    }
  }, [laneSuggestions])

  function handleTopicChange(value: string) {
    if (value === topic) return
    setTopic(value)
    // Changing the topic invalidates any lanes designed for the old one.
    if (draftLanes.length > 0) setDraftLanes([])
    lastAppliedSuggestionsRef.current = null
    onResetLaneSuggestions?.()
  }

  function updateLane(key: string, patch: Partial<DraftLane>) {
    setDraftLanes((prev) => prev.map((lane) => (lane.key === key ? { ...lane, ...patch } : lane)))
  }

  function removeLane(key: string) {
    setDraftLanes((prev) => prev.filter((lane) => lane.key !== key))
  }

  function addLane() {
    setDraftLanes((prev) =>
      prev.length >= MAX_LANES
        ? prev
        : [...prev, { key: nextDraftLaneKey(), label: '', question: '', query: '', enabled: true }],
    )
  }

  const laneCountValid = draftLanes.length >= 1 && draftLanes.length <= MAX_LANES
  const hasEnabledLane = draftLanes.some((lane) => lane.enabled)
  const allRowsValid = draftLanes.every(laneRowValid)
  const lanesValid = laneCountValid && hasEnabledLane && allRowsValid

  const canStartSingle = !!trimmedTopic && !overLimit
  const canStartLanes = canStartSingle && lanesValid && !laneSuggestionLoading

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (mode === 'lanes') {
      if (!canStartLanes) return
      onSubmit(
        trimmedTopic,
        targetCount,
        draftLanes.map((lane) => ({
          label: lane.label.trim(),
          question: lane.question.trim(),
          query: lane.query.trim(),
          enabled: lane.enabled,
        })),
      )
      return
    }
    if (!canStartSingle) return
    onSubmit(trimmedTopic, targetCount)
  }

  function handleSuggest() {
    if (!trimmedTopic || overLimit || laneSuggestionLoading) return
    onSuggestLanes?.(trimmedTopic)
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3">
      {researchLanesAvailable && (
        <div role="group" aria-label="Search mode" className="flex rounded-md border border-border p-0.5 text-xs">
          <button
            type="button"
            data-testid="new-review-mode-single"
            aria-pressed={mode === 'single'}
            onClick={() => setMode('single')}
            className={`flex-1 rounded px-2 py-1 font-medium ${
              mode === 'single' ? 'bg-accent text-accent-fg' : 'text-text-secondary hover:text-text'
            }`}
          >
            Single search
          </button>
          <button
            type="button"
            data-testid="new-review-mode-lanes"
            aria-pressed={mode === 'lanes'}
            onClick={() => setMode('lanes')}
            className={`flex-1 rounded px-2 py-1 font-medium ${
              mode === 'lanes' ? 'bg-accent text-accent-fg' : 'text-text-secondary hover:text-text'
            }`}
          >
            Research lanes
          </button>
        </div>
      )}

      <label className="flex flex-col gap-1 text-xs text-text-secondary">
        Topic
        <input
          autoFocus
          data-testid="new-review-topic"
          value={topic}
          onChange={(e) => handleTopicChange(e.target.value)}
          placeholder="e.g. parameter-efficient fine-tuning"
          maxLength={MAX_TOPIC_LENGTH}
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

      {researchLanesAvailable && mode === 'lanes' && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <button
              type="button"
              data-testid="new-review-suggest-lanes"
              onClick={handleSuggest}
              disabled={!trimmedTopic || overLimit || laneSuggestionLoading}
              className="rounded-md border border-border px-2.5 py-1 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              {draftLanes.length > 0 ? 'Suggest lanes again' : 'Suggest lanes'}
            </button>
            {laneSuggestionLoading && (
              <span
                role="status"
                aria-live="polite"
                data-testid="lane-suggestion-status"
                className="flex items-center gap-1.5 text-xs text-text-muted"
              >
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                Designing research lanes…
              </span>
            )}
          </div>

          <p className="text-[11px] text-text-muted">
            Suggestions only design a search plan — no papers are searched until you start.
          </p>

          {laneSuggestionError && (
            <p
              role="alert"
              data-testid="lane-suggestion-error"
              className="rounded-md border border-danger/30 bg-danger-soft px-2 py-1.5 text-xs text-danger"
            >
              {laneSuggestionError}
            </p>
          )}

          <fieldset className="flex flex-col gap-1">
            <legend className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
              Research lanes ({draftLanes.length}/{MAX_LANES})
            </legend>
            {draftLanes.length === 0 && (
              <p className="text-[11px] text-text-muted">
                Use “Suggest lanes” above, or add lanes manually.
              </p>
            )}
            <div className="divide-y divide-border">
              {draftLanes.map((lane, index) => (
                  <div key={lane.key} data-testid={`lane-row-${index}`} className="flex flex-col gap-1.5 py-2">
                    <div className="flex items-center gap-2">
                      <input
                        data-testid={`lane-label-${index}`}
                        aria-label={`Lane ${index + 1} label`}
                        value={lane.label}
                        onChange={(e) => updateLane(lane.key, { label: e.target.value })}
                        placeholder="Label"
                        maxLength={LANE_LABEL_MAX}
                        className="min-w-0 flex-1 rounded-md border border-border bg-panel-alt px-2 py-1 text-sm text-text outline-none focus:border-accent"
                      />
                      <label className="flex shrink-0 items-center gap-1 text-[11px] text-text-secondary">
                        <input
                          type="checkbox"
                          data-testid={`lane-enabled-${index}`}
                          checked={lane.enabled}
                          onChange={(e) => updateLane(lane.key, { enabled: e.target.checked })}
                          className="h-3.5 w-3.5 rounded border-border"
                        />
                        Enabled
                      </label>
                      <button
                        type="button"
                        data-testid={`lane-remove-${index}`}
                        onClick={() => removeLane(lane.key)}
                        disabled={draftLanes.length <= 1}
                        aria-label={`Remove lane ${index + 1}`}
                        className="shrink-0 rounded p-1 text-text-muted hover:bg-danger-soft hover:text-danger disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                      </button>
                    </div>
                    <input
                      data-testid={`lane-question-${index}`}
                      aria-label={`Lane ${index + 1} research question`}
                      value={lane.question}
                      onChange={(e) => updateLane(lane.key, { question: e.target.value })}
                      placeholder="Research question"
                      maxLength={LANE_QUESTION_MAX}
                      className="w-full max-w-full rounded-md border border-border bg-panel-alt px-2 py-1 text-xs text-text outline-none focus:border-accent"
                    />
                    <input
                      data-testid={`lane-query-${index}`}
                      aria-label={`Lane ${index + 1} search query`}
                      value={lane.query}
                      onChange={(e) => updateLane(lane.key, { query: e.target.value })}
                      placeholder="Search query"
                      maxLength={LANE_QUERY_MAX}
                      className="w-full max-w-full rounded-md border border-border bg-panel-alt px-2 py-1 text-xs text-text outline-none focus:border-accent"
                    />
                  </div>
                ))}
              </div>
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  data-testid="lane-add"
                  onClick={addLane}
                  disabled={draftLanes.length >= MAX_LANES}
                  className="flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Plus className="h-3.5 w-3.5" aria-hidden="true" />
                  Add lane
                </button>
              {draftLanes.length > 0 && !lanesValid && (
                <span data-testid="lane-validation-hint" className="text-[11px] text-text-muted">
                  {!hasEnabledLane ? 'Enable at least one lane' : 'Each lane needs a label and a search query'}
                </span>
              )}
            </div>
          </fieldset>
        </div>
      )}

      <div className="mt-1 flex gap-2">
        <button
          type="submit"
          data-testid="new-review-start"
          disabled={mode === 'lanes' ? !canStartLanes : !canStartSingle}
          className="flex-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-fg disabled:opacity-40"
        >
          {mode === 'lanes' ? 'Start lane research' : 'Start'}
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
