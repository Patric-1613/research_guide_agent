import { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'
import type { CurationStateResponse } from '../../types'
import type { TurnEvent } from '../../hooks/useCurationSession'
import { TurnDivider } from '../TurnFeed/TurnBlock'
import { PaperCard } from '../PaperPool/PaperCard'
import { mergeSelectedPaperIds } from '../../lib/selection'
import { aggregateKeywords, canonicalKeywordKey, type KeywordOption } from '../../lib/keywords'
import { buildLaneLabelMap, laneLabelsForPaper } from '../../lib/lanes'

// Paper Keywords and Filtering, K4.2: "Popular" is capped and gated so the
// filter UI never becomes the flat wall-of-checkboxes it replaced --
// count-one keywords (the overwhelming majority in a 10-paper batch) never
// occupy a Popular slot, and Popular never grows past this even if more
// than 12 keywords genuinely appear on 2+ papers.
const POPULAR_MIN_PAPER_COUNT = 2
const POPULAR_MAX_OPTIONS = 12

function sortKeywordOptions(options: KeywordOption[]): KeywordOption[] {
  return [...options].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
}

// Mirrors research_agent/query_expansion.py's BATCH_SIZE -- not exposed
// by the API (nothing currently needs it to be), so kept here as the
// one place the frontend needs to know "is this a full batch or a
// partial one" for the turn-divider messaging below.
const BATCH_SIZE = 10

interface ReviewModePanelProps {
  state: CurationStateResponse
  turnEvents: TurnEvent[]
  stagedPickIds: string[]
  disabled: boolean
  onAdd: (paperId: string) => void
  onRemoveStaged: (paperId: string) => void
  // stop=true submits whatever's staged and ends curation right away --
  // the backend (submitPicks) already supports this; the old UI never
  // exposed it, forcing users to keep hitting target_count exactly.
  // requestRefill (Phase 9d): the explicit "search for more now" action --
  // forces a fresh search even when the pool isn't truly exhausted yet.
  onSubmitPicks: (refinement: string | undefined, stop: boolean, requestRefill?: boolean) => Promise<void>
  // UXH.2: which of this panel's three submitPicks shapes (plain
  // Continue, requestRefill, stop) is currently in flight -- each
  // optional and defaulting to false so every pre-existing render call
  // in this component's own tests keeps working unchanged. At most one
  // is ever true at a time (useCurationSession's curationAction is a
  // single value), but they're passed as three separate booleans, not
  // one shared enum prop, matching this codebase's existing convention
  // for per-feature streaming flags (e.g. ChatModePanel's chatStreamActive/
  // chatStreamPhase as separate props rather than a combined object).
  continuingReview?: boolean
  searchingMore?: boolean
  finishingReview?: boolean
}

function stopReasonMessage(state: CurationStateResponse): string {
  const count = state.selected_papers.length
  if (state.stop_reason === 'exhausted' && count < state.target_count) {
    // Phase 9d, the actual reported bug: this used to look identical to a
    // clean finish, with no way to tell "the search ran dry" apart from
    // "you reached your target" -- and no recourse either way.
    return `The search ran out of new candidates at ${count} of ${state.target_count} selected — nothing more to fetch right now.`
  }
  return `Curation complete — ${count} papers selected.`
}

export function ReviewModePanel({
  state,
  turnEvents,
  stagedPickIds,
  disabled,
  onAdd,
  onRemoveStaged,
  onSubmitPicks,
  continuingReview = false,
  searchingMore = false,
  finishingReview = false,
}: ReviewModePanelProps) {
  const [refinement, setRefinement] = useState('')
  const pendingBatch = state.pending_batch
  const stagedSet = new Set(stagedPickIds)
  // UXH.1 (UX-01): deduplicated union -- see mergeSelectedPaperIds' own
  // docstring for why this replaced plain `a.length + b.length`.
  const totalSelected = mergeSelectedPaperIds(state.selected_paper_ids, stagedPickIds).length

  // Research Lanes (RL5): lane-aware display only. A single-query session
  // has no lanes -> foundVia() is always [] -> PaperCard renders no
  // "Found via" row, and the refill wording below stays unchanged.
  const isLaneSession = (state.lanes?.length ?? 0) > 0
  const laneLabels = buildLaneLabelMap(state.lanes)
  const foundVia = (paperId: string) => laneLabelsForPaper(paperId, state.paper_lane_ids, laneLabels)
  const LANE_SEARCH_LABEL = 'Searching across research lanes…'

  // Scroll the batch back to the top whenever the SET OF PAPER IDS in
  // pending_batch actually changes (a new batch was served, whether via
  // "Get next batch" or "Search for more candidates") -- not on every
  // re-render (staging a pick, loading toggling, etc. leave batchKey
  // unchanged, so the effect correctly no-ops then).
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const batchKey = pendingBatch?.map((p) => p.paper_id).join(',') ?? null
  useEffect(() => {
    if (scrollContainerRef.current) scrollContainerRef.current.scrollTop = 0
  }, [batchKey])

  // Paper Keywords and Filtering, K4.2: presentation-only keyword filter
  // for the CURRENT pending batch -- canonical keyword key (see
  // lib/keywords.ts's canonicalKeywordKey, which merges hyphen/space/case
  // surface variants) -> its own display label, never a Set of raw
  // strings, so a removable chip/checkbox can show a stable label without
  // a second lookup. Reset whenever the batch this filter applies to is
  // genuinely no longer the same one: state.session_id (switching
  // reviews) or batchKey (a new batch served, reusing the SAME stable key
  // the scroll-reset effect above already established) -- deliberately
  // NOT pendingBatch itself, whose array reference changes on every fresh
  // fetch even when its own paper-id set (and therefore batchKey) is
  // unchanged; keying off the array reference would silently clear a
  // still-valid filter on an ordinary same-batch refresh (staging a pick,
  // a loading-state toggle, etc.).
  const [selectedKeywords, setSelectedKeywords] = useState<Map<string, string>>(new Map())
  const [keywordFilterOpen, setKeywordFilterOpen] = useState(false)
  // Browse-all mode (search across every keyword, including count-one
  // ones) replaces the Popular checkbox list in place, rather than the
  // two ever rendering side by side -- see the panel JSX below.
  const [browseAllOpen, setBrowseAllOpen] = useState(false)
  const [keywordSearch, setKeywordSearch] = useState('')
  useEffect(() => {
    setSelectedKeywords(new Map())
    setKeywordFilterOpen(false)
    setBrowseAllOpen(false)
    setKeywordSearch('')
  }, [state.session_id, batchKey])

  // Closing the outer disclosure resets browse/search back to Popular's
  // default view (the simplest predictable behavior -- reopening always
  // starts from the same place) but deliberately leaves selectedKeywords
  // alone, so active filters survive a close/reopen within the same batch.
  function toggleKeywordFilterOpen() {
    setKeywordFilterOpen((prev) => {
      const next = !prev
      if (!next) {
        setBrowseAllOpen(false)
        setKeywordSearch('')
      }
      return next
    })
  }

  function toggleBrowseAll() {
    setBrowseAllOpen((prev) => !prev)
    setKeywordSearch('')
  }

  // zero-selection-curation-dead-end fix: `pending_batch === null` alone
  // does NOT mean curation genuinely finished -- that's only true when
  // the backend has actually recorded stage="synthesize" (reached only
  // via an explicit stop=True resume, per curation_loop.py's own module
  // docstring). A real corrupted session was found where pending_batch
  // was null (its LangGraph interrupt had been destroyed by an
  // out-of-band checkpoint write while genuinely mid-curation) but
  // stage was STILL "curate" -- the old code here rendered
  // stopReasonMessage()'s "Curation complete — 0 papers selected."
  // regardless, fabricating a completion that never actually happened
  // and leaving no Continue/Search/Finish control in sight. Splitting on
  // stage instead makes the honest, distinct case visible rather than
  // silently misreporting it as success.
  if (!pendingBatch && state.stage === 'synthesize') {
    // Phase 8, item 3: true both right after curation just finished AND
    // every time an already-completed review is reopened -- state.selected_papers
    // is already in the API response either way, it just wasn't being
    // rendered here before.
    return (
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-3">
          <p className="mb-3 text-center text-sm text-text-secondary">{stopReasonMessage(state)}</p>
          <div className="flex flex-col gap-2">
            {state.selected_papers.map((paper) => (
              <PaperCard
                key={paper.paper_id}
                paper={paper}
                showAbstract
                foundViaLabels={foundVia(paper.paper_id)}
                action={{ kind: 'none' }}
              />
            ))}
          </div>
        </div>
        <p className="border-t border-border bg-panel px-4 py-3 text-center text-sm text-text-muted">
          Switch to the Report tab on the left to generate your literature review.
        </p>
      </div>
    )
  }

  if (!pendingBatch) {
    // stage is still "curate" (not "synthesize") but the backend has no
    // batch to present -- this session is stuck, not finished. Never
    // claim completion here, and never offer any action that would
    // select/delete/mutate a paper -- there is currently no safe,
    // generalized recovery for this exact state, so this is honest
    // status only, not a repair.
    return (
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-3">
          <div data-testid="review-stalled-banner" className="rounded-md border border-danger/30 bg-danger-soft px-4 py-3 text-sm text-danger">
            <p className="font-medium">This review's active batch couldn&apos;t be loaded.</p>
            <p className="mt-1 text-text-secondary">
              Curation hasn&apos;t actually finished — {state.selected_paper_ids.length} paper
              {state.selected_paper_ids.length === 1 ? '' : 's'} selected so far, with candidates still in the pool.
              This can happen if the session was changed outside the app. Nothing has been changed here; please
              contact support to resume this review.
            </p>
          </div>
        </div>
      </div>
    )
  }

  async function handleContinue() {
    await onSubmitPicks(refinement.trim() || undefined, false)
    setRefinement('')
  }

  async function handleStop() {
    await onSubmitPicks(refinement.trim() || undefined, true)
    setRefinement('')
  }

  async function handleRequestRefill() {
    await onSubmitPicks(refinement.trim() || undefined, false, true)
    setRefinement('')
  }

  // curation-editable-until-locked Phase 10b/10d: an exhausted search or
  // a target-reached batch no longer ends curation -- the backend just
  // presents an empty (or ordinary) batch and leaves the review open, so
  // the frontend needs to spell out clearly what happened instead of
  // silently rendering zero PaperCards.
  const isEmptyBatch = pendingBatch.length === 0
  const targetReached = totalSelected >= state.target_count

  // Phase 9d/9e, the actual reported bug: a partial batch (fewer than
  // BATCH_SIZE, not refilled this turn) used to be indistinguishable
  // from "the pool ran fully dry" -- and the old auto-refill-below-
  // BATCH_SIZE behavior meant this message rarely even had a chance to
  // show. Now that a partial batch serves as-is, say so plainly, and
  // point at the explicit action instead of implying one already happened.
  const isPartialBatch = !state.refilled && !isEmptyBatch && pendingBatch.length < BATCH_SIZE
  const turnMessage = isEmptyBatch
    ? "No new candidates found for this search. Try refining below, or click \"I'm done\" if you're satisfied with what you have."
    : state.refilled
      ? isLaneSession
        ? `Searched across your research lanes and found ${pendingBatch.length} to show you this turn.`
        : `Searched for more candidates and found ${pendingBatch.length} to show you this turn.`
      : isPartialBatch
        ? `Only ${pendingBatch.length} candidate${pendingBatch.length === 1 ? '' : 's'} left in the already-fetched pool.`
        : `Showing ${pendingBatch.length} candidates from the pool already fetched — no new search needed.`

  // Paper Keywords and Filtering, K4.2: derived fresh from pendingBatch
  // every render -- cheap (at most BATCH_SIZE papers, a handful of
  // keywords each), so a plain computation rather than a memoized one;
  // nothing here writes to state, so there's no risk of it feeding back
  // into the reset effect above. aggregateKeywords does the canonical
  // (hyphen/space/case-insensitive) grouping and per-paper-once counting
  // -- see lib/keywords.ts for the exact rules.
  const keywordOptions = aggregateKeywords(pendingBatch)
  const popularOptions = sortKeywordOptions(keywordOptions.filter((o) => o.count >= POPULAR_MIN_PAPER_COUNT)).slice(
    0,
    POPULAR_MAX_OPTIONS,
  )
  const allOptionsSorted = sortKeywordOptions(keywordOptions)
  const searchKey = canonicalKeywordKey(keywordSearch)
  const browseAllResults = searchKey ? allOptionsSorted.filter((o) => o.key.includes(searchKey)) : allOptionsSorted

  const visibleBatch = selectedKeywords.size === 0
    ? pendingBatch
    : pendingBatch.filter((paper) => paper.keywords.some((keyword) => selectedKeywords.has(canonicalKeywordKey(keyword))))
  const noKeywordFilterMatches = selectedKeywords.size > 0 && visibleBatch.length === 0

  function toggleKeywordFilter(key: string, label: string) {
    setSelectedKeywords((prev) => {
      const next = new Map(prev)
      if (next.has(key)) next.delete(key)
      else next.set(key, label)
      return next
    })
  }

  function clearKeywordFilter() {
    setSelectedKeywords(new Map())
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div ref={scrollContainerRef} data-testid="review-batch-scroll" className="flex-1 overflow-y-auto px-4 py-3">
        {/* Only the active/current turn renders here -- past turns live in
            the "Browse past turns" browser instead of stacking inline. */}
        <TurnDivider turnNumber={turnEvents.length + 1} refilled={state.refilled} />
        {targetReached && (
          <p
            data-testid="review-target-reached-banner"
            className="mb-3 rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent"
          >
            You&apos;ve reached your target of {state.target_count} papers ({totalSelected} selected). Keep
            curating if you&apos;d like more, or click &quot;I&apos;m done&quot; below to finish.
          </p>
        )}
        <p data-testid="review-turn-message" className="mb-3 text-sm text-text-secondary">
          {isEmptyBatch ? (
            turnMessage
          ) : (
            <>
              {turnMessage} Select the ones you want, then continue{isPartialBatch ? ', or search for more below' : ''}.
            </>
          )}
        </p>
        {/* Paper Keywords and Filtering, K4.2: only worth showing at all
            when there's at least one real keyword to filter by -- an
            empty batch, or a batch where every paper has keywords:[] (no
            abstract, or a too-short one), gets no filter control rather
            than an empty, useless one. Collapsed by default: the options
            panel itself only renders while keywordFilterOpen, never a
            permanent list of every keyword above the paper list -- and
            it stays unframed/plain (no nested card), a bare flex column,
            same as before. Active-filter chips/summary/clear live in
            this OUTER row, outside the collapsible panel, so they stay
            visible regardless of open/closed state or Popular/Browse-all
            mode. */}
        {!isEmptyBatch && keywordOptions.length > 0 && (
          <div className="mb-3 flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                data-testid="keyword-filter-toggle"
                aria-expanded={keywordFilterOpen}
                aria-controls="keyword-filter-panel"
                onClick={toggleKeywordFilterOpen}
                className="rounded-md border border-border px-2.5 py-1 text-xs text-text-secondary hover:border-accent hover:text-accent"
              >
                Filter keywords{selectedKeywords.size > 0 ? ` (${selectedKeywords.size})` : ''}
              </button>
              {selectedKeywords.size > 0 && (
                <>
                  {Array.from(selectedKeywords.entries()).map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => toggleKeywordFilter(key, label)}
                      aria-label={`Remove ${label} filter`}
                      className="flex items-center gap-1 rounded-full bg-accent-soft px-2 py-0.5 text-[11px] font-medium text-accent"
                    >
                      {label}
                      <span aria-hidden="true">×</span>
                    </button>
                  ))}
                  <span data-testid="keyword-filter-summary" className="text-xs text-text-muted">
                    Showing {visibleBatch.length} of {pendingBatch.length} papers
                  </span>
                  <button
                    type="button"
                    data-testid="keyword-filter-clear"
                    onClick={clearKeywordFilter}
                    className="text-xs text-text-secondary underline decoration-dotted hover:text-text"
                  >
                    Clear filters
                  </button>
                </>
              )}
            </div>
            {keywordFilterOpen && (
              <div
                id="keyword-filter-panel"
                data-testid="keyword-filter-panel"
                aria-label="Filter by keyword"
                className="flex w-full max-w-full flex-col gap-2 rounded-md border border-border p-2"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-text-secondary">
                    {browseAllOpen ? 'All keywords' : 'Popular keywords'}
                  </p>
                  <button
                    type="button"
                    data-testid="keyword-browse-all-toggle"
                    aria-expanded={browseAllOpen}
                    aria-controls="keyword-browse-all-panel"
                    onClick={toggleBrowseAll}
                    className="text-xs text-text-secondary underline decoration-dotted hover:text-text"
                  >
                    {browseAllOpen ? 'Back to popular keywords' : `Browse all keywords (${keywordOptions.length})`}
                  </button>
                </div>

                {!browseAllOpen && (
                  popularOptions.length > 0 ? (
                    <div className="flex max-h-48 flex-col gap-0.5 overflow-y-auto">
                      {popularOptions.map((option) => (
                        <KeywordCheckbox
                          key={option.key}
                          option={option}
                          checked={selectedKeywords.has(option.key)}
                          onToggle={toggleKeywordFilter}
                        />
                      ))}
                    </div>
                  ) : (
                    <p data-testid="keyword-no-popular" className="text-xs text-text-muted">
                      No keyword appears in more than one paper in this batch yet.
                    </p>
                  )
                )}

                {browseAllOpen && (
                  <div id="keyword-browse-all-panel" data-testid="keyword-browse-all-panel" className="flex flex-col gap-2">
                    <div className="flex flex-col gap-1">
                      <label htmlFor="keyword-search-input" className="text-xs text-text-secondary">
                        Search keywords
                      </label>
                      <input
                        id="keyword-search-input"
                        data-testid="keyword-search-input"
                        type="text"
                        value={keywordSearch}
                        onChange={(e) => setKeywordSearch(e.target.value)}
                        placeholder="e.g. retrieval augmented"
                        className="w-full max-w-full rounded-md border border-border bg-panel-alt px-2 py-1 text-xs text-text outline-none focus:border-accent"
                      />
                    </div>
                    <div className="flex max-h-48 flex-col gap-0.5 overflow-y-auto">
                      {browseAllResults.length === 0 ? (
                        <p data-testid="keyword-search-empty" className="text-xs text-text-muted">
                          No keywords match your search.
                        </p>
                      ) : (
                        browseAllResults.map((option) => (
                          <KeywordCheckbox
                            key={option.key}
                            option={option}
                            checked={selectedKeywords.has(option.key)}
                            onToggle={toggleKeywordFilter}
                          />
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {isEmptyBatch ? (
          <p className="text-center text-sm italic text-text-muted">No candidates to show for this search.</p>
        ) : noKeywordFilterMatches ? (
          <div data-testid="keyword-filter-empty-state" className="text-center text-sm text-text-muted">
            <p className="mb-2">No papers match the selected keywords.</p>
            <button
              type="button"
              onClick={clearKeywordFilter}
              className="text-xs text-text-secondary underline decoration-dotted hover:text-text"
            >
              Clear filters
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {visibleBatch.map((paper) =>
              stagedSet.has(paper.paper_id) ? (
                <PaperCard
                  key={paper.paper_id}
                  paper={paper}
                  showAbstract
                  foundViaLabels={foundVia(paper.paper_id)}
                  action={{ kind: 'remove', onRemove: () => onRemoveStaged(paper.paper_id) }}
                />
              ) : (
                <PaperCard
                  key={paper.paper_id}
                  paper={paper}
                  isNew
                  showAbstract
                  foundViaLabels={foundVia(paper.paper_id)}
                  action={{ kind: 'add', onAdd: () => onAdd(paper.paper_id) }}
                />
              ),
            )}
          </div>
        )}
      </div>

      <div className="flex flex-col gap-2 border-t border-border bg-panel p-3">
        <input
          data-testid="review-refinement-input"
          value={refinement}
          onChange={(e) => setRefinement(e.target.value)}
          disabled={disabled}
          placeholder="Refine what you're looking for (optional)..."
          // Usage Protection M2.3 Part D: mirrors research_agent/config/
          // limits.py's max_text_length -- preventative UX only.
          maxLength={2000}
          className="rounded-md border border-border bg-panel-alt px-3 py-2 text-sm text-text outline-none focus:border-accent disabled:opacity-60"
        />
        <div className="flex items-center justify-between gap-2">
          {/* UXH.2: each button's OWN busy label replaces its idle text
              entirely (never a separate status alongside it) -- wrapped in
              role="status"/aria-live="polite" only while busy, so an AT
              hears "Finding next papers…" etc. the moment it appears, but
              ordinary idle-state changes (e.g. totalSelected ticking up as
              the user stages picks) are never announced as if they were
              progress. */}
          <button
            type="button"
            data-testid="review-stop"
            onClick={handleStop}
            disabled={disabled || totalSelected === 0}
            className="flex items-center gap-1.5 text-xs text-text-secondary underline decoration-dotted hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          >
            {finishingReview ? (
              <span role="status" aria-live="polite" className="flex items-center gap-1.5">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" />
                Finishing review…
              </span>
            ) : (
              `I'm done — finish with ${totalSelected} selected`
            )}
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              data-testid="review-request-refill"
              onClick={handleRequestRefill}
              disabled={disabled}
              title="Search for more candidates now, even though the current pool isn't empty"
              className="flex items-center gap-1.5 rounded-md border border-border px-3 py-2 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              {searchingMore ? (
                <span role="status" aria-live="polite" className="flex items-center gap-1.5">
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" />
                  {isLaneSession ? LANE_SEARCH_LABEL : 'Searching for more papers…'}
                </span>
              ) : (
                'Search for more candidates'
              )}
            </button>
            <button
              type="button"
              data-testid="review-continue"
              onClick={handleContinue}
              disabled={disabled}
              className="flex items-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg disabled:opacity-40"
            >
              {continuingReview ? (
                <span role="status" aria-live="polite" className="flex items-center gap-1.5">
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" />
                  {isLaneSession && state.reserve_remaining === 0 ? LANE_SEARCH_LABEL : 'Finding next papers…'}
                </span>
              ) : stagedPickIds.length > 0 ? (
                `Continue with ${stagedPickIds.length} added`
              ) : isEmptyBatch ? (
                'Try again'
              ) : (
                'Get next batch'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Paper Keywords and Filtering, K4.2: shared row markup for BOTH the
// Popular list and the Browse-all results list -- the two never render at
// the same time (see the panel JSX above), so reusing the same
// data-testid pattern for both never produces two simultaneous controls
// for one option.
function KeywordCheckbox({
  option,
  checked,
  onToggle,
}: {
  option: KeywordOption
  checked: boolean
  onToggle: (key: string, label: string) => void
}) {
  return (
    <label
      data-testid={`keyword-filter-option-${option.key}`}
      className="flex items-center gap-1.5 rounded px-1 py-0.5 text-xs text-text-secondary hover:bg-panel-alt"
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={() => onToggle(option.key, option.label)}
        className="h-3.5 w-3.5 shrink-0 rounded border-border"
      />
      <span className="max-w-full whitespace-normal break-words">{option.label}</span>
      <span className="shrink-0 text-text-muted">({option.count})</span>
    </label>
  )
}
