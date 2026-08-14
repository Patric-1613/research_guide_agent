import type { ComponentProps } from 'react'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReviewModePanel } from './ReviewModePanel'
import type { CurationStateResponse, PaperOut } from '../../types'
import type { TurnEvent } from '../../hooks/useCurationSession'

function paper(id: string, title: string, abstract: string | null = null, keywords: string[] = []): PaperOut {
  return {
    paper_id: id, title, authors: ['A'], year: 2024, venue: 'arXiv',
    abstract, url: null, doi: null, citation_count: 10, source: 'arxiv',
    source_urls: {}, score: null, keywords,
  }
}

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', display_title: 't', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    report_versions: [], active_report_version_id: null, chat_references: [],
    ...overrides,
  }
}

describe('ReviewModePanel', () => {
  it('renders pending_batch candidates with their abstracts, so users can judge them before picking', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One', 'An abstract about X.')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('An abstract about X.')).toBeInTheDocument()
  })

  it('a staged paper shows a remove affordance instead of add', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={['p1']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'Remove Paper One' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '+ Add to review' })).toBeInTheDocument()
  })

  it('clicking Continue submits staged picks and any typed refinement, with stop=false', async () => {
    const user = userEvent.setup()
    const onSubmitPicks = vi.fn().mockResolvedValue(undefined)
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={['p1']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={onSubmitPicks}
      />,
    )

    await user.type(screen.getByTestId('review-refinement-input'), 'focus on recent work')
    await user.click(screen.getByTestId('review-continue'))

    expect(onSubmitPicks).toHaveBeenCalledWith('focus on recent work', false)
  })

  it('clicking "Search for more candidates" submits with requestRefill=true (Phase 9d/9e)', async () => {
    const user = userEvent.setup()
    const onSubmitPicks = vi.fn().mockResolvedValue(undefined)
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={onSubmitPicks}
      />,
    )

    await user.click(screen.getByTestId('review-request-refill'))

    expect(onSubmitPicks).toHaveBeenCalledWith(undefined, false, true)
  })

  it('clicking "I\'m done" submits with stop=true, and is disabled when nothing is selected yet', async () => {
    const user = userEvent.setup()
    const onSubmitPicks = vi.fn().mockResolvedValue(undefined)
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')], selected_paper_ids: ['x'] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={onSubmitPicks}
      />,
    )

    const stopButton = screen.getByTestId('review-stop')
    expect(stopButton).toBeEnabled()
    await user.click(stopButton)

    expect(onSubmitPicks).toHaveBeenCalledWith(undefined, true)
  })

  it('the stop action is disabled when nothing is staged or already selected', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByTestId('review-stop')).toBeDisabled()
  })

  it('no pending batch (curation finished, or a completed review reopened): shows the completion message AND the selected papers themselves (Phase 8, item 3)', () => {
    const state = baseState({
      stage: 'synthesize',
      pending_batch: null,
      selected_papers: [paper('p1', 'Paper One', 'An abstract about Y.'), paper('p2', 'Paper Two')],
      selected_paper_ids: ['p1', 'p2'],
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/Curation complete/)).toBeInTheDocument()
    expect(screen.getByText(/2 papers selected/)).toBeInTheDocument()
    // The whole point of the fix: the papers themselves render, with their
    // abstracts, not just a count -- this is what a reopened completed
    // review was previously missing entirely.
    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('An abstract about Y.')).toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()
    // Already-confirmed picks have no add/remove affordance -- curation is over.
    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Remove/ })).not.toBeInTheDocument()
  })

  it('shows only the active/current turn -- past turnEvents no longer stack inline in the main panel', () => {
    const events: TurnEvent[] = [
      { turnNumber: 1, refilled: false, batchSize: 10, reserveRemainingAfter: 5, pickedPaperIds: ['a', 'b'] },
    ]
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')], refilled: true })
    render(
      <ReviewModePanel
        state={state} turnEvents={events} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    // Turn numbering for the CURRENT turn is preserved (still derived from
    // turnEvents.length + 1)...
    expect(screen.getByText('Turn 2 — new search')).toBeInTheDocument()
    // ...but no content from the past turn (turn 1's own divider, or its
    // "AGENT" / "You selected" summary) renders in the main panel anymore.
    expect(screen.queryByText('Turn 1 — from existing pool (no new search)')).not.toBeInTheDocument()
    expect(screen.queryByText('AGENT')).not.toBeInTheDocument()
    expect(screen.queryByText(/You selected/)).not.toBeInTheDocument()
    // The live batch itself still renders.
    expect(screen.getByText('Paper One')).toBeInTheDocument()
  })

  it('scrolls the batch container back to the top when pending_batch changes to a different set of paper IDs', () => {
    const state1 = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    const { rerender } = render(
      <ReviewModePanel
        state={state1} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )
    const scrollEl = screen.getByTestId('review-batch-scroll')
    scrollEl.scrollTop = 400

    const state2 = baseState({ pending_batch: [paper('p2', 'Paper Two')] })
    rerender(
      <ReviewModePanel
        state={state2} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(scrollEl.scrollTop).toBe(0)
  })

  it('does NOT reset scroll position when pending_batch is re-rendered with the same paper IDs', () => {
    const state1 = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    const { rerender } = render(
      <ReviewModePanel
        state={state1} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )
    const scrollEl = screen.getByTestId('review-batch-scroll')
    scrollEl.scrollTop = 400

    // Same paper IDs, e.g. a re-render triggered by staging a pick or a
    // loading-state toggle -- not a new batch.
    const state1Again = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    rerender(
      <ReviewModePanel
        state={state1Again} turnEvents={[]} stagedPickIds={['p1']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(scrollEl.scrollTop).toBe(400)
  })

  it('a partial batch (fewer than BATCH_SIZE, not refilled) gets a distinct message pointing at "search for more" (Phase 9e)', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')], // only 2, refilled: false
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/Only 2 candidates left in the already-fetched pool/)).toBeInTheDocument()
    expect(screen.getByText(/search for more below/)).toBeInTheDocument()
    expect(screen.queryByText(/no new search needed/)).not.toBeInTheDocument()
  })

  it('a full batch (exactly BATCH_SIZE, not refilled) keeps the existing "no new search needed" message unchanged', () => {
    const fullBatch = Array.from({ length: 10 }, (_, i) => paper(`p${i}`, `Paper ${i}`))
    const state = baseState({ pending_batch: fullBatch })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/no new search needed/)).toBeInTheDocument()
    expect(screen.queryByText(/Only \d+ candidates left/)).not.toBeInTheDocument()
  })

  it('a refilled batch keeps the existing "searched for more" message regardless of size', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')], refilled: true })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/Searched for more candidates and found 2/)).toBeInTheDocument()
    expect(screen.queryByText(/Only \d+ candidates left/)).not.toBeInTheDocument()
  })

  it('stopped short of target due to exhaustion: shows a distinct message, not indistinguishable from a clean finish (Phase 9e, the actual reported bug)', () => {
    const state = baseState({
      stage: 'synthesize', pending_batch: null, stop_reason: 'exhausted', target_count: 10,
      selected_papers: [paper('p1', 'Paper One')], selected_paper_ids: ['p1'],
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/ran out of new candidates at 1 of 10/)).toBeInTheDocument()
    expect(screen.queryByText(/^Curation complete/)).not.toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10d: an empty batch (search came back dry, still curating) shows a clear message and no PaperCards, not a silent blank list', () => {
    const state = baseState({ pending_batch: [], target_count: 10, selected_paper_ids: ['p1'] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/No new candidates found for this search/)).toBeInTheDocument()
    expect(screen.getByText(/Try refining below/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
    // The primary action relabels to reflect there's nothing to "get next" --
    // it's a retry, reusing the same refinement bar to search again.
    expect(screen.getByTestId('review-continue')).toHaveTextContent('Try again')
  })

  it('curation-editable-until-locked Phase 10d: an empty batch still stays fully editable -- "I\'m done" and "Search for more" remain available', () => {
    const state = baseState({ pending_batch: [], selected_paper_ids: ['p1'] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByTestId('review-stop')).toBeEnabled()
    expect(screen.getByTestId('review-request-refill')).toBeEnabled()
  })

  it('curation-editable-until-locked Phase 10d: reaching target_count while still curating shows a nudge banner, not a lock', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One')], target_count: 2, selected_paper_ids: ['a', 'b'],
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByTestId('review-target-reached-banner')).toHaveTextContent("reached your target of 2")
    // Still fully editable -- the new batch is still shown, picks still work.
    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '+ Add to review' })).toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10d: no nudge banner while still short of target_count', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One')], target_count: 10, selected_paper_ids: ['a'],
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.queryByTestId('review-target-reached-banner')).not.toBeInTheDocument()
  })

  it('UXH.1 (UX-01): "I\'m done" reflects staged picks, deduplicated against persisted ones, not double-counted', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')],
      selected_paper_ids: ['x', 'p1'],
    })
    render(
      <ReviewModePanel
        // p1 is staged again on top of being already-persisted (defensive
        // case) alongside a genuinely new staged pick p2 -- expect 3
        // distinct papers (x, p1, p2), not 4.
        state={state} turnEvents={[]} stagedPickIds={['p1', 'p2']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText("I'm done — finish with 3 selected")).toBeInTheDocument()
  })

  it('UXH.1 (UX-01): the target-reached nudge banner also counts staged picks toward the target, deduplicated', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One')], target_count: 2, selected_paper_ids: ['a'],
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={['b']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByTestId('review-target-reached-banner')).toHaveTextContent('reached your target of 2')
  })

  it('UXH.2: continuingReview replaces the Continue label with a truthful, accessible busy status', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
        continuingReview
      />,
    )

    const button = screen.getByTestId('review-continue')
    expect(button).toHaveTextContent('Finding next papers…')
    expect(button).not.toHaveTextContent('Get next batch')
    const status = screen.getByRole('status')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveTextContent('Finding next papers…')
  })

  it('UXH.2: searchingMore replaces the "Search for more candidates" label, not called streaming', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
        searchingMore
      />,
    )

    const button = screen.getByTestId('review-request-refill')
    expect(button).toHaveTextContent('Searching for more papers…')
    expect(button).not.toHaveTextContent('Search for more candidates')
    expect(screen.queryByText(/stream/i)).not.toBeInTheDocument()
  })

  it('UXH.2: finishingReview replaces the "I\'m done" label with "Finishing review…"', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')], selected_paper_ids: ['p1'] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
        finishingReview
      />,
    )

    const button = screen.getByTestId('review-stop')
    expect(button).toHaveTextContent('Finishing review…')
    expect(button).not.toHaveTextContent("I'm done")
  })

  it('UXH.2: only one busy label renders at a time; idle buttons keep their normal text', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')], selected_paper_ids: ['p1'] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
        finishingReview
      />,
    )

    expect(screen.getByTestId('review-continue')).toHaveTextContent('Get next batch')
    expect(screen.getByTestId('review-request-refill')).toHaveTextContent('Search for more candidates')
    expect(screen.getAllByRole('status')).toHaveLength(1)
  })

  it('UXH.2: no busy status at all when no curation action is active (all three flags default false)', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')], selected_paper_ids: ['p1'] })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('a clean finish (target_met) still shows the plain "Curation complete" message', () => {
    const state = baseState({
      stage: 'synthesize', pending_batch: null, stop_reason: 'target_met', target_count: 1,
      selected_papers: [paper('p1', 'Paper One')], selected_paper_ids: ['p1'],
    })
    render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByText(/Curation complete — 1 papers selected/)).toBeInTheDocument()
  })
})

describe('ReviewModePanel -- Paper Keywords and Filtering, K2: keyword filter', () => {
  function renderPanel(overrides: Partial<CurationStateResponse> = {}, props: Partial<ComponentProps<typeof ReviewModePanel>> = {}) {
    const onAdd = vi.fn()
    const onRemoveStaged = vi.fn()
    const onSubmitPicks = vi.fn().mockResolvedValue(undefined)
    const state = baseState(overrides)
    const utils = render(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={onAdd} onRemoveStaged={onRemoveStaged} onSubmitPicks={onSubmitPicks}
        {...props}
      />,
    )
    return { ...utils, onAdd, onRemoveStaged, onSubmitPicks, state }
  }

  it('the filter control is absent when no paper in the batch has any keywords', () => {
    renderPanel({ pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')] })

    expect(screen.queryByTestId('keyword-filter-toggle')).not.toBeInTheDocument()
  })

  it('the filter control appears once at least one paper has keywords', () => {
    renderPanel({ pending_batch: [paper('p1', 'Paper One', null, ['graph neural networks'])] })

    expect(screen.getByTestId('keyword-filter-toggle')).toBeInTheDocument()
  })

  it('keyword options are deduplicated case-insensitively, counted across both casings', async () => {
    const user = userEvent.setup()
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['Neural Networks']),
        paper('p2', 'Paper Two', null, ['neural networks']),
        paper('p3', 'Paper Three', null, ['neural networks']),
      ],
    })

    await user.click(screen.getByTestId('keyword-filter-toggle'))

    // Exactly one option for the case-varied keyword, count 3 -- not two
    // separate options ("Neural Networks" (1) and "neural networks" (2)).
    const options = screen.getAllByTestId(/^keyword-filter-option-/)
    expect(options).toHaveLength(1)
    expect(options[0]).toHaveTextContent('Neural Networks') // first-seen casing wins as the display label
    expect(options[0]).toHaveTextContent('(3)')
  })

  it('options are sorted by descending paper count, then alphabetically by label for ties', async () => {
    const user = userEvent.setup()
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['zebra topic', 'common topic']),
        paper('p2', 'Paper Two', null, ['apple topic', 'common topic']),
        paper('p3', 'Paper Three', null, ['common topic']),
      ],
    })

    await user.click(screen.getByTestId('keyword-filter-toggle'))

    const options = screen.getAllByTestId(/^keyword-filter-option-/)
    const labels = options.map((el) => el.textContent)
    // "common topic" (count 3) first; then the two count-1 ties break
    // alphabetically: "apple topic" before "zebra topic".
    expect(labels[0]).toContain('common topic')
    expect(labels[0]).toContain('(3)')
    expect(labels[1]).toContain('apple topic')
    expect(labels[2]).toContain('zebra topic')
  })

  it('one selected keyword filters the visible batch correctly, preserving original order', async () => {
    const user = userEvent.setup()
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
        paper('p3', 'Paper Three', null, ['topic a']),
      ],
    })

    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))

    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('Paper Three')).toBeInTheDocument()
    expect(screen.queryByText('Paper Two')).not.toBeInTheDocument()
    // Source order preserved: p1 before p3, not re-sorted.
    const cards = screen.getAllByTestId(/^paper-card-/)
    expect(cards.map((el) => el.dataset.testid)).toEqual(['paper-card-p1', 'paper-card-p3'])
  })

  it('multiple selected keywords use OR semantics -- a paper matching ANY selected keyword is visible', async () => {
    const user = userEvent.setup()
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
        paper('p3', 'Paper Three', null, ['topic c']),
      ],
    })

    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic b')).getByRole('checkbox'))

    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()
    expect(screen.queryByText('Paper Three')).not.toBeInTheDocument()
    expect(screen.getByTestId('keyword-filter-summary')).toHaveTextContent('Showing 2 of 3 papers')
  })

  it('empty selection shows the full pending batch', () => {
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
      ],
    })

    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()
    expect(screen.queryByTestId('keyword-filter-summary')).not.toBeInTheDocument()
  })

  it('"Clear filters" restores the complete batch', async () => {
    const user = userEvent.setup()
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
      ],
    })

    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    expect(screen.queryByText('Paper Two')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('keyword-filter-clear'))

    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()
    expect(screen.queryByTestId('keyword-filter-summary')).not.toBeInTheDocument()
  })

  it('removing one active chip updates the result, keeping any other active filter', async () => {
    const user = userEvent.setup()
    renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
        paper('p3', 'Paper Three', null, ['topic c']),
      ],
    })

    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic b')).getByRole('checkbox'))
    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Remove topic a filter' }))

    expect(screen.queryByText('Paper One')).not.toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()
    expect(screen.queryByText('Paper Three')).not.toBeInTheDocument()
  })

  it('zero matches renders the empty state when the only selected keyword matches nothing (defensive: a same-batch-key update drops the keyword a selection depended on)', async () => {
    const user = userEvent.setup()
    const { rerender, state } = renderPanel({
      pending_batch: [paper('p1', 'Paper One', null, ['topic a', 'topic b'])],
    })
    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    expect(screen.getByText('Paper One')).toBeInTheDocument()

    // Same batchKey (still just p1) but its OWN keywords shrink to no
    // longer include "topic a" -- a same-session, same-paper-id-set
    // update (a re-render this filter's own reset effect does NOT fire
    // for, since batchKey is paper-id-based, not keyword-based).
    rerender(
      <ReviewModePanel
        state={{ ...state, pending_batch: [paper('p1', 'Paper One', null, ['topic b'])] }}
        turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByTestId('keyword-filter-empty-state')).toBeInTheDocument()
    expect(screen.getByText('No papers match the selected keywords.')).toBeInTheDocument()
  })

  it('the filter resets when the session changes, even with the same paper IDs', async () => {
    const user = userEvent.setup()
    const { rerender, state } = renderPanel({
      session_id: 's1', pending_batch: [paper('p1', 'Paper One', null, ['topic a'])],
    })
    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    expect(screen.getByTestId('keyword-filter-summary')).toBeInTheDocument()

    rerender(
      <ReviewModePanel
        state={{ ...state, session_id: 's2' }} turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.queryByTestId('keyword-filter-summary')).not.toBeInTheDocument()
    expect(screen.queryByTestId('keyword-filter-panel')).not.toBeInTheDocument()
  })

  it('the filter resets when the pending-batch paper-id set genuinely changes', async () => {
    const user = userEvent.setup()
    const { rerender, state } = renderPanel({
      pending_batch: [paper('p1', 'Paper One', null, ['topic a'])],
    })
    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    expect(screen.getByTestId('keyword-filter-summary')).toBeInTheDocument()

    rerender(
      <ReviewModePanel
        state={{ ...state, pending_batch: [paper('p2', 'Paper Two', null, ['topic a'])] }}
        turnEvents={[]} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.queryByTestId('keyword-filter-summary')).not.toBeInTheDocument()
  })

  it('an ordinary re-render with the SAME paper-id set (e.g. staging a pick) does not reset an active filter', async () => {
    const user = userEvent.setup()
    const { rerender, state } = renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
      ],
    })
    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))
    expect(screen.queryByText('Paper Two')).not.toBeInTheDocument()

    // Same batchKey (p1,p2 unchanged) -- only stagedPickIds changed, the
    // same kind of re-render staging a pick already causes today.
    rerender(
      <ReviewModePanel
        state={state} turnEvents={[]} stagedPickIds={['p1']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSubmitPicks={vi.fn()}
      />,
    )

    expect(screen.getByTestId('keyword-filter-summary')).toBeInTheDocument()
    expect(screen.queryByText('Paper Two')).not.toBeInTheDocument()
  })

  it('filtering never changes selected counts, target progress, or the Continue/Stop submission payload', async () => {
    const user = userEvent.setup()
    const { onSubmitPicks } = renderPanel({
      pending_batch: [
        paper('p1', 'Paper One', null, ['topic a']),
        paper('p2', 'Paper Two', null, ['topic b']),
      ],
      target_count: 2, selected_paper_ids: ['x'],
    }, { stagedPickIds: ['p1'] })

    expect(screen.getByText("I'm done — finish with 2 selected")).toBeInTheDocument()

    await user.click(screen.getByTestId('keyword-filter-toggle'))
    await user.click(within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox'))

    // Filtering hid Paper Two, but the counts/banner (derived from
    // selected_paper_ids/stagedPickIds, never from the filtered view) are
    // unchanged, and Continue still submits the same payload regardless.
    expect(screen.getByText("I'm done — finish with 2 selected")).toBeInTheDocument()
    expect(screen.getByTestId('review-target-reached-banner')).toHaveTextContent('reached your target of 2')

    await user.click(screen.getByTestId('review-continue'))
    expect(onSubmitPicks).toHaveBeenCalledWith(undefined, false)
  })

  it('the disclosure and checkboxes expose correct accessible state', async () => {
    const user = userEvent.setup()
    renderPanel({ pending_batch: [paper('p1', 'Paper One', null, ['topic a'])] })

    const toggle = screen.getByTestId('keyword-filter-toggle')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(toggle).toHaveAttribute('aria-controls', 'keyword-filter-panel')
    expect(screen.queryByTestId('keyword-filter-panel')).not.toBeInTheDocument()

    await user.click(toggle)

    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    const panel = screen.getByTestId('keyword-filter-panel')
    expect(panel).toHaveAttribute('id', 'keyword-filter-panel')
    const checkbox = within(screen.getByTestId('keyword-filter-option-topic a')).getByRole('checkbox')
    expect(checkbox).not.toBeChecked()

    await user.click(checkbox)
    expect(checkbox).toBeChecked()
  })
})
