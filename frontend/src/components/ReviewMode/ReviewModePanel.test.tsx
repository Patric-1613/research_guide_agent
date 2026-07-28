import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReviewModePanel } from './ReviewModePanel'
import type { CurationStateResponse, PaperOut } from '../../api/types'
import type { TurnEvent } from '../../hooks/useCurationSession'

function paper(id: string, title: string, abstract: string | null = null): PaperOut {
  return {
    paper_id: id, title, authors: ['A'], year: 2024, venue: 'arXiv',
    abstract, url: null, doi: null, citation_count: 10, source: 'arxiv',
    source_urls: {}, score: null,
  }
}

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', display_title: 't', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null,
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

  it('renders past TurnBlocks above the current batch, with correct turn numbering', () => {
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

    expect(screen.getByText(/Turn 1/)).toBeInTheDocument()
    expect(screen.getByText('Turn 2 — new search')).toBeInTheDocument()
  })
})
