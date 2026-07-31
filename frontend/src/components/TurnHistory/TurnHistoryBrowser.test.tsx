import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { TurnHistoryBrowser } from './TurnHistoryBrowser'
import type { CurationStateResponse, PaperOut } from '../../types'

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
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    ...overrides,
  }
}

describe('TurnHistoryBrowser', () => {
  it('renders every past turn, most recent first, with abstracts', () => {
    const state = baseState({
      turn_history: [
        { turn_number: 1, refilled: false, batch: [paper('p0', 'Turn One Paper', 'Abstract one.')] },
        { turn_number: 2, refilled: true, batch: [paper('p1', 'Turn Two Paper', 'Abstract two.')] },
      ],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    expect(screen.getByText('Past turns (2)')).toBeInTheDocument()
    expect(screen.getByText('Turn One Paper')).toBeInTheDocument()
    expect(screen.getByText('Abstract one.')).toBeInTheDocument()
    expect(screen.getByText('Turn Two Paper')).toBeInTheDocument()
    // Most recent turn (2) listed before turn 1.
    const headings = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(headings[0]).toContain('Turn 2')
    expect(headings[1]).toContain('Turn 1')
    expect(screen.getByText(/Turn 2.*new search/)).toBeInTheDocument()
    expect(screen.getByText(/Turn 1.*from existing pool/)).toBeInTheDocument()
  })

  it('an already-selected paper shows no action at all', () => {
    const state = baseState({
      selected_paper_ids: ['p0'],
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'Already Selected')] }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Remove/ })).not.toBeInTheDocument()
  })

  it('stage=="curate": clicking add STAGES the paper (via onAdd), does not call onSelectFromHistory', async () => {
    const user = userEvent.setup()
    const onAdd = vi.fn()
    const onSelectFromHistory = vi.fn()
    const state = baseState({
      stage: 'curate',
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={onAdd} onRemoveStaged={vi.fn()} onSelectFromHistory={onSelectFromHistory} onClose={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: '+ Add to review' }))

    expect(onAdd).toHaveBeenCalledWith('p0')
    expect(onSelectFromHistory).not.toHaveBeenCalled()
    // Curate-stage note: picks apply on the next Continue, not immediately.
    expect(screen.getByText(/added the next time you hit Continue/)).toBeInTheDocument()
  })

  it('stage=="curate": a staged historical paper shows remove, wired to onRemoveStaged', async () => {
    const user = userEvent.setup()
    const onRemoveStaged = vi.fn()
    const state = baseState({
      stage: 'curate',
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={['p0']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={onRemoveStaged} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Remove From Turn 1' }))

    expect(onRemoveStaged).toHaveBeenCalledWith('p0')
  })

  it('stage=="synthesize": clicking add calls onSelectFromHistory IMMEDIATELY, not onAdd', async () => {
    const user = userEvent.setup()
    const onAdd = vi.fn()
    const onSelectFromHistory = vi.fn().mockResolvedValue(undefined)
    const state = baseState({
      stage: 'synthesize',
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={onAdd} onRemoveStaged={vi.fn()} onSelectFromHistory={onSelectFromHistory} onClose={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: '+ Add to review' }))

    expect(onSelectFromHistory).toHaveBeenCalledWith('p0')
    expect(onAdd).not.toHaveBeenCalled()
    // Synthesize-stage note must NOT show -- picks are immediate here.
    expect(screen.queryByText(/added the next time you hit Continue/)).not.toBeInTheDocument()
  })

  it('reported bug: once a report has been generated, an unselected historical paper is view-only, not "+ Add to review"', () => {
    const onAdd = vi.fn()
    const onSelectFromHistory = vi.fn()
    const state = baseState({
      stage: 'synthesize',
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
      report: {
        findings: { content: 'f', cited_papers: [], cited_web_articles: [] },
        limitations: { content: '', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: '', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
      },
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={onAdd} onRemoveStaged={vi.fn()} onSelectFromHistory={onSelectFromHistory} onClose={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
    expect(screen.getByText(/This review is locked/)).toBeInTheDocument()
  })

  it('reported bug: once chat has started, an unselected historical paper is view-only', () => {
    const state = baseState({
      stage: 'synthesize',
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
      chat_history: [{ role: 'user', content: 'hi' }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
    expect(screen.getByText(/This review is locked/)).toBeInTheDocument()
  })

  it('once locked, staged-but-not-yet-submitted historical picks also render as view-only, not "Remove"', () => {
    const state = baseState({
      stage: 'synthesize',
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
      chat_history: [{ role: 'user', content: 'hi' }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={['p0']} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: /Remove/ })).not.toBeInTheDocument()
  })

  it('not locked (no report, no chat) still behaves as before: synthesize stage picks immediately', () => {
    const state = baseState({
      stage: 'synthesize', report: null, chat_history: [],
      turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p0', 'From Turn 1')] }],
    })
    render(
      <TurnHistoryBrowser
        state={state} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: '+ Add to review' })).toBeInTheDocument()
    expect(screen.queryByText(/This review is locked/)).not.toBeInTheDocument()
  })

  it('clicking Close calls onClose', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(
      <TurnHistoryBrowser
        state={baseState()} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={onClose}
      />,
    )

    await user.click(screen.getByTestId('close-turn-history'))

    expect(onClose).toHaveBeenCalled()
  })

  it('shows an empty-state message with no turn history', () => {
    render(
      <TurnHistoryBrowser
        state={baseState()} stagedPickIds={[]} disabled={false}
        onAdd={vi.fn()} onRemoveStaged={vi.fn()} onSelectFromHistory={vi.fn()} onClose={vi.fn()}
      />,
    )

    expect(screen.getByText('No turns yet.')).toBeInTheDocument()
  })
})
