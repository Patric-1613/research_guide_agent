import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PaperPool } from './PaperPool'
import type { CurationStateResponse, PaperOut } from '../../api/types'

function paper(id: string, title: string): PaperOut {
  return {
    paper_id: id, title, authors: ['A'], year: 2024, venue: 'arXiv',
    abstract: null, url: null, doi: null, citation_count: 10, source: 'arxiv',
    source_urls: {}, score: null,
  }
}

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null,
    ...overrides,
  }
}

describe('PaperPool', () => {
  it('lists pending_batch papers under "New this turn" and confirmed picks under "Selected"', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')],
      selected_papers: [paper('p0', 'Already Picked')],
      selected_paper_ids: ['p0'],
    })

    render(<PaperPool state={state} stagedPickIds={[]} onAdd={vi.fn()} onRemoveStaged={vi.fn()} />)

    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByText('Paper Two')).toBeInTheDocument()
    expect(screen.getByText('Already Picked')).toBeInTheDocument()
  })

  it('a paper moves from "New this turn" to "Selected" once staged, without a backend round trip', () => {
    const state = baseState({ pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')] })

    const { rerender } = render(
      <PaperPool state={state} stagedPickIds={[]} onAdd={vi.fn()} onRemoveStaged={vi.fn()} />,
    )
    // Both start under "New this turn" -- confirmed by an Add button per card.
    expect(screen.getAllByRole('button', { name: '+ Add to review' })).toHaveLength(2)

    rerender(<PaperPool state={state} stagedPickIds={['p1']} onAdd={vi.fn()} onRemoveStaged={vi.fn()} />)

    // p1 now only has a remove (×) affordance; only p2 still offers Add.
    expect(screen.getAllByRole('button', { name: '+ Add to review' })).toHaveLength(1)
    expect(screen.getByRole('button', { name: 'Remove Paper One' })).toBeInTheDocument()
  })

  it('clicking + Add to review calls onAdd with that paper\'s id', async () => {
    const user = userEvent.setup()
    const onAdd = vi.fn()
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })

    render(<PaperPool state={state} stagedPickIds={[]} onAdd={onAdd} onRemoveStaged={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: '+ Add to review' }))

    expect(onAdd).toHaveBeenCalledWith('p1')
  })

  it('clicking × on a staged paper calls onRemoveStaged with that paper\'s id', async () => {
    const user = userEvent.setup()
    const onRemoveStaged = vi.fn()
    const state = baseState({ pending_batch: [paper('p1', 'Paper One')] })

    render(<PaperPool state={state} stagedPickIds={['p1']} onAdd={vi.fn()} onRemoveStaged={onRemoveStaged} />)
    await user.click(screen.getByRole('button', { name: 'Remove Paper One' }))

    expect(onRemoveStaged).toHaveBeenCalledWith('p1')
  })

  it('already-confirmed selected papers (from the backend) have no remove option -- unpicking isn\'t a supported backend action', () => {
    const state = baseState({ selected_papers: [paper('p0', 'Already Picked')], selected_paper_ids: ['p0'] })

    render(<PaperPool state={state} stagedPickIds={[]} onAdd={vi.fn()} onRemoveStaged={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Remove Already Picked' })).not.toBeInTheDocument()
  })

  it('shows the real reserve_remaining count in the pool status line', () => {
    const state = baseState({ reserve_remaining: 16 })
    render(<PaperPool state={state} stagedPickIds={[]} onAdd={vi.fn()} onRemoveStaged={vi.fn()} />)
    expect(screen.getByText(/16 more candidates already fetched/)).toBeInTheDocument()
  })
})
