import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PoolSummaryPanel } from './PoolSummaryPanel'
import type { CurationStateResponse, PaperOut } from '../../types'

function paper(id: string, title: string): PaperOut {
  return {
    paper_id: id, title, authors: ['A'], year: 2024, venue: 'arXiv',
    abstract: null, url: null, doi: null, citation_count: 10, source: 'arxiv',
    source_urls: {}, score: null, keywords: [],
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

describe('PoolSummaryPanel', () => {
  it('shows big-number counts for new-this-turn and selected, not full cards', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Paper One'), paper('p2', 'Paper Two')],
      selected_papers: [paper('p0', 'Already Picked')],
      selected_paper_ids: ['p0'],
    })
    render(<PoolSummaryPanel state={state} stagedPickIds={['p1']} />)

    // p1 is staged (no longer "new"), so only p2 counts as new-this-turn.
    expect(screen.getByTestId('stat-new-this-turn')).toHaveTextContent('1')
    // p0 confirmed + p1 staged = 2 selected total.
    expect(screen.getByTestId('stat-selected')).toHaveTextContent('2')
    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
  })

  it('lists selected paper titles compactly, staged ones distinguished by color not chrome', () => {
    const state = baseState({
      pending_batch: [paper('p1', 'Staged Paper')],
      selected_papers: [paper('p0', 'Confirmed Paper')],
      selected_paper_ids: ['p0'],
    })
    render(<PoolSummaryPanel state={state} stagedPickIds={['p1']} />)

    expect(screen.getByText('Staged Paper')).toBeInTheDocument()
    expect(screen.getByText('Confirmed Paper')).toBeInTheDocument()
  })

  it('empty state: shows "Nothing selected yet" when nothing is staged or confirmed', () => {
    render(<PoolSummaryPanel state={baseState()} stagedPickIds={[]} />)
    expect(screen.getByText('Nothing selected yet.')).toBeInTheDocument()
  })

  it('shows the real reserve_remaining count in the pool status line', () => {
    const state = baseState({ reserve_remaining: 16 })
    render(<PoolSummaryPanel state={state} stagedPickIds={[]} />)
    expect(screen.getByText(/16 more candidates already fetched/)).toBeInTheDocument()
  })

  it('UXH.1 (UX-01): an id present in both selected_paper_ids and stagedPickIds counts once, in both the header and the stat tile', () => {
    const state = baseState({
      selected_paper_ids: ['p0', 'p1'],
      selected_papers: [paper('p0', 'Already Picked'), paper('p1', 'Also Already Picked')],
      target_count: 10,
    })
    // p1 staged again (structurally shouldn't happen via the real UI, but
    // the count must stay correct defensively regardless) alongside a
    // genuinely new staged id p2.
    render(<PoolSummaryPanel state={state} stagedPickIds={['p1', 'p2']} />)

    // 3 distinct ids (p0, p1, p2), not 4 (2 persisted + 2 staged).
    expect(screen.getByTestId('stat-selected')).toHaveTextContent('3')
    expect(screen.getByText('Paper pool — 3/10 target')).toBeInTheDocument()
  })

  it('UXH.1 (UX-01): the pool header count includes staged picks, not persisted alone (previously under-counted)', () => {
    const state = baseState({ selected_paper_ids: ['p0'], target_count: 10 })
    render(<PoolSummaryPanel state={state} stagedPickIds={['p1', 'p2']} />)

    expect(screen.getByText('Paper pool — 3/10 target')).toBeInTheDocument()
  })
})
