import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TurnFeed } from './TurnFeed'
import type { CurationStateResponse, PaperOut } from '../../api/types'
import type { TurnEvent } from '../../hooks/useCurationSession'

function paper(id: string): PaperOut {
  return {
    paper_id: id, title: `Paper ${id}`, authors: [], year: 2024, venue: null,
    abstract: null, url: null, doi: null, citation_count: null, source: 'arxiv',
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

describe('TurnFeed', () => {
  it('curate stage: renders past TurnBlocks plus the currently-pending batch, with correct turn numbering', () => {
    const events: TurnEvent[] = [
      { turnNumber: 1, refilled: false, batchSize: 10, reserveRemainingAfter: 5, pickedPaperIds: ['a', 'b'] },
    ]
    const state = baseState({ pending_batch: [paper('p1')], refilled: true })

    render(<TurnFeed state={state} turnEvents={events} />)

    expect(screen.getByText(/Turn 1/)).toBeInTheDocument()
    expect(screen.getByText(/from existing pool/)).toBeInTheDocument()
    expect(screen.getByText(/You selected/)).toBeInTheDocument()
    // The pending (not-yet-completed) turn is numbered 2, and reflects
    // the CURRENT refilled flag (true), not the completed turn's (false).
    expect(screen.getByText('Turn 2 — new search')).toBeInTheDocument()
  })

  it('synthesize stage with no report yet: shows a completion message, not chat bubbles', () => {
    const state = baseState({ stage: 'synthesize', selected_papers: [paper('p1'), paper('p2')], selected_paper_ids: ['p1', 'p2'] })
    render(<TurnFeed state={state} turnEvents={[]} />)
    expect(screen.getByText(/Curation complete/)).toBeInTheDocument()
    expect(screen.getByText(/2 papers selected/)).toBeInTheDocument()
  })

  it('synthesize stage with a report: renders chat_history as message bubbles', () => {
    const state = baseState({
      stage: 'synthesize',
      report: { findings: { content: 'f', cited_papers: [], cited_web_articles: [] }, limitations: { content: '', cited_papers: [], cited_web_articles: [] }, future_scope: { content: '', cited_papers: [], cited_web_articles: [] }, skipped_paper_ids: [] },
      chat_history: [
        { role: 'user', content: 'what is this about?' },
        { role: 'assistant', content: 'It is about X.' },
      ],
    })
    render(<TurnFeed state={state} turnEvents={[]} />)
    expect(screen.getByText('what is this about?')).toBeInTheDocument()
    expect(screen.getByText('It is about X.')).toBeInTheDocument()
  })
})
