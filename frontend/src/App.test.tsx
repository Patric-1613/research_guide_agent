import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { useCurationSession } from './hooks/useCurationSession'
import { curationApi } from './api/client'
import type { CurationStateResponse } from './api/types'

vi.mock('./hooks/useCurationSession')
vi.mock('./api/client', () => ({
  curationApi: {
    listReviews: vi.fn().mockResolvedValue([]),
  },
}))

function fullState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 'transformers', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: [], refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null,
    ...overrides,
  }
}

function mockSession(state: CurationStateResponse | null, overrides: Partial<ReturnType<typeof useCurationSession>> = {}) {
  vi.mocked(useCurationSession).mockReturnValue({
    sessionId: state?.session_id ?? null,
    state,
    loading: false,
    error: null,
    turnEvents: [],
    openReview: vi.fn(),
    startReview: vi.fn(),
    submitPicks: vi.fn(),
    generateReport: vi.fn(),
    regenerateReport: vi.fn(),
    sendChatMessage: vi.fn(),
    refresh: vi.fn(),
    ...overrides,
  })
}

describe('App', () => {
  afterEach(() => {
    window.history.pushState({}, '', '/')
  })

  it('renders the persistent app title regardless of mode', () => {
    mockSession(fullState())
    render(<App />)
    expect(screen.getByText('Research Helper Agent')).toBeInTheDocument()
  })

  it('review mode: shows the candidate browser center panel and the pool summary on the right', () => {
    mockSession(fullState())
    render(<App />)

    expect(screen.getByTestId('review-continue')).toBeInTheDocument()
    expect(screen.getByTestId('stat-selected')).toBeInTheDocument()
  })

  it('auto-switches from Review to Report the moment curation finishes (stage becomes synthesize)', () => {
    mockSession(fullState({ stage: 'curate', pending_batch: [] }))
    const { rerender } = render(<App />)
    expect(screen.getByTestId('review-continue')).toBeInTheDocument()

    mockSession(fullState({ stage: 'synthesize', pending_batch: null }))
    rerender(<App />)

    expect(screen.getByTestId('generate-report')).toBeInTheDocument()
    // The pool summary (review-mode-only) must be gone once we've moved
    // off Review -- the whole point of point #6 in the redesign brief.
    expect(screen.queryByTestId('stat-selected')).not.toBeInTheDocument()
  })

  it('chat mode: the center panel shows only the conversation, no paper pool alongside it', async () => {
    const user = userEvent.setup()
    mockSession(fullState({ stage: 'synthesize', pending_batch: null, report: {
      findings: { content: 'f', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'l', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 's', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
    } }))
    render(<App />)

    await user.click(screen.getByTestId('workspace-mode-chat'))

    expect(screen.getByTestId('persistent-input')).toBeInTheDocument()
    expect(screen.queryByTestId('stat-selected')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '+ Add to review' })).not.toBeInTheDocument()
  })

  it('selecting a different review resets the workspace mode back to Review', async () => {
    const user = userEvent.setup()
    const openReview = vi.fn()
    vi.mocked(curationApi.listReviews).mockResolvedValue([
      { session_id: 's1', topic: 'transformers', stage: 'synthesize', selected_count: 5, target_count: 5, has_report: false, has_chat: false },
      { session_id: 's2', topic: 'other', stage: 'curate', selected_count: 0, target_count: 5, has_report: false, has_chat: false },
    ])
    mockSession(fullState({ stage: 'synthesize', pending_batch: null }), { openReview })
    render(<App />)

    await user.click(screen.getByTestId('workspace-mode-report'))
    expect(screen.getByTestId('generate-report')).toBeInTheDocument()

    await user.click(await screen.findByTestId('review-card-s2'))
    expect(openReview).toHaveBeenCalledWith('s2')
  })

  it('a reload with ?mode=chat in the URL stays on Chat once the (already-unlocked) state loads', () => {
    window.history.pushState({}, '', '/?session=s1&mode=chat')
    mockSession(fullState({ stage: 'synthesize', pending_batch: null }))
    render(<App />)

    expect(screen.getByTestId('persistent-input')).toBeInTheDocument()
    expect(screen.queryByTestId('generate-report')).not.toBeInTheDocument()
  })

  it('a stale ?mode=chat URL from before curation finished falls back to Review, not a broken locked view', () => {
    window.history.pushState({}, '', '/?session=s1&mode=chat')
    mockSession(fullState({ stage: 'curate', pending_batch: [] }))
    render(<App />)

    expect(screen.getByTestId('review-continue')).toBeInTheDocument()
  })
})
