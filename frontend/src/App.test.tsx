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
  // display_title defaults to whatever topic ends up being unless
  // separately overridden -- TopicHeader now renders display_title, not
  // topic, so tests that only override `topic` still see that text.
  const topic = overrides.topic ?? 'transformers'
  return {
    session_id: 's1', topic, display_title: topic, stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: [], refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
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
    deleteReview: vi.fn(),
    selectFromHistory: vi.fn(),
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

  it('shows the canonicalized display_title in the topic header, not the raw topic (Phase 8, item 5)', () => {
    mockSession(fullState({ topic: 'cars cooling system', display_title: 'Automotive Engine Cooling Systems' }))
    render(<App />)

    expect(screen.getByText('Automotive Engine Cooling Systems')).toBeInTheDocument()
    expect(screen.queryByText('cars cooling system')).not.toBeInTheDocument()
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

  it('reopening an already-finished review (mode reset to review by handleSelectReview) stays on Review, not bounced to Report (Phase 8, item 3 fallout)', async () => {
    const user = userEvent.setup()
    vi.mocked(curationApi.listReviews).mockResolvedValue([
      { session_id: 's2', topic: 'already done', display_title: 'already done', stage: 'synthesize', selected_count: 3, target_count: 3, has_report: true, has_chat: false },
    ])
    // Mount already on a DIFFERENT, still-in-progress session first -- the
    // bug only reproduces when a session that's unlocked from the moment
    // it's opened lands in 'review' mode, as handleSelectReview does.
    mockSession(fullState({ session_id: 's1', stage: 'curate', pending_batch: [] }))
    render(<App />)
    expect(screen.getByTestId('review-continue')).toBeInTheDocument()

    // Reopen an already-finished review: same shape handleSelectReview
    // produces (mode reset to 'review', state already unlocked).
    mockSession(fullState({
      session_id: 's2', stage: 'synthesize', pending_batch: null,
      selected_papers: [{
        paper_id: 'p1', title: 'Already Selected', authors: [], year: null, venue: null,
        abstract: null, url: null, doi: null, citation_count: null, source: 'arxiv',
        source_urls: {}, score: null,
      }],
      selected_paper_ids: ['p1'],
    }))
    await user.click(await screen.findByTestId('review-card-s2'))

    // Must show the papers in Review mode -- NOT get bounced to Report.
    // (Renders in both the center card and the right-panel selected list,
    // hence findAllByText rather than a single-match query.)
    expect((await screen.findAllByText('Already Selected')).length).toBeGreaterThan(0)
    expect(screen.queryByTestId('generate-report')).not.toBeInTheDocument()
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
      { session_id: 's1', topic: 'transformers', display_title: 'transformers', stage: 'synthesize', selected_count: 5, target_count: 5, has_report: false, has_chat: false },
      { session_id: 's2', topic: 'other', display_title: 'other', stage: 'curate', selected_count: 0, target_count: 5, has_report: false, has_chat: false },
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

  it('the "Browse past turns" toggle only appears when there is real turn history', () => {
    mockSession(fullState({ turn_history: [] }))
    const { rerender } = render(<App />)
    expect(screen.queryByTestId('open-turn-history')).not.toBeInTheDocument()

    mockSession(fullState({
      turn_history: [{ turn_number: 1, refilled: false, batch: [] }],
    }))
    rerender(<App />)
    expect(screen.getByText('Browse past turns (1)')).toBeInTheDocument()
  })

  it('opening turn history replaces the mode panel AND hides the pool summary, from any workspace mode (Phase 9f)', async () => {
    const user = userEvent.setup()
    mockSession(fullState({
      turn_history: [{
        turn_number: 1, refilled: false,
        batch: [{
          paper_id: 'p0', title: 'Historical Paper', authors: [], year: null, venue: null,
          abstract: null, url: null, doi: null, citation_count: null, source: 'arxiv',
          source_urls: {}, score: null,
        }],
      }],
    }))
    render(<App />)

    expect(screen.getByTestId('stat-selected')).toBeInTheDocument() // review mode's pool summary, present initially

    await user.click(screen.getByTestId('open-turn-history'))

    expect(screen.getByText('Historical Paper')).toBeInTheDocument()
    expect(screen.getByTestId('close-turn-history')).toBeInTheDocument()
    // Review mode's own panel and pool summary are both gone while browsing history.
    expect(screen.queryByTestId('review-continue')).not.toBeInTheDocument()
    expect(screen.queryByTestId('stat-selected')).not.toBeInTheDocument()
  })

  it('closing turn history returns to the previously-active workspace mode', async () => {
    const user = userEvent.setup()
    mockSession(fullState({
      turn_history: [{ turn_number: 1, refilled: false, batch: [] }],
    }))
    render(<App />)

    await user.click(screen.getByTestId('open-turn-history'))
    expect(screen.queryByTestId('review-continue')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('close-turn-history'))
    expect(screen.getByTestId('review-continue')).toBeInTheDocument()
  })
})
