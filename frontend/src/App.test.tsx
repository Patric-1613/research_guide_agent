import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { useCurationSession } from './hooks/useCurationSession'
import { curationApi } from './lib/api/client'
import type { CurationStateResponse } from './types'

vi.mock('./hooks/useCurationSession')
vi.mock('./lib/api/client', () => ({
  curationApi: {
    listReviews: vi.fn().mockResolvedValue([]),
    // report-quality Phase R5B: CurationWorkspacePage calls this
    // unconditionally on every render (not just when a report exists),
    // so the mock needs it even though these App-level tests never
    // assert on export behavior themselves.
    getReportExportUrl: vi.fn().mockReturnValue('http://test.local/curation/s1/report/export?format=markdown'),
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
    report_versions: [], active_report_version_id: null, chat_references: [],
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
    lastChatSearchMeta: null,
    reportPossiblyStale: false,
    lastAddToReportResult: null,
    dismissReportStaleWarning: vi.fn(),
    openReview: vi.fn(),
    startReview: vi.fn(),
    submitPicks: vi.fn(),
    generateReport: vi.fn(),
    regenerateReport: vi.fn(),
    activateReportVersion: vi.fn(),
    sendChatMessage: vi.fn(),
    chatStreamActive: false,
    chatStreamPhase: null,
    chatStreamText: '',
    chatStreamSyncFailed: false,
    sendChatMessageStreaming: vi.fn(),
    cancelChatStream: vi.fn(),
    deleteExchanges: vi.fn(),
    addExchangesToReport: vi.fn(),
    editExchange: vi.fn(),
    deleteReview: vi.fn(),
    selectFromHistory: vi.fn(),
    reopenReview: vi.fn(),
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

  it('chat UX fixes, bug 1: "Browse past turns" stays visible in Chat mode', async () => {
    const user = userEvent.setup()
    mockSession(fullState({
      stage: 'synthesize', pending_batch: null,
      turn_history: [{ turn_number: 1, refilled: false, batch: [] }],
    }))
    render(<App />)

    await user.click(screen.getByTestId('workspace-mode-chat'))

    expect(screen.getByText('Browse past turns (1)')).toBeInTheDocument()
  })

  it('chat UX fixes, bug 1: "Browse past turns" is hidden in Report mode, even with real turn history', async () => {
    const user = userEvent.setup()
    mockSession(fullState({
      stage: 'synthesize', pending_batch: null,
      turn_history: [{ turn_number: 1, refilled: false, batch: [] }],
    }))
    render(<App />)

    await user.click(screen.getByTestId('workspace-mode-report'))

    expect(screen.queryByTestId('open-turn-history')).not.toBeInTheDocument()
  })

  it('chat UX fixes, bug 1: switching to Report mode closes the turn history browser if it was already open', async () => {
    const user = userEvent.setup()
    mockSession(fullState({
      stage: 'synthesize', pending_batch: null,
      turn_history: [{ turn_number: 1, refilled: false, batch: [] }],
    }))
    render(<App />)

    await user.click(screen.getByTestId('open-turn-history'))
    expect(screen.getByTestId('close-turn-history')).toBeInTheDocument()

    await user.click(screen.getByTestId('workspace-mode-report'))

    expect(screen.queryByTestId('close-turn-history')).not.toBeInTheDocument()
    expect(screen.getByTestId('generate-report')).toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10e: "Reopen for more curation" appears once stopped, as long as nothing has been chatted/reported yet', () => {
    mockSession(fullState({ stage: 'synthesize', pending_batch: null, report: null, chat_history: [] }))
    render(<App />)
    expect(screen.getByTestId('reopen-review')).toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10e: no reopen action while still curating', () => {
    mockSession(fullState({ stage: 'curate' }))
    render(<App />)
    expect(screen.queryByTestId('reopen-review')).not.toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10e: no reopen action once a report exists', () => {
    mockSession(fullState({
      stage: 'synthesize', pending_batch: null, chat_history: [],
      report: {
        findings: { content: 'f', cited_papers: [], cited_web_articles: [] },
        limitations: { content: '', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: '', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
      },
    }))
    render(<App />)
    expect(screen.queryByTestId('reopen-review')).not.toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10e: no reopen action once chat has started', () => {
    mockSession(fullState({
      stage: 'synthesize', pending_batch: null, report: null,
      chat_history: [{ role: 'user', content: 'hi' }],
    }))
    render(<App />)
    expect(screen.queryByTestId('reopen-review')).not.toBeInTheDocument()
  })

  it('curation-editable-until-locked Phase 10e: clicking "Reopen for more curation" calls reopenReview()', async () => {
    const user = userEvent.setup()
    const reopenReview = vi.fn().mockResolvedValue(undefined)
    mockSession(fullState({ stage: 'synthesize', pending_batch: null, report: null, chat_history: [] }), { reopenReview })
    render(<App />)

    await user.click(screen.getByTestId('reopen-review'))

    expect(reopenReview).toHaveBeenCalled()
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

  // Usage Protection M2.3 Part C/F: the shared error banner every
  // curation action (search/start, curation chat, picks/refill, report
  // generate/regenerate, chat add-to-report) surfaces through, since
  // they all go through useCurationSession's one `error` state.
  describe('usage-protection error banner', () => {
    it('renders an accessible alert with the message when useCurationSession reports an error', () => {
      mockSession(fullState(), { error: 'Another action is already running for this review. Please wait and try again.' })
      render(<App />)

      const alert = screen.getByRole('alert')
      expect(alert).toHaveTextContent('Another action is already running for this review. Please wait and try again.')
      expect(screen.getByTestId('curation-error-banner')).toBe(alert)
    })

    it('renders no alert when there is no error', () => {
      mockSession(fullState())
      render(<App />)

      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    })

    it('existing report/chat/selection state remains rendered alongside the error banner', () => {
      mockSession(
        fullState({
          stage: 'synthesize', pending_batch: null,
          selected_papers: [{
            paper_id: 'p1', title: 'Still Selected', authors: [], year: null, venue: null,
            abstract: null, url: null, doi: null, citation_count: null, source: 'arxiv',
            source_urls: {}, score: null,
          }],
          selected_paper_ids: ['p1'],
          report: {
            findings: { content: 'f', cited_papers: [], cited_web_articles: [] },
            limitations: { content: 'l', cited_papers: [], cited_web_articles: [] },
            future_scope: { content: 's', cited_papers: [], cited_web_articles: [] },
            skipped_paper_ids: [],
          },
        }),
        { error: 'This review has reached its selected-paper limit. Remove some papers before adding more.' },
      )
      render(<App />)

      expect(screen.getByRole('alert')).toBeInTheDocument()
      // Renders in both the center card and the right-panel selected
      // list (same multi-match reasoning as this file's own pre-existing
      // "Already Selected" test above).
      expect(screen.getAllByText('Still Selected').length).toBeGreaterThan(0)
    })

    it('export links remain present and usable after a paid-action rejection', () => {
      mockSession(
        fullState({
          stage: 'synthesize', pending_batch: null,
          report: {
            findings: { content: 'f', cited_papers: [], cited_web_articles: [] },
            limitations: { content: 'l', cited_papers: [], cited_web_articles: [] },
            future_scope: { content: 's', cited_papers: [], cited_web_articles: [] },
            skipped_paper_ids: [],
          },
        }),
        { error: 'Usage protection is temporarily unavailable, so no paid action was started. Please try again shortly.' },
      )
      render(<App />)

      expect(screen.getByRole('alert')).toBeInTheDocument()
      expect(vi.mocked(curationApi.getReportExportUrl)).toHaveBeenCalled()
    })

    it('loading is false after a failed action, leaving controls enabled', () => {
      mockSession(fullState(), { error: 'x', loading: false })
      render(<App />)

      expect(screen.getByTestId('review-continue')).not.toBeDisabled()
    })
  })

  describe('client-side preventative text limits (Usage Protection M2.3 Part D)', () => {
    it('the new-review topic input has maxLength=2000', async () => {
      const user = userEvent.setup()
      vi.mocked(curationApi.listReviews).mockResolvedValue([])
      mockSession(null)
      render(<App />)

      await user.click(screen.getByRole('button', { name: '+ New review' }))
      expect(screen.getByTestId('new-review-topic')).toHaveAttribute('maxlength', '2000')
    })

    it('the chat message input has maxLength=2000', async () => {
      const user = userEvent.setup()
      mockSession(fullState({ stage: 'synthesize', pending_batch: null }))
      render(<App />)

      await user.click(screen.getByTestId('workspace-mode-chat'))
      expect(screen.getByTestId('persistent-input')).toHaveAttribute('maxlength', '2000')
    })

    it('the review refinement input has maxLength=2000', () => {
      mockSession(fullState())
      render(<App />)

      expect(screen.getByTestId('review-refinement-input')).toHaveAttribute('maxlength', '2000')
    })

    it('pasted input at exactly the limit is not silently truncated below it by application logic', async () => {
      const user = userEvent.setup()
      mockSession(fullState({ stage: 'synthesize', pending_batch: null }))
      render(<App />)
      await user.click(screen.getByTestId('workspace-mode-chat'))

      const input = screen.getByTestId('persistent-input') as HTMLInputElement
      const exactly2000 = 'a'.repeat(2000)
      await user.click(input)
      await user.paste(exactly2000)

      expect(input.value.length).toBe(2000) // the full, untruncated text the maxLength attribute allows
    })
  })
})
