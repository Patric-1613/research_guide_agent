import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useCurationSession, getSessionIdFromUrl } from './useCurationSession'
import { curationApi } from '../lib/api/client'
import type { CurationChatResponse, CurationStateResponse, CurationTurnResponse } from '../types'

vi.mock('../lib/api/client', () => ({
  curationApi: {
    getState: vi.fn(),
    start: vi.fn(),
    picks: vi.fn(),
    generateReport: vi.fn(),
    regenerateReport: vi.fn(),
    activateReportVersion: vi.fn(),
    chat: vi.fn(),
    listReviews: vi.fn(),
    deleteReview: vi.fn(),
    deleteChatExchanges: vi.fn(),
    addChatExchangesToReport: vi.fn(),
    editChatExchange: vi.fn(),
  },
}))

function fullState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 'transformers', display_title: 'transformers', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    report_versions: [], active_report_version_id: null,
    ...overrides,
  }
}

function chatResponse(overrides: Partial<CurationChatResponse> = {}): CurationChatResponse {
  return {
    answer: 'the answer', answerable: true, cited_papers: [], cited_web_articles: [],
    web_offer_made: false, web_offer_declined: false, web_search_used: false, new_web_articles_found: null,
    report_update_offer_made: false, report_update_declined: false, report_updated: false, chat_history: [],
    ...overrides,
  }
}

beforeEach(() => {
  window.history.pushState({}, '', '/')
  vi.clearAllMocks()
})

afterEach(() => {
  window.history.pushState({}, '', '/')
})

describe('useCurationSession', () => {
  it('loads state from the backend on mount when the URL already names a session (the refresh case)', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ selected_paper_ids: ['p1', 'p2'] }))

    const { result } = renderHook(() => useCurationSession())

    expect(result.current.sessionId).toBe('s1')
    await waitFor(() => expect(result.current.state).not.toBeNull())

    expect(curationApi.getState).toHaveBeenCalledWith('s1')
    expect(result.current.state?.selected_paper_ids).toEqual(['p1', 'p2'])
  })

  it('does not call getState at all when no session is named in the URL', async () => {
    const { result } = renderHook(() => useCurationSession())
    expect(result.current.sessionId).toBeNull()
    expect(result.current.state).toBeNull()
    expect(curationApi.getState).not.toHaveBeenCalled()
  })

  it('startReview puts the new session_id in the URL and loads full state', async () => {
    const turnResponse: CurationTurnResponse = {
      session_id: 'new-session', stage: 'curate', target_count: 10,
      selected_paper_ids: [], batch: [{ paper_id: 'p1' } as never], stop_reason: null,
      refilled: false, reserve_remaining: 5, refinement_notes: [],
    }
    vi.mocked(curationApi.start).mockResolvedValue(turnResponse)
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 'new-session' }))

    const { result } = renderHook(() => useCurationSession())

    await act(async () => {
      await result.current.startReview('transformers', 10)
    })

    expect(curationApi.start).toHaveBeenCalledWith({ topic: 'transformers', target_count: 10 })
    expect(getSessionIdFromUrl()).toBe('new-session')
    expect(result.current.sessionId).toBe('new-session')
    expect(result.current.state?.session_id).toBe('new-session')
  })

  it('report-quality Phase R3: activateReportVersion calls the API and refreshes state to the newly active version', async () => {
    vi.mocked(curationApi.getState)
      .mockResolvedValueOnce(fullState({
        session_id: 's1',
        report_versions: [
          { version_id: 'v1', version_number: 1, created_at: null, report_template: 'analytical', generation_reason: 'initial', is_active: false },
          { version_id: 'v2', version_number: 2, created_at: null, report_template: 'analytical', generation_reason: 'regenerate', is_active: true },
        ],
        active_report_version_id: 'v2',
      }))
      .mockResolvedValueOnce(fullState({
        session_id: 's1',
        report_versions: [
          { version_id: 'v1', version_number: 1, created_at: null, report_template: 'analytical', generation_reason: 'initial', is_active: true },
          { version_id: 'v2', version_number: 2, created_at: null, report_template: 'analytical', generation_reason: 'regenerate', is_active: false },
        ],
        active_report_version_id: 'v1',
      }))
    vi.mocked(curationApi.activateReportVersion).mockResolvedValue({
      findings: { content: '', cited_papers: [], cited_web_articles: [] },
      limitations: { content: '', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: '', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
    })
    window.history.pushState({}, '', '/?session=s1')

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state?.active_report_version_id).toBe('v2'))

    await act(async () => {
      await result.current.activateReportVersion('v1')
    })

    expect(curationApi.activateReportVersion).toHaveBeenCalledWith('s1', 'v1')
    expect(result.current.state?.active_report_version_id).toBe('v1')
  })

  it('submitPicks records a TurnEvent from the PRIOR state before reloading fresh state', async () => {
    window.history.pushState({}, '', '/?session=s1')
    const pendingBatch = [{ paper_id: 'p1' }, { paper_id: 'p2' }] as never
    vi.mocked(curationApi.getState).mockResolvedValue(
      fullState({ pending_batch: pendingBatch, refilled: true, reserve_remaining: 7 }),
    )

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.picks).mockResolvedValue({
      session_id: 's1', stage: 'curate', target_count: 10, selected_paper_ids: ['p1'],
      batch: [], stop_reason: null, refilled: false, reserve_remaining: 6, refinement_notes: [],
    })
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ selected_paper_ids: ['p1'] }))

    await act(async () => {
      await result.current.submitPicks(['p1'])
    })

    expect(curationApi.picks).toHaveBeenCalledWith('s1', { picked_paper_ids: ['p1'], stop: false })
    expect(result.current.turnEvents).toHaveLength(1)
    // The recorded turn reflects the batch that WAS pending before this
    // call, not anything from the post-pick response.
    expect(result.current.turnEvents[0]).toMatchObject({
      turnNumber: 1, refilled: true, batchSize: 2, reserveRemainingAfter: 7, pickedPaperIds: ['p1'],
    })
  })

  it('turnEvents resets to empty when loadState resolves for a DIFFERENT session_id than before', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state?.session_id).toBe('s1'))

    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's2', topic: 'other topic' }))
    await act(async () => {
      await result.current.openReview('s2')
    })

    expect(result.current.turnEvents).toEqual([])
    expect(result.current.state?.session_id).toBe('s2')
  })

  it('surfaces a clean error message when an action fails, without crashing', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockRejectedValue(new Error('network down'))

    const { result } = renderHook(() => useCurationSession())

    await waitFor(() => expect(result.current.error).toBe('Error: network down'))
    expect(result.current.loading).toBe(false)
  })

  it('deleteReview: deleting the CURRENTLY OPEN session clears sessionId/state and the URL (Phase 8, item 1)', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1' }))
    vi.mocked(curationApi.deleteReview).mockResolvedValue({ session_id: 's1', deleted: true })

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state?.session_id).toBe('s1'))

    await act(async () => {
      await result.current.deleteReview('s1')
    })

    expect(curationApi.deleteReview).toHaveBeenCalledWith('s1')
    expect(result.current.sessionId).toBeNull()
    expect(result.current.state).toBeNull()
    expect(getSessionIdFromUrl()).toBeNull()
  })

  it('deleteReview: deleting a DIFFERENT session leaves the currently open one untouched', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1' }))
    vi.mocked(curationApi.deleteReview).mockResolvedValue({ session_id: 's2', deleted: true })

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state?.session_id).toBe('s1'))

    await act(async () => {
      await result.current.deleteReview('s2')
    })

    expect(curationApi.deleteReview).toHaveBeenCalledWith('s2')
    expect(result.current.sessionId).toBe('s1')
    expect(result.current.state?.session_id).toBe('s1')
    expect(getSessionIdFromUrl()).toBe('s1')
  })

  it('chat-ux-fixes bug 2: sendChatMessage captures web_search_used/new_web_articles_found from the response as lastChatSearchMeta', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockResolvedValue(chatResponse({ web_search_used: true, new_web_articles_found: 2 }))

    await act(async () => {
      await result.current.sendChatMessage('yes')
    })

    expect(result.current.lastChatSearchMeta).toEqual({ webSearchUsed: true, newWebArticlesFound: 2 })
  })

  it('chat-ux-fixes bug 2: lastChatSearchMeta is null when the reply did not use a web search, even after a PRIOR reply did', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockResolvedValue(chatResponse({ web_search_used: true, new_web_articles_found: 1 }))
    await act(async () => {
      await result.current.sendChatMessage('yes')
    })
    expect(result.current.lastChatSearchMeta).not.toBeNull()

    vi.mocked(curationApi.chat).mockResolvedValue(chatResponse({ web_search_used: false, new_web_articles_found: null }))
    await act(async () => {
      await result.current.sendChatMessage('a normal follow-up question')
    })

    expect(result.current.lastChatSearchMeta).toBeNull()
  })

  it('chat-ux-fixes bug 2: lastChatSearchMeta resets to null when switching to a DIFFERENT session (mirrors turnEvents)', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1', stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state?.session_id).toBe('s1'))

    vi.mocked(curationApi.chat).mockResolvedValue(chatResponse({ web_search_used: true, new_web_articles_found: 1 }))
    await act(async () => {
      await result.current.sendChatMessage('yes')
    })
    expect(result.current.lastChatSearchMeta).not.toBeNull()

    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's2', topic: 'other topic' }))
    await act(async () => {
      await result.current.openReview('s2')
    })

    expect(result.current.lastChatSearchMeta).toBeNull()
  })

  it('chat-ux-polish Phase A: lastChatSearchMeta auto-clears ~5s after being set, with no further action', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockResolvedValue(chatResponse({ web_search_used: true, new_web_articles_found: 2 }))
    await act(async () => {
      await result.current.sendChatMessage('yes')
    })
    expect(result.current.lastChatSearchMeta).not.toBeNull()

    await act(async () => {
      vi.advanceTimersByTime(5000)
    })

    expect(result.current.lastChatSearchMeta).toBeNull()
    vi.useRealTimers()
  })

  it('chat-ux-polish Phase A: lastAddToReportResult auto-clears ~5s after being set', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.addChatExchangesToReport).mockResolvedValue({
      added_exchange_ids: ['ex-1'], skipped_exchange_ids: [], source_count: 1, chat_history: [], report: {} as never,
    })
    await act(async () => {
      await result.current.addExchangesToReport(['ex-1'])
    })
    expect(result.current.lastAddToReportResult).not.toBeNull()

    await act(async () => {
      vi.advanceTimersByTime(5000)
    })

    expect(result.current.lastAddToReportResult).toBeNull()
    vi.useRealTimers()
  })

  it('chat-ux-polish Phase A: starting a new chat action clears a still-fresh (not yet auto-cleared) notice from a prior action', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockResolvedValue(chatResponse({ web_search_used: true, new_web_articles_found: 2 }))
    await act(async () => {
      await result.current.sendChatMessage('yes')
    })
    expect(result.current.lastChatSearchMeta).not.toBeNull()

    vi.mocked(curationApi.deleteChatExchanges).mockResolvedValue({ deleted_exchange_ids: ['ex-1'], report_possibly_stale: false, chat_history: [] })
    await act(async () => {
      await result.current.deleteExchanges(['ex-1'])
    })

    // deleteExchanges never sets lastChatSearchMeta itself -- the only way
    // it could be null here is clearActionNotices firing at the top of
    // deleteExchanges, exactly as Phase A's notice lifecycle promises.
    expect(result.current.lastChatSearchMeta).toBeNull()
  })

  it('chat-ux-polish Phase A: dismissReportStaleWarning clears reportPossiblyStale on demand', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.deleteChatExchanges).mockResolvedValue({ deleted_exchange_ids: ['ex-1'], report_possibly_stale: true, chat_history: [] })
    await act(async () => {
      await result.current.deleteExchanges(['ex-1'])
    })
    expect(result.current.reportPossiblyStale).toBe(true)

    act(() => {
      result.current.dismissReportStaleWarning()
    })

    expect(result.current.reportPossiblyStale).toBe(false)
  })

  it('chat-ux-polish Phase A: a successful addExchangesToReport clears reportPossiblyStale (it just regenerated the report for real)', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ stage: 'synthesize' }))
    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.deleteChatExchanges).mockResolvedValue({ deleted_exchange_ids: ['ex-1'], report_possibly_stale: true, chat_history: [] })
    await act(async () => {
      await result.current.deleteExchanges(['ex-1'])
    })
    expect(result.current.reportPossiblyStale).toBe(true)

    vi.mocked(curationApi.addChatExchangesToReport).mockResolvedValue({
      added_exchange_ids: ['ex-2'], skipped_exchange_ids: [], source_count: 1, chat_history: [], report: {} as never,
    })
    await act(async () => {
      await result.current.addExchangesToReport(['ex-2'])
    })

    expect(result.current.reportPossiblyStale).toBe(false)
  })
})
