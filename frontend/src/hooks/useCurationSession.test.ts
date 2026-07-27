import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useCurationSession, getSessionIdFromUrl } from './useCurationSession'
import { curationApi } from '../api/client'
import type { CurationStateResponse, CurationTurnResponse } from '../api/types'

vi.mock('../api/client', () => ({
  curationApi: {
    getState: vi.fn(),
    start: vi.fn(),
    picks: vi.fn(),
    generateReport: vi.fn(),
    regenerateReport: vi.fn(),
    chat: vi.fn(),
    listReviews: vi.fn(),
  },
}))

function fullState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 'transformers', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null,
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
      refilled: false, reserve_remaining: 5,
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
      batch: [], stop_reason: null, refilled: false, reserve_remaining: 6,
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
})
