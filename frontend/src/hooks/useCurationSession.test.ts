import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useCurationSession, getSessionIdFromUrl } from './useCurationSession'
import { curationApi } from '../lib/api/client'
import { streamCurationChat, ChatStreamTransportError } from '../lib/api/chatStream'
import { ApiError } from '../types'
import type { ChatStreamServerEvent, CurationChatResponse, CurationStateResponse, CurationTurnResponse } from '../types'

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

vi.mock('../lib/api/chatStream', async () => {
  const actual = await vi.importActual<typeof import('../lib/api/chatStream')>('../lib/api/chatStream')
  return {
    streamCurationChat: vi.fn(),
    ChatStreamTransportError: actual.ChatStreamTransportError,
  }
})

// Turns a fixed array of events into an async generator that honors the
// AbortSignal streamCurationChat's own real fetch()/reader.read() would --
// rejecting with a genuine AbortError once the signal fires, exactly what
// controller.abort() (via cancelChatStream) must trigger.
async function* fakeStream(events: ChatStreamServerEvent[], signal: AbortSignal): AsyncGenerator<ChatStreamServerEvent> {
  for (const event of events) {
    if (signal.aborted) throw new DOMException('The operation was aborted.', 'AbortError')
    await Promise.resolve()
    yield event
  }
}

// A stream that fails before yielding anything at all -- used to
// simulate a preflight ApiError or a raw transport failure (e.g. fetch()
// itself rejecting) without relying on unreachable code after a throw to
// satisfy the AsyncGenerator return type.
// eslint-disable-next-line require-yield
async function* throwingStream(err: unknown): AsyncGenerator<ChatStreamServerEvent> {
  throw err
}

// A stream that fails after yielding a few events -- simulates a
// malformed/truncated SSE stream discovered mid-response.
async function* streamThatFailsMidway(
  events: ChatStreamServerEvent[],
  err: unknown,
): AsyncGenerator<ChatStreamServerEvent> {
  for (const event of events) {
    yield event
  }
  throw err
}

// A stream that yields `started` then hangs until the signal is aborted
// (mirroring a real in-flight fetch a user cancels mid-response) or the
// test calls `resume()`.
function controllableStream() {
  let resolveGate: (() => void) | null = null
  const gate = new Promise<void>((resolve) => { resolveGate = resolve })
  async function* gen(signal: AbortSignal): AsyncGenerator<ChatStreamServerEvent> {
    yield { type: 'started', data: {} }
    await Promise.race([
      gate,
      new Promise<never>((_, reject) => {
        if (signal.aborted) { reject(new DOMException('The operation was aborted.', 'AbortError')); return }
        signal.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')), { once: true })
      }),
    ])
    yield { type: 'phase', data: { phase: 'generating' } }
  }
  return { gen, resume: () => resolveGate?.() }
}

function fullState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 'transformers', display_title: 'transformers', stage: 'curate', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    report_versions: [], active_report_version_id: null, chat_references: [],
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

  it('a new failed action REPLACES a stale error from a previous failed action, rather than stacking it', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1' }))

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockRejectedValueOnce(new Error('first failure'))
    await act(async () => {
      await result.current.sendChatMessage('hello')
    })
    expect(result.current.error).toBe('Error: first failure')

    vi.mocked(curationApi.chat).mockRejectedValueOnce(new Error('second failure'))
    await act(async () => {
      await result.current.sendChatMessage('hello again')
    })
    expect(result.current.error).toBe('Error: second failure')
    expect(result.current.error).not.toContain('first failure')
  })

  it('a successful action clears an existing error left over from a prior failed action', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1' }))

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockRejectedValueOnce(new Error('boom'))
    await act(async () => {
      await result.current.sendChatMessage('hello')
    })
    expect(result.current.error).toBe('Error: boom')

    vi.mocked(curationApi.chat).mockResolvedValueOnce(chatResponse())
    await act(async () => {
      await result.current.sendChatMessage('retry')
    })
    expect(result.current.error).toBeNull()
  })

  it('a real ApiError rejection (e.g. action_in_progress) surfaces the safe mapped message, never the raw reason_code', async () => {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1' }))

    const { result } = renderHook(() => useCurationSession())
    await waitFor(() => expect(result.current.state).not.toBeNull())

    vi.mocked(curationApi.chat).mockRejectedValueOnce(
      new ApiError(409, {
        detail: {
          reason_code: 'action_in_progress',
          message: 'Another request is already in progress for this session. Please wait for it to finish.',
        },
      }),
    )
    await act(async () => {
      await result.current.sendChatMessage('hello')
    })

    expect(result.current.error).toBe('Another action is already running for this review. Please wait and try again.')
    expect(result.current.error).not.toContain('action_in_progress')
    expect(result.current.error).not.toMatch(/[{}[\]]/)
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

describe('useCurationSession -- Usage Protection M4.2B: chat-streaming lifecycle', () => {
  async function mountAtS1() {
    window.history.pushState({}, '', '/?session=s1')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1', stage: 'synthesize' }))
    const rendered = renderHook(() => useCurationSession())
    await waitFor(() => expect(rendered.result.current.state).not.toBeNull())
    return rendered
  }

  it('started -> phase -> delta -> completed -> done: accumulates phase/text, then reloads canonical state and clears the preview', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) =>
      fakeStream(
        [
          { type: 'started', data: {} },
          { type: 'phase', data: { phase: 'generating' } },
          { type: 'delta', data: { text: 'Hello' } },
          { type: 'completed', data: { answer: 'Hello', answerable: true, cited_papers: [], cited_web_articles: [] } },
          { type: 'done', data: {} },
        ],
        opts.signal,
      ),
    )
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1', stage: 'synthesize' }))

    await act(async () => {
      await result.current.sendChatMessageStreaming('hi')
    })

    expect(streamCurationChat).toHaveBeenCalledWith('s1', { message: 'hi' }, expect.objectContaining({ signal: expect.any(AbortSignal) }))
    expect(result.current.chatStreamActive).toBe(false)
    expect(result.current.chatStreamText).toBe('')
    expect(result.current.chatStreamSyncFailed).toBe(false)
    expect(result.current.error).toBeNull()
    // The canonical reload (getState) ran again beyond the initial mount load.
    expect(curationApi.getState).toHaveBeenCalledTimes(2)
  })

  it('a final-only answer (zero deltas) still completes successfully', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) =>
      fakeStream(
        [
          { type: 'started', data: {} },
          { type: 'completed', data: { answer: '', answerable: false, cited_papers: [], cited_web_articles: [] } },
          { type: 'done', data: {} },
        ],
        opts.signal,
      ),
    )

    await act(async () => {
      await result.current.sendChatMessageStreaming('anything?')
    })

    expect(result.current.error).toBeNull()
    expect(result.current.chatStreamActive).toBe(false)
  })

  it('one large final delta accumulates into chatStreamText DURING the stream, before completed/done arrive', async () => {
    const { result } = await mountAtS1()
    let releaseDone: (() => void) | null = null
    const doneGate = new Promise<void>((resolve) => { releaseDone = resolve })
    vi.mocked(streamCurationChat).mockImplementation(() =>
      (async function* () {
        yield { type: 'started', data: {} } as ChatStreamServerEvent
        yield { type: 'delta', data: { text: 'The full answer arrives all at once.' } } as ChatStreamServerEvent
        await doneGate
        yield {
          type: 'completed',
          data: { answer: 'The full answer arrives all at once.', answerable: true, cited_papers: [], cited_web_articles: [] },
        } as ChatStreamServerEvent
        yield { type: 'done', data: {} } as ChatStreamServerEvent
      })(),
    )

    let sendPromise: Promise<void> = Promise.resolve()
    act(() => {
      sendPromise = result.current.sendChatMessageStreaming('q')
    })

    await waitFor(() => expect(result.current.chatStreamText).toBe('The full answer arrives all at once.'))
    expect(result.current.chatStreamActive).toBe(true) // still mid-stream

    releaseDone!()
    await act(async () => {
      await sendPromise
    })

    expect(result.current.error).toBeNull()
  })

  it('multiple deltas accumulate in order without duplication', async () => {
    const { result } = await mountAtS1()
    let releaseDone: (() => void) | null = null
    const doneGate = new Promise<void>((resolve) => { releaseDone = resolve })
    vi.mocked(streamCurationChat).mockImplementation(() =>
      (async function* () {
        yield { type: 'started', data: {} } as ChatStreamServerEvent
        yield { type: 'delta', data: { text: 'Hel' } } as ChatStreamServerEvent
        yield { type: 'delta', data: { text: 'lo ' } } as ChatStreamServerEvent
        yield { type: 'delta', data: { text: 'world' } } as ChatStreamServerEvent
        await doneGate
        yield { type: 'completed', data: { answer: 'Hello world', answerable: true, cited_papers: [], cited_web_articles: [] } } as ChatStreamServerEvent
        yield { type: 'done', data: {} } as ChatStreamServerEvent
      })(),
    )

    let sendPromise: Promise<void> = Promise.resolve()
    act(() => {
      sendPromise = result.current.sendChatMessageStreaming('q')
    })

    await waitFor(() => expect(result.current.chatStreamText).toBe('Hello world'))

    releaseDone!()
    await act(async () => {
      await sendPromise
    })

    expect(result.current.error).toBeNull()
  })

  it('reloads canonical state only after completed THEN done, not merely after completed', async () => {
    const { result } = await mountAtS1()
    let releaseDone: (() => void) | null = null
    const doneGate = new Promise<void>((resolve) => { releaseDone = resolve })
    vi.mocked(streamCurationChat).mockImplementation(() =>
      (async function* () {
        yield { type: 'started', data: {} } as ChatStreamServerEvent
        yield { type: 'completed', data: { answer: 'Hi', answerable: true, cited_papers: [], cited_web_articles: [] } } as ChatStreamServerEvent
        await doneGate
        yield { type: 'done', data: {} } as ChatStreamServerEvent
      })(),
    )

    let sendPromise: Promise<void> = Promise.resolve()
    act(() => {
      sendPromise = result.current.sendChatMessageStreaming('hi')
    })

    await waitFor(() => expect(vi.mocked(curationApi.getState).mock.calls.length).toBeGreaterThanOrEqual(1))
    // Only the initial mount load so far -- completed alone must not
    // have triggered a reload yet.
    expect(curationApi.getState).toHaveBeenCalledTimes(1)

    releaseDone!()
    await act(async () => {
      await sendPromise
    })

    expect(curationApi.getState).toHaveBeenCalledTimes(2)
  })

  it('rejects a second concurrent chat stream while one is already active', async () => {
    const { result } = await mountAtS1()
    const { gen } = controllableStream()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) => gen(opts.signal))

    let firstPromise: Promise<void> = Promise.resolve()
    act(() => {
      firstPromise = result.current.sendChatMessageStreaming('first')
    })
    await waitFor(() => expect(result.current.chatStreamActive).toBe(true))

    await act(async () => {
      await result.current.sendChatMessageStreaming('second')
    })

    expect(streamCurationChat).toHaveBeenCalledTimes(1)

    // Clean up: cancel the still-active first stream.
    act(() => {
      result.current.cancelChatStream()
    })
    await act(async () => {
      await firstPromise
    })
  })

  it('a handled SSE error -> done sets the safe backend message, clears the preview, and never reloads', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) =>
      fakeStream(
        [
          { type: 'started', data: {} },
          { type: 'phase', data: { phase: 'generating' } },
          { type: 'error', data: { reason_code: 'provider_error', message: 'The model provider returned an error.' } },
          { type: 'done', data: {} },
        ],
        opts.signal,
      ),
    )

    await act(async () => {
      await result.current.sendChatMessageStreaming('hi')
    })

    expect(result.current.error).toBe('The model provider returned an error.')
    expect(result.current.chatStreamActive).toBe(false)
    // No reload for a handled failure -- the backend guarantees nothing
    // was persisted, so only the initial mount load happened.
    expect(curationApi.getState).toHaveBeenCalledTimes(1)
  })

  it('a malformed/truncated stream stops locally, shows a safe retryable message, and reloads canonical state', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation(() =>
      streamThatFailsMidway(
        [{ type: 'started', data: {} }, { type: 'phase', data: { phase: 'generating' } }],
        new ChatStreamTransportError('Received a malformed message from the server.'),
      ),
    )
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1', stage: 'synthesize' }))

    await act(async () => {
      await result.current.sendChatMessageStreaming('hi')
    })

    expect(result.current.chatStreamActive).toBe(false)
    expect(result.current.error).toMatch(/lost the connection/i)
    expect(result.current.error).not.toMatch(/malformed message/i)
    expect(curationApi.getState).toHaveBeenCalledTimes(2)
  })

  it('a raw transport failure (e.g. the underlying fetch rejecting) is treated the same as a malformed stream: safe message + reload', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation(() => throwingStream(new TypeError('Failed to fetch')))
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1', stage: 'synthesize' }))

    await act(async () => {
      await result.current.sendChatMessageStreaming('hi')
    })

    expect(result.current.chatStreamActive).toBe(false)
    expect(result.current.error).toMatch(/lost the connection/i)
    expect(result.current.error).not.toContain('Failed to fetch')
    expect(curationApi.getState).toHaveBeenCalledTimes(2)
  })

  it('a preflight ApiError (e.g. 429) uses the existing safe error mapping and never reloads', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation(() =>
      throwingStream(
        new ApiError(429, {
          detail: { reason_code: 'session_hourly_limit_reached', message: 'This session has reached its usage limit.' },
        }),
      ),
    )

    await act(async () => {
      await result.current.sendChatMessageStreaming('hi')
    })

    expect(result.current.error).toBe('This session has reached its usage limit.')
    expect(curationApi.getState).toHaveBeenCalledTimes(1)
  })

  it('reload failure after a successful completed/done keeps the answer visible and sets chatStreamSyncFailed', async () => {
    const { result } = await mountAtS1()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) =>
      fakeStream(
        [
          { type: 'started', data: {} },
          { type: 'delta', data: { text: 'Hello' } },
          { type: 'completed', data: { answer: 'Hello', answerable: true, cited_papers: [], cited_web_articles: [] } },
          { type: 'done', data: {} },
        ],
        opts.signal,
      ),
    )
    vi.mocked(curationApi.getState).mockRejectedValueOnce(new Error('network down'))

    await act(async () => {
      await result.current.sendChatMessageStreaming('hi')
    })

    expect(result.current.chatStreamSyncFailed).toBe(true)
    expect(result.current.chatStreamActive).toBe(false)
    // The completed answer stays visible -- the preview is NOT cleared.
    expect(result.current.chatStreamText).toBe('Hello')
  })

  it('cancellation: aborts the stream, does not set a visible error, and reloads canonical state', async () => {
    const { result } = await mountAtS1()
    const { gen } = controllableStream()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) => gen(opts.signal))

    let sendPromise: Promise<void> = Promise.resolve()
    act(() => {
      sendPromise = result.current.sendChatMessageStreaming('hi')
    })
    await waitFor(() => expect(result.current.chatStreamActive).toBe(true))

    act(() => {
      result.current.cancelChatStream()
    })
    await act(async () => {
      await sendPromise
    })

    expect(result.current.error).toBeNull()
    expect(result.current.chatStreamActive).toBe(false)
    expect(result.current.chatStreamText).toBe('')
    expect(curationApi.getState).toHaveBeenCalledTimes(2)
  })

  it('unmounting the hook aborts an in-flight chat stream', async () => {
    const { gen } = controllableStream()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) => gen(opts.signal))
    const { result, unmount } = await mountAtS1()

    act(() => {
      void result.current.sendChatMessageStreaming('hi')
    })
    await waitFor(() => expect(result.current.chatStreamActive).toBe(true))

    const abortSpy = vi.spyOn(AbortController.prototype, 'abort')
    unmount()

    expect(abortSpy).toHaveBeenCalled()
    abortSpy.mockRestore()
  })

  it('switching to a different session aborts the in-flight chat stream for the old one', async () => {
    const { gen } = controllableStream()
    vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) => gen(opts.signal))
    const { result } = await mountAtS1()

    act(() => {
      void result.current.sendChatMessageStreaming('hi')
    })
    await waitFor(() => expect(result.current.chatStreamActive).toBe(true))

    const abortSpy = vi.spyOn(AbortController.prototype, 'abort')
    vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's2', topic: 'other' }))
    await act(async () => {
      await result.current.openReview('s2')
    })

    expect(abortSpy).toHaveBeenCalled()
    abortSpy.mockRestore()
  })

  describe('invalid event ordering is rejected safely (never crashes, never surfaces raw internals)', () => {
    async function expectOrderingViolationHandledSafely(events: ChatStreamServerEvent[]) {
      const { result } = await mountAtS1()
      vi.mocked(streamCurationChat).mockImplementation((_sessionId, _req, opts) => fakeStream(events, opts.signal))
      vi.mocked(curationApi.getState).mockResolvedValue(fullState({ session_id: 's1', stage: 'synthesize' }))

      await act(async () => {
        await result.current.sendChatMessageStreaming('hi')
      })

      expect(result.current.chatStreamActive).toBe(false)
      expect(result.current.error).toMatch(/lost the connection/i)
      expect(curationApi.getState).toHaveBeenCalledTimes(2)
    }

    it('a delta before started', async () => {
      await expectOrderingViolationHandledSafely([{ type: 'delta', data: { text: 'x' } }])
    })

    it('an event after done', async () => {
      await expectOrderingViolationHandledSafely([
        { type: 'started', data: {} },
        { type: 'done', data: {} },
        { type: 'phase', data: { phase: 'saving' } },
      ])
    })

    it('more than one completed event', async () => {
      await expectOrderingViolationHandledSafely([
        { type: 'started', data: {} },
        { type: 'completed', data: { answer: 'a', answerable: true, cited_papers: [], cited_web_articles: [] } },
        { type: 'completed', data: { answer: 'b', answerable: true, cited_papers: [], cited_web_articles: [] } },
        { type: 'done', data: {} },
      ])
    })

    it('both error and completed for the same turn', async () => {
      await expectOrderingViolationHandledSafely([
        { type: 'started', data: {} },
        { type: 'error', data: { reason_code: 'provider_error', message: 'The model provider returned an error.' } },
        { type: 'completed', data: { answer: 'a', answerable: true, cited_papers: [], cited_web_articles: [] } },
        { type: 'done', data: {} },
      ])
    })

    it('the stream ends without a done event at all', async () => {
      await expectOrderingViolationHandledSafely([
        { type: 'started', data: {} },
        { type: 'completed', data: { answer: 'a', answerable: true, cited_papers: [], cited_web_articles: [] } },
      ])
    })
  })
})
