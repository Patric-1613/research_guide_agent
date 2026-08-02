import { useCallback, useEffect, useRef, useState } from 'react'
import { curationApi } from '../lib/api/client'
import { ApiError } from '../types'
import type { CurationStateResponse } from '../types'

// One completed curation turn, for the center panel's scrollback. This is
// deliberately client-only, accumulated as turns happen during THIS page
// load -- the backend does not persist a per-turn message log (only
// current/cumulative state: selected_paper_ids, pending_batch, refilled
// for the CURRENT turn). A page refresh therefore does not replay
// history for turns completed before the refresh; it starts a fresh,
// empty scrollback and accumulates new turns from that point on. The
// state properties that actually matter for correctness after a refresh
// (which papers are selected, what's pending, the progress count) all
// come from the backend via loadState() below, not from this list.
export interface TurnEvent {
  turnNumber: number
  refilled: boolean
  batchSize: number
  reserveRemainingAfter: number
  pickedPaperIds: string[]
}

// chat-ux-fixes bug 2: web_search_used/new_web_articles_found are already
// returned by POST /chat's response on every turn, but were previously
// discarded entirely -- captured here as an annotation on the MOST
// RECENT reply, the same "client-only, not persisted server-side" model
// TurnEvent above already uses (this one-shot fact genuinely isn't part
// of PaperPoolSession; a refresh loses it, same as TurnEvent's own
// scrollback). Always describes the latest reply only -- overwritten (to
// null if that reply didn't use a web search) on every subsequent
// sendChatMessage call, never accumulated into a list.
export interface ChatSearchMeta {
  webSearchUsed: boolean
  newWebArticlesFound: number | null
}

const SESSION_PARAM = 'session'

export function getSessionIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get(SESSION_PARAM)
}

function setSessionIdInUrl(sessionId: string): void {
  const url = new URL(window.location.href)
  url.searchParams.set(SESSION_PARAM, sessionId)
  window.history.pushState({}, '', url)
}

function clearSessionIdInUrl(): void {
  const url = new URL(window.location.href)
  url.searchParams.delete(SESSION_PARAM)
  window.history.pushState({}, '', url)
}

interface UseCurationSessionResult {
  sessionId: string | null
  state: CurationStateResponse | null
  loading: boolean
  error: string | null
  turnEvents: TurnEvent[]
  lastChatSearchMeta: ChatSearchMeta | null
  // curation-chat-delete Phase 3: true if the most recent deleteExchanges
  // call removed an answer that had been added to the report. Same
  // "latest action only, client-only" model as lastChatSearchMeta --
  // Phase 3 never persists or acts on this, it's purely a signal for the
  // frontend to show a "report may be stale" warning.
  reportPossiblyStale: boolean
  openReview: (sessionId: string) => void
  startReview: (topic: string, targetCount: number) => Promise<void>
  submitPicks: (pickedIds: string[], stop?: boolean, refinement?: string, requestRefill?: boolean) => Promise<void>
  // Return the freshly-loaded state (not void) so a caller can react to
  // exactly what changed as a RESULT of this action -- e.g. App.tsx uses
  // this to know how many web_articles_added a report generation/
  // regeneration just covered, without racing React's own render timing.
  generateReport: () => Promise<CurationStateResponse | undefined>
  regenerateReport: () => Promise<CurationStateResponse | undefined>
  sendChatMessage: (message: string) => Promise<void>
  // curation-chat-delete Phase 3: exchange_ids, not individual message
  // ids -- deleting an exchange always removes both the user question and
  // assistant answer that share it (see the backend's own
  // delete_chat_exchanges() docstring for why).
  deleteExchanges: (exchangeIds: string[]) => Promise<void>
  // curation-review-management Phase 8, item 1: deletes for real via the
  // backend, then -- ONLY if the deleted id was the currently-open
  // session -- clears sessionId/state/URL so the UI falls back to the
  // empty "select a review" state instead of showing a session that no
  // longer exists. Deleting a DIFFERENT review while one is open must not
  // disturb the open one at all.
  deleteReview: (sessionId: string) => Promise<void>
  // curation-turn-history Phase 9c: synthesize-stage only (enforced
  // server-side) -- see client.ts's own comment for why.
  selectFromHistory: (paperId: string) => Promise<void>
  // curation-editable-until-locked Phase 10e: reopens a stopped-but-
  // untouched review back into active curation -- server-side gated
  // (stage=="synthesize", no report, no chat yet), see client.ts's
  // reopen() for the full reasoning.
  reopenReview: () => Promise<void>
  refresh: () => Promise<void>
}

export function useCurationSession(): UseCurationSessionResult {
  const [sessionId, setSessionIdState] = useState<string | null>(() => getSessionIdFromUrl())
  const [state, setState] = useState<CurationStateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [turnEvents, setTurnEvents] = useState<TurnEvent[]>([])
  const [lastChatSearchMeta, setLastChatSearchMeta] = useState<ChatSearchMeta | null>(null)
  const [reportPossiblyStale, setReportPossiblyStale] = useState(false)
  const turnEventsSessionRef = useRef<string | null>(null)

  const loadState = useCallback(async (id: string): Promise<CurationStateResponse> => {
    const fresh = await curationApi.getState(id)
    setState(fresh)
    if (turnEventsSessionRef.current !== id) {
      turnEventsSessionRef.current = id
      setTurnEvents([])
      setLastChatSearchMeta(null)
    }
    return fresh
  }, [])

  const runAction = useCallback(async <T,>(action: () => Promise<T>): Promise<T | undefined> => {
    setLoading(true)
    setError(null)
    try {
      return await action()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
      return undefined
    } finally {
      setLoading(false)
    }
  }, [])

  // THE Phase 6d property: whatever session_id the URL names -- including
  // right after a hard page refresh, which re-runs this effect from
  // scratch with no prior in-memory state at all -- is what gets loaded
  // FROM THE BACKEND here. Nothing in this hook reads from
  // localStorage/sessionStorage or any other browser-only store; the URL
  // plus this one GET call is the entire source of truth on load.
  useEffect(() => {
    if (sessionId) {
      runAction(() => loadState(sessionId))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  useEffect(() => {
    const onPopState = () => setSessionIdState(getSessionIdFromUrl())
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  const openReview = useCallback((id: string) => {
    setSessionIdInUrl(id)
    setSessionIdState(id)
  }, [])

  const startReview = useCallback(
    (topic: string, targetCount: number) =>
      runAction(async () => {
        const response = await curationApi.start({ topic, target_count: targetCount })
        setSessionIdInUrl(response.session_id)
        setSessionIdState(response.session_id)
        await loadState(response.session_id)
      }),
    [runAction, loadState],
  )

  const submitPicks = useCallback(
    (pickedIds: string[], stop = false, refinement?: string, requestRefill?: boolean) =>
      runAction(async () => {
        if (!sessionId || !state) return
        const completedTurn: TurnEvent = {
          turnNumber: turnEvents.length + 1,
          refilled: state.refilled,
          batchSize: state.pending_batch?.length ?? 0,
          reserveRemainingAfter: state.reserve_remaining,
          pickedPaperIds: pickedIds,
        }
        await curationApi.picks(sessionId, { picked_paper_ids: pickedIds, stop, refinement, request_refill: requestRefill })
        setTurnEvents((prev) => [...prev, completedTurn])
        await loadState(sessionId)
      }),
    [runAction, sessionId, state, turnEvents.length, loadState],
  )

  const generateReport = useCallback(
    () =>
      runAction(async () => {
        if (!sessionId) return undefined
        await curationApi.generateReport(sessionId)
        return loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const regenerateReport = useCallback(
    () =>
      runAction(async () => {
        if (!sessionId) return undefined
        await curationApi.regenerateReport(sessionId)
        return loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const sendChatMessage = useCallback(
    (message: string) =>
      runAction(async () => {
        if (!sessionId) return
        const response = await curationApi.chat(sessionId, { message })
        setLastChatSearchMeta(
          response.web_search_used
            ? { webSearchUsed: true, newWebArticlesFound: response.new_web_articles_found }
            : null,
        )
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const deleteExchanges = useCallback(
    (exchangeIds: string[]) =>
      runAction(async () => {
        if (!sessionId) return
        const response = await curationApi.deleteChatExchanges(sessionId, { exchange_ids: exchangeIds })
        setReportPossiblyStale(response.report_possibly_stale)
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const refresh = useCallback(
    () =>
      runAction(async () => {
        if (!sessionId) return
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const deleteReview = useCallback(
    (idToDelete: string) =>
      runAction(async () => {
        await curationApi.deleteReview(idToDelete)
        if (idToDelete === sessionId) {
          clearSessionIdInUrl()
          setSessionIdState(null)
          setState(null)
        }
      }),
    [runAction, sessionId],
  )

  // Phase 9c/9f: the SYNTHESIZE-STAGE-ONLY way to add a paper from an
  // earlier turn -- immediate, no "next turn" wait. Picking from history
  // WHILE stage=="curate" instead bundles the paper_id into submitPicks'
  // pickedIds (see App.tsx's turn history browser), since that's the
  // only channel safe to use while a real interrupt is pending.
  const selectFromHistory = useCallback(
    (paperId: string) =>
      runAction(async () => {
        if (!sessionId) return
        await curationApi.selectFromHistory(sessionId, paperId)
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const reopenReview = useCallback(
    () =>
      runAction(async () => {
        if (!sessionId) return
        await curationApi.reopen(sessionId)
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  return {
    sessionId, state, loading, error, turnEvents, lastChatSearchMeta, reportPossiblyStale,
    openReview, startReview, submitPicks, generateReport, regenerateReport, sendChatMessage, deleteExchanges,
    deleteReview, selectFromHistory, reopenReview, refresh,
  }
}
