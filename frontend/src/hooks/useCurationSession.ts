import { useCallback, useEffect, useRef, useState } from 'react'
import { curationApi } from '../api/client'
import { ApiError } from '../api/types'
import type { CurationStateResponse } from '../api/types'

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

const SESSION_PARAM = 'session'

export function getSessionIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get(SESSION_PARAM)
}

function setSessionIdInUrl(sessionId: string): void {
  const url = new URL(window.location.href)
  url.searchParams.set(SESSION_PARAM, sessionId)
  window.history.pushState({}, '', url)
}

interface UseCurationSessionResult {
  sessionId: string | null
  state: CurationStateResponse | null
  loading: boolean
  error: string | null
  turnEvents: TurnEvent[]
  openReview: (sessionId: string) => void
  startReview: (topic: string, targetCount: number) => Promise<void>
  submitPicks: (pickedIds: string[], stop?: boolean, refinement?: string) => Promise<void>
  // Return the freshly-loaded state (not void) so a caller can react to
  // exactly what changed as a RESULT of this action -- e.g. App.tsx uses
  // this to know how many web_articles_added a report generation/
  // regeneration just covered, without racing React's own render timing.
  generateReport: () => Promise<CurationStateResponse | undefined>
  regenerateReport: () => Promise<CurationStateResponse | undefined>
  sendChatMessage: (message: string) => Promise<void>
  refresh: () => Promise<void>
}

export function useCurationSession(): UseCurationSessionResult {
  const [sessionId, setSessionIdState] = useState<string | null>(() => getSessionIdFromUrl())
  const [state, setState] = useState<CurationStateResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [turnEvents, setTurnEvents] = useState<TurnEvent[]>([])
  const turnEventsSessionRef = useRef<string | null>(null)

  const loadState = useCallback(async (id: string): Promise<CurationStateResponse> => {
    const fresh = await curationApi.getState(id)
    setState(fresh)
    if (turnEventsSessionRef.current !== id) {
      turnEventsSessionRef.current = id
      setTurnEvents([])
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
    (pickedIds: string[], stop = false, refinement?: string) =>
      runAction(async () => {
        if (!sessionId || !state) return
        const completedTurn: TurnEvent = {
          turnNumber: turnEvents.length + 1,
          refilled: state.refilled,
          batchSize: state.pending_batch?.length ?? 0,
          reserveRemainingAfter: state.reserve_remaining,
          pickedPaperIds: pickedIds,
        }
        await curationApi.picks(sessionId, { picked_paper_ids: pickedIds, stop, refinement })
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
        await curationApi.chat(sessionId, { message })
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

  return {
    sessionId, state, loading, error, turnEvents,
    openReview, startReview, submitPicks, generateReport, regenerateReport, sendChatMessage, refresh,
  }
}
