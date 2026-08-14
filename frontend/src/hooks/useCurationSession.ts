import { useCallback, useEffect, useRef, useState } from 'react'
import { curationApi } from '../lib/api/client'
import { ChatStreamTransportError, streamCurationChat } from '../lib/api/chatStream'
import { getUserFacingErrorMessage } from '../lib/api/errorMessages'
import { ReportStreamTransportError, streamGenerateReport, streamRegenerateReport } from '../lib/api/reportStream'
import { ApiError } from '../types'
import type {
  ChatStreamPhase, CurationGenerateReportRequest, CurationRegenerateReportRequest, CurationStateResponse,
  RefinementMode, ReportStreamCompletionNotice, ReportStreamPhase, ReportStreamServerEvent, ReportTemplate,
} from '../types'

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
// UXH.2: the synchronous curation actions (start a review, submit picks in
// their three distinct shapes) previously shared nothing but the generic
// `loading` flag -- enough to disable controls, but not enough for a
// caller to know WHICH action is actually running, so every button either
// said nothing truthful or fell back to a vague "Loading…". This union
// names exactly the four actions that can set it; `null` means none is
// active. Deliberately NOT reused for chat/report streaming, which already
// have their own richer, longer-lived lifecycle state (chatStream*/
// reportStream* below) -- this is only for the short request/response
// curation actions that had no per-action state at all before this phase.
export type CurationActionKind = 'starting_review' | 'continuing_review' | 'searching_more' | 'finishing_review'

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

// curation-chat-add-to-report Phase 4: success-feedback summary of the
// most recent addExchangesToReport call, same client-only/latest-action
// model as ChatSearchMeta above.
export interface AddToReportResult {
  addedCount: number
  skippedCount: number
  sourceCount: number
}

// chat-ux-polish Phase A: notice lifecycle for the "success/info" class
// of ephemeral state (lastChatSearchMeta, lastAddToReportResult) --
// distinct from reportPossiblyStale, which is a WARNING and intentionally
// does NOT use this (it persists until dismissed or a report-changing
// action explicitly clears it, see dismissReportStaleWarning/generateReport/
// regenerateReport/addExchangesToReport below). A success/info notice:
//   - auto-clears ~5s after being set, so it doesn't linger forever
//   - gets replaced/cleared immediately whenever a NEW chat action starts
//     (see clearActionNotices, called at the top of every chat action),
//     so an old, unrelated success note can't keep sitting next to a
//     newer, unrelated one
const NOTICE_AUTO_CLEAR_MS = 5000

// report-progress-observability: the report-stream completion notice
// (see reportStreamCompletionNotice below) reuses this same auto-clearing
// mechanic but names its own duration -- a distinct constant, even though
// it happens to share NOTICE_AUTO_CLEAR_MS's value, so each call site's
// intent reads clearly on its own.
const REPORT_STREAM_SUCCESS_NOTICE_MS = 5000

function useAutoClearingState<T>(clearAfterMs: number = NOTICE_AUTO_CLEAR_MS): [T | null, (value: T | null) => void] {
  const [value, setValue] = useState<T | null>(null)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const set = useCallback((next: T | null) => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    setValue(next)
    if (next !== null) {
      timeoutRef.current = setTimeout(() => {
        timeoutRef.current = null
        setValue(null)
      }, clearAfterMs)
    }
  }, [clearAfterMs])

  useEffect(
    () => () => {
      if (timeoutRef.current !== null) clearTimeout(timeoutRef.current)
    },
    [],
  )

  return [value, set]
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
  // curation-chat-add-to-report Phase 4: same "latest action only,
  // client-only" model as lastChatSearchMeta/reportPossiblyStale -- set
  // fresh on every successful addExchangesToReport call, lost on refresh.
  lastAddToReportResult: AddToReportResult | null
  // chat-ux-polish Phase A: the only way to clear reportPossiblyStale
  // other than a report-changing action succeeding (generateReport/
  // regenerateReport/addExchangesToReport all clear it on success, since
  // each one just regenerated the report for real).
  dismissReportStaleWarning: () => void
  // UXH.2: which of the four synchronous curation actions is currently
  // in flight, or null when none is -- see CurationActionKind's own
  // docstring above. Set synchronously before the action's request
  // begins and cleared on every settle path (success or failure), same
  // guarantee `loading` already provides, just specific enough to drive
  // a truthful per-action label instead of one generic spinner.
  curationAction: CurationActionKind | null
  openReview: (sessionId: string) => void
  startReview: (topic: string, targetCount: number) => Promise<void>
  submitPicks: (pickedIds: string[], stop?: boolean, refinement?: string, requestRefill?: boolean) => Promise<void>
  // Return the freshly-loaded state (not void) so a caller can react to
  // exactly what changed as a RESULT of this action -- e.g. App.tsx uses
  // this to know how many web_articles_added a report generation/
  // regeneration just covered, without racing React's own render timing.
  // report-quality Phase R2C: reportTemplate is optional on both --
  // omitted preserves exactly the prior behavior (generate defaults to
  // analytical server-side, regenerate preserves the existing report's
  // template server-side).
  // report-quality Phase R4.1: refinementMode is optional the same way
  // -- omitted/"off" preserves exactly the prior (no refinement)
  // behavior.
  generateReport: (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) => Promise<CurationStateResponse | undefined>
  regenerateReport: (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) => Promise<CurationStateResponse | undefined>
  // Usage Protection M4.3B: streaming lifecycle state/actions -- the
  // normal UI Generate/Regenerate path now uses generateReportStreaming/
  // regenerateReportStreaming (see CurationWorkspacePage.tsx); generate
  // Report/regenerateReport above remain available unchanged for
  // backward compatibility.
  reportStreamActive: boolean
  reportStreamOperation: 'generate' | 'regenerate' | null
  reportStreamPhase: ReportStreamPhase | null
  // report-progress-observability: the ordered, deduplicated phases
  // genuinely observed so far for the current active stream -- see this
  // field's own state declaration above for the full "why alongside
  // reportStreamPhase, not instead of it" reasoning.
  reportStreamPhaseHistory: ReportStreamPhase[]
  reportStreamStopping: boolean
  reportStreamError: string | null
  reportStreamSyncFailed: boolean
  // report-progress-observability: set once, briefly, after a genuinely
  // completed report-stream turn's canonical reload has succeeded -- see
  // ReportStreamCompletionNotice's own docstring above.
  reportStreamCompletionNotice: ReportStreamCompletionNotice | null
  generateReportStreaming: (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) => Promise<void>
  regenerateReportStreaming: (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) => Promise<void>
  cancelReportStream: () => void
  // report-quality Phase R3: switches which report version is active --
  // same return convention as generate/regenerateReport above.
  activateReportVersion: (versionId: string) => Promise<CurationStateResponse | undefined>
  sendChatMessage: (message: string) => Promise<void>
  // Usage Protection M4.2B: streaming lifecycle state/actions -- the
  // normal UI submission path now uses sendChatMessageStreaming (see
  // CurationWorkspacePage.tsx); sendChatMessage above remains available
  // unchanged for backward compatibility.
  chatStreamActive: boolean
  chatStreamPhase: ChatStreamPhase | null
  chatStreamText: string
  chatStreamSyncFailed: boolean
  sendChatMessageStreaming: (message: string) => Promise<void>
  cancelChatStream: () => void
  // curation-chat-delete Phase 3: exchange_ids, not individual message
  // ids -- deleting an exchange always removes both the user question and
  // assistant answer that share it (see the backend's own
  // delete_chat_exchanges() docstring for why).
  deleteExchanges: (exchangeIds: string[]) => Promise<void>
  // curation-chat-add-to-report Phase 4: same exchange_id-based batching
  // as deleteExchanges -- approves the requested exchanges' cited web
  // sources and regenerates the report through the existing selective
  // path (see the backend's regenerate_report_with_approved_web_sources).
  addExchangesToReport: (exchangeIds: string[]) => Promise<void>
  // curation-chat-edit Phase 5: truncate-and-regenerate -- replaces
  // exchangeId's question, discards its old answer and every later
  // exchange, and regenerates a fresh answer. The edited exchange gets a
  // NEW exchange_id (see the backend's edit_chat_exchange() docstring).
  editExchange: (exchangeId: string, question: string) => Promise<void>
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
  const [lastChatSearchMeta, setLastChatSearchMeta] = useAutoClearingState<ChatSearchMeta>()
  const [reportPossiblyStale, setReportPossiblyStale] = useState(false)
  const [lastAddToReportResult, setLastAddToReportResult] = useAutoClearingState<AddToReportResult>()
  const turnEventsSessionRef = useRef<string | null>(null)

  // UXH.2: see CurationActionKind's own docstring. The ref alongside the
  // state mirrors chatStreamActiveRef/reportStreamActiveRef's own
  // rationale below -- runAction's own reentrancy guard (see below) reads
  // THIS synchronously, not the state, so a rapid double-click can't slip
  // a second curation action in ahead of a state update that hasn't
  // re-rendered (and therefore hasn't disabled the button) yet.
  const [curationAction, setCurationActionState] = useState<CurationActionKind | null>(null)
  const curationActionRef = useRef<CurationActionKind | null>(null)
  const setCurationAction = useCallback((action: CurationActionKind | null) => {
    curationActionRef.current = action
    setCurationActionState(action)
  }, [])

  // UXH.1 (UX-04): the session id a caller most recently expressed intent
  // to load -- updated synchronously by setSessionId below, at the exact
  // moment openReview/startReview/deleteReview/popstate change it, NOT
  // via an effect (which would only catch up after React's next commit,
  // too late to protect against a response that resolves in that gap).
  // loadState checks a response's own requested id against this ref
  // before publishing it: a response for a session the user has since
  // navigated away from is stale and must never overwrite what's on
  // screen for the session actually open now (a late success OR a late
  // failure alike -- see loadState and the sessionId-effect below).
  const currentSessionIdRef = useRef<string | null>(sessionId)
  // UXH.1 (UX-04): which session the sessionId-change effect further
  // below last actually displayed -- see that effect's own comment for
  // why this is a second, separate ref from currentSessionIdRef above.
  const lastLoadedSessionIdRef = useRef<string | null>(null)
  // UXH.1 (UX-04): guards against publishing state after this hook's
  // owning component has unmounted -- a late-resolving loadState call
  // (from any caller, not just the sessionId effect below) must not call
  // setState once there's no component left to receive it.
  const isMountedRef = useRef(true)
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  // The one place sessionId is ever assigned -- keeps currentSessionIdRef
  // perfectly in sync with the state setter, synchronously, so nothing
  // that reads the ref later can ever observe a stale value React just
  // hasn't re-rendered yet.
  const setSessionId = useCallback((id: string | null) => {
    currentSessionIdRef.current = id
    setSessionIdState(id)
  }, [])

  // Usage Protection M4.2B: chat-streaming lifecycle state, owned here
  // (not inside ChatModePanel) per this phase's own explicit requirement
  // -- only what's genuinely needed to render/control an in-flight
  // stream: whether one is active, its current phase, the accumulated
  // safe answer text so far, and whether the post-completion canonical
  // reload failed. Citation/exchange metadata is deliberately NOT
  // tracked here -- see sendChatMessageStreaming's own docstring for why
  // chatStreamText alone is always enough.
  const [chatStreamActive, setChatStreamActiveState] = useState(false)
  const [chatStreamPhase, setChatStreamPhase] = useState<ChatStreamPhase | null>(null)
  const [chatStreamText, setChatStreamText] = useState('')
  const [chatStreamSyncFailed, setChatStreamSyncFailed] = useState(false)
  // A ref alongside the state above: sendChatMessageStreaming's own
  // "reject a second concurrent stream" guard reads THIS, not the state,
  // since a rapid double-invocation (e.g. a fast double-click) could
  // otherwise race ahead of a state update that hasn't re-rendered yet.
  const chatStreamActiveRef = useRef(false)
  const chatStreamAbortRef = useRef<AbortController | null>(null)

  const setChatStreamActive = useCallback((active: boolean) => {
    chatStreamActiveRef.current = active
    setChatStreamActiveState(active)
  }, [])

  // The one place the streaming preview is ever cleared -- called on a
  // successful canonical reload (loadState below, which every action
  // already calls on its own success; once fresh chat_history has
  // landed, the temporary preview is superseded, per this phase's own
  // "replace it with canonical history after reload" requirement), on a
  // handled SSE error, and on cancellation.
  const clearChatStreamPreview = useCallback(() => {
    setChatStreamActive(false)
    setChatStreamPhase(null)
    setChatStreamText('')
    setChatStreamSyncFailed(false)
    chatStreamAbortRef.current = null
  }, [setChatStreamActive])

  // Usage Protection M4.3B: report-generation/regeneration streaming
  // lifecycle state, owned here (not inside ReportModePanel) -- same
  // "hook owns lifecycle, component stays presentational" split M4.2B
  // established for chat. Unlike chat, no local preview TEXT is ever
  // held (report content is never streamed) -- reportStreamError is its
  // own dedicated field, deliberately NOT the shared `error` banner, so
  // a handled report-stream failure surfaces inline in the report panel
  // rather than hijacking the app-wide error surface. reportStream
  // Stopping is a distinct third state (see cancelReportStream's own
  // docstring below): true only during the "cancellation requested, not
  // yet settled" window, so the UI can show a stable "Stopping" label
  // instead of implying cancellation is instantaneous.
  const [reportStreamActive, setReportStreamActiveState] = useState(false)
  const [reportStreamOperation, setReportStreamOperation] = useState<'generate' | 'regenerate' | null>(null)
  const [reportStreamPhase, setReportStreamPhase] = useState<ReportStreamPhase | null>(null)
  // report-progress-observability: the ordered, deduplicated phases
  // genuinely observed so far for the CURRENT active stream -- unlike
  // reportStreamPhase (the single latest phase, kept unchanged for any
  // existing caller/test that only needs that), this accumulates so
  // ReportModePanel can render every phase that actually happened, not
  // just the most recent one. Reset to [] wherever clearReportStreamPreview
  // already runs (a new operation starting, or existing settle/cleanup
  // paths) -- see runReportStreamOperation below for where entries are
  // appended via a functional update (so back-to-back phase events within
  // the same React batch can never overwrite one another).
  const [reportStreamPhaseHistory, setReportStreamPhaseHistory] = useState<ReportStreamPhase[]>([])
  const [reportStreamStopping, setReportStreamStopping] = useState(false)
  const [reportStreamError, setReportStreamError] = useState<string | null>(null)
  const [reportStreamSyncFailed, setReportStreamSyncFailed] = useState(false)
  // report-progress-observability: a temporary, completed-status-only
  // confirmation ("Report regenerated · Evaluated · Saved") -- same
  // auto-clearing mechanic as lastChatSearchMeta/lastAddToReportResult
  // above, just with its own named duration (REPORT_STREAM_SUCCESS_
  // NOTICE_MS). Deliberately NOT reset by clearReportStreamPreview (see
  // that function's own comment below) -- it is set explicitly, once,
  // only after loadState's own post-stream reload has already succeeded,
  // and cleared explicitly wherever a NEW report action starts or the
  // session changes (see runReportStreamOperation and the session-change
  // effect below).
  const [reportStreamCompletionNotice, setReportStreamCompletionNotice] =
    useAutoClearingState<ReportStreamCompletionNotice>(REPORT_STREAM_SUCCESS_NOTICE_MS)
  // Same ref-alongside-state rationale as chatStreamActiveRef above.
  const reportStreamActiveRef = useRef(false)
  const reportStreamAbortRef = useRef<AbortController | null>(null)

  const setReportStreamActive = useCallback((active: boolean) => {
    reportStreamActiveRef.current = active
    setReportStreamActiveState(active)
  }, [])

  // Deliberately does NOT reset reportStreamError -- unlike every other
  // field here, an error message describes something that already
  // finished; a LATER successful reload (whether from cancellation
  // cleanup, an unrelated action's own loadState call, or anything
  // else) doesn't retroactively un-happen the failure, so clearing it
  // here would silently erase a message the user hasn't acted on yet.
  // reportStreamError is instead reset only where a NEW report-stream
  // attempt actually begins (see runReportStreamOperation's own start).
  const clearReportStreamPreview = useCallback(() => {
    setReportStreamActive(false)
    setReportStreamOperation(null)
    setReportStreamPhase(null)
    setReportStreamPhaseHistory([])
    setReportStreamStopping(false)
    setReportStreamSyncFailed(false)
    reportStreamAbortRef.current = null
  }, [setReportStreamActive])

  // chat-ux-polish Phase A: called at the start of every chat action
  // (sendChatMessage/deleteExchanges/editExchange/addExchangesToReport)
  // so a success/info notice from a DIFFERENT, earlier action can't keep
  // sitting on screen once something new is happening. Deliberately
  // does NOT touch reportPossiblyStale -- that one has its own,
  // different lifecycle (see dismissReportStaleWarning below).
  const clearActionNotices = useCallback(() => {
    setLastChatSearchMeta(null)
    setLastAddToReportResult(null)
  }, [setLastChatSearchMeta, setLastAddToReportResult])

  const dismissReportStaleWarning = useCallback(() => {
    setReportPossiblyStale(false)
  }, [])

  const loadState = useCallback(async (id: string): Promise<CurationStateResponse> => {
    const fresh = await curationApi.getState(id)
    // UXH.1 (UX-04): id is the session THIS call was requested for -- if
    // the user has since switched to a different session (or this hook
    // has unmounted) while the request was in flight, publishing fresh
    // here would overwrite whatever's actually current on screen with a
    // stale result. Every caller of loadState (the sessionId-change
    // effect below, and every action -- submitPicks, chat/report
    // streaming, deleteExchanges, etc.) shares this one guard, so none of
    // them need their own staleness check. `fresh` is still returned
    // either way -- a caller mid-action for the (no longer current)
    // session it started against may still want the raw response even
    // though it's not being published to shared state.
    if (!isMountedRef.current || currentSessionIdRef.current !== id) {
      return fresh
    }
    setState(fresh)
    if (turnEventsSessionRef.current !== id) {
      turnEventsSessionRef.current = id
      setTurnEvents([])
      setLastChatSearchMeta(null)
    }
    // Usage Protection M4.2B/M4.3B: canonical state is now current -- any
    // leftover streaming preview (chat or report, from this call's own
    // caller, or from an earlier turn) is superseded. Safe to call
    // unconditionally: a no-op when there was nothing to clear.
    clearChatStreamPreview()
    clearReportStreamPreview()
    return fresh
  }, [clearChatStreamPreview, clearReportStreamPreview])

  // UXH.2: actionKind is optional -- every OTHER runAction caller
  // (generateReport, deleteExchanges, deleteReview, etc.) omits it and is
  // completely unaffected, same before/after behavior as always. Passed,
  // it does two things: (1) a synchronous reentrancy guard, exactly like
  // chatStreamActiveRef/reportStreamActiveRef's own guards, so a second
  // curation action can never start while one is already running, even
  // across a fast double-click the UI's own disabled-button re-render
  // hasn't caught up with yet; (2) sets/clears curationAction alongside
  // the existing loading/error lifecycle, on every settle path (success
  // or failure alike) via the same finally block already used for
  // loading.
  const runAction = useCallback(async <T,>(action: () => Promise<T>, actionKind?: CurationActionKind): Promise<T | undefined> => {
    if (actionKind) {
      if (curationActionRef.current) return undefined
      setCurationAction(actionKind)
    }
    setLoading(true)
    setError(null)
    try {
      return await action()
    } catch (err) {
      // Usage Protection M2.3: the one shared translation from a caught
      // failure to safe, user-facing text -- see lib/api/errorMessages.ts.
      // `state`/`turnEvents`/etc. are untouched here (they're only ever
      // updated by loadState() on a CONFIRMED backend success elsewhere
      // in this hook), so whatever was already rendered before this
      // action stays exactly as it was.
      setError(getUserFacingErrorMessage(err))
      return undefined
    } finally {
      setLoading(false)
      if (actionKind) setCurationAction(null)
    }
  }, [setCurationAction])

  // THE Phase 6d property: whatever session_id the URL names -- including
  // right after a hard page refresh, which re-runs this effect from
  // scratch with no prior in-memory state at all -- is what gets loaded
  // FROM THE BACKEND here. Nothing in this hook reads from
  // localStorage/sessionStorage or any other browser-only store; the URL
  // plus this one GET call is the entire source of truth on load.
  //
  // UXH.1 (UX-04): lastLoadedSessionIdRef records which session this
  // EFFECT last actually displayed (distinct from currentSessionIdRef,
  // which setSessionId updates immediately at the moment of intent, well
  // before this effect gets a chance to run) -- comparing the two tells
  // this run whether it's a genuine switch AWAY from a different,
  // already-displayed session (clear state so the old review can't be
  // mistaken for the new one while the fetch is in flight) versus the
  // first-ever load for this hook instance or a startReview-style
  // null -> id transition (state is already null; nothing to clear, and
  // clearing it here would race startReview's own direct loadState call
  // for no benefit). A plain `runAction(() => loadState(sessionId))`
  // (the prior implementation) doesn't distinguish stale from current
  // here, so it's replaced with logic that only publishes error/loading
  // completion for the request THIS run actually issued -- loadState
  // itself already guards the success path the same way for every
  // caller, not just this one.
  useEffect(() => {
    const previousId = lastLoadedSessionIdRef.current
    lastLoadedSessionIdRef.current = sessionId
    if (!sessionId) {
      if (previousId !== null) setState(null)
      return
    }
    const requestedId = sessionId
    if (previousId !== null && previousId !== requestedId) {
      setState(null)
    }
    setError(null)
    setLoading(true)
    loadState(requestedId)
      .catch((err) => {
        if (isMountedRef.current && currentSessionIdRef.current === requestedId) {
          setError(getUserFacingErrorMessage(err))
        }
      })
      .finally(() => {
        if (isMountedRef.current && currentSessionIdRef.current === requestedId) {
          setLoading(false)
        }
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  // Usage Protection M4.2B: aborts any in-flight chat stream whenever the
  // open session changes -- a stream belongs to the session it was
  // started for, and switching sessions must not let it keep running and
  // writing into now-stale UI state -- or when this hook itself
  // unmounts. One cleanup function covers both: React runs it before
  // re-running this effect for a new sessionId, and again on unmount.
  useEffect(() => {
    return () => {
      chatStreamAbortRef.current?.abort()
    }
  }, [sessionId])

  // Usage Protection M4.3B: same reasoning as the chat-stream effect
  // above, for a report-generation/regeneration stream -- a stream
  // belongs to the session it was started for.
  // report-progress-observability: also clears a still-showing completion
  // notice on the same edge -- it describes a report-stream turn that
  // finished for the session being LEFT, and must not keep sitting on
  // screen once the user has switched (or the hook itself unmounts).
  useEffect(() => {
    return () => {
      reportStreamAbortRef.current?.abort()
      setReportStreamCompletionNotice(null)
    }
  }, [sessionId, setReportStreamCompletionNotice])

  useEffect(() => {
    const onPopState = () => setSessionId(getSessionIdFromUrl())
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [setSessionId])

  const openReview = useCallback((id: string) => {
    setSessionIdInUrl(id)
    setSessionId(id)
  }, [setSessionId])

  const startReview = useCallback(
    (topic: string, targetCount: number) =>
      runAction(async () => {
        const response = await curationApi.start({ topic, target_count: targetCount })
        setSessionIdInUrl(response.session_id)
        setSessionId(response.session_id)
        await loadState(response.session_id)
      }, 'starting_review'),
    [runAction, loadState, setSessionId],
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
        // UXH.2: stop takes priority over requestRefill in the (today
        // unreachable via the UI, since ReviewModePanel's own handlers
        // never combine them) case both are somehow true -- finishing the
        // review is the more consequential of the two to describe
        // truthfully.
      }, stop ? 'finishing_review' : requestRefill ? 'searching_more' : 'continuing_review'),
    [runAction, sessionId, state, turnEvents.length, loadState],
  )

  const generateReport = useCallback(
    (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) =>
      runAction(async () => {
        if (!sessionId) return undefined
        await curationApi.generateReport(sessionId, reportTemplate, refinementMode)
        // chat-ux-polish Phase A: a fresh generation resolves whatever
        // prompted the stale warning (if anything did) -- this is a real
        // report-changing action succeeding, the explicit clear condition
        // reportPossiblyStale's own docs promise.
        setReportPossiblyStale(false)
        return loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const regenerateReport = useCallback(
    (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) =>
      runAction(async () => {
        if (!sessionId) return undefined
        await curationApi.regenerateReport(sessionId, reportTemplate, refinementMode)
        setReportPossiblyStale(false)
        return loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  // report-quality Phase R3: a pure pointer switch, not a report-
  // changing action -- but still routed through the same runAction/
  // loadState refresh pattern as generateReport/regenerateReport above,
  // since the activated version's content only becomes visible once
  // state is reloaded from the backend.
  const activateReportVersion = useCallback(
    (versionId: string) =>
      runAction(async () => {
        if (!sessionId) return undefined
        await curationApi.activateReportVersion(sessionId, versionId)
        return loadState(sessionId)
      }),
    [runAction, sessionId, loadState],
  )

  const sendChatMessage = useCallback(
    (message: string) =>
      runAction(async () => {
        if (!sessionId) return
        // chat-ux-polish Phase A: a new chat action starting clears any
        // stale success/info notice from a DIFFERENT, earlier action --
        // see clearActionNotices' own docs. Deliberately before the
        // await, so the UI reflects "something new is happening" right
        // away, not only once the round trip finishes.
        clearActionNotices()
        const response = await curationApi.chat(sessionId, { message })
        setLastChatSearchMeta(
          response.web_search_used
            ? { webSearchUsed: true, newWebArticlesFound: response.new_web_articles_found }
            : null,
        )
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState, clearActionNotices, setLastChatSearchMeta],
  )

  // Usage Protection M4.2B: the streaming counterpart to sendChatMessage
  // above -- POST /curation/{session_id}/chat/stream via lib/api/
  // chatStream.ts's streamCurationChat, driving this hook's own
  // chatStream* state instead of resolving once like every other action
  // here. sendChatMessage itself is untouched and stays fully callable
  // (non-streaming backward compatibility is an explicit requirement of
  // this phase) -- this is simply what the UI's normal Send/offer-button
  // path now calls instead (see CurationWorkspacePage.tsx's own
  // handleSendMessage).
  //
  // Event-sequencing is enforced HERE, not in chatStream.ts (which only
  // validates SSE framing) -- a violation of the frozen M4.1/M4.2A
  // contract (delta before started, anything after done, more than one
  // completed/error, both completed AND error, done missing entirely) is
  // treated exactly like a transport failure: the stream is abandoned, a
  // safe message is shown, and canonical state is reloaded to discover
  // what (if anything) the backend actually persisted. Real backend
  // behavior never produces this -- it would only come from a corrupted
  // transport.
  //
  // Does not run through runAction/the shared loading flag (unlike every
  // other action here) -- this needs its own incremental phase/delta
  // updates DURING the round trip, not runAction's single before/after
  // toggle, and a cancellation must never populate the shared `error`
  // banner. chatStreamActiveRef guards against a second concurrent
  // stream; CurationWorkspacePage additionally disables the whole panel
  // (loading || chatStreamActive) while one is running.
  const sendChatMessageStreaming = useCallback(
    async (message: string): Promise<void> => {
      // Usage Protection M4.3B: also refuses to start while a report
      // stream is active for this session -- the two are mutually
      // exclusive (same "one paid/blocking action at a time" invariant
      // `loading` already enforces for every non-streaming action), not
      // just independently guarded.
      if (!sessionId || chatStreamActiveRef.current || reportStreamActiveRef.current) return
      clearActionNotices()
      setError(null)
      const controller = new AbortController()
      chatStreamAbortRef.current = controller
      setChatStreamActive(true)
      setChatStreamPhase(null)
      setChatStreamText('')
      setChatStreamSyncFailed(false)

      const syncMessage = 'Lost the connection while the assistant was responding. Checking what was saved…'
      const reloadBestEffort = async () => {
        try {
          await loadState(sessionId)
        } catch {
          // Best-effort only -- the next successful action's own
          // loadState call will catch canonical state up regardless.
        }
      }

      let sawStarted = false
      let sawDone = false
      let sawCompleted = false
      let sawError = false

      try {
        for await (const event of streamCurationChat(sessionId, { message }, { signal: controller.signal })) {
          if (sawDone) {
            throw new ChatStreamTransportError('Received data after the stream had already ended.')
          }
          if (event.type === 'started') {
            if (sawStarted) throw new ChatStreamTransportError('Received a duplicate start signal.')
            sawStarted = true
            continue
          }
          if (!sawStarted) {
            throw new ChatStreamTransportError('Received data before the stream had started.')
          }
          switch (event.type) {
            case 'phase':
              setChatStreamPhase(event.data.phase)
              break
            case 'delta':
              setChatStreamText((prev) => prev + event.data.text)
              break
            case 'completed':
              if (sawCompleted || sawError) throw new ChatStreamTransportError('Received more than one terminal signal.')
              sawCompleted = true
              break
            case 'error':
              if (sawCompleted || sawError) throw new ChatStreamTransportError('Received more than one terminal signal.')
              sawError = true
              setError(event.data.message)
              break
            case 'done':
              sawDone = true
              break
          }
        }
      } catch (err) {
        if (controller.signal.aborted) {
          // Deliberate user/unmount/session-change cancellation -- never
          // a visible application error. The backend's own M4.2A
          // lifecycle hardening guarantees a genuinely cancelled turn
          // never leaves a half-persisted result (research_agent/
          // curation_chat_streaming.py's own _persist_and_complete waits
          // for any in-flight save to settle before the lease is
          // released), so reloading here can only ever reveal "nothing
          // changed" or a save that had already committed -- either way,
          // canonical state is what must be shown, never the abandoned
          // local preview.
          clearChatStreamPreview()
          await reloadBestEffort()
          return
        }
        clearChatStreamPreview()
        if (err instanceof ApiError) {
          // A preflight rejection (404/400/409/422/429/503) -- research_
          // agent/services/curation_chat_service.py::stream_answer_
          // curation_chat's own synchronous checks all run BEFORE any
          // StreamingResponse is constructed, so nothing was ever
          // started; the existing shared error-message mapping applies
          // and there is nothing to reload.
          setError(getUserFacingErrorMessage(err))
          return
        }
        // Any other failure once an attempt to read the stream began --
        // a ChatStreamTransportError (malformed/truncated SSE) or a raw
        // transport failure (e.g. the underlying fetch() itself
        // rejecting) alike -- is the same "something broke mid-response"
        // bucket: a safe, fixed retry message, never automatically
        // resending the POST, and a reload to discover whether anything
        // was actually persisted before the failure.
        setError(syncMessage)
        await reloadBestEffort()
        return
      }

      if (!sawDone || (!sawCompleted && !sawError)) {
        // The server closed the connection without the one required
        // clean-termination signal (`done`), or without ever reaching a
        // terminal outcome at all -- same "malformed/truncated" bucket
        // as the catch block above.
        clearChatStreamPreview()
        setError(syncMessage)
        await reloadBestEffort()
        return
      }

      if (sawError) {
        // A handled SSE error -> done: the backend's own M4.2A hardening
        // guarantees nothing was persisted for a handled failure (see
        // research_agent/curation_chat_streaming.py's own
        // HandledStreamFailure docstring), so no reload is needed here --
        // the safe message is already set above.
        clearChatStreamPreview()
        return
      }

      // Success: completed -> done. chatStreamText already holds the
      // exact final answer text -- the adapter's own "sum of deltas ==
      // final answer" invariant (research_agent/chat_streaming.py's own
      // stream_chat_answer docstring) -- so there is nothing further to
      // capture from the completed event's own payload before reloading
      // canonical state, which is authoritative for everything the
      // completed payload deliberately does not carry (exchange_id,
      // citations, relevance metadata, offer state, chat_references).
      setChatStreamPhase(null)
      try {
        await loadState(sessionId)
        // loadState's own success path already calls
        // clearChatStreamPreview() -- see above.
      } catch {
        setChatStreamSyncFailed(true)
        setChatStreamActive(false)
      }
    },
    [sessionId, loadState, clearActionNotices, clearChatStreamPreview, setChatStreamActive],
  )

  const cancelChatStream = useCallback(() => {
    chatStreamAbortRef.current?.abort()
  }, [])

  // Usage Protection M4.3B: shared orchestration for generateReport
  // Streaming/regenerateReportStreaming below -- identical event-state-
  // machine/cancellation/reload handling either way; the only difference
  // between the two public actions is which adapter function to call and
  // which `generation_reason`-adjacent label ('generate'/'regenerate')
  // to track. Neither public action duplicates this body, so the two can
  // never drift apart. Mirrors sendChatMessageStreaming's own shape
  // closely, with two deliberate differences: (1) failures set the
  // dedicated reportStreamError field, never the shared `error` banner;
  // (2) cancellation shows a stable "stopping" window (reportStream
  // Stopping) while the post-cancellation reload is still in flight,
  // rather than clearing the preview immediately -- the backend may need
  // to wait out an active synchronous provider call before releasing its
  // lease, so pretending cancellation is instantaneous here would be
  // dishonest.
  const runReportStreamOperation = useCallback(
    async (
      operation: 'generate' | 'regenerate',
      streamFn: (
        sessionId: string,
        req: CurationGenerateReportRequest | CurationRegenerateReportRequest,
        options: { signal: AbortSignal },
      ) => AsyncGenerator<ReportStreamServerEvent>,
      reportTemplate?: ReportTemplate,
      refinementMode?: RefinementMode,
    ): Promise<void> => {
      // Usage Protection M4.3B: mutually exclusive with an active chat
      // stream too -- same cross-domain guard sendChatMessageStreaming's
      // own now carries in the other direction.
      if (!sessionId || reportStreamActiveRef.current || chatStreamActiveRef.current) return
      setError(null)
      const controller = new AbortController()
      reportStreamAbortRef.current = controller
      setReportStreamActive(true)
      setReportStreamOperation(operation)
      setReportStreamPhase(null)
      setReportStreamPhaseHistory([])
      setReportStreamStopping(false)
      setReportStreamError(null)
      setReportStreamSyncFailed(false)
      // report-progress-observability: a NEW report action starting always
      // clears a still-showing completion notice from an EARLIER turn --
      // same "something new is happening" rule clearActionNotices already
      // applies to the chat-side success/info notices.
      setReportStreamCompletionNotice(null)

      const syncMessage = 'Lost the connection while the report was being generated. Checking what was saved…'
      const reloadBestEffort = async () => {
        try {
          await loadState(sessionId)
        } catch {
          // Best-effort only -- the next successful action's own
          // loadState call will catch canonical state up regardless.
        }
      }
      // report-quality Phase R2C/R4.1 parity: the non-streaming generate
      // Report/regenerateReport actions send report_template/
      // refinement_mode only when meaningfully set (an explicit template,
      // and refinementMode !== 'off') -- same omission convention
      // preserved here so the backend's own None-means-"preserve
      // existing"/"off" resolution behaves identically either way.
      const req: CurationGenerateReportRequest | CurationRegenerateReportRequest = {
        ...(reportTemplate ? { report_template: reportTemplate } : {}),
        ...(refinementMode && refinementMode !== 'off' ? { refinement_mode: refinementMode } : {}),
      }

      let sawStarted = false
      let sawDone = false
      let sawCompleted = false
      let sawError = false
      // report-progress-observability: a plain local array, NOT read from
      // reportStreamPhaseHistory React state -- this async function's own
      // closure would otherwise see whatever value was current at CALL
      // time, never the state's later updates (the same reason sawStarted/
      // sawDone/etc. above are local variables, not state). Used at the
      // very end, once completed -> done has genuinely succeeded, to seed
      // the completion notice with exactly the phases this turn actually
      // observed. setReportStreamPhaseHistory (React-visible, for the
      // active trail) is still updated via a FUNCTIONAL update below, so
      // rapid consecutive phase events within the same React batch can
      // never overwrite one another.
      const observedPhases: ReportStreamPhase[] = []

      try {
        for await (const event of streamFn(sessionId, req, { signal: controller.signal })) {
          if (sawDone) {
            throw new ReportStreamTransportError('Received data after the stream had already ended.')
          }
          if (event.type === 'started') {
            if (sawStarted) throw new ReportStreamTransportError('Received a duplicate start signal.')
            sawStarted = true
            continue
          }
          if (!sawStarted) {
            throw new ReportStreamTransportError('Received data before the stream had started.')
          }
          switch (event.type) {
            case 'phase': {
              const phase = event.data.phase
              setReportStreamPhase(phase)
              // Defensive dedup -- see this block's own comment above and
              // the ReportStreamCompletionNotice docstring for why a
              // repeated phase must never appear twice or reorder history.
              if (!observedPhases.includes(phase)) observedPhases.push(phase)
              setReportStreamPhaseHistory((prev) => (prev.includes(phase) ? prev : [...prev, phase]))
              break
            }
            case 'completed':
              if (sawCompleted || sawError) throw new ReportStreamTransportError('Received more than one terminal signal.')
              sawCompleted = true
              break
            case 'error':
              if (sawCompleted || sawError) throw new ReportStreamTransportError('Received more than one terminal signal.')
              sawError = true
              setReportStreamError(event.data.message)
              break
            case 'done':
              sawDone = true
              break
          }
        }
      } catch (err) {
        if (controller.signal.aborted) {
          // Deliberate user/unmount/session-change cancellation -- never
          // a visible application error. reportStreamStopping stays true
          // (reportStreamActive/operation/phase all stay as they were)
          // through the reload below, so the UI can keep showing a
          // stable "Stopping" state instead of an instant snap back to
          // idle -- the backend's own M4.3A lifecycle hardening may
          // still be waiting out an in-flight synchronous provider call
          // before it can release the lease. clearReportStreamPreview()
          // only runs once that reload has genuinely settled (success or
          // failure), which is also the only point controls are safe to
          // re-enable.
          setReportStreamStopping(true)
          await reloadBestEffort()
          clearReportStreamPreview()
          return
        }
        clearReportStreamPreview()
        if (err instanceof ApiError) {
          // A preflight rejection (404/400/409/422/429/503) -- research_
          // agent/services/curation_report_service.py's own synchronous
          // checks all run BEFORE any StreamingResponse is constructed,
          // so nothing was ever started; nothing to reload.
          setReportStreamError(getUserFacingErrorMessage(err))
          return
        }
        // Any other failure once an attempt to read the stream began --
        // a ReportStreamTransportError (malformed/truncated SSE) or a
        // raw transport failure alike -- is the same "something broke
        // mid-response" bucket: a safe, fixed retry message, never
        // automatically resending the POST, and a reload to discover
        // whether anything was actually persisted before the failure.
        setReportStreamError(syncMessage)
        await reloadBestEffort()
        return
      }

      if (!sawDone || (!sawCompleted && !sawError)) {
        // The server closed the connection without the one required
        // clean-termination signal (`done`), or without ever reaching a
        // terminal outcome at all -- same "malformed/truncated" bucket
        // as the catch block above.
        clearReportStreamPreview()
        setReportStreamError(syncMessage)
        await reloadBestEffort()
        return
      }

      if (sawError) {
        // A handled SSE error -> done: the backend's own M4.3A hardening
        // guarantees nothing was persisted for a handled failure, so no
        // reload is needed here -- the safe message is already set above.
        clearReportStreamPreview()
        return
      }

      // Success: completed -> done. The completed payload's own report
      // content is deliberately never captured into local state here --
      // canonical state (loaded fresh below) is authoritative for the
      // report body, version, active version, refinement metadata,
      // generation reason, references, and sections; nothing here ever
      // constructs a report or appends a version client-side.
      setReportStreamPhase(null)
      setReportPossiblyStale(false)
      try {
        await loadState(sessionId)
        // loadState's own success path already calls
        // clearReportStreamPreview() -- see above.
        // report-progress-observability: set the completion notice only
        // now -- after loadState has genuinely succeeded, per this
        // phase's own "mark successful completion only after the existing
        // valid completed -> done contract AND a successful reload" rule.
        // Guarded against the session having changed while this reload
        // was in flight (loadState itself already guards its OWN state
        // publish the same way) -- without this, a session switch that
        // raced ahead of this await would otherwise resurrect a notice
        // for a turn that belongs to a review the user has since left,
        // moments after the session-change effect above already cleared
        // it.
        if (currentSessionIdRef.current === sessionId) {
          setReportStreamCompletionNotice({ operation, phases: observedPhases })
        }
      } catch {
        setReportStreamSyncFailed(true)
        setReportStreamActive(false)
        setReportStreamOperation(null)
        setReportStreamPhase(null)
      }
    },
    [sessionId, loadState, clearReportStreamPreview, setReportStreamActive, setReportStreamCompletionNotice],
  )

  const generateReportStreaming = useCallback(
    (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) =>
      runReportStreamOperation('generate', streamGenerateReport, reportTemplate, refinementMode),
    [runReportStreamOperation],
  )

  const regenerateReportStreaming = useCallback(
    (reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) =>
      runReportStreamOperation('regenerate', streamRegenerateReport, reportTemplate, refinementMode),
    [runReportStreamOperation],
  )

  const cancelReportStream = useCallback(() => {
    reportStreamAbortRef.current?.abort()
  }, [])

  const deleteExchanges = useCallback(
    (exchangeIds: string[]) =>
      runAction(async () => {
        if (!sessionId) return
        clearActionNotices()
        const response = await curationApi.deleteChatExchanges(sessionId, { exchange_ids: exchangeIds })
        setReportPossiblyStale(response.report_possibly_stale)
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState, clearActionNotices],
  )

  const addExchangesToReport = useCallback(
    (exchangeIds: string[]) =>
      runAction(async () => {
        if (!sessionId) return
        clearActionNotices()
        const response = await curationApi.addChatExchangesToReport(sessionId, { exchange_ids: exchangeIds })
        setLastAddToReportResult({
          addedCount: response.added_exchange_ids.length,
          skippedCount: response.skipped_exchange_ids.length,
          sourceCount: response.source_count,
        })
        // chat-ux-polish Phase A: this call just regenerated the report
        // for real (over the approved set) -- whatever staleness concern
        // was outstanding is resolved by this success, same as generate/
        // regenerateReport above.
        setReportPossiblyStale(false)
        // On failure, curationApi.addChatExchangesToReport above throws --
        // runAction's own catch sets the shared error and this line never
        // runs, so state (and therefore every badge) stays exactly as it
        // was. loadState() only ever runs after a confirmed backend
        // success, never optimistically.
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState, clearActionNotices, setLastAddToReportResult],
  )

  const editExchange = useCallback(
    (exchangeId: string, question: string) =>
      runAction(async () => {
        if (!sessionId) return
        clearActionNotices()
        const response = await curationApi.editChatExchange(sessionId, { exchange_id: exchangeId, question })
        // Same reused signal as deleteExchanges (Phase 3) -- editing away
        // a report-included exchange is the same kind of staleness.
        setReportPossiblyStale(response.report_possibly_stale)
        await loadState(sessionId)
      }),
    [runAction, sessionId, loadState, clearActionNotices],
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
          setSessionId(null)
          setState(null)
        }
      }),
    [runAction, sessionId, setSessionId],
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
    sessionId, state, loading, error, turnEvents, lastChatSearchMeta, reportPossiblyStale, lastAddToReportResult,
    dismissReportStaleWarning, curationAction,
    openReview, startReview, submitPicks, generateReport, regenerateReport,
    reportStreamActive, reportStreamOperation, reportStreamPhase, reportStreamPhaseHistory, reportStreamStopping,
    reportStreamError, reportStreamSyncFailed, reportStreamCompletionNotice,
    generateReportStreaming, regenerateReportStreaming, cancelReportStream,
    activateReportVersion, sendChatMessage,
    chatStreamActive, chatStreamPhase, chatStreamText, chatStreamSyncFailed, sendChatMessageStreaming, cancelChatStream,
    deleteExchanges, addExchangesToReport, editExchange, deleteReview, selectFromHistory, reopenReview, refresh,
  }
}
