import { useEffect, useRef, useState } from 'react'
import { useCurationSession } from '../hooks/useCurationSession'
import { curationApi } from '../lib/api/client'
import { AppHeader } from '../components/AppHeader/AppHeader'
import { ReviewsList } from '../components/ReviewsList/ReviewsList'
import { TopicHeader } from '../components/TurnFeed/TopicHeader'
import { ReviewModePanel } from '../components/ReviewMode/ReviewModePanel'
import { PoolSummaryPanel } from '../components/ReviewMode/PoolSummaryPanel'
import { ChatModePanel } from '../components/ChatMode/ChatModePanel'
import { ReportModePanel } from '../components/ReportMode/ReportModePanel'
import { TurnHistoryBrowser } from '../components/TurnHistory/TurnHistoryBrowser'
import type { WorkspaceMode } from '../components/WorkspaceMode/WorkspaceModeSwitcher'
import type { RefinementMode, ReportTemplate } from '../types'

const MODE_PARAM = 'mode'

// Usage Protection M2.3 Part D: preventative-UX mirrors of the backend's
// own provisional policy values (research_agent/config/limits.py:
// max_picked_ids_per_mutation, max_selected_papers_per_session) -- NOT
// re-derivations of backend logic, just the same two numbers, so a
// future backend policy change (still out of scope for this phase) is
// the only place this app's actual enforcement lives. See handleAdd
// below for where these are applied.
const MAX_PICKED_IDS_PER_MUTATION = 30
const MAX_SELECTED_PAPERS_PER_SESSION = 60

// Mirrors the `session` URL param pattern in useCurationSession.ts --
// workspaceMode is UI-only state, but it still needs to survive a real
// browser reload (e.g. mid-chat), or a refresh would silently bounce the
// user back out of Chat/Report, regressing the already-tested "state
// survives refresh" property.
function getModeFromUrl(): WorkspaceMode {
  const raw = new URLSearchParams(window.location.search).get(MODE_PARAM)
  return raw === 'chat' || raw === 'report' ? raw : 'review'
}

function setModeInUrl(mode: WorkspaceMode): void {
  const url = new URL(window.location.href)
  if (mode === 'review') url.searchParams.delete(MODE_PARAM)
  else url.searchParams.set(MODE_PARAM, mode)
  window.history.replaceState({}, '', url)
}

export default function CurationWorkspacePage() {
  const {
    sessionId, state, loading, error, turnEvents, lastChatSearchMeta, reportPossiblyStale, lastAddToReportResult,
    dismissReportStaleWarning,
    openReview, startReview, submitPicks, activateReportVersion,
    chatStreamActive, chatStreamPhase, chatStreamText, chatStreamSyncFailed, sendChatMessageStreaming, cancelChatStream,
    reportStreamActive, reportStreamOperation, reportStreamPhase, reportStreamStopping, reportStreamError,
    reportStreamSyncFailed, generateReportStreaming, regenerateReportStreaming, cancelReportStream,
    deleteExchanges, addExchangesToReport, editExchange, deleteReview, selectFromHistory, reopenReview, refresh,
  } = useCurationSession()

  const [stagedPickIds, setStagedPickIds] = useState<string[]>([])
  const [reviewsRefreshToken, setReviewsRefreshToken] = useState(0)
  const [workspaceMode, setWorkspaceModeState] = useState<WorkspaceMode>(() => getModeFromUrl())
  // curation-turn-history Phase 9f: deliberately NOT a 4th WorkspaceMode --
  // reachable from Review/Chat/Report alike (the whole point: being
  // "stuck" in Report mode no longer means the earlier turns are gone),
  // so it's an overlay toggle on top of whichever mode is active, not a
  // peer of the tri-tab switcher's own lock semantics.
  const [showHistory, setShowHistory] = useState(false)

  // Chat and Report unlock together, purely on curation being finished
  // (stage === "synthesize") -- chat_turn()'s own guard has no
  // dependency on a report existing yet, so gating on has_report would
  // be stricter than what the backend actually requires.
  const unlocked = state?.stage === 'synthesize'

  // curation-editable-until-locked Phase 10e: the ONLY gate on going
  // back to active curation, mirroring reopen_curation_session()'s own
  // server-side check exactly (research_agent/curation_session.py) --
  // this is just the UI-hiding half, the backend re-validates for real.
  // Once reopened, `unlocked` above naturally goes back to false on the
  // next loadState() (stage flips back to "curate"), which the existing
  // effect below already handles by bouncing workspaceMode back to
  // 'review' -- no separate re-lock logic needed here.
  const reopenEligible = !!state && state.stage === 'synthesize' && state.report === null && state.chat_history.length === 0

  // Staged ("+ Add to review") picks are scoped to whichever batch is
  // CURRENTLY pending -- reset whenever the pending batch's identity
  // changes, including "there's no longer a pending batch at all"
  // (curation just finished).
  const pendingBatchKey = state?.pending_batch?.map((p) => p.paper_id).join(',') ?? null
  useEffect(() => {
    setStagedPickIds([])
  }, [pendingBatchKey])

  function setWorkspaceMode(mode: WorkspaceMode) {
    setWorkspaceModeState(mode)
    setModeInUrl(mode)
    // Chat UX fixes, bug 1: the turn history browser is Review/Chat-only
    // (redundant in Review's own inline turn feed but still useful there;
    // never made sense in Report, which has no notion of "turns" at all)
    // -- switching TO Report must close it, not just hide the toggle
    // button that opens it, so navigating there via a tab click can't
    // leave it stuck open from an earlier Review/Chat session.
    if (mode === 'report') setShowHistory(false)
  }

  // Re-sync from the URL on back/forward navigation, mirroring the
  // session-id popstate handling in useCurationSession.ts.
  useEffect(() => {
    function onPopState() {
      setWorkspaceModeState(getModeFromUrl())
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  // Phase 8, item 3 fallout: the auto-jump-to-Report below must fire
  // EXACTLY ONCE, at the live moment a session transitions from
  // in-progress to just-finished -- not every time this effect happens to
  // re-run while already unlocked. It used to fire unconditionally
  // whenever (unlocked && workspaceMode === 'review'), which meant a
  // completed review could never actually be VIEWED in Review mode at
  // all: reopening one (handleSelectReview resets mode to 'review'),
  // deep-linking with ?mode=review, or manually clicking the Review tab
  // all got instantly bounced back to Report before render even settled
  // -- silently making ReviewModePanel's now-fixed selected-papers view
  // unreachable through any normal navigation path. Tracked per
  // session_id so opening a DIFFERENT already-finished review doesn't
  // read as a "live transition" either.
  const prevUnlockRef = useRef<{ sessionId: string | null; unlocked: boolean }>({ sessionId: null, unlocked: false })

  // Two safety/UX rules:
  //  - The moment curation finishes WHILE THIS SESSION IS ALREADY OPEN,
  //    jump straight to Report once -- that's the natural next step right
  //    after the last pick. Never clobber a manual Chat/Report navigation,
  //    and never re-fire just because the review happens to still be
  //    unlocked on a later render.
  //  - If Chat/Report is showing but curation ISN'T actually finished
  //    (e.g. a stale/bookmarked `?mode=chat` URL from before this review
  //    reached that stage), fall back to Review rather than rendering a
  //    locked tab's content.
  useEffect(() => {
    // While state is still loading (null), we don't yet know whether
    // `unlocked` is real or just the null-state default of false --
    // clamping here would force a `?mode=chat` reload back to Review
    // before the real (possibly-unlocked) state ever arrives.
    if (!state) return

    const sameSession = prevUnlockRef.current.sessionId === state.session_id
    const justUnlocked = sameSession && !prevUnlockRef.current.unlocked && unlocked
    prevUnlockRef.current = { sessionId: state.session_id, unlocked }

    if (!unlocked && workspaceMode !== 'review') {
      setWorkspaceMode('review')
    } else if (justUnlocked && workspaceMode === 'review') {
      setWorkspaceMode('report')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, unlocked, workspaceMode])

  function handleSelectReview(id: string) {
    setStagedPickIds([])
    setWorkspaceMode('review')
    setShowHistory(false)
    openReview(id)
  }

  async function handleStartReview(topic: string, targetCount: number) {
    setWorkspaceMode('review')
    setShowHistory(false)
    await startReview(topic, targetCount)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleDeleteReview(id: string) {
    await deleteReview(id)
    setReviewsRefreshToken((t) => t + 1)
  }

  // curation-refinement-and-auto-offer Phase 6f-2: refinement text (if
  // any was typed) rides in the SAME picks submission, forcing a fresh,
  // refinement-guided search on this turn rather than waiting for the
  // pool to run low on its own.
  async function handleSubmitPicks(refinement: string | undefined, stop: boolean, requestRefill?: boolean) {
    await submitPicks(stagedPickIds, stop, refinement, requestRefill)
    setReviewsRefreshToken((t) => t + 1)
  }

  // Usage Protection M4.3B: streaming is now the normal submission path
  // for both Generate and Regenerate (ReportModePanel's own onGenerate
  // Report/onRegenerateReport props, unchanged in shape) -- generate
  // Report/regenerateReport (non-streaming) remain fully callable on the
  // hook for backward compatibility, just no longer wired to this UI's
  // own normal submission path. Neither handler awaits all the way to a
  // canonical reload the way the old non-streaming versions did in one
  // shot -- generateReportStreaming/regenerateReportStreaming resolve
  // once the whole streamed turn (including its own canonical reload)
  // has settled, so this refresh token bump still lands at the right
  // time either way.
  async function handleGenerateReport(reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) {
    await generateReportStreaming(reportTemplate, refinementMode)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleRegenerateReport(reportTemplate?: ReportTemplate, refinementMode?: RefinementMode) {
    await regenerateReportStreaming(reportTemplate, refinementMode)
    setReviewsRefreshToken((t) => t + 1)
  }

  function handleRetryReportSync() {
    void refresh()
  }

  // report-quality Phase R3: a pure pointer switch -- doesn't change
  // has_report/has_chat/stage, so unlike generate/regenerate above,
  // there's nothing here the reviews sidebar needs to refresh for.
  async function handleActivateReportVersion(versionId: string) {
    await activateReportVersion(versionId)
  }

  // curation-refinement-and-auto-offer Phase 6f-4: report regeneration
  // is no longer triggered by a passive banner/button here -- it's
  // offered conversationally (chat_turn() appends the offer to the
  // answer and sets pending_report_update automatically once a new web
  // source makes the report stale), and accepted via the SAME Yes/No
  // buttons -> onSendMessage path ChatModePanel already uses for the
  // web-search offer, not a separate mechanism.
  // Usage Protection M4.2B: streaming is now the normal submission path
  // for the persistent-input Send button and the web-search/report-
  // update offer Yes/No buttons alike (ChatModePanel's own onSendMessage
  // prop, unchanged in shape) -- sendChatMessage (non-streaming) remains
  // fully callable on the hook for backward compatibility, just no
  // longer wired to this UI's own normal submission path.
  async function handleSendMessage(message: string) {
    await sendChatMessageStreaming(message)
    setReviewsRefreshToken((t) => t + 1)
  }

  function handleRetryChatSync() {
    void refresh()
  }

  async function handleDeleteExchanges(exchangeIds: string[]) {
    await deleteExchanges(exchangeIds)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleAddExchangesToReport(exchangeIds: string[]) {
    await addExchangesToReport(exchangeIds)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleEditExchange(exchangeId: string, question: string) {
    await editExchange(exchangeId, question)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleSelectFromHistory(paperId: string) {
    await selectFromHistory(paperId)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleReopenReview() {
    await reopenReview()
    setReviewsRefreshToken((t) => t + 1)
  }

  function handleAdd(paperId: string) {
    setStagedPickIds((prev) => {
      if (prev.includes(paperId)) return prev
      // Usage Protection M2.3 Part D: preventative-only UX mirrors of the
      // backend's own provisional caps (research_agent/config/limits.py)
      // -- picked_paper_ids per /picks mutation (30) and unique selected
      // papers per session (60). The backend remains authoritative and
      // re-checks both for real (session_limits.py); this only avoids an
      // avoidable round trip for the common case of staging past either
      // cap through this UI's own turn-history "add from any past turn"
      // path, the one place more than a single batch's worth (10) can
      // accumulate before one submission.
      if (prev.length >= MAX_PICKED_IDS_PER_MUTATION) return prev
      const existingSelected = state?.selected_paper_ids.length ?? 0
      if (existingSelected + prev.length + 1 > MAX_SELECTED_PAPERS_PER_SESSION) return prev
      return [...prev, paperId]
    })
  }

  function handleRemoveStaged(paperId: string) {
    setStagedPickIds((prev) => prev.filter((id) => id !== paperId))
  }

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-bg text-text">
      <AppHeader />
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <ReviewsList
          activeSessionId={sessionId}
          onSelectReview={handleSelectReview}
          onStartReview={handleStartReview}
          onDeleteReview={handleDeleteReview}
          refreshToken={reviewsRefreshToken}
          workspaceMode={workspaceMode}
          workspaceUnlocked={unlocked}
          onWorkspaceModeChange={setWorkspaceMode}
        />

        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          {!state && (
            <div className="flex flex-1 items-center justify-center text-sm text-text-muted">
              {loading ? 'Loading…' : 'Select a review on the left, or start a new one.'}
            </div>
          )}
          {/* Usage Protection M2.3 Part C: the one shared error surface for
              every curation action (useCurationSession's own runAction sets
              `error` on failure, clears it at the start of the next action
              AND on any success -- see that hook for the full lifecycle).
              role="alert" gives it accessible live-region semantics without
              adding a toast/alert library; a new action or a successful
              retry is what clears it, never a timer, so it stays visible
              exactly as long as nothing else has happened yet. */}
          {error && (
            <div
              role="alert"
              data-testid="curation-error-banner"
              className="border-b border-danger/30 bg-danger-soft px-4 py-2 text-sm text-danger"
            >
              {error}
            </div>
          )}
          {state && (
            <>
              <TopicHeader topic={state.display_title} selectedCount={state.selected_paper_ids.length} targetCount={state.target_count} />
              {!showHistory && (reopenEligible || (state.turn_history.length > 0 && workspaceMode !== 'report')) && (
                <div className="flex items-center justify-between border-b border-border bg-panel px-4 py-1.5">
                  <div>
                    {reopenEligible && (
                      <button
                        type="button"
                        data-testid="reopen-review"
                        onClick={handleReopenReview}
                        disabled={loading}
                        className="text-xs text-text-secondary underline decoration-dotted hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        Reopen for more curation
                      </button>
                    )}
                  </div>
                  {/* Chat UX fixes, bug 1: Review + Chat only, never Report -- see setWorkspaceMode above. */}
                  {state.turn_history.length > 0 && workspaceMode !== 'report' && (
                    <button
                      type="button"
                      data-testid="open-turn-history"
                      onClick={() => setShowHistory(true)}
                      className="text-xs text-text-secondary underline decoration-dotted hover:text-accent"
                    >
                      Browse past turns ({state.turn_history.length})
                    </button>
                  )}
                </div>
              )}
              {showHistory ? (
                <TurnHistoryBrowser
                  state={state}
                  stagedPickIds={stagedPickIds}
                  disabled={loading}
                  onAdd={handleAdd}
                  onRemoveStaged={handleRemoveStaged}
                  onSelectFromHistory={handleSelectFromHistory}
                  onClose={() => setShowHistory(false)}
                />
              ) : (
                <>
                  {workspaceMode === 'review' && (
                    <ReviewModePanel
                      state={state}
                      turnEvents={turnEvents}
                      stagedPickIds={stagedPickIds}
                      disabled={loading}
                      onAdd={handleAdd}
                      onRemoveStaged={handleRemoveStaged}
                      onSubmitPicks={handleSubmitPicks}
                    />
                  )}
                  {workspaceMode === 'chat' && (
                    <ChatModePanel
                      state={state}
                      disabled={loading || chatStreamActive || reportStreamActive}
                      onSendMessage={handleSendMessage}
                      lastSearchMeta={lastChatSearchMeta}
                      onDeleteExchanges={handleDeleteExchanges}
                      reportPossiblyStale={reportPossiblyStale}
                      onAddExchangesToReport={handleAddExchangesToReport}
                      lastAddToReportResult={lastAddToReportResult}
                      onEditExchange={handleEditExchange}
                      onDismissReportStaleWarning={dismissReportStaleWarning}
                      chatStreamActive={chatStreamActive}
                      chatStreamPhase={chatStreamPhase}
                      chatStreamText={chatStreamText}
                      chatStreamSyncFailed={chatStreamSyncFailed}
                      onCancelChatStream={cancelChatStream}
                      onRetrySync={handleRetryChatSync}
                    />
                  )}
                  {workspaceMode === 'report' && (
                    <ReportModePanel
                      state={state}
                      disabled={loading || reportStreamActive || chatStreamActive}
                      onGenerateReport={handleGenerateReport}
                      onRegenerateReport={handleRegenerateReport}
                      onActivateReportVersion={handleActivateReportVersion}
                      exportUrls={{
                        markdown: curationApi.getReportExportUrl(state.session_id, 'markdown'),
                        pdf: curationApi.getReportExportUrl(state.session_id, 'pdf'),
                        docx: curationApi.getReportExportUrl(state.session_id, 'docx'),
                      }}
                      reportStreamActive={reportStreamActive}
                      reportStreamOperation={reportStreamOperation}
                      reportStreamPhase={reportStreamPhase}
                      reportStreamStopping={reportStreamStopping}
                      reportStreamError={reportStreamError}
                      reportStreamSyncFailed={reportStreamSyncFailed}
                      onCancelReportStream={cancelReportStream}
                      onRetryReportSync={handleRetryReportSync}
                    />
                  )}
                </>
              )}
            </>
          )}
        </main>

        {state && workspaceMode === 'review' && !showHistory && (
          <PoolSummaryPanel state={state} stagedPickIds={stagedPickIds} />
        )}
      </div>
    </div>
  )
}
