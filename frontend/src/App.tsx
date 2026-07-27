import { useEffect, useState } from 'react'
import { useCurationSession } from './hooks/useCurationSession'
import { AppHeader } from './components/AppHeader/AppHeader'
import { ReviewsList } from './components/ReviewsList/ReviewsList'
import { TopicHeader } from './components/TurnFeed/TopicHeader'
import { ReviewModePanel } from './components/ReviewMode/ReviewModePanel'
import { PoolSummaryPanel } from './components/ReviewMode/PoolSummaryPanel'
import { ChatModePanel } from './components/ChatMode/ChatModePanel'
import { ReportModePanel } from './components/ReportMode/ReportModePanel'
import type { WorkspaceMode } from './components/WorkspaceMode/WorkspaceModeSwitcher'

const MODE_PARAM = 'mode'

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

export default function App() {
  const {
    sessionId, state, loading, error, turnEvents,
    openReview, startReview, submitPicks, generateReport, regenerateReport, sendChatMessage,
  } = useCurationSession()

  const [stagedPickIds, setStagedPickIds] = useState<string[]>([])
  const [reviewsRefreshToken, setReviewsRefreshToken] = useState(0)
  const [workspaceMode, setWorkspaceModeState] = useState<WorkspaceMode>(() => getModeFromUrl())

  // Chat and Report unlock together, purely on curation being finished
  // (stage === "synthesize") -- chat_turn()'s own guard has no
  // dependency on a report existing yet, so gating on has_report would
  // be stricter than what the backend actually requires.
  const unlocked = state?.stage === 'synthesize'

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

  // Two safety/UX rules, both keyed off `unlocked`:
  //  - The moment curation finishes, jump straight to Report -- that's
  //    the natural next step, and Review mode has nothing left to do (no
  //    pending batch). Never clobber a manual Chat/Report navigation the
  //    user already made.
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
    if (!unlocked && workspaceMode !== 'review') {
      setWorkspaceMode('review')
    } else if (unlocked && workspaceMode === 'review') {
      setWorkspaceMode('report')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, unlocked, workspaceMode])

  function handleSelectReview(id: string) {
    setStagedPickIds([])
    setWorkspaceMode('review')
    openReview(id)
  }

  async function handleStartReview(topic: string, targetCount: number) {
    setWorkspaceMode('review')
    await startReview(topic, targetCount)
    setReviewsRefreshToken((t) => t + 1)
  }

  // curation-refinement-and-auto-offer Phase 6f-2: refinement text (if
  // any was typed) rides in the SAME picks submission, forcing a fresh,
  // refinement-guided search on this turn rather than waiting for the
  // pool to run low on its own.
  async function handleSubmitPicks(refinement: string | undefined, stop: boolean) {
    await submitPicks(stagedPickIds, stop, refinement)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleGenerateReport() {
    await generateReport()
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleRegenerateReport() {
    await regenerateReport()
    setReviewsRefreshToken((t) => t + 1)
  }

  // curation-refinement-and-auto-offer Phase 6f-4: report regeneration
  // is no longer triggered by a passive banner/button here -- it's
  // offered conversationally (chat_turn() appends the offer to the
  // answer and sets pending_report_update automatically once a new web
  // source makes the report stale), and accepted via the SAME Yes/No
  // buttons -> onSendMessage path ChatModePanel already uses for the
  // web-search offer, not a separate mechanism.
  async function handleSendMessage(message: string) {
    await sendChatMessage(message)
    setReviewsRefreshToken((t) => t + 1)
  }

  function handleAdd(paperId: string) {
    setStagedPickIds((prev) => (prev.includes(paperId) ? prev : [...prev, paperId]))
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
          {error && (
            <div className="border-b border-danger/30 bg-danger-soft px-4 py-2 text-sm text-danger">{error}</div>
          )}
          {state && (
            <>
              <TopicHeader topic={state.topic} selectedCount={state.selected_paper_ids.length} targetCount={state.target_count} />
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
                <ChatModePanel state={state} disabled={loading} onSendMessage={handleSendMessage} />
              )}
              {workspaceMode === 'report' && (
                <ReportModePanel
                  state={state}
                  disabled={loading}
                  onGenerateReport={handleGenerateReport}
                  onRegenerateReport={handleRegenerateReport}
                />
              )}
            </>
          )}
        </main>

        {state && workspaceMode === 'review' && (
          <PoolSummaryPanel state={state} stagedPickIds={stagedPickIds} />
        )}
      </div>
    </div>
  )
}
