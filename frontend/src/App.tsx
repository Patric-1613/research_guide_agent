import { useEffect, useState } from 'react'
import { useCurationSession } from './hooks/useCurationSession'
import { ReviewsList } from './components/ReviewsList/ReviewsList'
import { TopicHeader } from './components/TurnFeed/TopicHeader'
import { TurnFeed } from './components/TurnFeed/TurnFeed'
import { PersistentInput } from './components/PersistentInput/PersistentInput'
import { PaperPool } from './components/PaperPool/PaperPool'

export default function App() {
  const {
    sessionId, state, loading, error, turnEvents,
    openReview, startReview, submitPicks, generateReport, regenerateReport, sendChatMessage,
  } = useCurationSession()

  const [stagedPickIds, setStagedPickIds] = useState<string[]>([])
  const [reviewsRefreshToken, setReviewsRefreshToken] = useState(0)
  // How many web_articles_added the report on screen actually covers --
  // NOT the same as "web_articles_added.length > 0" (that stays true
  // forever after the first web source is ever approved, even once the
  // report has already been regenerated to include it -- a real bug an
  // e2e run caught: the banner never went away after clicking
  // "Regenerate report"). Reset to the CURRENT count whenever a
  // different session loads (assume an existing persisted report
  // already covers whatever it was generated against), then bumped
  // explicitly only when a generate/regenerate call actually succeeds.
  const [reportCoveredWebArticleCount, setReportCoveredWebArticleCount] = useState(0)

  // Staged ("+ Add to review") picks are scoped to whichever batch is
  // CURRENTLY pending -- reset whenever the pending batch's identity
  // changes, including "there's no longer a pending batch at all"
  // (curation just finished).
  const pendingBatchKey = state?.pending_batch?.map((p) => p.paper_id).join(',') ?? null
  useEffect(() => {
    setStagedPickIds([])
  }, [pendingBatchKey])

  useEffect(() => {
    if (state) setReportCoveredWebArticleCount(state.web_articles_added.length)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.session_id])

  function handleSelectReview(id: string) {
    setStagedPickIds([])
    openReview(id)
  }

  async function handleStartReview(topic: string, targetCount: number) {
    await startReview(topic, targetCount)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleSubmitPicks() {
    await submitPicks(stagedPickIds)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleGenerateReport() {
    const fresh = await generateReport()
    if (fresh) setReportCoveredWebArticleCount(fresh.web_articles_added.length)
    setReviewsRefreshToken((t) => t + 1)
  }

  async function handleRegenerateReport() {
    const fresh = await regenerateReport()
    if (fresh) setReportCoveredWebArticleCount(fresh.web_articles_added.length)
    setReviewsRefreshToken((t) => t + 1)
  }

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

  const showRegenerateBanner =
    !!state?.report && state.stage === 'synthesize' && state.web_articles_added.length > reportCoveredWebArticleCount

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-text">
      <ReviewsList
        activeSessionId={sessionId}
        onSelectReview={handleSelectReview}
        onStartReview={handleStartReview}
        refreshToken={reviewsRefreshToken}
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
            {showRegenerateBanner && (
              <div className="flex items-center justify-between border-b border-border bg-panel-alt px-4 py-2 text-xs text-text-secondary">
                <span>New web sources were added — the report doesn't reflect them yet.</span>
                <button
                  type="button"
                  onClick={handleRegenerateReport}
                  disabled={loading}
                  className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-fg disabled:opacity-40"
                >
                  Regenerate report
                </button>
              </div>
            )}
            <TurnFeed state={state} turnEvents={turnEvents} />
            <PersistentInput
              stage={state.stage}
              hasReport={state.report != null}
              pendingWebOffer={state.pending_web_offer}
              disabled={loading}
              stagedPickCount={stagedPickIds.length}
              onSubmitPicks={handleSubmitPicks}
              onGenerateReport={handleGenerateReport}
              onSendMessage={handleSendMessage}
            />
          </>
        )}
      </main>

      {state && (
        <PaperPool state={state} stagedPickIds={stagedPickIds} onAdd={handleAdd} onRemoveStaged={handleRemoveStaged} />
      )}
    </div>
  )
}
