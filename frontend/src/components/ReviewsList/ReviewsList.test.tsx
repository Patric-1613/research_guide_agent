import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReviewsList } from './ReviewsList'
import { curationApi } from '../../lib/api/client'
import type { CurationReviewSummary } from '../../types'

vi.mock('../../lib/api/client', () => ({
  curationApi: {
    listReviews: vi.fn(),
  },
}))

function review(overrides: Partial<CurationReviewSummary> = {}): CurationReviewSummary {
  // display_title defaults to whatever topic ends up being (mirroring the
  // real backend's own pre-Phase-8/canonicalize-failure fallback) unless
  // a test explicitly overrides it separately -- so the many existing
  // tests here that only override `topic` still see that same string
  // rendered, since ReviewCard now displays display_title, not topic.
  const topic = overrides.topic ?? 'topic'
  return {
    session_id: 's1', topic, display_title: topic, stage: 'curate', selected_count: 0, target_count: 10,
    has_report: false, has_chat: false,
    ...overrides,
  }
}

describe('ReviewsList', () => {
  it('groups reviews under status section headers, in most-active-first order, skipping empty sections', async () => {
    vi.mocked(curationApi.listReviews).mockResolvedValue([
      review({ session_id: 'a', topic: 'Curating one', stage: 'curate' }),
      review({ session_id: 'b', topic: 'Has report', stage: 'synthesize', has_report: true }),
      review({ session_id: 'c', topic: 'Report and chat', stage: 'synthesize', has_report: true, has_chat: true }),
    ])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )

    await waitFor(() => expect(screen.getByText('Curating one')).toBeInTheDocument())

    // "Ready for report" has zero reviews here and must not render as an
    // empty section header.
    expect(screen.queryByText(/Ready for report/)).not.toBeInTheDocument()

    const headers = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(headers.some((h) => h?.startsWith('Curating'))).toBe(true)
    expect(headers.some((h) => h?.startsWith('Report ('))).toBe(true)
    expect(headers.some((h) => h?.startsWith('Report + Chat'))).toBe(true)
  })

  it('a chatted-but-no-report review groups under "Chatted", not "Ready for report" (Phase 8, previously-hidden state)', async () => {
    vi.mocked(curationApi.listReviews).mockResolvedValue([
      review({ session_id: 'x', topic: 'Chatted only', stage: 'synthesize', has_report: false, has_chat: true }),
    ])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )

    await waitFor(() => expect(screen.getByText('Chatted only')).toBeInTheDocument())
    const headers = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(headers.some((h) => h?.startsWith('Chatted ('))).toBe(true)
    expect(screen.queryByText(/Ready for report/)).not.toBeInTheDocument()
  })

  it('the workspace mode switcher only renders once a review is active', async () => {
    vi.mocked(curationApi.listReviews).mockResolvedValue([])

    const { rerender } = render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(curationApi.listReviews).toHaveBeenCalled())
    expect(screen.queryByTestId('workspace-mode-review')).not.toBeInTheDocument()

    rerender(
      <ReviewsList
        activeSessionId="s1" onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    expect(screen.getByTestId('workspace-mode-review')).toBeInTheDocument()
  })

  it('shows the canonicalized display_title, not the raw topic (Phase 8, item 5)', async () => {
    vi.mocked(curationApi.listReviews).mockResolvedValue([
      review({ session_id: 's1', topic: 'cars cooling system', display_title: 'Automotive Engine Cooling Systems' }),
    ])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )

    await waitFor(() => expect(screen.getByText('Automotive Engine Cooling Systems')).toBeInTheDocument())
    expect(screen.queryByText('cars cooling system')).not.toBeInTheDocument()
  })

  it('clicking a review card calls onSelectReview with that session id', async () => {
    const user = userEvent.setup()
    const onSelectReview = vi.fn()
    vi.mocked(curationApi.listReviews).mockResolvedValue([review({ session_id: 's1', topic: 'My topic' })])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={onSelectReview} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('My topic')).toBeInTheDocument())
    await user.click(screen.getByTestId('review-card-s1'))

    expect(onSelectReview).toHaveBeenCalledWith('s1')
  })

  it('clicking delete opens the in-app ConfirmDialog (not window.confirm), naming the review', async () => {
    const user = userEvent.setup()
    const onSelectReview = vi.fn()
    const onDeleteReview = vi.fn()
    const confirmSpy = vi.spyOn(window, 'confirm')
    vi.mocked(curationApi.listReviews).mockResolvedValue([review({ session_id: 's1', topic: 'My topic' })])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={onSelectReview} onStartReview={vi.fn()} onDeleteReview={onDeleteReview} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('My topic')).toBeInTheDocument())
    await user.click(screen.getByTestId('delete-review-s1'))

    expect(confirmSpy).not.toHaveBeenCalled()
    expect(screen.getByTestId('confirm-dialog')).toHaveTextContent('My topic')
    expect(onDeleteReview).not.toHaveBeenCalled() // not yet -- still needs confirmation
    // The click must not also bubble into the card's own onSelect.
    expect(onSelectReview).not.toHaveBeenCalled()
  })

  it('confirming the dialog calls onDeleteReview WITHOUT also selecting the review (Phase 8, item 1)', async () => {
    const user = userEvent.setup()
    const onSelectReview = vi.fn()
    const onDeleteReview = vi.fn()
    vi.mocked(curationApi.listReviews).mockResolvedValue([review({ session_id: 's1', topic: 'My topic' })])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={onSelectReview} onStartReview={vi.fn()} onDeleteReview={onDeleteReview} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('My topic')).toBeInTheDocument())
    await user.click(screen.getByTestId('delete-review-s1'))
    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    expect(onDeleteReview).toHaveBeenCalledWith('s1')
    expect(onSelectReview).not.toHaveBeenCalled()
    expect(screen.queryByTestId('confirm-dialog')).not.toBeInTheDocument()
  })

  it('UXH.2: startingReview shows an accessible "Starting new review…" status and hides the New Review trigger', async () => {
    vi.mocked(curationApi.listReviews).mockResolvedValue([])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
        startingReview
      />,
    )

    const status = screen.getByTestId('starting-review-status')
    expect(status).toHaveTextContent('Starting new review…')
    expect(status).toHaveAttribute('role', 'status')
    expect(status).toHaveAttribute('aria-live', 'polite')
    // No interactive affordance left that could submit a second start.
    expect(screen.queryByTestId('new-review-trigger')).not.toBeInTheDocument()
    expect(screen.queryByTestId('new-review-start')).not.toBeInTheDocument()
  })

  it('UXH.2: the New Review trigger is available again once startingReview clears', async () => {
    vi.mocked(curationApi.listReviews).mockResolvedValue([])

    const { rerender } = render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
        startingReview
      />,
    )
    expect(screen.getByTestId('starting-review-status')).toBeInTheDocument()

    rerender(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
        startingReview={false}
      />,
    )

    expect(screen.queryByTestId('starting-review-status')).not.toBeInTheDocument()
    expect(screen.getByTestId('new-review-trigger')).toBeInTheDocument()
  })

  it('starting a new review calls onStartReview with the submitted topic and target count', async () => {
    const user = userEvent.setup()
    const onStartReview = vi.fn()
    vi.mocked(curationApi.listReviews).mockResolvedValue([])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={onStartReview} onDeleteReview={vi.fn()} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )

    await user.click(screen.getByTestId('new-review-trigger'))
    await user.type(screen.getByTestId('new-review-topic'), 'parameter-efficient fine-tuning')
    await user.click(screen.getByTestId('new-review-start'))

    expect(onStartReview).toHaveBeenCalledWith('parameter-efficient fine-tuning', 10)
  })

  it('cancelling the dialog does not call onDeleteReview, and closes the dialog', async () => {
    const user = userEvent.setup()
    const onDeleteReview = vi.fn()
    vi.mocked(curationApi.listReviews).mockResolvedValue([review({ session_id: 's1', topic: 'My topic' })])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={onDeleteReview} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('My topic')).toBeInTheDocument())
    await user.click(screen.getByTestId('delete-review-s1'))
    await user.click(screen.getByTestId('confirm-dialog-cancel'))

    expect(onDeleteReview).not.toHaveBeenCalled()
    expect(screen.queryByTestId('confirm-dialog')).not.toBeInTheDocument()
  })
})
