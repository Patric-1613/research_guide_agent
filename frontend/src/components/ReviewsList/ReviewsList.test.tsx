import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReviewsList } from './ReviewsList'
import { curationApi } from '../../api/client'
import type { CurationReviewSummary } from '../../api/types'

vi.mock('../../api/client', () => ({
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

  it('clicking delete asks for confirmation, then calls onDeleteReview WITHOUT also selecting the review (Phase 8, item 1)', async () => {
    const user = userEvent.setup()
    const onSelectReview = vi.fn()
    const onDeleteReview = vi.fn()
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    vi.mocked(curationApi.listReviews).mockResolvedValue([review({ session_id: 's1', topic: 'My topic' })])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={onSelectReview} onStartReview={vi.fn()} onDeleteReview={onDeleteReview} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('My topic')).toBeInTheDocument())
    await user.click(screen.getByTestId('delete-review-s1'))

    expect(confirmSpy).toHaveBeenCalledWith(expect.stringContaining('My topic'))
    expect(onDeleteReview).toHaveBeenCalledWith('s1')
    // The click must not also bubble into the card's own onSelect.
    expect(onSelectReview).not.toHaveBeenCalled()
    confirmSpy.mockRestore()
  })

  it('declining the confirmation does not call onDeleteReview', async () => {
    const user = userEvent.setup()
    const onDeleteReview = vi.fn()
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    vi.mocked(curationApi.listReviews).mockResolvedValue([review({ session_id: 's1', topic: 'My topic' })])

    render(
      <ReviewsList
        activeSessionId={null} onSelectReview={vi.fn()} onStartReview={vi.fn()} onDeleteReview={onDeleteReview} refreshToken={0}
        workspaceMode="review" workspaceUnlocked={false} onWorkspaceModeChange={vi.fn()}
      />,
    )
    await waitFor(() => expect(screen.getByText('My topic')).toBeInTheDocument())
    await user.click(screen.getByTestId('delete-review-s1'))

    expect(onDeleteReview).not.toHaveBeenCalled()
    confirmSpy.mockRestore()
  })
})
