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

describe('ReviewsList -- Research Lanes (RL5b): drafts survive a failed start', () => {
  const THREE = [
    { lane_id: 'a', label: 'Retrieval', question: 'q1', query: 'retrieval augmented', enabled: true, origin: 'suggested', generation_version: 1 },
    { lane_id: 'b', label: 'Evaluation', question: 'q2', query: 'evaluating grounding', enabled: true, origin: 'suggested', generation_version: 1 },
    { lane_id: 'c', label: 'Failure modes', question: 'q3', query: 'faithfulness failures', enabled: true, origin: 'suggested', generation_version: 1 },
  ]

  function baseProps(overrides: Partial<React.ComponentProps<typeof ReviewsList>> = {}) {
    return {
      activeSessionId: null, onSelectReview: vi.fn(), onStartReview: vi.fn(), onDeleteReview: vi.fn(),
      refreshToken: 0, workspaceMode: 'review' as const, workspaceUnlocked: false, onWorkspaceModeChange: vi.fn(),
      ...overrides,
    }
  }

  it('a failed single-search start keeps the form open with topic and target intact', async () => {
    const user = userEvent.setup()
    vi.mocked(curationApi.listReviews).mockResolvedValue([])
    const onStartReview = vi.fn().mockResolvedValue(undefined) // failure -> no session id

    render(<ReviewsList {...baseProps({ onStartReview })} />)
    await user.click(screen.getByTestId('new-review-trigger'))
    await user.clear(screen.getByTestId('new-review-target-count'))
    await user.type(screen.getByTestId('new-review-target-count'), '15')
    await user.type(screen.getByTestId('new-review-topic'), 'my topic')
    await user.click(screen.getByTestId('new-review-start'))

    expect(onStartReview).toHaveBeenCalledWith('my topic', 15)
    // Form still mounted, values untouched.
    expect((screen.getByTestId('new-review-topic') as HTMLInputElement).value).toBe('my topic')
    expect((screen.getByTestId('new-review-target-count') as HTMLInputElement).value).toBe('15')
    expect(screen.queryByTestId('new-review-trigger')).not.toBeInTheDocument()
  })

  it('a failed lane-mode start restores every edited lane value and does not re-request suggestions', async () => {
    const user = userEvent.setup()
    vi.mocked(curationApi.listReviews).mockResolvedValue([])
    const onStartReview = vi.fn().mockResolvedValue(undefined)
    const onSuggestLanes = vi.fn()

    render(<ReviewsList {...baseProps({ onStartReview, researchLanesAvailable: true, laneSuggestions: THREE, onSuggestLanes })} />)
    await user.click(screen.getByTestId('new-review-trigger'))
    await user.click(screen.getByTestId('new-review-mode-lanes'))
    await user.type(screen.getByTestId('new-review-topic'), 'quantum error correction') // clears seeded rows
    await user.click(screen.getByTestId('lane-add'))
    await user.type(screen.getByTestId('lane-label-0'), 'My edited lane')
    await user.type(screen.getByTestId('lane-question-0'), 'does it converge?')
    await user.type(screen.getByTestId('lane-query-0'), 'surface code threshold')
    await user.click(screen.getByTestId('new-review-start'))

    expect(onStartReview).toHaveBeenCalledTimes(1)
    // Form still shows the lane editor with the exact edited values.
    expect((screen.getByTestId('lane-label-0') as HTMLInputElement).value).toBe('My edited lane')
    expect((screen.getByTestId('lane-question-0') as HTMLInputElement).value).toBe('does it converge?')
    expect((screen.getByTestId('lane-query-0') as HTMLInputElement).value).toBe('surface code threshold')
    expect(screen.getByTestId('lane-enabled-0')).toBeChecked()
    expect((screen.getByTestId('new-review-topic') as HTMLInputElement).value).toBe('quantum error correction')
    // No second suggestion request from the failed start.
    expect(onSuggestLanes).not.toHaveBeenCalled()
  })

  it('while a start is in flight the form is replaced by the status but its draft state survives', async () => {
    const user = userEvent.setup()
    vi.mocked(curationApi.listReviews).mockResolvedValue([])

    const { rerender } = render(<ReviewsList {...baseProps({ startingReview: false })} />)
    await user.click(screen.getByTestId('new-review-trigger'))
    await user.type(screen.getByTestId('new-review-topic'), 'a durable topic')

    rerender(<ReviewsList {...baseProps({ startingReview: true })} />)
    // Fields replaced by the one status; no second Start action.
    expect(screen.getByTestId('starting-review-status')).toHaveTextContent('Starting new review…')
    expect(screen.queryByTestId('new-review-topic')).not.toBeInTheDocument()
    expect(screen.queryByTestId('new-review-start')).not.toBeInTheDocument()

    // Start failed -> back to the form with the draft intact.
    rerender(<ReviewsList {...baseProps({ startingReview: false })} />)
    expect((screen.getByTestId('new-review-topic') as HTMLInputElement).value).toBe('a durable topic')
  })

  it('the form closes and resets only after a genuinely successful start (a truthy session id)', async () => {
    const user = userEvent.setup()
    vi.mocked(curationApi.listReviews).mockResolvedValue([])
    const onStartReview = vi.fn().mockResolvedValue('sess-42')
    const onResetLaneSuggestions = vi.fn()

    render(<ReviewsList {...baseProps({ onStartReview, onResetLaneSuggestions })} />)
    await user.click(screen.getByTestId('new-review-trigger'))
    await user.type(screen.getByTestId('new-review-topic'), 'topic')
    await user.click(screen.getByTestId('new-review-start'))

    await waitFor(() => expect(screen.getByTestId('new-review-trigger')).toBeInTheDocument())
    expect(onResetLaneSuggestions).toHaveBeenCalled()

    // Re-opening the form starts from an empty draft.
    await user.click(screen.getByTestId('new-review-trigger'))
    expect((screen.getByTestId('new-review-topic') as HTMLInputElement).value).toBe('')
  })
})
