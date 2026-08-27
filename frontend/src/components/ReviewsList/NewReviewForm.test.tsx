import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { NewReviewForm } from './NewReviewForm'
import type { ResearchLaneOut } from '../../types'

function lane(overrides: Partial<ResearchLaneOut> = {}): ResearchLaneOut {
  return {
    lane_id: 'srv-x', label: 'Retrieval', question: 'How does retrieval help?',
    query: 'retrieval augmented generation', enabled: true, origin: 'suggested', generation_version: 1,
    ...overrides,
  }
}

const THREE: ResearchLaneOut[] = [
  lane({ lane_id: 'a', label: 'Retrieval', query: 'retrieval augmented generation' }),
  lane({ lane_id: 'b', label: 'Evaluation', query: 'evaluating factual grounding' }),
  lane({ lane_id: 'c', label: 'Failure modes', query: 'faithfulness failure modes' }),
]

describe('NewReviewForm -- single search (unchanged)', () => {
  it('renders no segmented control when research lanes are unavailable', () => {
    render(<NewReviewForm onSubmit={vi.fn()} onCancel={vi.fn()} />)
    expect(screen.queryByTestId('new-review-mode-single')).not.toBeInTheDocument()
    expect(screen.getByTestId('new-review-start')).toHaveTextContent('Start')
  })

  it('submits exactly (topic, targetCount) with no lanes argument', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<NewReviewForm onSubmit={onSubmit} onCancel={vi.fn()} />)

    await user.type(screen.getByTestId('new-review-topic'), 'transformers')
    await user.click(screen.getByTestId('new-review-start'))

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith('transformers', 10)
  })

  it('defaults to single mode even when lanes are available', () => {
    render(<NewReviewForm onSubmit={vi.fn()} onCancel={vi.fn()} researchLanesAvailable />)
    expect(screen.getByTestId('new-review-mode-single')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('new-review-mode-lanes')).toHaveAttribute('aria-pressed', 'false')
    expect(screen.queryByTestId('new-review-suggest-lanes')).not.toBeInTheDocument()
  })
})

describe('NewReviewForm -- research lanes', () => {
  async function openLanes(props: Partial<React.ComponentProps<typeof NewReviewForm>> = {}) {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    const onSuggestLanes = vi.fn()
    const onResetLaneSuggestions = vi.fn()
    const utils = render(
      <NewReviewForm
        onSubmit={onSubmit}
        onCancel={vi.fn()}
        researchLanesAvailable
        onSuggestLanes={onSuggestLanes}
        onResetLaneSuggestions={onResetLaneSuggestions}
        {...props}
      />,
    )
    await user.click(screen.getByTestId('new-review-mode-lanes'))
    return { user, onSubmit, onSuggestLanes, onResetLaneSuggestions, ...utils }
  }

  it('the Suggest lanes button requests suggestions for the current topic', async () => {
    const { user, onSuggestLanes } = await openLanes()
    await user.type(screen.getByTestId('new-review-topic'), '  reducing hallucination  ')
    await user.click(screen.getByTestId('new-review-suggest-lanes'))
    expect(onSuggestLanes).toHaveBeenCalledWith('reducing hallucination')
  })

  it('shows a polite "Designing research lanes…" status and disables Suggest while loading', async () => {
    await openLanes({ laneSuggestionLoading: true })
    const status = screen.getByTestId('lane-suggestion-status')
    expect(status).toHaveTextContent('Designing research lanes…')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(screen.getByTestId('new-review-suggest-lanes')).toBeDisabled()
  })

  it('a suggestion failure shows a safe error and preserves the typed topic', async () => {
    const user = userEvent.setup()
    const { rerender } = render(
      <NewReviewForm onSubmit={vi.fn()} onCancel={vi.fn()} researchLanesAvailable onSuggestLanes={vi.fn()} />,
    )
    await user.click(screen.getByTestId('new-review-mode-lanes'))
    await user.type(screen.getByTestId('new-review-topic'), 'my topic')

    rerender(
      <NewReviewForm
        onSubmit={vi.fn()} onCancel={vi.fn()} researchLanesAvailable onSuggestLanes={vi.fn()}
        laneSuggestionError="Research lane suggestions are unavailable right now."
      />,
    )

    expect(screen.getByTestId('lane-suggestion-error')).toHaveTextContent('unavailable right now')
    expect((screen.getByTestId('new-review-topic') as HTMLInputElement).value).toBe('my topic')
  })

  it('server suggestions seed editable divided rows (label / question / query / enabled)', async () => {
    await openLanes({ laneSuggestions: THREE })
    expect(screen.getByTestId('lane-row-0')).toBeInTheDocument()
    expect(screen.getByTestId('lane-row-2')).toBeInTheDocument()
    expect((screen.getByTestId('lane-label-0') as HTMLInputElement).value).toBe('Retrieval')
    expect((screen.getByTestId('lane-query-1') as HTMLInputElement).value).toBe('evaluating factual grounding')
    expect(screen.getByTestId('lane-enabled-0')).toBeChecked()
  })

  it('editing a label, toggling enabled, adding and removing lanes all work; max is four', async () => {
    const { user } = await openLanes({ laneSuggestions: THREE })

    await user.clear(screen.getByTestId('lane-label-0'))
    await user.type(screen.getByTestId('lane-label-0'), 'Architectures')
    expect((screen.getByTestId('lane-label-0') as HTMLInputElement).value).toBe('Architectures')

    await user.click(screen.getByTestId('lane-enabled-1'))
    expect(screen.getByTestId('lane-enabled-1')).not.toBeChecked()

    await user.click(screen.getByTestId('lane-add'))
    expect(screen.getByTestId('lane-row-3')).toBeInTheDocument()
    expect(screen.getByTestId('lane-add')).toBeDisabled() // now at 4

    await user.click(screen.getByTestId('lane-remove-3'))
    expect(screen.queryByTestId('lane-row-3')).not.toBeInTheDocument()
    expect(screen.getByTestId('lane-add')).toBeEnabled()
  })

  it('the last remaining lane row cannot be removed', async () => {
    const { user } = await openLanes({ laneSuggestions: [THREE[0]] })
    expect(screen.getByTestId('lane-remove-0')).toBeDisabled()
    void user
  })

  it('Start lane research is blocked until 1-4 rows, >=1 enabled, and every row has a label + query', async () => {
    const { user, onSubmit } = await openLanes()
    const start = screen.getByTestId('new-review-start')
    expect(start).toHaveTextContent('Start lane research')
    expect(start).toBeDisabled() // no topic, no lanes

    await user.type(screen.getByTestId('new-review-topic'), 'a topic')
    await user.click(screen.getByTestId('lane-add'))
    expect(start).toBeDisabled() // row has no label/query yet

    await user.type(screen.getByTestId('lane-label-0'), 'L')
    await user.type(screen.getByTestId('lane-query-0'), 'q')
    expect(start).toBeEnabled()

    await user.click(screen.getByTestId('lane-enabled-0')) // disable the only lane
    expect(start).toBeDisabled()

    await user.click(screen.getByTestId('lane-enabled-0')) // re-enable
    await user.click(start)
    expect(onSubmit).toHaveBeenCalledWith('a topic', 10, [{ label: 'L', question: '', query: 'q', enabled: true }])
  })

  it('changing the topic clears draft lanes and asks the hook to reset server suggestions', async () => {
    const { user, onResetLaneSuggestions } = await openLanes({ laneSuggestions: THREE })
    expect(screen.getByTestId('lane-row-0')).toBeInTheDocument()

    await user.type(screen.getByTestId('new-review-topic'), 'x')

    expect(screen.queryByTestId('lane-row-0')).not.toBeInTheDocument()
    expect(onResetLaneSuggestions).toHaveBeenCalled()
  })

  it('the start payload from edited server suggestions strips identity metadata (lane_id / origin / generation_version)', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(
      <NewReviewForm
        onSubmit={onSubmit} onCancel={vi.fn()} researchLanesAvailable
        onSuggestLanes={vi.fn()} onResetLaneSuggestions={vi.fn()} laneSuggestions={THREE}
      />,
    )
    await user.click(screen.getByTestId('new-review-mode-lanes'))
    // Typing the topic clears the seeded rows by design (a lane set is
    // specific to one topic), so the flow is: set topic, then build/edit
    // the lanes, then start.
    await user.type(screen.getByTestId('new-review-topic'), 'kept topic')
    await user.click(screen.getByTestId('lane-add'))
    await user.type(screen.getByTestId('lane-label-0'), 'Only lane')
    await user.type(screen.getByTestId('lane-query-0'), 'the query')
    await user.click(screen.getByTestId('new-review-start'))

    expect(onSubmit).toHaveBeenCalledWith('kept topic', 10, [
      { label: 'Only lane', question: '', query: 'the query', enabled: true },
    ])
    expect(JSON.stringify(onSubmit.mock.calls[0][2])).not.toMatch(/lane_id|origin|generation_version/)
  })
})
