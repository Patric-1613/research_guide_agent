import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PersistentInput } from './PersistentInput'

describe('PersistentInput', () => {
  it('the Yes button sends the literal string "yes" through onSendMessage — the same prop a typed "yes" + Send uses', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()

    render(
      <PersistentInput
        stage="synthesize"
        hasReport
        pendingWebOffer={{ question: 'what about scaling laws?' }}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={vi.fn()}
        onSendMessage={onSendMessage}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Yes' }))

    expect(onSendMessage).toHaveBeenCalledTimes(1)
    expect(onSendMessage).toHaveBeenCalledWith('yes')
  })

  it('the No button sends the literal string "no" through onSendMessage', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()

    render(
      <PersistentInput
        stage="synthesize"
        hasReport
        pendingWebOffer={{ question: 'what about scaling laws?' }}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={vi.fn()}
        onSendMessage={onSendMessage}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'No' }))

    expect(onSendMessage).toHaveBeenCalledTimes(1)
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('typing "yes" into the text field and pressing Send calls onSendMessage with the EXACT SAME argument the Yes button sends — proving the button is convenience, not a separate mechanism', async () => {
    const user = userEvent.setup()
    const onSendMessageViaButton = vi.fn()
    const onSendMessageViaTyping = vi.fn()

    const { unmount } = render(
      <PersistentInput
        stage="synthesize"
        hasReport
        pendingWebOffer={{ question: 'what about scaling laws?' }}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={vi.fn()}
        onSendMessage={onSendMessageViaButton}
      />,
    )
    await user.click(screen.getByRole('button', { name: 'Yes' }))
    unmount()

    render(
      <PersistentInput
        stage="synthesize"
        hasReport
        pendingWebOffer={{ question: 'what about scaling laws?' }}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={vi.fn()}
        onSendMessage={onSendMessageViaTyping}
      />,
    )
    await user.type(screen.getByPlaceholderText('Ask a question about the selected papers...'), 'yes')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    // Both paths must have called onSendMessage with byte-identical
    // arguments -- if the button ever grew its own separate call shape
    // (e.g. a dedicated acceptOffer() action), this assertion is what
    // would catch it.
    expect(onSendMessageViaButton.mock.calls).toEqual(onSendMessageViaTyping.mock.calls)
    expect(onSendMessageViaButton).toHaveBeenCalledWith('yes')
  })

  it('Yes/No buttons only render in chat mode with a pending offer -- not during curation or before a report exists', () => {
    render(
      <PersistentInput
        stage="curate"
        hasReport={false}
        pendingWebOffer={{ question: 'irrelevant here' }}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={vi.fn()}
        onSendMessage={vi.fn()}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Yes' })).not.toBeInTheDocument()
  })

  it('curate mode: Send calls onSubmitPicks (not onSendMessage), regardless of typed text', async () => {
    const user = userEvent.setup()
    const onSubmitPicks = vi.fn()
    const onSendMessage = vi.fn()

    render(
      <PersistentInput
        stage="curate"
        hasReport={false}
        pendingWebOffer={null}
        disabled={false}
        stagedPickCount={3}
        onSubmitPicks={onSubmitPicks}
        onGenerateReport={vi.fn()}
        onSendMessage={onSendMessage}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(onSubmitPicks).toHaveBeenCalledTimes(1)
    expect(onSendMessage).not.toHaveBeenCalled()
    expect(screen.getByText('3 added')).toBeInTheDocument()
  })

  it('generate-report mode: Send calls onGenerateReport', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()

    render(
      <PersistentInput
        stage="synthesize"
        hasReport={false}
        pendingWebOffer={null}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={onGenerateReport}
        onSendMessage={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(onGenerateReport).toHaveBeenCalledTimes(1)
  })

  it('chat mode: Send does nothing on empty/whitespace-only text', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()

    render(
      <PersistentInput
        stage="synthesize"
        hasReport
        pendingWebOffer={null}
        disabled={false}
        stagedPickCount={0}
        onSubmitPicks={vi.fn()}
        onGenerateReport={vi.fn()}
        onSendMessage={onSendMessage}
      />,
    )

    await user.type(screen.getByPlaceholderText('Ask a question about the selected papers...'), '   ')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(onSendMessage).not.toHaveBeenCalled()
  })
})
