import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatModePanel } from './ChatModePanel'
import type { CurationStateResponse } from '../../types'

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', display_title: 't', stage: 'synthesize', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    ...overrides,
  }
}

describe('ChatModePanel', () => {
  it('renders chat_history as message bubbles', () => {
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'what is this about?' },
        { role: 'assistant', content: 'It is about X.' },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    expect(screen.getByText('what is this about?')).toBeInTheDocument()
    expect(screen.getByText('It is about X.')).toBeInTheDocument()
  })

  it('typing a message and pressing Send calls onSendMessage and clears the input', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    const input = screen.getByTestId('persistent-input')
    await user.type(input, 'tell me more')
    await user.click(screen.getByTestId('persistent-input-send'))

    expect(onSendMessage).toHaveBeenCalledWith('tell me more')
    expect(input).toHaveValue('')
  })

  it('Send does nothing on empty/whitespace-only text', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    await user.type(screen.getByTestId('persistent-input'), '   ')
    await user.click(screen.getByTestId('persistent-input-send'))

    expect(onSendMessage).not.toHaveBeenCalled()
  })

  it('the web-search offer Yes/No buttons send the literal "yes"/"no" through onSendMessage', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    const state = baseState({ pending_web_offer: { question: 'what about scaling laws?' } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    await user.click(screen.getByTestId('web-offer-yes'))
    expect(onSendMessage).toHaveBeenCalledWith('yes')

    await user.click(screen.getByTestId('web-offer-no'))
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('the report-update offer renders its own prompt and reuses the same Yes/No path', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    const state = baseState({ pending_report_update: { new_article_count: 1 } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    expect(screen.getByText('Update the report to include the newly approved source(s)?')).toBeInTheDocument()
    await user.click(screen.getByTestId('web-offer-no'))
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('no offers pending: no Yes/No buttons render', () => {
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)
    expect(screen.queryByTestId('web-offer-yes')).not.toBeInTheDocument()
  })

  it('chat-ux-fixes bug 3: sending a message shows it immediately, without waiting for the round trip to resolve', async () => {
    const user = userEvent.setup()
    let resolveSend: () => void = () => {}
    const onSendMessage = vi.fn(() => new Promise<void>((resolve) => { resolveSend = resolve }))
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    await user.type(screen.getByTestId('persistent-input'), 'tell me more')
    await user.click(screen.getByTestId('persistent-input-send'))

    // Visible right away -- state.chat_history hasn't changed at all (this
    // test never updates it), so this can ONLY be the optimistic bubble.
    expect(screen.getByTestId('pending-message')).toHaveTextContent('tell me more')

    resolveSend()
    await waitFor(() => expect(screen.queryByTestId('pending-message')).not.toBeInTheDocument())
  })

  it('chat-ux-fixes bug 3: the optimistic bubble also appears for the Yes/No offer buttons, not just typed messages', async () => {
    const user = userEvent.setup()
    let resolveSend: () => void = () => {}
    const onSendMessage = vi.fn(() => new Promise<void>((resolve) => { resolveSend = resolve }))
    const state = baseState({ pending_web_offer: { question: 'what about scaling laws?' } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    await user.click(screen.getByTestId('web-offer-yes'))

    expect(screen.getByTestId('pending-message')).toHaveTextContent('yes')
    resolveSend()
    await waitFor(() => expect(screen.queryByTestId('pending-message')).not.toBeInTheDocument())
  })

  it('chat-ux-fixes bug 3: the optimistic bubble is cleared even if onSendMessage rejects, not left stuck', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn().mockRejectedValue(new Error('boom'))
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} />)

    await user.type(screen.getByTestId('persistent-input'), 'tell me more')
    await user.click(screen.getByTestId('persistent-input-send'))

    await waitFor(() => expect(screen.queryByTestId('pending-message')).not.toBeInTheDocument())
  })

  it('chat-ux-fixes bug 2: shows how many new web sources were found, when the last reply searched the web', () => {
    render(
      <ChatModePanel
        state={baseState()} disabled={false} onSendMessage={vi.fn()}
        lastSearchMeta={{ webSearchUsed: true, newWebArticlesFound: 2 }}
      />,
    )

    expect(screen.getByTestId('web-search-meta-note')).toHaveTextContent('Searched the web and found 2 new sources.')
  })

  it('chat-ux-fixes bug 2: singular phrasing for exactly one new source', () => {
    render(
      <ChatModePanel
        state={baseState()} disabled={false} onSendMessage={vi.fn()}
        lastSearchMeta={{ webSearchUsed: true, newWebArticlesFound: 1 }}
      />,
    )

    expect(screen.getByTestId('web-search-meta-note')).toHaveTextContent('Searched the web and found 1 new source.')
  })

  it('chat-ux-fixes bug 2: says so plainly when a real web search found nothing new -- distinguishable from the button doing nothing', () => {
    render(
      <ChatModePanel
        state={baseState()} disabled={false} onSendMessage={vi.fn()}
        lastSearchMeta={{ webSearchUsed: true, newWebArticlesFound: 0 }}
      />,
    )

    expect(screen.getByTestId('web-search-meta-note')).toHaveTextContent("Searched the web, but didn't find anything new.")
  })

  it('chat-ux-fixes bug 2: no note at all when the last reply never searched the web', () => {
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)
    expect(screen.queryByTestId('web-search-meta-note')).not.toBeInTheDocument()
  })
})
