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

  it('curation-chat-metadata Phase 1: shows the one-time hint next to the first web-backed assistant answer', () => {
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'what is this about?' },
        { role: 'assistant', content: 'It is about X [Paper 1].', used_web_search: false, added_to_report: false },
        { role: 'user', content: 'anything recent?' },
        { role: 'assistant', content: 'Per [Web 1], ...', used_web_search: true, added_to_report: false },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    expect(screen.getByTestId('web-metadata-hint')).toHaveTextContent(
      'This answer used web sources — report-inclusion controls will be added to the message menu in a future update.',
    )
  })

  it('curation-chat-metadata Phase 1: no hint at all when no assistant answer has used web search', () => {
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'what is this about?' },
        { role: 'assistant', content: 'It is about X [Paper 1].', used_web_search: false, added_to_report: false },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    expect(screen.queryByTestId('web-metadata-hint')).not.toBeInTheDocument()
  })

  it('curation-chat-metadata Phase 1: the hint does not repeat on a later web-backed answer', () => {
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'q1' },
        { role: 'assistant', content: 'a1 [Web 1]', used_web_search: true, added_to_report: false },
        { role: 'user', content: 'q2' },
        { role: 'assistant', content: 'a2 [Web 1]', used_web_search: true, added_to_report: false },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    expect(screen.getAllByTestId('web-metadata-hint')).toHaveLength(1)
  })
})

describe('ChatModePanel -- curation-chat-select Phase 2: message menu + select mode', () => {
  function exchangeState(): CurationStateResponse {
    return baseState({
      chat_history: [
        { role: 'user', content: 'what is RoCoFT?', exchange_id: 'ex-1' },
        { role: 'assistant', content: 'A PEFT method [Paper 1].', exchange_id: 'ex-1', used_web_search: false, added_to_report: false },
        { role: 'user', content: 'anything recent?', exchange_id: 'ex-2' },
        { role: 'assistant', content: 'Per [Web 1], ...', exchange_id: 'ex-2', used_web_search: true, added_to_report: false },
      ],
    })
  }

  it('renders a "..." message menu button for every chat message', () => {
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)
    expect(screen.getAllByTestId('message-menu-button')).toHaveLength(4)
  })

  it('the menu button has an accessible label', () => {
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)
    expect(screen.getAllByLabelText('Message actions')).toHaveLength(4)
  })

  it('opening a message menu shows Select, Delete, and Add to report -- all present, Delete/Add to report disabled', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])

    expect(screen.getByTestId('message-menu-select')).toBeEnabled()
    expect(screen.getByTestId('message-menu-delete')).toBeDisabled()
    expect(screen.getByTestId('message-menu-add-to-report')).toBeDisabled()
  })

  it('Edit appears only on the user-question side of an exchange, and is disabled', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)
    const menuButtons = screen.getAllByTestId('message-menu-button')

    await user.click(menuButtons[0]) // user message
    expect(screen.getByTestId('message-menu-edit')).toBeDisabled()

    await user.click(menuButtons[0]) // close it
    await user.click(menuButtons[1]) // assistant message
    expect(screen.queryByTestId('message-menu-edit')).not.toBeInTheDocument()
  })

  it('clicking Select in a message menu enters select mode with checkboxes, and no bulk bar yet', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    expect(screen.queryAllByTestId('exchange-select-checkbox')).toHaveLength(0)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))

    // Select mode is on for the whole panel -- every message row now shows
    // a checkbox, not just the one whose menu was used.
    expect(screen.getAllByTestId('exchange-select-checkbox')).toHaveLength(4)
    // The message that triggered Select is pre-selected -- the bulk bar
    // shows immediately with "1 selected".
    expect(screen.getByTestId('bulk-selected-count')).toHaveTextContent('1 selected')
  })

  it('selectable exchanges can be checked and unchecked, and the count updates', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    expect(screen.getByTestId('bulk-selected-count')).toHaveTextContent('1 selected')

    const checkboxes = screen.getAllByTestId('exchange-select-checkbox')
    // Checking the OTHER exchange's checkbox (index 2/3 belong to ex-2) --
    // both entries of ex-1 are already implicitly selected via exchange_id.
    await user.click(checkboxes[2])
    expect(screen.getByTestId('bulk-selected-count')).toHaveTextContent('2 selected')

    await user.click(checkboxes[2])
    expect(screen.getByTestId('bulk-selected-count')).toHaveTextContent('1 selected')
  })

  it('checking either message of the same exchange reflects as one shared selection', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))

    const checkboxes = screen.getAllByTestId('exchange-select-checkbox')
    // ex-1's user (index 0) and assistant (index 1) entries should both
    // already read as checked -- same exchange_id.
    expect(checkboxes[0]).toBeChecked()
    expect(checkboxes[1]).toBeChecked()
  })

  it('the bulk action bar shows Delete selected and Add selected to report, both disabled', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))

    expect(screen.getByTestId('bulk-action-bar')).toBeInTheDocument()
    expect(screen.getByTestId('bulk-delete')).toBeDisabled()
    expect(screen.getByTestId('bulk-add-to-report')).toBeDisabled()
  })

  it('Clear selection empties the selection, hides the bulk bar, and exits select mode', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    expect(screen.getByTestId('bulk-action-bar')).toBeInTheDocument()

    await user.click(screen.getByTestId('bulk-clear-selection'))

    expect(screen.queryByTestId('bulk-action-bar')).not.toBeInTheDocument()
    expect(screen.queryAllByTestId('exchange-select-checkbox')).toHaveLength(0)
  })

  it('old entries without exchange_id render a disabled, non-selectable checkbox once select mode is on', async () => {
    const user = userEvent.setup()
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'pre-Phase-1 question' },
        { role: 'assistant', content: 'pre-Phase-1 answer' },
        { role: 'user', content: 'a new question', exchange_id: 'ex-1' },
        { role: 'assistant', content: 'a new answer', exchange_id: 'ex-1', used_web_search: false, added_to_report: false },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[2]) // the new question's menu
    await user.click(screen.getByTestId('message-menu-select'))

    const checkboxes = screen.getAllByTestId('exchange-select-checkbox')
    expect(checkboxes[0]).toBeDisabled() // old entry, no exchange_id
    expect(checkboxes[1]).toBeDisabled()
    expect(checkboxes[2]).toBeEnabled()
    expect(checkboxes[3]).toBeEnabled()
  })

  it('Select is disabled in the menu for an old entry without exchange_id', async () => {
    const user = userEvent.setup()
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'pre-Phase-1 question' },
        { role: 'assistant', content: 'pre-Phase-1 answer' },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    expect(screen.getByTestId('message-menu-select')).toBeDisabled()
  })

  it('the web badge and hint still render normally with the menu/select UI layered on top', () => {
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} />)

    expect(screen.getByTestId('chat-web-badge')).toBeInTheDocument()
    expect(screen.getByTestId('web-metadata-hint')).toBeInTheDocument()
  })
})
