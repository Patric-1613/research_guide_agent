import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ChatModePanel } from './ChatModePanel'
import type { CurationStateResponse } from '../../types'

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', display_title: 't', stage: 'synthesize', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    report_versions: [], active_report_version_id: null,
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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    expect(screen.getByText('what is this about?')).toBeInTheDocument()
    expect(screen.getByText('It is about X.')).toBeInTheDocument()
  })

  it('typing a message and pressing Send calls onSendMessage and clears the input', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    const input = screen.getByTestId('persistent-input')
    await user.type(input, 'tell me more')
    await user.click(screen.getByTestId('persistent-input-send'))

    expect(onSendMessage).toHaveBeenCalledWith('tell me more')
    expect(input).toHaveValue('')
  })

  it('Send does nothing on empty/whitespace-only text', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.type(screen.getByTestId('persistent-input'), '   ')
    await user.click(screen.getByTestId('persistent-input-send'))

    expect(onSendMessage).not.toHaveBeenCalled()
  })

  it('the web-search offer Yes/No buttons send the literal "yes"/"no" through onSendMessage', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    const state = baseState({ pending_web_offer: { question: 'what about scaling laws?' } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getByTestId('web-offer-yes'))
    expect(onSendMessage).toHaveBeenCalledWith('yes')

    await user.click(screen.getByTestId('web-offer-no'))
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('the report-update offer renders its own prompt and reuses the same Yes/No path', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    const state = baseState({ pending_report_update: { new_article_count: 1 } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    expect(screen.getByText('Update the report to include the newly approved source(s)?')).toBeInTheDocument()
    await user.click(screen.getByTestId('web-offer-no'))
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('no offers pending: no Yes/No buttons render', () => {
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)
    expect(screen.queryByTestId('web-offer-yes')).not.toBeInTheDocument()
  })

  it('chat-ux-fixes bug 3: sending a message shows it immediately, without waiting for the round trip to resolve', async () => {
    const user = userEvent.setup()
    let resolveSend: () => void = () => {}
    const onSendMessage = vi.fn(() => new Promise<void>((resolve) => { resolveSend = resolve }))
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getByTestId('web-offer-yes'))

    expect(screen.getByTestId('pending-message')).toHaveTextContent('yes')
    resolveSend()
    await waitFor(() => expect(screen.queryByTestId('pending-message')).not.toBeInTheDocument())
  })

  it('chat-ux-fixes bug 3: the optimistic bubble is cleared even if onSendMessage rejects, not left stuck', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn().mockRejectedValue(new Error('boom'))
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.type(screen.getByTestId('persistent-input'), 'tell me more')
    await user.click(screen.getByTestId('persistent-input-send'))

    await waitFor(() => expect(screen.queryByTestId('pending-message')).not.toBeInTheDocument())
  })

  it('chat-ux-fixes bug 2: shows how many new web sources were found, when the last reply searched the web', () => {
    render(
      <ChatModePanel
        state={baseState()} disabled={false} onSendMessage={vi.fn()}
        lastSearchMeta={{ webSearchUsed: true, newWebArticlesFound: 2 }} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    expect(screen.getByTestId('web-search-meta-note')).toHaveTextContent('Searched the web and found 2 new sources.')
  })

  it('chat-ux-fixes bug 2: singular phrasing for exactly one new source', () => {
    render(
      <ChatModePanel
        state={baseState()} disabled={false} onSendMessage={vi.fn()}
        lastSearchMeta={{ webSearchUsed: true, newWebArticlesFound: 1 }} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    expect(screen.getByTestId('web-search-meta-note')).toHaveTextContent('Searched the web and found 1 new source.')
  })

  it('chat-ux-fixes bug 2: says so plainly when a real web search found nothing new -- distinguishable from the button doing nothing', () => {
    render(
      <ChatModePanel
        state={baseState()} disabled={false} onSendMessage={vi.fn()}
        lastSearchMeta={{ webSearchUsed: true, newWebArticlesFound: 0 }} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    expect(screen.getByTestId('web-search-meta-note')).toHaveTextContent("Searched the web, but didn't find anything new.")
  })

  it('chat-ux-fixes bug 2: no note at all when the last reply never searched the web', () => {
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)
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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)
    expect(screen.getAllByTestId('message-menu-button')).toHaveLength(4)
  })

  it('the menu button has an accessible label', () => {
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)
    expect(screen.getAllByLabelText('Message actions')).toHaveLength(4)
  })

  it('curation-chat-delete Phase 3: opening a message menu shows Select and Delete active; Add to report disabled here because this message (a paper-only exchange) is ineligible', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0]) // ex-1's user message

    expect(screen.getByTestId('message-menu-select')).toBeEnabled()
    expect(screen.getByTestId('message-menu-delete')).toBeEnabled()
    expect(screen.getByTestId('message-menu-add-to-report')).toBeDisabled()
  })

  it('curation-chat-edit Phase 5: Edit appears only on the user-question side of an exchange, and is enabled for a real exchange_id', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)
    const menuButtons = screen.getAllByTestId('message-menu-button')

    await user.click(menuButtons[0]) // user message, exchange_id: 'ex-1'
    expect(screen.getByTestId('message-menu-edit')).toBeEnabled()

    await user.click(menuButtons[0]) // close it
    await user.click(menuButtons[1]) // assistant message
    expect(screen.queryByTestId('message-menu-edit')).not.toBeInTheDocument()
  })

  it('clicking Select in a message menu enters select mode with checkboxes, and no bulk bar yet', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))

    const checkboxes = screen.getAllByTestId('exchange-select-checkbox')
    // ex-1's user (index 0) and assistant (index 1) entries should both
    // already read as checked -- same exchange_id.
    expect(checkboxes[0]).toBeChecked()
    expect(checkboxes[1]).toBeChecked()
  })

  it('curation-chat-delete Phase 3: the bulk action bar shows Delete selected active; Add selected to report disabled because the selection (ex-1, paper-only) has no eligible exchange', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))

    expect(screen.getByTestId('bulk-action-bar')).toBeInTheDocument()
    expect(screen.getByTestId('bulk-delete')).toBeEnabled()
    expect(screen.getByTestId('bulk-add-to-report')).toBeDisabled()
  })

  it('Clear selection empties the selection, hides the bulk bar, and exits select mode', async () => {
    const user = userEvent.setup()
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    expect(screen.getByTestId('message-menu-select')).toBeDisabled()
  })

  it('the web badge and hint still render normally with the menu/select UI layered on top', () => {
    render(<ChatModePanel state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    expect(screen.getByTestId('chat-web-badge')).toBeInTheDocument()
    expect(screen.getByTestId('web-metadata-hint')).toBeInTheDocument()
  })

  it('Delete is disabled in the menu, and no checkbox is shown as selectable, for an old entry without exchange_id', async () => {
    const user = userEvent.setup()
    const state = baseState({
      chat_history: [
        { role: 'user', content: 'pre-Phase-1 question' },
        { role: 'assistant', content: 'pre-Phase-1 answer' },
      ],
    })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null} onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()} />)

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    expect(screen.getByTestId('message-menu-delete')).toBeDisabled()
  })
})

describe('ChatModePanel -- curation-chat-menu-dismiss: closing the "..." menu on outside interaction', () => {
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

  function renderPanel() {
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
  }

  it('opens on "..." click', async () => {
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getAllByTestId('message-menu-button')[0])

    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)
  })

  it('clicking the same "..." button again closes it', async () => {
    const user = userEvent.setup()
    renderPanel()
    const menuButtons = screen.getAllByTestId('message-menu-button')

    await user.click(menuButtons[0])
    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)

    await user.click(menuButtons[0])
    expect(screen.queryAllByTestId('message-menu')).toHaveLength(0)
  })

  it('clicking outside the open menu closes it', async () => {
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)

    // Blank space -- not any message row, not any menu.
    await user.click(document.body)

    await waitFor(() => expect(screen.queryAllByTestId('message-menu')).toHaveLength(0))
  })

  it('pressing Escape closes the open menu', async () => {
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)

    await user.keyboard('{Escape}')

    await waitFor(() => expect(screen.queryAllByTestId('message-menu')).toHaveLength(0))
  })

  it('clicking another row\'s "..." button closes the first menu and opens the second, never both at once', async () => {
    const user = userEvent.setup()
    renderPanel()
    const menuButtons = screen.getAllByTestId('message-menu-button')

    await user.click(menuButtons[0]) // ex-1's user message menu
    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)
    expect(screen.getByTestId('message-menu-select')).toBeInTheDocument()

    await user.click(menuButtons[2]) // ex-2's user message menu -- a DIFFERENT row

    // Still exactly one menu open at a time -- never two overlapping.
    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)
  })

  it('clicking a menu action closes the menu after dispatching the action', async () => {
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))

    expect(screen.queryAllByTestId('message-menu')).toHaveLength(0)
    // The Select action's own effect (entering select mode) still ran.
    expect(screen.getAllByTestId('exchange-select-checkbox').length).toBeGreaterThan(0)
  })

  it('checking a select-mode checkbox on another row does not leave a previous menu open', async () => {
    const user = userEvent.setup()
    renderPanel()
    const menuButtons = screen.getAllByTestId('message-menu-button')

    await user.click(menuButtons[0])
    await user.click(screen.getByTestId('message-menu-select')) // enters select mode, menu closes itself

    // Re-open a menu, then interact with a checkbox on a DIFFERENT row instead
    // of that menu's own actions -- the still-open menu must not linger.
    await user.click(menuButtons[2])
    expect(screen.getAllByTestId('message-menu')).toHaveLength(1)

    const checkboxes = screen.getAllByTestId('exchange-select-checkbox')
    await user.click(checkboxes[0])

    await waitFor(() => expect(screen.queryAllByTestId('message-menu')).toHaveLength(0))
  })
})

describe('ChatModePanel -- curation-chat-delete Phase 3: deleting exchanges', () => {
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

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('per-message Delete opens an in-app confirm dialog (not window.confirm); confirming calls onDeleteExchanges with just that exchange', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm')
    const onDeleteExchanges = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0]) // ex-1's user message
    await user.click(screen.getByTestId('message-menu-delete'))

    expect(confirmSpy).not.toHaveBeenCalled()
    expect(screen.getByTestId('confirm-dialog')).toBeInTheDocument()
    expect(onDeleteExchanges).not.toHaveBeenCalled() // not yet -- still needs confirmation

    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    expect(onDeleteExchanges).toHaveBeenCalledWith(['ex-1'])
    expect(screen.queryByTestId('confirm-dialog')).not.toBeInTheDocument()
  })

  it('cancelling the per-message delete dialog does nothing -- no onDeleteExchanges call, dialog closes', async () => {
    const user = userEvent.setup()
    const onDeleteExchanges = vi.fn()
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-delete'))
    await user.click(screen.getByTestId('confirm-dialog-cancel'))

    expect(onDeleteExchanges).not.toHaveBeenCalled()
    expect(screen.queryByTestId('confirm-dialog')).not.toBeInTheDocument()
  })

  it('bulk Delete selected opens an in-app confirm dialog naming the selected count; confirming calls onDeleteExchanges with every selected exchange id', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm')
    const onDeleteExchanges = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select')) // selects ex-1
    const checkboxes = screen.getAllByTestId('exchange-select-checkbox')
    await user.click(checkboxes[2]) // also select ex-2

    await user.click(screen.getByTestId('bulk-delete'))

    expect(confirmSpy).not.toHaveBeenCalled()
    expect(screen.getByTestId('confirm-dialog')).toHaveTextContent('Delete 2 selected exchanges?')

    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    expect(onDeleteExchanges).toHaveBeenCalledWith(expect.arrayContaining(['ex-1', 'ex-2']))
    expect((onDeleteExchanges.mock.calls[0][0] as string[]).length).toBe(2)
  })

  it('cancelling the bulk delete dialog does not call onDeleteExchanges and leaves the selection intact', async () => {
    const user = userEvent.setup()
    const onDeleteExchanges = vi.fn()
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    await user.click(screen.getByTestId('bulk-delete'))
    await user.click(screen.getByTestId('confirm-dialog-cancel'))

    expect(onDeleteExchanges).not.toHaveBeenCalled()
    expect(screen.getByTestId('bulk-action-bar')).toBeInTheDocument() // still selected
  })

  it('after a successful bulk delete, selection clears and select mode exits', async () => {
    const user = userEvent.setup()
    const onDeleteExchanges = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    await user.click(screen.getByTestId('bulk-delete'))
    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    await waitFor(() => expect(screen.queryByTestId('bulk-action-bar')).not.toBeInTheDocument())
    expect(screen.queryAllByTestId('exchange-select-checkbox')).toHaveLength(0)
  })

  it('after a successful per-message delete, that exchange is de-selected if it had been checked', async () => {
    const user = userEvent.setup()
    const onDeleteExchanges = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    // Select ex-1 AND ex-2 first.
    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    await user.click(screen.getAllByTestId('exchange-select-checkbox')[2])
    expect(screen.getByTestId('bulk-selected-count')).toHaveTextContent('2 selected')

    // Delete just ex-2 via its own message menu.
    await user.click(screen.getAllByTestId('message-menu-button')[2])
    await user.click(screen.getByTestId('message-menu-delete'))
    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    await waitFor(() => expect(screen.getByTestId('bulk-selected-count')).toHaveTextContent('1 selected'))
  })

  it('the "..." menu button remains visible and accessible via its aria-label', () => {
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    const menuButton = screen.getAllByLabelText('Message actions')[0]
    expect(menuButton.getAttribute('class')).toContain('border')
    expect(menuButton.getAttribute('class')).toContain('font-bold')
  })

  it('renders the report-possibly-stale warning when the prop is true', () => {
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale
        onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    expect(screen.getByTestId('report-possibly-stale-warning')).toBeInTheDocument()
  })

  it('no report-possibly-stale warning when the prop is false', () => {
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    expect(screen.queryByTestId('report-possibly-stale-warning')).not.toBeInTheDocument()
  })

  it('the web badge still renders correctly on remaining messages alongside active delete controls', () => {
    render(
      <ChatModePanel
        state={exchangeState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    expect(screen.getByTestId('chat-web-badge')).toBeInTheDocument()
  })
})

describe('ChatModePanel -- curation-chat-add-to-report Phase 4: adding exchanges to the report', () => {
  function mixedEligibilityState(): CurationStateResponse {
    return baseState({
      chat_history: [
        // ex-1: eligible web-backed exchange.
        { role: 'user', content: 'anything recent?', exchange_id: 'ex-1' },
        {
          role: 'assistant', content: 'Per [Web 1], ...', exchange_id: 'ex-1',
          used_web_search: true, cited_web_articles: [{ url: 'https://a.com', title: 'A' }], added_to_report: false,
        },
        // ex-2: paper-only -- ineligible.
        { role: 'user', content: 'what is RoCoFT?', exchange_id: 'ex-2' },
        { role: 'assistant', content: 'A PEFT method [Paper 1].', exchange_id: 'ex-2', used_web_search: false, added_to_report: false },
        // ex-3: already added -- ineligible.
        { role: 'user', content: 'anything else?', exchange_id: 'ex-3' },
        {
          role: 'assistant', content: 'Per [Web 2], ...', exchange_id: 'ex-3',
          used_web_search: true, cited_web_articles: [{ url: 'https://b.com', title: 'B' }], added_to_report: true,
        },
        // Old, pre-Phase-1 entries -- no exchange_id, ineligible.
        { role: 'user', content: 'an old question' },
        { role: 'assistant', content: 'an old answer' },
      ],
    })
  }

  it('Add to report is enabled in the menu for an eligible web-backed assistant message', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[1]) // ex-1's assistant message
    expect(screen.getByTestId('message-menu-add-to-report')).toBeEnabled()
  })

  it('Add to report is disabled for a user message, a paper-only answer, an already-added answer, and an old entry', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    const menuButtons = screen.getAllByTestId('message-menu-button')

    for (const index of [0, 2, 3, 5, 6, 7]) {
      await user.click(menuButtons[index])
      expect(screen.getByTestId('message-menu-add-to-report')).toBeDisabled()
      await user.click(menuButtons[index]) // close
    }
  })

  it('clicking Add to report calls onAddExchangesToReport with just that exchange, no confirmation prompt', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm')
    const onAddExchangesToReport = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false}
        onAddExchangesToReport={onAddExchangesToReport} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[1])
    await user.click(screen.getByTestId('message-menu-add-to-report'))

    expect(onAddExchangesToReport).toHaveBeenCalledWith(['ex-1'])
    expect(confirmSpy).not.toHaveBeenCalled()
    confirmSpy.mockRestore()
  })

  it('bulk Add selected to report is disabled when the selection has no eligible exchange', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[2]) // ex-2, paper-only
    await user.click(screen.getByTestId('message-menu-select'))

    expect(screen.getByTestId('bulk-add-to-report')).toBeDisabled()
  })

  it('bulk Add selected to report is enabled once the selection includes at least one eligible exchange', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    // Select ex-2 (ineligible) first, then also check ex-1 (eligible).
    await user.click(screen.getAllByTestId('message-menu-button')[2])
    await user.click(screen.getByTestId('message-menu-select'))
    expect(screen.getByTestId('bulk-add-to-report')).toBeDisabled()

    await user.click(screen.getAllByTestId('exchange-select-checkbox')[0]) // ex-1's user entry
    expect(screen.getByTestId('bulk-add-to-report')).toBeEnabled()
  })

  it('clicking bulk Add selected to report calls onAddExchangesToReport with every selected id, no confirmation prompt', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm')
    const onAddExchangesToReport = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false}
        onAddExchangesToReport={onAddExchangesToReport} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0]) // ex-1's user message
    await user.click(screen.getByTestId('message-menu-select')) // selects ex-1
    await user.click(screen.getByTestId('bulk-add-to-report'))

    expect(onAddExchangesToReport).toHaveBeenCalledWith(['ex-1'])
    expect(confirmSpy).not.toHaveBeenCalled()
    confirmSpy.mockRestore()
  })

  it('after a successful bulk add-to-report, selection clears and select mode exits', async () => {
    const user = userEvent.setup()
    const onAddExchangesToReport = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false}
        onAddExchangesToReport={onAddExchangesToReport} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    await user.click(screen.getByTestId('bulk-add-to-report'))

    await waitFor(() => expect(screen.queryByTestId('bulk-action-bar')).not.toBeInTheDocument())
    expect(screen.queryAllByTestId('exchange-select-checkbox')).toHaveLength(0)
  })

  it('shows a success note reflecting the latest add-to-report result', () => {
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={{ addedCount: 1, skippedCount: 0, sourceCount: 1 }} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    expect(screen.getByTestId('add-to-report-success-note')).toHaveTextContent('Added 1 exchange (1 source) to the report.')
  })

  it('no success note when lastAddToReportResult is null', () => {
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    expect(screen.queryByTestId('add-to-report-success-note')).not.toBeInTheDocument()
  })

  it('the web badge stays exactly as the state prop says while a call is in flight (never greyed optimistically) -- state only ever comes from the caller', async () => {
    const user = userEvent.setup()
    // ChatModePanel holds no local mirror of chat_history/added_to_report
    // -- badges are rendered purely from the state prop. A never-resolved
    // promise here proves nothing flips the badge synchronously/
    // optimistically on click; the real hook (useCurationSession's
    // runAction) only ever updates state via loadState() AFTER a
    // confirmed backend success, which is exactly what this simulates by
    // simply never resolving.
    const onAddExchangesToReport = vi.fn(() => new Promise<void>(() => {}))
    const state = mixedEligibilityState()
    render(
      <ChatModePanel
        state={state} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false}
        onAddExchangesToReport={onAddExchangesToReport} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[1]) // ex-1's assistant message
    await user.click(screen.getByTestId('message-menu-add-to-report'))

    expect(onAddExchangesToReport).toHaveBeenCalledWith(['ex-1'])
    // ex-1's own badge is still blue (added_to_report is still false in
    // the state prop this component was given -- nothing local changed it).
    const badge = screen.getAllByTestId('chat-web-badge')[0]
    expect(badge.getAttribute('class')).toContain('text-blue-400')
  })

  it('Delete still works alongside add-to-report controls', async () => {
    const user = userEvent.setup()
    const onDeleteExchanges = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={mixedEligibilityState()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false}
        onAddExchangesToReport={vi.fn()} lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[1])
    await user.click(screen.getByTestId('message-menu-delete'))
    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    expect(onDeleteExchanges).toHaveBeenCalledWith(['ex-1'])
  })
})

describe('ChatModePanel -- curation-chat-edit Phase 5: editing a user question', () => {
  function stateWithEditableAndOldEntries(): CurationStateResponse {
    return baseState({
      chat_history: [
        { role: 'user', content: 'the original question', exchange_id: 'ex-1' },
        { role: 'assistant', content: 'the original answer [Paper 1].', exchange_id: 'ex-1', used_web_search: false, added_to_report: false },
        // Old, pre-Phase-1 entries -- no exchange_id, ineligible for edit.
        { role: 'user', content: 'an old question' },
        { role: 'assistant', content: 'an old answer' },
      ],
    })
  }

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('Edit is enabled only for a user message with a real exchange_id, not for an old entry', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    const menuButtons = screen.getAllByTestId('message-menu-button')

    await user.click(menuButtons[0]) // ex-1's user message
    expect(screen.getByTestId('message-menu-edit')).toBeEnabled()
    await user.click(menuButtons[0]) // close

    await user.click(menuButtons[2]) // old user message, no exchange_id
    expect(screen.getByTestId('message-menu-edit')).toBeDisabled()
  })

  it('Edit is not rendered at all for an assistant message', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[1]) // ex-1's assistant message
    expect(screen.queryByTestId('message-menu-edit')).not.toBeInTheDocument()
  })

  it('Edit opens an in-app dialog, not window.prompt', async () => {
    const user = userEvent.setup()
    const promptSpy = vi.spyOn(window, 'prompt')
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-edit'))

    expect(promptSpy).not.toHaveBeenCalled()
    expect(screen.getByTestId('edit-dialog')).toBeInTheDocument()
  })

  it('cancelling the edit dialog does not call onEditExchange, and closes the dialog', async () => {
    const user = userEvent.setup()
    const onEditExchange = vi.fn()
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={onEditExchange} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-edit'))
    await user.click(screen.getByTestId('edit-dialog-cancel'))

    expect(onEditExchange).not.toHaveBeenCalled()
    expect(screen.queryByTestId('edit-dialog')).not.toBeInTheDocument()
  })

  it('submitting a blank/whitespace-only edit does not call onEditExchange -- Save stays disabled', async () => {
    const user = userEvent.setup()
    const onEditExchange = vi.fn()
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={onEditExchange} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-edit'))
    await user.clear(screen.getByTestId('edit-dialog-textarea'))
    await user.type(screen.getByTestId('edit-dialog-textarea'), '   ')

    expect(screen.getByTestId('edit-dialog-save')).toBeDisabled()
    await user.click(screen.getByTestId('edit-dialog-save'))

    expect(onEditExchange).not.toHaveBeenCalled()
  })

  it('the edit dialog textarea is pre-filled with the existing question text', async () => {
    const user = userEvent.setup()
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-edit'))

    expect(screen.getByTestId('edit-dialog-textarea')).toHaveValue('the original question')
  })

  it('submitting a real edited question calls onEditExchange with the exchange id and trimmed text', async () => {
    const user = userEvent.setup()
    const onEditExchange = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={onEditExchange} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-edit'))
    await user.clear(screen.getByTestId('edit-dialog-textarea'))
    await user.type(screen.getByTestId('edit-dialog-textarea'), '  an edited question  ')
    await user.click(screen.getByTestId('edit-dialog-save'))

    expect(onEditExchange).toHaveBeenCalledWith('ex-1', 'an edited question')
  })

  it('after a successful edit, any active selection clears and select mode exits', async () => {
    const user = userEvent.setup()
    const onEditExchange = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={onEditExchange} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    // Select ex-1 first via its own menu.
    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-select'))
    expect(screen.getByTestId('bulk-action-bar')).toBeInTheDocument()

    // Then edit it via the same message's menu.
    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-edit'))
    await user.clear(screen.getByTestId('edit-dialog-textarea'))
    await user.type(screen.getByTestId('edit-dialog-textarea'), 'an edited question')
    await user.click(screen.getByTestId('edit-dialog-save'))

    await waitFor(() => expect(screen.queryByTestId('bulk-action-bar')).not.toBeInTheDocument())
    expect(screen.queryAllByTestId('exchange-select-checkbox')).toHaveLength(0)
  })

  it('shows the stale-report warning when the edit response signals report_possibly_stale (same reused prop as Phase 3 delete)', () => {
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )
    expect(screen.getByTestId('report-possibly-stale-warning')).toBeInTheDocument()
  })

  it('Delete and Add to report still work on the same state alongside Edit', async () => {
    const user = userEvent.setup()
    const onDeleteExchanges = vi.fn().mockResolvedValue(undefined)
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={onDeleteExchanges} reportPossiblyStale={false} onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={vi.fn()}
      />,
    )

    await user.click(screen.getAllByTestId('message-menu-button')[0])
    await user.click(screen.getByTestId('message-menu-delete'))
    await user.click(screen.getByTestId('confirm-dialog-confirm'))

    expect(onDeleteExchanges).toHaveBeenCalledWith(['ex-1'])
  })

  it('dismissing the stale-report warning calls onDismissReportStaleWarning', async () => {
    const user = userEvent.setup()
    const onDismissReportStaleWarning = vi.fn()
    render(
      <ChatModePanel
        state={stateWithEditableAndOldEntries()} disabled={false} onSendMessage={vi.fn()} lastSearchMeta={null}
        onDeleteExchanges={vi.fn()} reportPossiblyStale onAddExchangesToReport={vi.fn()}
        lastAddToReportResult={null} onEditExchange={vi.fn()} onDismissReportStaleWarning={onDismissReportStaleWarning}
      />,
    )

    await user.click(screen.getByTestId('dismiss-stale-warning'))

    expect(onDismissReportStaleWarning).toHaveBeenCalled()
  })
})
