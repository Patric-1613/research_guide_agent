import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ChatModePanel } from './ChatModePanel'
import type { CurationStateResponse } from '../../api/types'

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', stage: 'synthesize', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null,
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
    render(<ChatModePanel state={state} disabled={false} onSendMessage={vi.fn()} />)

    expect(screen.getByText('what is this about?')).toBeInTheDocument()
    expect(screen.getByText('It is about X.')).toBeInTheDocument()
  })

  it('typing a message and pressing Send calls onSendMessage and clears the input', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} />)

    const input = screen.getByTestId('persistent-input')
    await user.type(input, 'tell me more')
    await user.click(screen.getByTestId('persistent-input-send'))

    expect(onSendMessage).toHaveBeenCalledWith('tell me more')
    expect(input).toHaveValue('')
  })

  it('Send does nothing on empty/whitespace-only text', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={onSendMessage} />)

    await user.type(screen.getByTestId('persistent-input'), '   ')
    await user.click(screen.getByTestId('persistent-input-send'))

    expect(onSendMessage).not.toHaveBeenCalled()
  })

  it('the web-search offer Yes/No buttons send the literal "yes"/"no" through onSendMessage', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    const state = baseState({ pending_web_offer: { question: 'what about scaling laws?' } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} />)

    await user.click(screen.getByTestId('web-offer-yes'))
    expect(onSendMessage).toHaveBeenCalledWith('yes')

    await user.click(screen.getByTestId('web-offer-no'))
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('the report-update offer renders its own prompt and reuses the same Yes/No path', async () => {
    const user = userEvent.setup()
    const onSendMessage = vi.fn()
    const state = baseState({ pending_report_update: { new_article_count: 1 } })
    render(<ChatModePanel state={state} disabled={false} onSendMessage={onSendMessage} />)

    expect(screen.getByText('Update the report to include the newly approved source(s)?')).toBeInTheDocument()
    await user.click(screen.getByTestId('web-offer-no'))
    expect(onSendMessage).toHaveBeenCalledWith('no')
  })

  it('no offers pending: no Yes/No buttons render', () => {
    render(<ChatModePanel state={baseState()} disabled={false} onSendMessage={vi.fn()} />)
    expect(screen.queryByTestId('web-offer-yes')).not.toBeInTheDocument()
  })
})
