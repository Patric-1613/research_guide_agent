import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ChatMessage } from './ChatMessage'
import type { ChatTurn } from '../../types'

describe('ChatMessage', () => {
  it('renders a blue web badge for a web-backed answer not yet added to the report', () => {
    const turn: ChatTurn = {
      role: 'assistant', content: 'Per [Web 1], ...',
      used_web_search: true, added_to_report: false,
    }
    render(<ChatMessage turn={turn} />)

    const badge = screen.getByTestId('chat-web-badge')
    expect(badge).toBeInTheDocument()
    expect(badge.className).toContain('text-blue-400')
  })

  it('renders a grey web badge once added_to_report is true', () => {
    const turn: ChatTurn = {
      role: 'assistant', content: 'Per [Web 1], ...',
      used_web_search: true, added_to_report: true,
    }
    render(<ChatMessage turn={turn} />)

    const badge = screen.getByTestId('chat-web-badge')
    expect(badge).toBeInTheDocument()
    expect(badge.className).not.toContain('text-blue-400')
    expect(badge.className).toContain('text-text-muted')
  })

  it('renders no badge for a paper-only assistant answer', () => {
    const turn: ChatTurn = {
      role: 'assistant', content: 'Per [Paper 1], ...',
      used_web_search: false, added_to_report: false,
    }
    render(<ChatMessage turn={turn} />)

    expect(screen.queryByTestId('chat-web-badge')).not.toBeInTheDocument()
  })

  it('renders no badge for a user message, even if used_web_search were somehow set', () => {
    const turn: ChatTurn = { role: 'user', content: 'what about recent work?', used_web_search: true }
    render(<ChatMessage turn={turn} />)

    expect(screen.queryByTestId('chat-web-badge')).not.toBeInTheDocument()
  })

  it('renders no badge when used_web_search is absent entirely (pre-Phase-1 / optimistic message shape)', () => {
    const turn: ChatTurn = { role: 'assistant', content: 'Fine [Paper 1].' }
    render(<ChatMessage turn={turn} />)

    expect(screen.queryByTestId('chat-web-badge')).not.toBeInTheDocument()
  })
})
