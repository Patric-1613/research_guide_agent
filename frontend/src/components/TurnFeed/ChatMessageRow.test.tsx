import { describe, expect, it } from 'vitest'
import { isEligibleForAddToReport } from './ChatMessageRow'
import type { ChatTurn } from '../../types'

// chat-web-relevance-guardrails R7C: direct unit tests for the shared
// eligibility predicate -- ChatModePanel.test.tsx additionally covers
// it through the rendered "Add to report" menu item's disabled state,
// but this proves the pure function's own return value directly.

function baseEligibleTurn(overrides: Partial<ChatTurn> = {}): ChatTurn {
  return {
    role: 'assistant', content: 'Per [Web 1], ...', exchange_id: 'ex-1',
    used_web_search: true, cited_web_articles: [{ url: 'https://a.com', title: 'A' }], added_to_report: false,
    ...overrides,
  }
}

describe('isEligibleForAddToReport', () => {
  it('returns true when every existing condition passes and web_relevance_verified is omitted (legacy)', () => {
    expect(isEligibleForAddToReport(baseEligibleTurn())).toBe(true)
  })

  it('returns true when web_relevance_verified is explicitly null (legacy, serialized)', () => {
    expect(isEligibleForAddToReport(baseEligibleTurn({ web_relevance_verified: null }))).toBe(true)
  })

  it('returns true when web_relevance_verified is true', () => {
    expect(isEligibleForAddToReport(baseEligibleTurn({ web_relevance_verified: true }))).toBe(true)
  })

  it('returns false when web_relevance_verified is false, even though every other condition passes', () => {
    expect(isEligibleForAddToReport(baseEligibleTurn({ web_relevance_verified: false }))).toBe(false)
  })
})
