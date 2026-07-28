import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReportModePanel } from './ReportModePanel'
import type { CurationStateResponse } from '../../api/types'

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', display_title: 't', stage: 'synthesize', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    ...overrides,
  }
}

describe('ReportModePanel', () => {
  it('no report yet: shows a Generate report CTA, not empty sections -- this is the fix for "no way to see the report"', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(
      <ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} />,
    )

    expect(screen.getByText('No report yet for this review.')).toBeInTheDocument()
    await user.click(screen.getByTestId('generate-report'))
    expect(onGenerateReport).toHaveBeenCalledTimes(1)
  })

  it('renders findings, limitations, and future scope content once a report exists', () => {
    const state = baseState({
      report: {
        findings: { content: 'Finding A.', cited_papers: [{ paper_id: 'p1', title: 'Paper One' }], cited_web_articles: [] },
        limitations: { content: 'Limitation A.', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'Future A.', cited_papers: [], cited_web_articles: [{ url: 'https://x.com', title: 'Web Article' }] },
        skipped_paper_ids: [],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} />)

    expect(screen.getByText('Finding A.')).toBeInTheDocument()
    expect(screen.getByText('Limitation A.')).toBeInTheDocument()
    expect(screen.getByText('Future A.')).toBeInTheDocument()
    expect(screen.getByText('Paper One')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Web Article' })).toHaveAttribute('href', 'https://x.com')
  })

  it('clicking Regenerate calls onRegenerateReport', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({
      report: {
        findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} />)

    await user.click(screen.getByTestId('regenerate-report'))
    expect(onRegenerateReport).toHaveBeenCalledTimes(1)
  })

  it('shows a skipped-papers note when skipped_paper_ids is non-empty', () => {
    const state = baseState({
      report: {
        findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: ['p9'],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} />)

    expect(screen.getByText(/1 selected paper skipped from synthesis/)).toBeInTheDocument()
  })
})
