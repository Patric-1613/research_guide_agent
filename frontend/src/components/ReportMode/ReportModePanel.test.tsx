import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReportModePanel } from './ReportModePanel'
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
        findings: { content: 'Finding A.', cited_papers: [{ paper_id: 'p1', title: 'Paper One' }], cited_web_articles: [], reference_numbers: [1] },
        limitations: { content: 'Limitation A.', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        future_scope: { content: 'Future A.', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        skipped_paper_ids: [],
        references: [
          { number: 1, kind: 'paper', title: 'Paper One', formatted: 'A. Uthor (2024). Paper One.', paper_id: 'p1', link_url: null },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} />)

    expect(screen.getByText('Finding A.')).toBeInTheDocument()
    expect(screen.getByText('Limitation A.')).toBeInTheDocument()
    expect(screen.getByText('Future A.')).toBeInTheDocument()
  })

  it('old citation pills are no longer rendered -- title-only pills are superseded by References', () => {
    const state = baseState({
      report: {
        findings: { content: 'Finding A.', cited_papers: [{ paper_id: 'p1', title: 'Paper One' }], cited_web_articles: [{ url: 'https://x.com', title: 'Web Article' }], reference_numbers: [1, 2] },
        limitations: { content: 'Limitation A.', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        future_scope: { content: 'Future A.', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        skipped_paper_ids: [],
        references: [
          { number: 1, kind: 'paper', title: 'Paper One', formatted: 'A. Uthor (2024). Paper One.', paper_id: 'p1', link_url: null },
          { number: 2, kind: 'web', title: 'Web Article', formatted: 'Web Article. x.com. https://x.com', url: 'https://x.com', link_url: 'https://x.com' },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} />)

    // The bare pill text (just a title, no citation/link) must not appear
    // on its own anymore -- only inside the References section's full
    // formatted citation.
    expect(screen.queryByText('Paper One')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Web Article' })).not.toBeInTheDocument()
  })

  it('renders a References section with formatted paper citations and hyperlinked web sources', () => {
    const state = baseState({
      report: {
        findings: {
          content: 'Per [1] and [2], X is true.',
          cited_papers: [{ paper_id: 'p1', title: 'Paper One' }],
          cited_web_articles: [{ url: 'https://x.com', title: 'Web Article' }],
          reference_numbers: [1, 2],
        },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        skipped_paper_ids: [],
        references: [
          { number: 1, kind: 'paper', title: 'Paper One', formatted: 'Uthor, A. (2024). Paper One. arXiv preprint.', paper_id: 'p1', link_url: 'https://doi.org/10.1/x' },
          { number: 2, kind: 'web', title: 'Web Article', formatted: 'Web Article. x.com. https://x.com', url: 'https://x.com', link_url: 'https://x.com' },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} />)

    const references = screen.getByTestId('report-references')
    expect(references).toBeInTheDocument()

    // Inline markers render as clickable anchors pointing into References.
    expect(screen.getByTestId('citation-marker-1')).toHaveAttribute('href', '#ref-1')
    expect(screen.getByTestId('citation-marker-2')).toHaveAttribute('href', '#ref-2')

    // Paper reference shows real formatted citation text, not a bare title.
    expect(screen.getByText('Uthor, A. (2024). Paper One. arXiv preprint.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Uthor, A. (2024). Paper One. arXiv preprint.' })).toHaveAttribute(
      'href', 'https://doi.org/10.1/x',
    )

    // Web reference is hyperlinked via link_url.
    expect(screen.getByRole('link', { name: 'Web Article. x.com. https://x.com' })).toHaveAttribute(
      'href', 'https://x.com',
    )
  })

  it('an old report with no references field at all still renders safely, with no References section', () => {
    const state = baseState({
      report: {
        findings: { content: 'Old prose, no markers.', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
        // references deliberately omitted entirely, matching an old,
        // pre-R1 report shape reaching the frontend.
      },
    })

    expect(() =>
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} />),
    ).not.toThrow()

    expect(screen.getByText('Old prose, no markers.')).toBeInTheDocument()
    expect(screen.queryByTestId('report-references')).not.toBeInTheDocument()
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
