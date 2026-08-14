import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ReportModePanel } from './ReportModePanel'
import type { CurationStateResponse, ReportStreamCompletionNotice } from '../../types'

function baseState(overrides: Partial<CurationStateResponse> = {}): CurationStateResponse {
  return {
    session_id: 's1', topic: 't', display_title: 't', stage: 'synthesize', target_count: 10,
    selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
    reserve_remaining: 0, refinement_notes: [], report: null, chat_history: [], web_articles_added: [],
    pending_web_offer: null, pending_report_update: null, turn_history: [], stop_reason: null,
    report_versions: [], active_report_version_id: null, chat_references: [],
    ...overrides,
  }
}

describe('ReportModePanel', () => {
  it('no report yet: shows a Generate report CTA, not empty sections -- this is the fix for "no way to see the report"', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(
      <ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />,
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    const references = screen.getByTestId('report-references')
    expect(references).toBeInTheDocument()

    // Inline markers render as clickable anchors pointing into References.
    expect(screen.getByTestId('citation-marker-1')).toHaveAttribute('href', '#ref-1')
    expect(screen.getByTestId('citation-marker-2')).toHaveAttribute('href', '#ref-2')

    // Paper reference shows real formatted citation text, not a bare title.
    expect(screen.getByText('Uthor, A. (2024). Paper One. arXiv preprint.')).toBeInTheDocument()
    const paperLink = screen.getByRole('link', { name: 'Uthor, A. (2024). Paper One. arXiv preprint.' })
    expect(paperLink).toHaveAttribute('href', 'https://doi.org/10.1/x')

    // Web reference is hyperlinked via link_url.
    const webLink = screen.getByRole('link', { name: 'Web Article. x.com. https://x.com' })
    expect(webLink).toHaveAttribute('href', 'https://x.com')

    // Both linked references must be visibly clickable AT REST -- an
    // underline, not just a color that changes on hover (easy to miss).
    for (const link of [paperLink, webLink]) {
      expect(link.className).toContain('underline')
    }

    // Web reference gets a subtle "Web source" indicator; the paper
    // reference (same list, same numbering) does not.
    expect(screen.getByTestId('reference-web-icon-2')).toHaveAttribute('aria-label', 'Web source')
    expect(screen.queryByTestId('reference-web-icon-1')).not.toBeInTheDocument()
  })

  it('a reference with no link_url renders as plain, non-linked text', () => {
    const state = baseState({
      report: {
        findings: {
          content: 'Per [1], X is true.', cited_papers: [{ paper_id: 'p1', title: 'Paper One' }],
          cited_web_articles: [], reference_numbers: [1],
        },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [], reference_numbers: [] },
        skipped_paper_ids: [],
        references: [
          { number: 1, kind: 'paper', title: 'Paper One', formatted: 'Uthor, A. (n.d.). Paper One.', paper_id: 'p1', link_url: null },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByText('Uthor, A. (n.d.). Paper One.')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Uthor, A. (n.d.). Paper One.' })).not.toBeInTheDocument()
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
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />),
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByText(/1 selected paper skipped from synthesis/)).toBeInTheDocument()
  })
})

describe('ReportModePanel -- report-quality Phase R2A: dynamic sections', () => {
  it('renders an arbitrary number of backend-provided sections, in backend order, with headings from title', () => {
    const state = baseState({
      report: {
        findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
        sections: [
          { key: 'executive_summary', title: 'Executive Summary', content: 'Summary content.', reference_numbers: [] },
          { key: 'thematic_findings', title: 'Thematic Findings', content: 'Findings content.', reference_numbers: [] },
          { key: 'gap_analysis', title: 'Gap Analysis', content: 'Gap content.', reference_numbers: [] },
          { key: 'conclusion', title: 'Conclusion', content: 'Conclusion content.', reference_numbers: [] },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    const headings = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(headings).toEqual(['Executive Summary', 'Thematic Findings', 'Gap Analysis', 'Conclusion'])
    expect(screen.getByText('Summary content.')).toBeInTheDocument()
    expect(screen.getByText('Conclusion content.')).toBeInTheDocument()

    // The old 3-name assumption is gone -- neither legacy content string
    // appears, proving rendering came from `sections`, not a fallback.
    expect(screen.queryByText('F')).not.toBeInTheDocument()
    expect(screen.queryByText('L')).not.toBeInTheDocument()
  })

  it('renders a section nav with one entry per section, linking to each section', () => {
    const state = baseState({
      report: {
        findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
        sections: [
          { key: 'executive_summary', title: 'Executive Summary', content: 'A', reference_numbers: [] },
          { key: 'conclusion', title: 'Conclusion', content: 'B', reference_numbers: [] },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    const nav = screen.getByTestId('report-section-nav')
    expect(nav).toBeInTheDocument()
    expect(screen.getByTestId('section-nav-link-executive_summary')).toHaveAttribute('href', '#section-executive_summary')
    expect(screen.getByTestId('section-nav-link-conclusion')).toHaveAttribute('href', '#section-conclusion')
  })

  it('References still render after all dynamic sections', () => {
    const state = baseState({
      report: {
        findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
        sections: [
          { key: 'executive_summary', title: 'Executive Summary', content: 'Per [1], X.', reference_numbers: [1] },
        ],
        references: [
          { number: 1, kind: 'paper', title: 'Paper One', formatted: 'Uthor, A. (2024). Paper One.', paper_id: 'p1', link_url: null },
        ],
      },
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('citation-marker-1')).toHaveAttribute('href', '#ref-1')
    expect(screen.getByTestId('report-references')).toBeInTheDocument()
    expect(screen.getByText('Uthor, A. (2024). Paper One.')).toBeInTheDocument()
  })

  it('a report with only legacy fields (sections absent) still renders safely via the same fallback path, with no section nav for just one implicit group', () => {
    const state = baseState({
      report: {
        findings: { content: 'Old findings prose.', cited_papers: [], cited_web_articles: [] },
        limitations: { content: 'Old limitations prose.', cited_papers: [], cited_web_articles: [] },
        future_scope: { content: 'Old future prose.', cited_papers: [], cited_web_articles: [] },
        skipped_paper_ids: [],
        // sections deliberately omitted -- simulates a raw pre-R2A shape
        // reaching the frontend, independent of trusting that the
        // backend always derives it.
      },
    })

    expect(() =>
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />),
    ).not.toThrow()

    expect(screen.getByText('Old findings prose.')).toBeInTheDocument()
    expect(screen.getByText('Old limitations prose.')).toBeInTheDocument()
    expect(screen.getByText('Old future prose.')).toBeInTheDocument()
    // Still three sections worth of headings, even via the fallback path.
    const headings = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(headings).toEqual(['Findings', 'Limitations', 'Future Scope'])
    expect(screen.getByTestId('report-section-nav')).toBeInTheDocument()
  })
})

describe('ReportModePanel -- report-quality Phase R2C: report templates', () => {
  function reportWithTemplate(template?: 'foundational' | 'analytical' | 'expert') {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
      ...(template ? { report_template: template } : {}),
    }
  }

  it('renders all three template options', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-template-option-foundational')).toBeInTheDocument()
    expect(screen.getByTestId('report-template-option-analytical')).toBeInTheDocument()
    expect(screen.getByTestId('report-template-option-expert')).toBeInTheDocument()
  })

  it('defaults the selector to Analytical when there is no report yet', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByTestId('report-template-option-foundational')).toHaveAttribute('aria-checked', 'false')
  })

  it('initializes the selector from an existing report_template', () => {
    const state = baseState({ report: reportWithTemplate('expert') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-template-option-expert')).toHaveAttribute('aria-checked', 'true')
  })

  it('selecting Foundational and clicking Generate calls onGenerateReport("foundational")', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('report-template-option-foundational'))
    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('foundational', 'off')
  })

  it('selecting Expert and clicking Regenerate calls onRegenerateReport("expert")', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({ report: reportWithTemplate('analytical') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('report-template-option-expert'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('expert', 'off')
  })

  it('shows a template badge for an existing report', () => {
    const state = baseState({ report: reportWithTemplate('foundational') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-template-badge')).toHaveTextContent('Foundational')
  })

  it('defaults the badge and selector to Analytical for an old report with no report_template field', () => {
    const state = baseState({ report: reportWithTemplate() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-template-badge')).toHaveTextContent('Analytical')
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')
  })

  it('syncs the selector when the active report changes to a different template', () => {
    const { rerender } = render(
      <ReportModePanel
        state={baseState({ report: reportWithTemplate('analytical') })}
        disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
      />,
    )
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')

    rerender(
      <ReportModePanel
        state={baseState({ report: reportWithTemplate('expert') })}
        disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
      />,
    )

    expect(screen.getByTestId('report-template-option-expert')).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'false')
  })
})

describe('ReportModePanel -- report-quality Phase R3: report version selector', () => {
  const V1 = {
    version_id: 'v1', version_number: 1, created_at: '2026-08-05T00:00:00+00:00',
    report_template: 'analytical' as const, generation_reason: 'initial', is_active: false,
  }
  const V2 = {
    version_id: 'v2', version_number: 2, created_at: '2026-08-05T01:00:00+00:00',
    report_template: 'expert' as const, generation_reason: 'regenerate', is_active: false,
  }
  const V3 = {
    version_id: 'v3', version_number: 3, created_at: '2026-08-05T02:00:00+00:00',
    report_template: 'foundational' as const, generation_reason: 'chat_add_to_report', is_active: true,
  }

  function reportStub() {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
    }
  }

  it('hides the version selector when report_versions is empty (old-report fixture, no version metadata)', () => {
    const state = baseState({ report: reportStub(), report_versions: [], active_report_version_id: null })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.queryByTestId('report-version-selector')).not.toBeInTheDocument()
    // Rest of the panel still renders safely without version metadata.
    expect(screen.getByText('F')).toBeInTheDocument()
  })

  it('renders one option per available version', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    const select = screen.getByTestId('report-version-selector') as HTMLSelectElement
    expect(select.options).toHaveLength(3)
  })

  it('selects the option matching active_report_version_id', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v2',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    const select = screen.getByTestId('report-version-selector') as HTMLSelectElement
    expect(select.value).toBe('v2')
  })

  it('labels include the version number, template, and generation reason', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByText('Version 1 — Analytical — Initial')).toBeInTheDocument()
    expect(screen.getByText('Version 2 — Expert — Regenerate')).toBeInTheDocument()
    expect(screen.getByText('Version 3 — Foundational — Chat add')).toBeInTheDocument()
  })

  it('changing the selector calls onActivateReportVersion with the chosen version_id', async () => {
    const user = userEvent.setup()
    const onActivateReportVersion = vi.fn()
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(
      <ReportModePanel
        state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={onActivateReportVersion} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
      />,
    )

    await user.selectOptions(screen.getByTestId('report-version-selector'), 'v1')

    expect(onActivateReportVersion).toHaveBeenCalledWith('v1')
  })

  it('the template selector remains independent of the version selector', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(
      <ReportModePanel
        state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport}
        onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
      />,
    )

    await user.click(screen.getByTestId('report-template-option-expert'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('expert', 'off')
  })

  it('is disabled while a report action is in progress, same as the other report controls', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={true} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-version-selector')).toBeDisabled()
  })
})

describe('ReportModePanel -- report-quality Phase R4.1: refinement toggle', () => {
  function reportStub(overrides: Record<string, unknown> = {}) {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
      ...overrides,
    }
  }

  it('renders the "Refine once" toggle before a report exists', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('refine-once-toggle')).toBeInTheDocument()
    expect(screen.getByText('Refine once')).toBeInTheDocument()
  })

  it('renders the "Refine once" toggle once a report exists', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('refine-once-toggle')).toBeInTheDocument()
  })

  it('the toggle defaults to off (unchecked)', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('refine-once-toggle')).not.toBeChecked()
  })

  it('clicking Generate with the toggle off calls onGenerateReport with "off"', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('analytical', 'off')
  })

  it('checking the toggle then clicking Generate calls onGenerateReport(template, "single")', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('refine-once-toggle'))
    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('analytical', 'single')
  })

  it('checking the toggle then clicking Regenerate calls onRegenerateReport(template, "single")', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('refine-once-toggle'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('analytical', 'single')
  })

  it('the toggle is disabled while a report action is in progress', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={true} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('refine-once-toggle')).toBeDisabled()
  })

  it('renders a compact refinement badge when a revision happened', () => {
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: null,
          issues: [], revision_instructions: '', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-refinement-badge')).toHaveTextContent('Refined once · score 40')
  })

  it('renders a compact refinement badge when the draft was only evaluated (no revision)', () => {
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 0, initial_score: 88, final_score: 88,
          issues: [], revision_instructions: '', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('report-refinement-badge')).toHaveTextContent('Evaluated · score 88')
  })

  it('does not render a refinement badge when the report has no refinement metadata', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.queryByTestId('report-refinement-badge')).not.toBeInTheDocument()
  })

  it('the badge alone never renders full issues or revision instructions', () => {
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: null,
          issues: [], revision_instructions: '', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.queryByText(/revision instructions/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/issues/i)).not.toBeInTheDocument()
  })
})

describe('ReportModePanel -- report-quality Phase R4.2: evaluation details disclosure', () => {
  function reportStub(overrides: Record<string, unknown> = {}) {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
      ...overrides,
    }
  }

  it('renders no disclosure toggle when the report has no refinement metadata', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.queryByTestId('evaluation-details-toggle')).not.toBeInTheDocument()
  })

  it('renders no disclosure toggle when refinement has no issues and no section_scores', () => {
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 0, initial_score: 90, final_score: 90,
          issues: [], revision_instructions: '', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.queryByTestId('evaluation-details-toggle')).not.toBeInTheDocument()
  })

  it('renders the disclosure toggle when issues exist, collapsed by default', () => {
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: null,
          issues: ['Gap Analysis overlaps with Future Research Directions'],
          revision_instructions: 'merge the two sections', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    expect(screen.getByTestId('evaluation-details-toggle')).toHaveTextContent('Evaluation details')
    expect(screen.queryByTestId('evaluation-details-panel')).not.toBeInTheDocument()
  })

  it('expanding the disclosure shows the draft-before-revision copy and the issues', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: null,
          issues: ['Issue one', 'Issue two'], revision_instructions: 'fix it', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const panel = screen.getByTestId('evaluation-details-panel')
    expect(panel).toHaveTextContent('Evaluator findings describe the draft before revision, not necessarily the final report.')
    expect(screen.getByTestId('evaluation-details-issues')).toHaveTextContent('Issue one')
    expect(screen.getByTestId('evaluation-details-issues')).toHaveTextContent('Issue two')
  })

  it('rounds===0: shows "Score N" and "No revision needed", not an Initial/Final score comparison', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 0, initial_score: 90, final_score: 90,
          issues: ['A minor note'], revision_instructions: '', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const summary = screen.getByTestId('evaluation-details-score-summary')
    expect(summary).toHaveTextContent('Score 90')
    expect(summary).toHaveTextContent('No revision needed')
    expect(summary).not.toHaveTextContent('Initial score')
    expect(summary).not.toHaveTextContent('Final score')
  })

  it('rounds>0 and final_score null: shows initial score, "Revised once", and "Final score not re-evaluated"', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: null,
          issues: ['Issue one'], revision_instructions: 'fix it', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const summary = screen.getByTestId('evaluation-details-score-summary')
    expect(summary).toHaveTextContent('Initial score 40')
    expect(summary).toHaveTextContent('Revised once')
    expect(summary).toHaveTextContent('Final score not re-evaluated')
  })

  it('rounds>0 and final_score present: shows the score transition "Score N → M"', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: 75,
          issues: ['Issue one'], revision_instructions: 'fix it', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const summary = screen.getByTestId('evaluation-details-score-summary')
    expect(summary).toHaveTextContent('Score 40 → 75')
    expect(summary).toHaveTextContent('Revised once')
  })

  it('shows only the first 5 issues, with a "+N more" line for the rest', async () => {
    const user = userEvent.setup()
    const issues = ['one', 'two', 'three', 'four', 'five', 'six', 'seven']
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 30, final_score: null,
          issues, revision_instructions: 'fix it', section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const issuesList = screen.getByTestId('evaluation-details-issues')
    for (const issue of ['one', 'two', 'three', 'four', 'five']) {
      expect(issuesList).toHaveTextContent(issue)
    }
    expect(issuesList).not.toHaveTextContent('six')
    expect(issuesList).not.toHaveTextContent('seven')
    expect(screen.getByTestId('evaluation-details-more-issues')).toHaveTextContent('+2 more')
  })

  it('renders section scores as separate rows, using real section titles when the key matches', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        // Only "findings"/"future_scope" scored -- a partial dict, the
        // report also has "limitations" with no score at all.
        refinement: {
          enabled: true, rounds: 0, initial_score: 85, final_score: 85,
          issues: [], revision_instructions: '',
          section_scores: { findings: 80, future_scope: 90 },
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const scores = screen.getByTestId('evaluation-details-section-scores')
    expect(scores).toHaveTextContent('Section scores')

    const findingsRow = screen.getByTestId('evaluation-details-section-score-findings')
    expect(findingsRow).toHaveTextContent('Findings')
    expect(findingsRow).toHaveTextContent('80')

    const futureScopeRow = screen.getByTestId('evaluation-details-section-score-future_scope')
    expect(futureScopeRow).toHaveTextContent('Future Scope')
    expect(futureScopeRow).toHaveTextContent('90')

    // Partial dict tolerated -- no row rendered for the unscored section.
    expect(screen.queryByTestId('evaluation-details-section-score-limitations')).not.toBeInTheDocument()
  })

  it('falls back to the raw key as the label when a section_scores key has no matching section', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 0, initial_score: 70, final_score: 70,
          issues: [], revision_instructions: '',
          section_scores: { some_unmapped_key: 65 },
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    const row = screen.getByTestId('evaluation-details-section-score-some_unmapped_key')
    expect(row).toHaveTextContent('some_unmapped_key')
    expect(row).toHaveTextContent('65')
  })

  it('never renders the raw revision_instructions text', async () => {
    const user = userEvent.setup()
    const state = baseState({
      report: reportStub({
        refinement: {
          enabled: true, rounds: 1, initial_score: 40, final_score: null,
          issues: ['Issue one'],
          revision_instructions: 'REWRITE THE INTRODUCTION TO EMPHASIZE X',
          section_scores: null,
        },
      }),
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportUrls={{ markdown: 'http://test.local/curation/s1/report/export?format=markdown', pdf: 'http://test.local/curation/s1/report/export?format=pdf', docx: 'http://test.local/curation/s1/report/export?format=docx' }} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null} reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false} onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()} />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    expect(screen.queryByText(/REWRITE THE INTRODUCTION/i)).not.toBeInTheDocument()
  })
})

describe('ReportModePanel -- report-quality Phase R5C.3: Export menu', () => {
  function reportStub() {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
    }
  }

  const EXPORT_URLS = {
    markdown: 'http://test.local/curation/s1/report/export?format=markdown',
    pdf: 'http://test.local/curation/s1/report/export?format=pdf',
    docx: 'http://test.local/curation/s1/report/export?format=docx',
  }

  function renderPanel({ disabled = false, withReport = true }: { disabled?: boolean; withReport?: boolean } = {}) {
    const state = baseState({ report: withReport ? reportStub() : null })
    return render(
      <ReportModePanel
        state={state} disabled={disabled} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportUrls={EXPORT_URLS}
        reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
        reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
        onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
      />,
    )
  }

  it('does not render the Export trigger when there is no report yet', () => {
    renderPanel({ withReport: false })

    expect(screen.queryByTestId('export-menu-trigger')).not.toBeInTheDocument()
  })

  it('renders the Export trigger once a report exists', () => {
    renderPanel()

    expect(screen.getByTestId('export-menu-trigger')).toBeInTheDocument()
    expect(screen.queryByTestId('export-menu')).not.toBeInTheDocument()
  })

  it('clicking the trigger opens the menu with Markdown, PDF, and DOCX options', async () => {
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByTestId('export-menu-trigger'))

    expect(screen.getByTestId('export-menu')).toBeInTheDocument()
    expect(screen.getByTestId('export-menu-option-markdown')).toHaveTextContent('Markdown')
    expect(screen.getByTestId('export-menu-option-pdf')).toHaveTextContent('PDF')
    expect(screen.getByTestId('export-menu-option-docx')).toHaveTextContent('DOCX')
  })

  it('each option is a real download link pointing at its own format URL', async () => {
    const user = userEvent.setup()
    renderPanel()
    await user.click(screen.getByTestId('export-menu-trigger'))

    for (const format of ['markdown', 'pdf', 'docx'] as const) {
      const option = screen.getByTestId(`export-menu-option-${format}`)
      expect(option).toHaveAttribute('href', EXPORT_URLS[format])
      expect(option).toHaveAttribute('download')
    }
  })

  it('the trigger is a real disabled button while a report action is in progress, so the menu can never open', async () => {
    const user = userEvent.setup()
    renderPanel({ disabled: true })

    const trigger = screen.getByTestId('export-menu-trigger')
    expect(trigger).toBeDisabled()

    await user.click(trigger)

    expect(screen.queryByTestId('export-menu')).not.toBeInTheDocument()
  })

  it('closes the menu after clicking an option', async () => {
    const user = userEvent.setup()
    renderPanel()
    await user.click(screen.getByTestId('export-menu-trigger'))
    expect(screen.getByTestId('export-menu')).toBeInTheDocument()

    // fireEvent, not userEvent -- jsdom attempts a real (unsupported)
    // navigation for a non-"#" <a href> click either way, which is just
    // console noise (not a jsdom API), but userEvent's extra pointer/
    // focus event sequence makes that noise more verbose for no benefit
    // here; the only thing under test is the onClick-driven menu-close.
    fireEvent.click(screen.getByTestId('export-menu-option-pdf'))

    expect(screen.queryByTestId('export-menu')).not.toBeInTheDocument()
  })

  it('closes the menu on an outside click', async () => {
    const user = userEvent.setup()
    renderPanel()
    await user.click(screen.getByTestId('export-menu-trigger'))
    expect(screen.getByTestId('export-menu')).toBeInTheDocument()

    await user.click(document.body)

    expect(screen.queryByTestId('export-menu')).not.toBeInTheDocument()
  })

  it('closes the menu on Escape', async () => {
    const user = userEvent.setup()
    renderPanel()
    await user.click(screen.getByTestId('export-menu-trigger'))
    expect(screen.getByTestId('export-menu')).toBeInTheDocument()

    await user.keyboard('{Escape}')

    expect(screen.queryByTestId('export-menu')).not.toBeInTheDocument()
  })

  it('a click inside the menu container but not on an option (e.g. the trigger itself) does not count as an outside click', async () => {
    const user = userEvent.setup()
    renderPanel()
    const trigger = screen.getByTestId('export-menu-trigger')
    await user.click(trigger)
    expect(screen.getByTestId('export-menu')).toBeInTheDocument()

    // Clicking the trigger again is its own explicit toggle-closed path,
    // not the outside-click listener -- both should agree the menu ends
    // up closed, but for different reasons, so this is worth its own test.
    await user.click(trigger)

    expect(screen.queryByTestId('export-menu')).not.toBeInTheDocument()
  })
})

describe('ReportModePanel -- Usage Protection M4.3B: report-generation progress streaming UI', () => {
  const EXPORT_URLS = {
    markdown: 'http://test.local/curation/s1/report/export?format=markdown',
    pdf: 'http://test.local/curation/s1/report/export?format=pdf',
    docx: 'http://test.local/curation/s1/report/export?format=docx',
  }

  function reportStub() {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
      report_template: 'analytical' as const,
      refinement: { enabled: true, rounds: 1, initial_score: 40, final_score: null, issues: ['too shallow'], revision_instructions: '', section_scores: null },
    }
  }

  interface StreamingOverrides {
    state?: CurationStateResponse
    disabled?: boolean
    reportStreamActive?: boolean
    reportStreamOperation?: 'generate' | 'regenerate' | null
    reportStreamPhase?: string | null
    reportStreamPhaseHistory?: string[]
    reportStreamStopping?: boolean
    reportStreamError?: string | null
    reportStreamSyncFailed?: boolean
    reportStreamCompletionNotice?: ReportStreamCompletionNotice | null
  }

  function renderStreaming(overrides: StreamingOverrides = {}) {
    const onGenerateReport = vi.fn()
    const onRegenerateReport = vi.fn()
    const onCancelReportStream = vi.fn()
    const onRetryReportSync = vi.fn()
    render(
      <ReportModePanel
        state={overrides.state ?? baseState()}
        disabled={overrides.disabled ?? false}
        onGenerateReport={onGenerateReport}
        onRegenerateReport={onRegenerateReport}
        onActivateReportVersion={vi.fn()}
        exportUrls={EXPORT_URLS}
        reportStreamActive={overrides.reportStreamActive ?? false}
        reportStreamOperation={overrides.reportStreamOperation ?? null}
        reportStreamPhase={(overrides.reportStreamPhase as never) ?? null}
        reportStreamPhaseHistory={overrides.reportStreamPhaseHistory as never}
        reportStreamStopping={overrides.reportStreamStopping ?? false}
        reportStreamError={overrides.reportStreamError ?? null}
        reportStreamSyncFailed={overrides.reportStreamSyncFailed ?? false}
        reportStreamCompletionNotice={overrides.reportStreamCompletionNotice ?? null}
        onCancelReportStream={onCancelReportStream}
        onRetryReportSync={onRetryReportSync}
      />,
    )
    return { onGenerateReport, onRegenerateReport, onCancelReportStream, onRetryReportSync }
  }

  describe('empty (Generate) view', () => {
    it('shows the progress area with a phase label while generating, and hides the "no report yet" copy', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating' })

      expect(screen.getByTestId('report-stream-progress')).toBeInTheDocument()
      expect(screen.getByTestId('report-stream-phase-label')).toHaveTextContent('Generating report')
      expect(screen.queryByText('No report yet for this review.')).not.toBeInTheDocument()
    })

    it('UXH.3: the phase label is a live region -- previously missing here, unlike its Regenerate sibling', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating' })

      const label = screen.getByTestId('report-stream-phase-label')
      expect(label).toHaveAttribute('role', 'status')
      expect(label).toHaveAttribute('aria-live', 'polite')
    })

    it('replaces Generate with Stop while active, in the same stable slot', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating' })

      expect(screen.queryByTestId('generate-report')).not.toBeInTheDocument()
      expect(screen.getByTestId('report-stream-stop')).toBeInTheDocument()
    })

    it('shows Generate (not Stop) when no stream is active', () => {
      renderStreaming()

      expect(screen.getByTestId('generate-report')).toBeInTheDocument()
      expect(screen.queryByTestId('report-stream-stop')).not.toBeInTheDocument()
    })

    it('keeps template and Refine Once controls visible while generating', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating', disabled: true })

      expect(screen.getByTestId('report-template-selector')).toBeInTheDocument()
      expect(screen.getByTestId('refine-once-toggle')).toBeInTheDocument()
    })

    it('renders no partial report content (no findings/sections/references) while generating', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'saving' })

      expect(screen.queryByTestId('report-references')).not.toBeInTheDocument()
      expect(screen.queryByText('F')).not.toBeInTheDocument()
    })

    it('shows no percentage/numeric progress indicator anywhere in the progress area', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating' })

      const progress = screen.getByTestId('report-stream-progress')
      expect(progress.textContent).not.toMatch(/%/)
    })

    it('clicking Stop calls onCancelReportStream', async () => {
      const user = userEvent.setup()
      const { onCancelReportStream } = renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating' })

      await user.click(screen.getByTestId('report-stream-stop'))

      expect(onCancelReportStream).toHaveBeenCalledTimes(1)
    })

    it('shows "Stopping" while cancellation is settling, and disables the Stop button against a double click', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'generating', reportStreamStopping: true })

      expect(screen.getByTestId('report-stream-phase-label')).toHaveTextContent('Stopping')
      expect(screen.getByTestId('report-stream-stop')).toBeDisabled()
    })

    it('shows a safe error message when a handled failure occurs', () => {
      renderStreaming({ reportStreamError: 'The model provider returned an error.' })

      expect(screen.getByTestId('report-stream-error')).toHaveTextContent('The model provider returned an error.')
    })

    it('shows a sync-retry notice, and clicking Retry calls onRetryReportSync', async () => {
      const user = userEvent.setup()
      const { onRetryReportSync } = renderStreaming({ reportStreamSyncFailed: true })

      expect(screen.getByTestId('report-stream-sync-failed')).toBeInTheDocument()
      await user.click(screen.getByTestId('report-stream-sync-retry'))

      expect(onRetryReportSync).toHaveBeenCalledTimes(1)
    })

    it('re-enables Generate once the stream is no longer active', () => {
      const { rerender } = render(
        <ReportModePanel
          state={baseState()} disabled onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="generate" reportStreamPhase="generating"
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      expect(screen.getByTestId('report-stream-stop')).toBeInTheDocument()

      rerender(
        <ReportModePanel
          state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(screen.getByTestId('generate-report')).toBeEnabled()
      expect(screen.queryByTestId('report-stream-stop')).not.toBeInTheDocument()
    })

    it('UXH.3: focus returns to Generate once a completed/cancelled stream removes Stop, when Stop had held focus', () => {
      const { rerender } = render(
        <ReportModePanel
          state={baseState()} disabled onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="generate" reportStreamPhase="generating"
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      screen.getByTestId('report-stream-stop').focus()
      expect(document.activeElement).toBe(screen.getByTestId('report-stream-stop'))

      rerender(
        <ReportModePanel
          state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(document.activeElement).toBe(screen.getByTestId('generate-report'))
    })

    it('UXH.3: a completed first-ever Generate falls back to focusing Regenerate once the report view replaces the empty view', () => {
      const { rerender } = render(
        <ReportModePanel
          state={baseState()} disabled onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="generate" reportStreamPhase="generating"
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      screen.getByTestId('report-stream-stop').focus()

      // The stream completed WITH a report -- the empty view (and its own
      // generateButtonRef target) is gone, replaced by the full report
      // view showing Regenerate instead.
      rerender(
        <ReportModePanel
          state={baseState({ report: reportStub() })} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(document.activeElement).toBe(screen.getByTestId('regenerate-report'))
    })
  })

  describe('regeneration view: existing report stays visible', () => {
    it('keeps the existing report fully visible while regenerating, not dimmed or cleared', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'evaluating' })

      expect(screen.getByText('F')).toBeInTheDocument()
      expect(screen.getByText('L')).toBeInTheDocument()
      expect(screen.getByText('S')).toBeInTheDocument()
    })

    it('shows a compact phase label in the header while regenerating', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'revising' })

      expect(screen.getByTestId('report-stream-phase-label')).toHaveTextContent('Revising report')
    })

    it('UXH.2: the header status is a prominent, accessible live region -- not the old easy-to-miss italic text', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'revising' })

      const label = screen.getByTestId('report-stream-phase-label')
      expect(label).toHaveAttribute('role', 'status')
      expect(label).toHaveAttribute('aria-live', 'polite')
      expect(label.className).not.toContain('italic')
    })

    it('UXH.2: Stop remains available in its existing slot throughout regeneration, never duplicated elsewhere', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'revising' })

      expect(screen.getAllByTestId('report-stream-stop')).toHaveLength(1)
      expect(screen.getAllByTestId('report-stream-phase-label')).toHaveLength(1)
    })

    it('replaces Regenerate with Stop in the same stable action slot', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'generating' })

      expect(screen.queryByTestId('regenerate-report')).not.toBeInTheDocument()
      expect(screen.getByTestId('report-stream-stop')).toBeInTheDocument()
    })

    it('shows Regenerate (not Stop) when no stream is active', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state })

      expect(screen.getByTestId('regenerate-report')).toBeInTheDocument()
      expect(screen.queryByTestId('report-stream-stop')).not.toBeInTheDocument()
    })

    it('disables template/refinement/version/export controls while regenerating (disabled prop composition)', () => {
      const state = baseState({
        report: reportStub(),
        report_versions: [{ version_id: 'v1', version_number: 1, created_at: null, report_template: 'analytical', generation_reason: 'initial', is_active: true }],
        active_report_version_id: 'v1',
      })
      renderStreaming({ state, disabled: true, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'generating' })

      expect(screen.getByTestId('report-version-selector')).toBeDisabled()
      expect(screen.getByTestId('report-template-option-analytical')).toBeDisabled()
      expect(screen.getByTestId('refine-once-toggle')).toBeDisabled()
      expect(screen.getByTestId('export-menu-trigger')).toBeDisabled()
    })

    it('re-enables controls once regeneration is no longer active', () => {
      const state = baseState({ report: reportStub() })
      const { rerender } = render(
        <ReportModePanel
          state={state} disabled onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="regenerate" reportStreamPhase="saving"
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      expect(screen.getByTestId('export-menu-trigger')).toBeDisabled()

      rerender(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(screen.getByTestId('export-menu-trigger')).toBeEnabled()
      expect(screen.getByTestId('regenerate-report')).toBeInTheDocument()
    })

    it('preserves existing badges, evaluation details, and references while regenerating', () => {
      const state = baseState({
        report: {
          ...reportStub(),
          references: [{ number: 1, kind: 'paper', title: 'Paper One', formatted: 'A. Uthor (2024). Paper One.', paper_id: 'p1', link_url: null }],
        },
      })
      renderStreaming({ state, reportStreamActive: true, reportStreamOperation: 'regenerate', reportStreamPhase: 'evaluating' })

      expect(screen.getByTestId('report-template-badge')).toBeInTheDocument()
      expect(screen.getByTestId('report-refinement-badge')).toBeInTheDocument()
      expect(screen.getByTestId('evaluation-details-toggle')).toBeInTheDocument()
      expect(screen.getByTestId('report-references')).toBeInTheDocument()
    })

    it('shows a safe error message inline in the header area on a handled failure', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state, reportStreamError: 'Failed to save the report.' })

      expect(screen.getByTestId('report-stream-error')).toHaveTextContent('Failed to save the report.')
    })

    it('shows a sync-retry notice without disturbing the visible report', () => {
      const state = baseState({ report: reportStub() })
      const { onRetryReportSync } = renderStreaming({ state, reportStreamSyncFailed: true })

      expect(screen.getByTestId('report-stream-sync-failed')).toBeInTheDocument()
      expect(screen.getByText('F')).toBeInTheDocument()
      expect(onRetryReportSync).not.toHaveBeenCalled()
    })

    it('all four phase labels map to their own concise text, using regenerate-specific wording for "generating"', () => {
      const state = baseState({ report: reportStub() })
      // report-progress-observability: "generating" now reads "Regenerating
      // report" for a regenerate operation (distinct from "Generating
      // report" for a fresh generate) -- see the empty-view case below for
      // the generate-operation wording.
      const cases: Array<[string, string]> = [
        ['generating', 'Regenerating report'],
        ['evaluating', 'Evaluating draft'],
        ['revising', 'Revising report'],
        ['saving', 'Saving report'],
      ]
      for (const [phase, label] of cases) {
        const { unmount } = render(
          <ReportModePanel
            state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
            exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="regenerate" reportStreamPhase={phase as never}
            reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
            onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
          />,
        )
        expect(screen.getByTestId('report-stream-phase-label')).toHaveTextContent(label)
        unmount()
      }
    })

    it('UXH.3: focus returns to Regenerate once a completed/cancelled stream removes Stop, when Stop had held focus', () => {
      const state = baseState({ report: reportStub() })
      const { rerender } = render(
        <ReportModePanel
          state={state} disabled onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="regenerate" reportStreamPhase="generating"
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      screen.getByTestId('report-stream-stop').focus()
      expect(document.activeElement).toBe(screen.getByTestId('report-stream-stop'))

      rerender(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(document.activeElement).toBe(screen.getByTestId('regenerate-report'))
    })

    it('UXH.3: does not steal focus from a control the user deliberately focused before the stream settled', () => {
      const state = baseState({ report: reportStub() })
      // disabled=false here is a component-level-only combination (the
      // real app always disables every other control while a report
      // stream is active) -- used purely to exercise "the user focused
      // something else and it stayed focusable," which this component's
      // own effect must still respect regardless of what the real
      // disabled-prop composition happens to be upstream.
      const { rerender } = render(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive reportStreamOperation="regenerate" reportStreamPhase="generating"
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      screen.getByTestId('report-template-option-foundational').focus()
      expect(document.activeElement).toBe(screen.getByTestId('report-template-option-foundational'))

      rerender(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(document.activeElement).toBe(screen.getByTestId('report-template-option-foundational'))
    })

    it('UXH.3: an ordinary re-render with no stream ever active never moves focus', () => {
      const state = baseState({ report: reportStub() })
      const { rerender } = render(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      screen.getByTestId('report-template-option-expert').focus()

      // A re-render triggered by something unrelated (e.g. a version list
      // update) -- reportStreamActive stays false throughout, so the
      // focus effect's own true -> false edge never fires.
      rerender(
        <ReportModePanel
          state={{ ...state, report_versions: [{ version_id: 'v1', version_number: 1, created_at: null, report_template: 'analytical', generation_reason: 'initial', is_active: true }] }}
          disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      expect(document.activeElement).toBe(screen.getByTestId('report-template-option-expert'))
    })
  })

  describe('report-progress-observability: accumulated phase trail + completion notice', () => {
    it('empty (Generate) view: renders one row per observed phase, checks for completed ones, spinner for the active one', () => {
      renderStreaming({
        reportStreamActive: true, reportStreamOperation: 'generate',
        reportStreamPhase: 'evaluating', reportStreamPhaseHistory: ['generating', 'evaluating'],
      })

      const trail = screen.getByTestId('report-stream-phase-label')
      expect(trail).toHaveTextContent('Report generated')
      expect(trail).toHaveTextContent('Evaluating draft')
      // Only two rows -- 'saving'/'revising' never happened, so they never render.
      expect(screen.queryByTestId('report-stream-phase-row-saving')).not.toBeInTheDocument()
      expect(screen.queryByTestId('report-stream-phase-row-revising')).not.toBeInTheDocument()
    })

    it('Generate uses "Generating report"/"Report generated"', () => {
      renderStreaming({
        reportStreamActive: true, reportStreamOperation: 'generate',
        reportStreamPhase: 'saving', reportStreamPhaseHistory: ['generating', 'saving'],
      })
      const trail = screen.getByTestId('report-stream-phase-label')
      expect(trail).toHaveTextContent('Report generated')
      expect(trail).not.toHaveTextContent('Report regenerated')
    })

    it('Regenerate uses "Regenerating report"/"Report regenerated"', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({
        state, reportStreamActive: true, reportStreamOperation: 'regenerate',
        reportStreamPhase: 'saving', reportStreamPhaseHistory: ['generating', 'saving'],
      })
      const trail = screen.getByTestId('report-stream-phase-label')
      expect(trail).toHaveTextContent('Report regenerated')
    })

    it('a completed phase shows a check icon; the active (last) phase shows a spinner, never both for the same row', () => {
      renderStreaming({
        reportStreamActive: true, reportStreamOperation: 'generate',
        reportStreamPhase: 'saving', reportStreamPhaseHistory: ['generating', 'evaluating', 'saving'],
      })

      const generatingRow = screen.getByTestId('report-stream-phase-row-generating')
      const evaluatingRow = screen.getByTestId('report-stream-phase-row-evaluating')
      const savingRow = screen.getByTestId('report-stream-phase-row-saving')

      expect(generatingRow.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument()
      expect(generatingRow).toHaveTextContent('Report generated')
      expect(evaluatingRow).toHaveTextContent('Draft evaluated')
      // The active row shows an animated spinner, distinguishing it from
      // the completed rows above (jsdom doesn't compute a real spin
      // animation, so the class name itself is the observable signal).
      expect(savingRow.querySelector('.animate-spin')).toBeInTheDocument()
      expect(generatingRow.querySelector('.animate-spin')).not.toBeInTheDocument()
      expect(evaluatingRow.querySelector('.animate-spin')).not.toBeInTheDocument()
    })

    it('never renders a "Revised"/revising row unless a revising event was actually received', () => {
      renderStreaming({
        reportStreamActive: true, reportStreamOperation: 'generate',
        reportStreamPhase: 'saving', reportStreamPhaseHistory: ['generating', 'evaluating', 'saving'],
      })

      expect(screen.queryByTestId('report-stream-phase-row-revising')).not.toBeInTheDocument()
      expect(screen.queryByText(/revised/i)).not.toBeInTheDocument()
    })

    it('renders a revising row, in order, once a revising event was received', () => {
      renderStreaming({
        reportStreamActive: true, reportStreamOperation: 'generate',
        reportStreamPhase: 'saving', reportStreamPhaseHistory: ['generating', 'evaluating', 'revising', 'saving'],
      })

      const rows = screen.getAllByTestId(/report-stream-phase-row-/).map((el) => el.dataset.testid)
      expect(rows).toEqual([
        'report-stream-phase-row-generating', 'report-stream-phase-row-evaluating',
        'report-stream-phase-row-revising', 'report-stream-phase-row-saving',
      ])
      expect(screen.getByTestId('report-stream-phase-row-revising')).toHaveTextContent('Report revised')
    })

    it('regeneration header: exactly one live region, whether showing the trail or nothing at all', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({
        state, reportStreamActive: true, reportStreamOperation: 'regenerate',
        reportStreamPhase: 'evaluating', reportStreamPhaseHistory: ['generating', 'evaluating'],
      })

      expect(screen.getAllByTestId('report-stream-phase-label')).toHaveLength(1)
      const region = screen.getByTestId('report-stream-phase-label')
      expect(region).toHaveAttribute('role', 'status')
      expect(region).toHaveAttribute('aria-live', 'polite')
    })

    it('regeneration header: shows the completion notice once the stream is no longer active', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({
        state, reportStreamActive: false, reportStreamOperation: null,
        reportStreamCompletionNotice: { operation: 'regenerate', phases: ['generating', 'evaluating', 'saving'] },
      })

      const notice = screen.getByTestId('report-stream-completion-notice')
      expect(notice).toHaveTextContent('Report regenerated · Evaluated · Saved')
      expect(notice).toHaveAttribute('role', 'status')
      expect(notice).toHaveAttribute('aria-live', 'polite')
      // The trail and the notice are never shown at the same time.
      expect(screen.queryByTestId('report-stream-phase-label')).not.toBeInTheDocument()
    })

    it('the notice text is derived only from the operation and phases actually received (generate, no evaluation)', () => {
      // A generate-operation notice only ever shows once state.report is
      // populated (the reload that precedes it just set it) -- same
      // "with report" header the regenerate case renders in.
      const state = baseState({ report: reportStub() })
      renderStreaming({
        state, reportStreamActive: false, reportStreamOperation: null,
        reportStreamCompletionNotice: { operation: 'generate', phases: ['generating', 'saving'] },
      })

      expect(screen.getByTestId('report-stream-completion-notice')).toHaveTextContent('Report generated · Saved')
    })

    it('the notice text includes "Revised" only when a revising event was part of the completed turn', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({
        state, reportStreamActive: false, reportStreamOperation: null,
        reportStreamCompletionNotice: { operation: 'regenerate', phases: ['generating', 'evaluating', 'revising', 'saving'] },
      })

      expect(screen.getByTestId('report-stream-completion-notice')).toHaveTextContent(
        'Report regenerated · Evaluated · Revised · Saved',
      )
    })

    it('a still-showing notice disappears once the hook clears it (simulated via re-render, under fake timers)', () => {
      vi.useFakeTimers()
      const state = baseState({ report: reportStub() })
      const { rerender } = render(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          reportStreamCompletionNotice={{ operation: 'regenerate', phases: ['generating', 'saving'] }}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )
      expect(screen.getByTestId('report-stream-completion-notice')).toBeInTheDocument()

      // The real 5s auto-clear timer lives in useCurationSession.ts (see
      // its own REPORT_STREAM_SUCCESS_NOTICE_MS-driven test) -- this panel
      // holds no timer of its own, only whatever prop it's given. Advancing
      // fake time here is a no-op for THIS component; what actually flips
      // the prop, exactly as the real hook eventually does, is this re-
      // render passing reportStreamCompletionNotice={null}.
      vi.advanceTimersByTime(5000)
      rerender(
        <ReportModePanel
          state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
          exportUrls={EXPORT_URLS} reportStreamActive={false} reportStreamOperation={null} reportStreamPhase={null}
          reportStreamStopping={false} reportStreamError={null} reportStreamSyncFailed={false}
          reportStreamCompletionNotice={null}
          onCancelReportStream={vi.fn()} onRetryReportSync={vi.fn()}
        />,
      )

      // No permanent empty status container left behind in its place.
      expect(screen.queryByTestId('report-stream-completion-notice')).not.toBeInTheDocument()
      expect(screen.queryByTestId('report-stream-phase-label')).not.toBeInTheDocument()
      vi.useRealTimers()
    })

    it('no permanent empty status container renders when neither a stream nor a notice is active', () => {
      const state = baseState({ report: reportStub() })
      renderStreaming({ state })

      expect(screen.queryByTestId('report-stream-phase-label')).not.toBeInTheDocument()
      expect(screen.queryByTestId('report-stream-completion-notice')).not.toBeInTheDocument()
    })

    it('falls back to a single-row trail from reportStreamPhase alone when reportStreamPhaseHistory is omitted (back-compat)', () => {
      renderStreaming({ reportStreamActive: true, reportStreamOperation: 'generate', reportStreamPhase: 'evaluating' })

      const trail = screen.getByTestId('report-stream-phase-label')
      expect(trail).toHaveTextContent('Evaluating draft')
      expect(screen.queryByTestId('report-stream-phase-row-generating')).not.toBeInTheDocument()
    })
  })
})
