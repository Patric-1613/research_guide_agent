import { fireEvent, render, screen } from '@testing-library/react'
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
    report_versions: [], active_report_version_id: null, chat_references: [],
    ...overrides,
  }
}

describe('ReportModePanel', () => {
  it('no report yet: shows a Generate report CTA, not empty sections -- this is the fix for "no way to see the report"', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(
      <ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />,
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />),
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />),
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
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('report-template-option-foundational')).toBeInTheDocument()
    expect(screen.getByTestId('report-template-option-analytical')).toBeInTheDocument()
    expect(screen.getByTestId('report-template-option-expert')).toBeInTheDocument()
  })

  it('defaults the selector to Analytical when there is no report yet', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByTestId('report-template-option-foundational')).toHaveAttribute('aria-checked', 'false')
  })

  it('initializes the selector from an existing report_template', () => {
    const state = baseState({ report: reportWithTemplate('expert') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('report-template-option-expert')).toHaveAttribute('aria-checked', 'true')
  })

  it('selecting Foundational and clicking Generate calls onGenerateReport("foundational")', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    await user.click(screen.getByTestId('report-template-option-foundational'))
    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('foundational', 'off')
  })

  it('selecting Expert and clicking Regenerate calls onRegenerateReport("expert")', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({ report: reportWithTemplate('analytical') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    await user.click(screen.getByTestId('report-template-option-expert'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('expert', 'off')
  })

  it('shows a template badge for an existing report', () => {
    const state = baseState({ report: reportWithTemplate('foundational') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('report-template-badge')).toHaveTextContent('Foundational')
  })

  it('defaults the badge and selector to Analytical for an old report with no report_template field', () => {
    const state = baseState({ report: reportWithTemplate() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('report-template-badge')).toHaveTextContent('Analytical')
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')
  })

  it('syncs the selector when the active report changes to a different template', () => {
    const { rerender } = render(
      <ReportModePanel
        state={baseState({ report: reportWithTemplate('analytical') })}
        disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown"
      />,
    )
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')

    rerender(
      <ReportModePanel
        state={baseState({ report: reportWithTemplate('expert') })}
        disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown"
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.queryByTestId('report-version-selector')).not.toBeInTheDocument()
    // Rest of the panel still renders safely without version metadata.
    expect(screen.getByText('F')).toBeInTheDocument()
  })

  it('renders one option per available version', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    const select = screen.getByTestId('report-version-selector') as HTMLSelectElement
    expect(select.options).toHaveLength(3)
  })

  it('selects the option matching active_report_version_id', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v2',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    const select = screen.getByTestId('report-version-selector') as HTMLSelectElement
    expect(select.value).toBe('v2')
  })

  it('labels include the version number, template, and generation reason', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
        onActivateReportVersion={onActivateReportVersion} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown"
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
        onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown"
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
    render(<ReportModePanel state={state} disabled={true} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('refine-once-toggle')).toBeInTheDocument()
    expect(screen.getByText('Refine once')).toBeInTheDocument()
  })

  it('renders the "Refine once" toggle once a report exists', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('refine-once-toggle')).toBeInTheDocument()
  })

  it('the toggle defaults to off (unchecked)', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('refine-once-toggle')).not.toBeChecked()
  })

  it('clicking Generate with the toggle off calls onGenerateReport with "off"', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('analytical', 'off')
  })

  it('checking the toggle then clicking Generate calls onGenerateReport(template, "single")', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    await user.click(screen.getByTestId('refine-once-toggle'))
    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('analytical', 'single')
  })

  it('checking the toggle then clicking Regenerate calls onRegenerateReport(template, "single")', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    await user.click(screen.getByTestId('refine-once-toggle'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('analytical', 'single')
  })

  it('the toggle is disabled while a report action is in progress', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={true} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    expect(screen.getByTestId('report-refinement-badge')).toHaveTextContent('Evaluated · score 88')
  })

  it('does not render a refinement badge when the report has no refinement metadata', () => {
    const state = baseState({ report: reportStub() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} exportMarkdownUrl="http://test.local/curation/s1/report/export?format=markdown" />)

    await user.click(screen.getByTestId('evaluation-details-toggle'))

    expect(screen.queryByText(/REWRITE THE INTRODUCTION/i)).not.toBeInTheDocument()
  })
})

describe('ReportModePanel -- report-quality Phase R5B: Export Markdown link', () => {
  function reportStub() {
    return {
      findings: { content: 'F', cited_papers: [], cited_web_articles: [] },
      limitations: { content: 'L', cited_papers: [], cited_web_articles: [] },
      future_scope: { content: 'S', cited_papers: [], cited_web_articles: [] },
      skipped_paper_ids: [],
    }
  }

  const EXPORT_URL = 'http://test.local/curation/s1/report/export?format=markdown'

  it('does not render the Export link when there is no report yet', () => {
    render(
      <ReportModePanel
        state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportMarkdownUrl={EXPORT_URL}
      />,
    )

    expect(screen.queryByTestId('export-markdown')).not.toBeInTheDocument()
  })

  it('renders the Export link once a report exists', () => {
    const state = baseState({ report: reportStub() })
    render(
      <ReportModePanel
        state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportMarkdownUrl={EXPORT_URL}
      />,
    )

    expect(screen.getByTestId('export-markdown')).toHaveTextContent('Export Markdown')
  })

  it('the Export link points to the exportMarkdownUrl prop it was given', () => {
    const state = baseState({ report: reportStub() })
    render(
      <ReportModePanel
        state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportMarkdownUrl={EXPORT_URL}
      />,
    )

    expect(screen.getByTestId('export-markdown')).toHaveAttribute('href', EXPORT_URL)
  })

  it('the Export link has a download attribute so the browser downloads rather than navigates', () => {
    const state = baseState({ report: reportStub() })
    render(
      <ReportModePanel
        state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportMarkdownUrl={EXPORT_URL}
      />,
    )

    expect(screen.getByTestId('export-markdown')).toHaveAttribute('download')
  })

  it('the Export link is suppressed (aria-disabled, click prevented) while a report action is in progress', () => {
    const state = baseState({ report: reportStub() })
    render(
      <ReportModePanel
        state={state} disabled={true} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportMarkdownUrl={EXPORT_URL}
      />,
    )

    const link = screen.getByTestId('export-markdown')
    expect(link).toHaveAttribute('aria-disabled', 'true')

    // <a> has no native disabled attribute -- the real guarantee is that
    // the click handler calls preventDefault(). fireEvent.click() returns
    // false exactly when the dispatched event's default action was
    // prevented (per the DOM spec), which is the precise thing to assert
    // here rather than something indirect like "no navigation happened"
    // (jsdom doesn't navigate on <a> clicks at all, disabled or not, so
    // that wouldn't actually prove anything).
    const notPrevented = fireEvent.click(link)
    expect(notPrevented).toBe(false)
  })

  it('the Export link is not suppressed when disabled is false', () => {
    const state = baseState({ report: reportStub() })
    render(
      <ReportModePanel
        state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()}
        onActivateReportVersion={vi.fn()} exportMarkdownUrl={EXPORT_URL}
      />,
    )

    expect(screen.getByTestId('export-markdown')).toHaveAttribute('aria-disabled', 'false')
    // Not asserting a real fireEvent.click() here -- jsdom attempts an
    // actual (unsupported) navigation for a non-prevented <a href>
    // click, which only adds console noise. The disabled-case test
    // above already proves the meaningful contrast (its click IS
    // prevented); an unprevented click is simply the default behavior
    // of an <a> with no disabling guard, not something that needs its
    // own separate proof.
  })
})
