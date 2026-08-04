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
    report_versions: [], active_report_version_id: null, chat_references: [],
    ...overrides,
  }
}

describe('ReportModePanel', () => {
  it('no report yet: shows a Generate report CTA, not empty sections -- this is the fix for "no way to see the report"', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(
      <ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />,
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />),
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
      render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />),
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
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.getByTestId('report-template-option-foundational')).toBeInTheDocument()
    expect(screen.getByTestId('report-template-option-analytical')).toBeInTheDocument()
    expect(screen.getByTestId('report-template-option-expert')).toBeInTheDocument()
  })

  it('defaults the selector to Analytical when there is no report yet', () => {
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByTestId('report-template-option-foundational')).toHaveAttribute('aria-checked', 'false')
  })

  it('initializes the selector from an existing report_template', () => {
    const state = baseState({ report: reportWithTemplate('expert') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.getByTestId('report-template-option-expert')).toHaveAttribute('aria-checked', 'true')
  })

  it('selecting Foundational and clicking Generate calls onGenerateReport("foundational")', async () => {
    const user = userEvent.setup()
    const onGenerateReport = vi.fn()
    render(<ReportModePanel state={baseState()} disabled={false} onGenerateReport={onGenerateReport} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    await user.click(screen.getByTestId('report-template-option-foundational'))
    await user.click(screen.getByTestId('generate-report'))

    expect(onGenerateReport).toHaveBeenCalledWith('foundational')
  })

  it('selecting Expert and clicking Regenerate calls onRegenerateReport("expert")', async () => {
    const user = userEvent.setup()
    const onRegenerateReport = vi.fn()
    const state = baseState({ report: reportWithTemplate('analytical') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={onRegenerateReport} onActivateReportVersion={vi.fn()} />)

    await user.click(screen.getByTestId('report-template-option-expert'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('expert')
  })

  it('shows a template badge for an existing report', () => {
    const state = baseState({ report: reportWithTemplate('foundational') })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.getByTestId('report-template-badge')).toHaveTextContent('Foundational')
  })

  it('defaults the badge and selector to Analytical for an old report with no report_template field', () => {
    const state = baseState({ report: reportWithTemplate() })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.getByTestId('report-template-badge')).toHaveTextContent('Analytical')
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')
  })

  it('syncs the selector when the active report changes to a different template', () => {
    const { rerender } = render(
      <ReportModePanel
        state={baseState({ report: reportWithTemplate('analytical') })}
        disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
      />,
    )
    expect(screen.getByTestId('report-template-option-analytical')).toHaveAttribute('aria-checked', 'true')

    rerender(
      <ReportModePanel
        state={baseState({ report: reportWithTemplate('expert') })}
        disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()}
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
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.queryByTestId('report-version-selector')).not.toBeInTheDocument()
    // Rest of the panel still renders safely without version metadata.
    expect(screen.getByText('F')).toBeInTheDocument()
  })

  it('renders one option per available version', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    const select = screen.getByTestId('report-version-selector') as HTMLSelectElement
    expect(select.options).toHaveLength(3)
  })

  it('selects the option matching active_report_version_id', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v2',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    const select = screen.getByTestId('report-version-selector') as HTMLSelectElement
    expect(select.value).toBe('v2')
  })

  it('labels include the version number, template, and generation reason', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={false} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

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
        onActivateReportVersion={onActivateReportVersion}
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
        onActivateReportVersion={vi.fn()}
      />,
    )

    await user.click(screen.getByTestId('report-template-option-expert'))
    await user.click(screen.getByTestId('regenerate-report'))

    expect(onRegenerateReport).toHaveBeenCalledWith('expert')
  })

  it('is disabled while a report action is in progress, same as the other report controls', () => {
    const state = baseState({
      report: reportStub(), report_versions: [V1, V2, V3], active_report_version_id: 'v3',
    })
    render(<ReportModePanel state={state} disabled={true} onGenerateReport={vi.fn()} onRegenerateReport={vi.fn()} onActivateReportVersion={vi.fn()} />)

    expect(screen.getByTestId('report-version-selector')).toBeDisabled()
  })
})
