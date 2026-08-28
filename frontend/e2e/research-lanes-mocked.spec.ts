import { test, expect, type Page, type Route } from '@playwright/test'

/**
 * RL6 Part B -- zero-cost, fully MOCKED browser journey for Research Lanes.
 *
 * Every backend response is intercepted with page.route(). No FastAPI
 * backend runs; no network or provider call is ever made. Validates the
 * end-to-end UI journey at desktop and narrow-mobile viewports (see the
 * two Playwright projects in playwright.mocked.config.ts).
 */

const LANE_A = { lane_id: 'srv-la', label: 'Retrieval architectures', question: 'Which retrieval designs cut hallucination?', query: 'retrieval augmented generation architectures', enabled: true, origin: 'suggested', generation_version: 1 }
const LANE_B = { lane_id: 'srv-lb', label: 'Evaluation of factuality', question: 'How is factual grounding measured?', query: 'measuring factual grounding in RAG', enabled: true, origin: 'suggested', generation_version: 1 }
const LANE_C = { lane_id: 'srv-lc', label: 'Failure modes', question: 'When does grounding still fail?', query: 'failure modes of RAG faithfulness', enabled: true, origin: 'suggested', generation_version: 1 }

function paper(id: string, title: string) {
  return {
    paper_id: id, title, authors: ['A. Author'], year: 2024, venue: 'arXiv', abstract: `Abstract for ${title}.`,
    url: null, doi: null, citation_count: 5, source: 'arxiv', source_urls: {}, score: 0.9, keywords: ['rag', 'grounding'],
  }
}

interface MockConfig {
  capabilityEnabled: boolean
  suggestDelayMs: number
  suggestStatus: number
  startStatus: number
  laneState: boolean // whether GET /curation/:id returns a lane session
}

const LANE_STATE = {
  session_id: 'sess-e2e', topic: 'rl6 e2e disposable topic', display_title: 'RL6 E2E Disposable Topic',
  stage: 'curate', target_count: 10, selected_paper_ids: [], selected_papers: [],
  pending_batch: [paper('p1', 'Dense Passage Retrieval'), paper('p2', 'Grounded Answer Evaluation')],
  refilled: false, reserve_remaining: 8, refinement_notes: [], report: null, chat_history: [],
  web_articles_added: [], pending_web_offer: null, pending_report_update: null,
  turn_history: [
    {
      turn_number: 1, refilled: false,
      batch: [paper('p1', 'Dense Passage Retrieval'), paper('p2', 'Grounded Answer Evaluation')],
      // FROZEN provenance -- deliberately NARROWER for p1 than the cumulative map below.
      paper_lane_ids: { p1: ['srv-la'], p2: ['srv-lb'] },
    },
  ],
  stop_reason: null, report_versions: [], active_report_version_id: null, chat_references: [],
  lanes: [LANE_A, LANE_B],
  // CUMULATIVE provenance -- p1 was later re-discovered via lane B too.
  paper_lane_ids: { p1: ['srv-la', 'srv-lb'], p2: ['srv-lb'] },
  lane_result_counts: { 'srv-la': 1, 'srv-lb': 2 },
}

const SINGLE_STATE = {
  ...LANE_STATE,
  turn_history: [{ turn_number: 1, refilled: false, batch: [paper('p1', 'Dense Passage Retrieval')] }],
  pending_batch: [paper('p1', 'Dense Passage Retrieval')],
  lanes: [], paper_lane_ids: {}, lane_result_counts: {},
}

async function installRoutes(page: Page, cfg: MockConfig) {
  await page.route('**/curation/capabilities', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ research_lanes_enabled: cfg.capabilityEnabled }) }),
  )
  await page.route('**/curation/reviews', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' }),
  )
  await page.route('**/curation/lanes/suggest', async (route: Route) => {
    if (cfg.suggestDelayMs) await new Promise((r) => setTimeout(r, cfg.suggestDelayMs))
    if (cfg.suggestStatus !== 200) {
      return route.fulfill({ status: cfg.suggestStatus, contentType: 'application/json', body: JSON.stringify({ detail: { error: 'curation_lane_suggest service unavailable' } }) })
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ lanes: [LANE_A, LANE_B, LANE_C] }) })
  })
  await page.route('**/curation/start', async (route: Route) => {
    if (cfg.startStatus !== 200) {
      return route.fulfill({ status: cfg.startStatus, contentType: 'application/json', body: JSON.stringify({ detail: 'Research lane service is temporarily unavailable.' }) })
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ session_id: 'sess-e2e', stage: 'curate', target_count: 10, selected_paper_ids: [], batch: [], stop_reason: null, refilled: false, reserve_remaining: 8, refinement_notes: [] }) })
  })
  await page.route(/\/curation\/sess-e2e(\?.*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(cfg.laneState ? LANE_STATE : SINGLE_STATE) }),
  )
}

function baseCfg(over: Partial<MockConfig> = {}): MockConfig {
  return { capabilityEnabled: true, suggestDelayMs: 0, suggestStatus: 200, startStatus: 200, laneState: true, ...over }
}

// Collects genuine app-level console errors and uncaught exceptions.
// "Failed to load resource" lines are the browser logging an HTTP status
// (this suite deliberately mocks some 4xx/5xx responses to exercise error
// UI) -- not an application error, so they are filtered out.
async function collectConsoleErrors(page: Page): Promise<string[]> {
  const errors: string[] = []
  page.on('console', (m) => {
    if (m.type() !== 'error') return
    const t = m.text()
    if (/Failed to load resource|favicon\.ico/i.test(t)) return
    errors.push(t)
  })
  page.on('pageerror', (e) => errors.push(String(e)))
  return errors
}

async function expectNoHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
}

async function expectNoNestedButtons(page: Page) {
  const nested = await page.locator('button button').count()
  expect(nested).toBe(0)
}

test.describe('Research Lanes -- mocked browser journey', () => {
  test('1. capability OFF -> only Single search, no lane affordances', async ({ page }) => {
    const errors = await collectConsoleErrors(page)
    await installRoutes(page, baseCfg({ capabilityEnabled: false }))
    await page.goto('/')
    await page.getByTestId('new-review-trigger').click()
    await expect(page.getByTestId('new-review-topic')).toBeVisible()
    await expect(page.getByTestId('new-review-mode-single')).toHaveCount(0)
    await expect(page.getByTestId('new-review-mode-lanes')).toHaveCount(0)
    await expect(page.getByTestId('new-review-start')).toHaveText('Start')
    await expectNoHorizontalOverflow(page)
    expect(errors).toEqual([])
  })

  test('2-6. capability ON: suggest states, edit/enable/add/remove/max-4, switch-to-single, failed start restores draft', async ({ page }) => {
    const errors = await collectConsoleErrors(page)
    const cfg = baseCfg()
    await installRoutes(page, cfg)
    await page.goto('/')
    await page.getByTestId('new-review-trigger').click()

    // 2. Research lanes segmented control is available.
    await expect(page.getByTestId('new-review-mode-lanes')).toBeVisible()
    await page.getByTestId('new-review-mode-lanes').click()
    await page.getByTestId('new-review-topic').fill('reducing hallucination in retrieval augmented generation')

    // 3a. Suggest -> loading status, then success -> 3 divided rows.
    cfg.suggestDelayMs = 400
    await page.getByTestId('new-review-suggest-lanes').click()
    await expect(page.getByTestId('lane-suggestion-status')).toHaveText(/Designing research lanes/)
    await expect(page.getByTestId('lane-row-0')).toBeVisible()
    await expect(page.getByTestId('lane-row-2')).toBeVisible()
    await expect(page.getByTestId('lane-label-0')).toHaveValue('Retrieval architectures')
    cfg.suggestDelayMs = 0

    // 4. Edit label, toggle enabled, add up to max four, remove.
    await page.getByTestId('lane-label-0').fill('Hybrid retrieval')
    await page.getByTestId('lane-enabled-1').uncheck()
    await expect(page.getByTestId('lane-enabled-1')).not.toBeChecked()
    await page.getByTestId('lane-add').click()
    await expect(page.getByTestId('lane-row-3')).toBeVisible()
    await expect(page.getByTestId('lane-add')).toBeDisabled() // max four
    await page.getByTestId('lane-remove-3').click()
    await expect(page.getByTestId('lane-row-3')).toHaveCount(0)
    await expect(page.getByTestId('lane-add')).toBeEnabled()

    // 5. Switch to Single search -> lane draft invalidated; back to lanes -> no rows.
    await page.getByTestId('new-review-mode-single').click()
    await page.getByTestId('new-review-mode-lanes').click()
    await expect(page.getByTestId('lane-row-0')).toHaveCount(0)

    // 3b. Suggestion error -> safe inline message, topic preserved, no raw JSON.
    cfg.suggestStatus = 503
    await page.getByTestId('new-review-suggest-lanes').click()
    await expect(page.getByTestId('lane-suggestion-error')).toBeVisible()
    const errText = await page.getByTestId('lane-suggestion-error').textContent()
    expect(errText).not.toMatch(/[{}[\]]/)
    await expect(page.getByTestId('new-review-topic')).toHaveValue('reducing hallucination in retrieval augmented generation')
    cfg.suggestStatus = 200

    // 6. Build a lane, fail the start, confirm the exact draft is restored.
    await page.getByTestId('new-review-suggest-lanes').click()
    await expect(page.getByTestId('lane-row-0')).toBeVisible()
    await page.getByTestId('lane-label-0').fill('My durable label')
    await page.getByTestId('lane-question-0').fill('does it converge?')
    await page.getByTestId('lane-query-0').fill('surface code threshold theorem')
    await page.getByTestId('lane-enabled-2').uncheck() // keep exactly two enabled
    cfg.startStatus = 503
    await page.getByTestId('new-review-start').click()
    await expect(page.getByTestId('curation-error-banner')).toBeVisible()
    await expect(page.getByTestId('lane-label-0')).toHaveValue('My durable label')
    await expect(page.getByTestId('lane-question-0')).toHaveValue('does it converge?')
    await expect(page.getByTestId('lane-query-0')).toHaveValue('surface code threshold theorem')
    await expect(page.getByTestId('lane-enabled-2')).not.toBeChecked()

    await expectNoHorizontalOverflow(page)
    await expectNoNestedButtons(page)
    expect(errors).toEqual([])
  })

  test('7-9. successful start -> read-only lane summary, Found-via chips, frozen provenance in Browse Past Turns', async ({ page }, testInfo) => {
    // The active-review workspace keeps a fixed 288px review sidebar next
    // to the main panel -- a pre-existing whole-app shell constraint, not
    // specific to Research Lanes -- so it is exercised at the desktop
    // width. The New Review lane editor (tests 1-6, 10) lives in that
    // sidebar and IS validated at 375px.
    test.skip(testInfo.project.name === 'mobile', 'active-review workspace needs the desktop layout (fixed sidebar)')
    const errors = await collectConsoleErrors(page)
    const cfg = baseCfg()
    await installRoutes(page, cfg)
    await page.goto('/')
    await page.getByTestId('new-review-trigger').click()
    await page.getByTestId('new-review-mode-lanes').click()
    await page.getByTestId('new-review-topic').fill('rl6 e2e disposable topic')
    await page.getByTestId('new-review-suggest-lanes').click()
    await expect(page.getByTestId('lane-row-0')).toBeVisible()
    await page.getByTestId('lane-enabled-2').uncheck() // exactly two enabled
    await page.getByTestId('new-review-start').click()

    // 7. Canonical, read-only lane summary. Two lanes enabled -> "2 active".
    await expect(page.getByTestId('lane-summary-toggle')).toHaveText(/Research lanes · 2 active/)
    await page.getByTestId('lane-summary-toggle').click()
    await expect(page.getByTestId('lane-summary-panel')).toBeVisible()
    await expect(page.getByTestId('lane-summary-item-srv-la')).toContainText('Retrieval architectures')
    await expect(page.getByTestId('lane-summary-item-srv-la')).toContainText('1 found')
    await expect(page.getByTestId('lane-summary-item-srv-lb')).toContainText('2 found')
    // No editable field anywhere in the summary.
    await expect(page.getByTestId('lane-summary-panel').locator('input')).toHaveCount(0)

    // 8. "Found via" chips use the CUMULATIVE map for the live batch.
    await expect(page.getByTestId('paper-lanes-p1')).toContainText('Retrieval architectures')
    await expect(page.getByTestId('paper-lanes-p1')).toContainText('Evaluation of factuality')
    await expect(page.getByTestId('paper-lanes-p2')).toContainText('Evaluation of factuality')
    await expect(page.getByTestId('paper-lanes-p2')).not.toContainText('Retrieval architectures')

    // 9. Browse Past Turns uses the turn's FROZEN provenance (narrower for p1).
    await page.getByTestId('open-turn-history').click()
    await expect(page.getByTestId('paper-lanes-p1')).toContainText('Retrieval architectures')
    await expect(page.getByTestId('paper-lanes-p1')).not.toContainText('Evaluation of factuality')

    await expectNoHorizontalOverflow(page)
    await expectNoNestedButtons(page)
    expect(errors).toEqual([])
  })

  test('10. single-query session renders no lane UI', async ({ page }) => {
    const errors = await collectConsoleErrors(page)
    await installRoutes(page, baseCfg({ laneState: false }))
    await page.goto('/?session=sess-e2e')
    await expect(page.getByTestId('paper-card-p1')).toBeVisible()
    await expect(page.getByTestId('lane-summary-toggle')).toHaveCount(0)
    await expect(page.locator('[data-testid^="paper-lanes-"]')).toHaveCount(0)
    await expectNoHorizontalOverflow(page)
    await expectNoNestedButtons(page)
    expect(errors).toEqual([])
  })
})
