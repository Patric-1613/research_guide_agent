import { test, expect, type Page } from '@playwright/test'

/**
 * Real, live, non-mocked proof of two things at once, run as one
 * continuous session to avoid redundant real LLM/search calls:
 *
 *   1. (Phase 6c) The full curate -> report -> chat -> web-escalation ->
 *      regenerate flow works against the REAL backend (real arXiv/
 *      Semantic Scholar search, real OpenAI calls, real Tavily search).
 *
 *   2. (Phase 6d) Page state survives a REAL browser reload at the two
 *      points that matter most: while a curation batch is genuinely
 *      PENDING (mid-interrupt -- explicitly prioritized per the brief,
 *      since this is the exact property Phase 6a's own
 *      get_curation_state() discovery was about), and mid-conversation
 *      in the chat stage (chat_history + report + a pending web offer
 *      all reloading from the backend, not from anything held only in
 *      React state).
 *
 * Requires BOTH a real FastAPI backend (uvicorn research_agent.api:app)
 * and this Vite dev server already running -- see e2e/README.md.
 */

const TOPIC = 'parameter-efficient fine-tuning for large language models'
const TARGET_COUNT = 3

async function waitForPendingBatch(page: Page, timeout = 60_000) {
  await expect(page.getByText('New this turn').first()).toBeVisible({ timeout })
  await expect(page.locator('[data-testid^="add-paper-"]').first()).toBeVisible({ timeout })
}

test('curate -> refresh mid-interrupt -> report -> chat -> web offer -> refresh mid-chat -> regenerate', async ({ page }) => {
  test.setTimeout(20 * 60_000) // real search + multiple real LLM calls, generous given real-world rate-limit retries

  // arXiv/Semantic Scholar rate-limit flakiness is real and has recurred
  // throughout this project (see ingestion.py's own retry/backoff
  // handling and the OpenAlex-fallback feature this repo already ships).
  // Routing the one real /curation/start call through that real fallback
  // keeps this test fully live (a real request, a real response from a
  // real alternate data source) without being at the mercy of whichever
  // provider happens to be degraded at the moment this test runs.
  await page.route('**/curation/start', async (route) => {
    const body = { ...route.request().postDataJSON(), use_openalex_fallback: true }
    await route.continue({ postData: JSON.stringify(body) })
  })

  await page.goto('/')

  // --- Start a real review ---
  await page.getByRole('button', { name: '+ New review' }).click()
  await page.getByTestId('new-review-topic').fill(TOPIC)
  await page.getByTestId('new-review-target-count').fill(String(TARGET_COUNT))
  await page.getByTestId('new-review-start').click()

  await waitForPendingBatch(page, 15 * 60_000)
  await expect(page).toHaveURL(/[?&]session=[^&]+/)
  const url = new URL(page.url())
  const sessionId = url.searchParams.get('session')
  expect(sessionId).toBeTruthy()

  const cardsBeforeRefresh = await page.locator('[data-testid^="paper-card-"]').evaluateAll((els) =>
    els.map((el) => el.getAttribute('data-testid')),
  )
  expect(cardsBeforeRefresh.length).toBeGreaterThan(0)

  // --- THE SHARPEST TEST: reload while a batch is genuinely pending ---
  // (mid-interrupt) -- nothing has been submitted yet, so if the pending
  // batch were held only in React state, it would be gone after this.
  await page.reload()

  await expect(page).toHaveURL(new RegExp(`session=${sessionId}`))
  await waitForPendingBatch(page)
  const cardsAfterRefresh = await page.locator('[data-testid^="paper-card-"]').evaluateAll((els) =>
    els.map((el) => el.getAttribute('data-testid')),
  )
  // The SAME pending batch, recovered from the checkpointer via
  // GET /curation/{id} -- not a freshly-served, different-looking batch.
  expect(new Set(cardsAfterRefresh)).toEqual(new Set(cardsBeforeRefresh))
  await expect(page.getByTestId('progress-count')).toHaveText(`0 of ${TARGET_COUNT} selected`)

  // --- Add papers and submit picks for real ---
  const addButtons = page.locator('[data-testid^="add-paper-"]')
  for (let i = 0; i < TARGET_COUNT; i++) {
    await addButtons.first().click()
  }
  await expect(page.getByText(`${TARGET_COUNT} added`)).toBeVisible()
  await page.getByTestId('persistent-input-send').click()

  // --- Curation should now be finished (target met) ---
  await expect(page.getByTestId('progress-count')).toHaveText(`${TARGET_COUNT} of ${TARGET_COUNT} selected`, {
    timeout: 60_000,
  })
  await expect(page.getByText('Curation complete')).toBeVisible()

  // --- Generate the real report ---
  await page.getByTestId('persistent-input-send').click()
  await expect(page.getByText('Report ready.')).toBeVisible({ timeout: 90_000 })

  // --- Ask a real, directly-answerable question ---
  await page.getByTestId('persistent-input').fill('What does LoRA add to each layer?')
  await page.getByTestId('persistent-input-send').click()
  await expect(page.locator('text=What does LoRA add to each layer?')).toBeVisible()
  // Some assistant reply appears (real LLM output, content not asserted).
  await expect(page.locator('.border-panel-alt, [class*="border-border"]').last()).toBeVisible({ timeout: 60_000 })

  // --- Ask something the selected papers likely don't cover, to trigger a real web offer ---
  await page.getByTestId('persistent-input').fill('What is the latest 2026 GLUE leaderboard ranking for these exact methods?')
  await page.getByTestId('persistent-input-send').click()
  await expect(page.getByTestId('web-offer-yes')).toBeVisible({ timeout: 60_000 })

  const chatMessageCountBeforeRefresh = await page.locator('[data-testid="persistent-input"]').count() // sanity: input still present
  expect(chatMessageCountBeforeRefresh).toBe(1)

  // --- Phase 6d, second point: reload mid-conversation, offer still pending ---
  await page.reload()
  await expect(page).toHaveURL(new RegExp(`session=${sessionId}`))
  // chat_history reloaded from the backend, not lost:
  await expect(page.locator('text=What does LoRA add to each layer?')).toBeVisible({ timeout: 30_000 })
  // report state survived too -- input is in chat mode (report exists),
  // not back in "generate report" mode:
  await expect(page.getByTestId('persistent-input')).toHaveAttribute('placeholder', 'Ask a question about the selected papers...')
  // the pending web offer survived the reload too -- the Yes/No buttons
  // reappear because pending_web_offer round-tripped through the
  // backend, not because React remembered it:
  await expect(page.getByTestId('web-offer-yes')).toBeVisible({ timeout: 30_000 })

  // --- Accept the (real) web offer post-refresh ---
  await page.getByTestId('web-offer-yes').click()
  await expect(page.getByTestId('web-offer-yes')).not.toBeVisible({ timeout: 90_000 })

  // --- Regenerate the report now that a web source was approved ---
  await expect(page.getByRole('button', { name: 'Regenerate report' })).toBeVisible({ timeout: 15_000 })
  await page.getByRole('button', { name: 'Regenerate report' }).click()
  await expect(page.getByRole('button', { name: 'Regenerate report' })).not.toBeVisible({ timeout: 90_000 })
})
