import { test, expect, type Page } from '@playwright/test'

/**
 * Real, live, non-mocked proof of the full curation-and-chat flow,
 * including both Phase 6f features, run as one continuous session to
 * avoid redundant real LLM/search calls:
 *
 *   1. (Phase 6c) The full curate -> report -> chat -> web-escalation ->
 *      regenerate flow works against the REAL backend (real arXiv/
 *      Semantic Scholar search, real OpenAI calls, real Tavily search).
 *
 *   2. (Phase 6d) Page state survives a REAL browser reload at the two
 *      points that matter most: while a curation batch is genuinely
 *      PENDING (mid-interrupt), and mid-conversation in the chat stage
 *      (chat_history + report + a pending offer all reloading from the
 *      backend, not from anything held only in React state).
 *
 *   3. (Phase 6f-2) Typed refinement text during curation forces a real,
 *      different batch -- not just accepted without error.
 *
 *   4. (Phase 6f-3/6f-4) The report-update offer appears as a real
 *      conversational message (not the old passive banner) after a web
 *      source is approved, and its Yes button drives a real report
 *      regeneration through the same dispatch path as typed text.
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

async function pendingBatchIds(page: Page): Promise<string[]> {
  return page.locator('[data-testid^="paper-card-"]').evaluateAll((els) => els.map((el) => el.getAttribute('data-testid')))
}

test('curate with refinement -> refresh mid-interrupt -> report -> chat -> web offer -> conversational report-update offer -> refresh mid-chat', async ({ page }) => {
  test.setTimeout(25 * 60_000) // real search + multiple real LLM calls, generous given real-world rate-limit retries

  // arXiv/Semantic Scholar rate-limit flakiness is real and has recurred
  // throughout this project -- routing the one real /curation/start call
  // through the real OpenAlex fallback keeps this test fully live (a
  // real request, a real response from a real alternate data source)
  // without being at the mercy of whichever provider is degraded today.
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

  const turn1BatchIds = await pendingBatchIds(page)
  expect(turn1BatchIds.length).toBeGreaterThan(0)

  // --- THE SHARPEST REFRESH TEST: reload while a batch is genuinely ---
  // pending (mid-interrupt) -- nothing has been submitted yet, so if the
  // pending batch were held only in React state, it would be gone after this.
  await page.reload()
  await expect(page).toHaveURL(new RegExp(`session=${sessionId}`))
  await waitForPendingBatch(page)
  const turn1BatchIdsAfterRefresh = await pendingBatchIds(page)
  expect(new Set(turn1BatchIdsAfterRefresh)).toEqual(new Set(turn1BatchIds))
  await expect(page.getByTestId('progress-count')).toHaveText(`0 of ${TARGET_COUNT} selected`)

  // --- Turn 1: pick ONE paper, no refinement -- still short of target ---
  await page.locator('[data-testid^="add-paper-"]').first().click()
  await expect(page.getByTestId('review-continue')).toHaveText('Continue with 1 added')
  await page.getByTestId('review-continue').click()
  await expect(page.getByTestId('progress-count')).toHaveText(`1 of ${TARGET_COUNT} selected`, { timeout: 60_000 })

  // --- Turn 2 (Phase 6f-2): type refinement text, add NOTHING, Continue -- ---
  // must force a real, different batch without submitting any new picks.
  await waitForPendingBatch(page, 60_000)
  const turn2BatchIdsBeforeRefinement = await pendingBatchIds(page)
  const firstCardBeforeRefinement = page.locator(`[data-testid="${turn2BatchIdsBeforeRefinement[0]}"]`)
  await page.getByTestId('review-refinement-input').fill('focus on more recent work published in 2024 or later')
  await page.getByTestId('review-continue').click()
  // waitForPendingBatch alone isn't enough here -- SOME batch stays
  // visible in the DOM the whole time (the old one, until React
  // re-renders), so it's trivially satisfied before AND after the real
  // update lands. Wait for the specific old card to actually be GONE,
  // proving the DOM genuinely swapped to a new batch, before reading it.
  await expect(firstCardBeforeRefinement).toHaveCount(0, { timeout: 90_000 })
  await expect(page.getByTestId('progress-count')).toHaveText(`1 of ${TARGET_COUNT} selected`)
  await waitForPendingBatch(page, 30_000)
  const turn3BatchIds = await pendingBatchIds(page)
  expect(new Set(turn3BatchIds)).not.toEqual(new Set(turn2BatchIdsBeforeRefinement))
  console.log('PASS: refinement text produced a genuinely different batch, with zero picks submitted alongside it.')

  // --- Finish curation from the refined batch ---
  const addButtons = page.locator('[data-testid^="add-paper-"]')
  for (let i = 0; i < TARGET_COUNT - 1; i++) {
    await addButtons.first().click()
  }
  await expect(page.getByTestId('review-continue')).toHaveText(`Continue with ${TARGET_COUNT - 1} added`)
  await page.getByTestId('review-continue').click()
  await expect(page.getByTestId('progress-count')).toHaveText(`${TARGET_COUNT} of ${TARGET_COUNT} selected`, { timeout: 60_000 })
  await expect(page.getByText('Curation complete')).toBeVisible()

  // --- Curation finishing auto-switches the workspace to Report (locked
  // tabs unlock together the instant stage flips to "synthesize") ---
  await expect(page.getByTestId('workspace-mode-chat')).toBeEnabled()
  await expect(page.getByTestId('generate-report')).toBeVisible()

  // --- Generate the real report ---
  await page.getByTestId('generate-report').click()
  await expect(page.getByText('Findings')).toBeVisible({ timeout: 90_000 })

  // --- Switch to Chat to ask questions about the selected papers ---
  await page.getByTestId('workspace-mode-chat').click()

  // --- Ask a real, directly-answerable question ---
  await page.getByTestId('persistent-input').fill('What does LoRA add to each layer?')
  await page.getByTestId('persistent-input-send').click()
  await expect(page.locator('text=What does LoRA add to each layer?')).toBeVisible()

  // --- Ask something the selected papers likely don't cover, to trigger a real web offer ---
  await page.getByTestId('persistent-input').fill('What is the latest 2026 GLUE leaderboard ranking for these exact methods?')
  await page.getByTestId('persistent-input-send').click()
  await expect(page.getByTestId('web-offer-yes')).toBeVisible({ timeout: 60_000 })

  // --- Phase 6d, second refresh point: reload mid-conversation, offer still pending ---
  // The workspace mode itself (?mode=chat, set when the Chat tab was
  // clicked above) must also survive the reload -- otherwise the user
  // would be silently bounced back to Report, even though the chat
  // history/report/offer all correctly reloaded from the backend.
  await expect(page).toHaveURL(/mode=chat/)
  await page.reload()
  await expect(page).toHaveURL(new RegExp(`session=${sessionId}`))
  await expect(page).toHaveURL(/mode=chat/)
  await expect(page.locator('text=What does LoRA add to each layer?')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('persistent-input')).toHaveAttribute('placeholder', 'Ask a question about the selected papers...')
  await expect(page.getByTestId('web-offer-yes')).toBeVisible({ timeout: 30_000 })

  // --- Accept the (real) web offer post-refresh ---
  // Note: the Yes/No buttons share one testid across BOTH offer types
  // (by design -- they're the same dispatch mechanism, not two separate
  // ones), so the button can go straight from "accept the web offer" to
  // immediately visible again for the report-update offer that this
  // acceptance itself triggers. Waiting for it to vanish and STAY gone
  // would be asserting on the wrong thing -- wait for the specific
  // report-update offer text instead, which only appears once the web
  // offer has actually resolved.
  await page.getByTestId('web-offer-yes').click()

  // --- Phase 6f-3/6f-4: the CONVERSATIONAL report-update offer, not the old banner ---
  await expect(page.getByText('Update the report to include the newly approved source(s)?')).toBeVisible({ timeout: 90_000 })
  await expect(page.getByRole('button', { name: 'Regenerate report' })).toHaveCount(0) // the old passive banner is gone
  await page.getByTestId('web-offer-yes').click() // same button/testid, now driving the report-update accept
  await expect(page.getByText('Update the report to include the newly approved source(s)?')).not.toBeVisible({ timeout: 90_000 })
  await expect(page.locator("text=Done — I've updated the report")).toBeVisible({ timeout: 5_000 })
})
