import { test, expect, type Request } from '@playwright/test'
import { writeFileSync } from 'node:fs'

/**
 * RL6 Part D -- ONE approved live journey. Real backend (:8001,
 * RESEARCH_LANES_ENABLED=true), real OpenAI + arXiv/Semantic Scholar.
 * Bounded: exactly one Suggest POST + one Start POST, two enabled lanes,
 * disposable topic. No Continue / refill / chat / report / export.
 */

const DISPOSABLE_TOPIC = 'RL6 E2E disposable — retrieval augmented generation hallucination reduction'

test('live: Suggest once, keep two lanes, Start once, verify canonical lanes + provenance', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (m) => {
    if (m.type() !== 'error') return
    const t = m.text()
    if (/Failed to load resource|favicon\.ico/i.test(t)) return
    // Pre-existing, unrelated: ReviewCard.tsx nests its delete <button>
    // inside the card <button> (last touched well before the Research
    // Lanes range; see its own `display: contents` comment). React warns.
    // Not Research Lanes code -- filtered so this spec's assertion covers
    // only lane-feature-originated errors.
    if (/ReviewCard|<button> button|cannot be a descendant of/i.test(t)) return
    consoleErrors.push(t)
  })
  page.on('pageerror', (e) => consoleErrors.push(String(e)))

  const posts: { url: string; body: unknown }[] = []
  page.on('request', (r: Request) => {
    if (r.method() !== 'POST') return
    if (!/\/curation\//.test(r.url())) return
    let body: unknown = null
    try { body = r.postDataJSON() } catch { body = r.postData() }
    posts.push({ url: r.url(), body })
  })

  const responses: Record<string, unknown> = {}
  page.on('response', async (res) => {
    const u = res.url()
    if (/\/curation\/lanes\/suggest$/.test(u)) responses.suggest = await res.json().catch(() => null)
    if (/\/curation\/start$/.test(u)) responses.start = await res.json().catch(() => null)
  })

  await page.goto('/')

  // --- New review -> Research lanes ---
  await page.getByTestId('new-review-trigger').click()
  await expect(page.getByTestId('new-review-mode-lanes')).toBeVisible()
  await page.getByTestId('new-review-mode-lanes').click()
  await page.getByTestId('new-review-topic').fill(DISPOSABLE_TOPIC)
  await page.getByTestId('new-review-target-count').fill('4')

  // --- Suggest once ---
  await page.getByTestId('new-review-suggest-lanes').click()
  await expect(page.getByTestId('lane-row-0')).toBeVisible({ timeout: 60_000 })
  await expect(page.getByTestId('lane-row-2')).toBeVisible()

  // --- Keep exactly two lanes enabled (disable the third) ---
  await page.getByTestId('lane-enabled-2').uncheck()
  await expect(page.getByTestId('lane-enabled-0')).toBeChecked()
  await expect(page.getByTestId('lane-enabled-1')).toBeChecked()
  await expect(page.getByTestId('lane-enabled-2')).not.toBeChecked()

  const draftLabels = await page.locator('[data-testid^="lane-label-"]').evaluateAll((els) =>
    els.map((e) => (e as HTMLInputElement).value),
  )

  // --- Start once ---
  await page.getByTestId('new-review-start').click()
  await expect(page.getByTestId('lane-summary-toggle')).toBeVisible({ timeout: 120_000 })

  const sessionId = new URL(page.url()).searchParams.get('session')
  expect(sessionId).toBeTruthy()

  // --- Exactly one Suggest POST, one Start POST, no duplicates ---
  const suggestPosts = posts.filter((p) => /\/curation\/lanes\/suggest$/.test(p.url))
  const startPosts = posts.filter((p) => /\/curation\/start$/.test(p.url))
  expect(suggestPosts).toHaveLength(1)
  expect(startPosts).toHaveLength(1)
  expect(posts.filter((p) => /\/picks$/.test(p.url))).toHaveLength(0)
  expect(posts.filter((p) => /\/chat/.test(p.url))).toHaveLength(0)
  expect(posts.filter((p) => /\/report/.test(p.url))).toHaveLength(0)

  // --- Start payload carries NO lane identity metadata ---
  const startBody = startPosts[0].body as { topic: string; lanes: unknown[] }
  expect(JSON.stringify(startBody)).not.toMatch(/lane_id|origin|generation_version/)
  expect(startBody.lanes).toHaveLength(3)
  expect((startBody.lanes as { enabled: boolean }[]).filter((l) => l.enabled)).toHaveLength(2)
  for (const l of startBody.lanes as Record<string, unknown>[]) {
    expect(Object.keys(l).sort()).toEqual(['enabled', 'label', 'query', 'question'].sort())
  }

  // --- Canonical, server-minted lane IDs replace the draft identity ---
  const state = responses.start ? await page.evaluate(async (sid) => {
    const r = await fetch(`http://localhost:8001/curation/${sid}`)
    return r.json()
  }, sessionId) : null
  expect(state.lanes).toHaveLength(3)
  for (const lane of state.lanes) {
    expect(lane.lane_id).toMatch(/^[0-9a-f]{32}$/) // uuid4 hex, server-minted
    expect(lane.origin).toBe('user')
    expect(lane.generation_version).toBe(1)
  }
  expect(state.lanes.filter((l: { enabled: boolean }) => l.enabled)).toHaveLength(2)

  // --- Lane summary: "2 active", per-lane cumulative counts ---
  await expect(page.getByTestId('lane-summary-toggle')).toHaveText(/Research lanes · 2 active/)
  await page.getByTestId('lane-summary-toggle').click()
  await expect(page.getByTestId('lane-summary-panel')).toBeVisible()
  await expect(page.getByTestId('lane-summary-panel').locator('input')).toHaveCount(0) // read-only

  // --- "Found via" chips render for the served batch ---
  const laneChipCount = await page.locator('[data-testid^="paper-lanes-"]').count()
  expect(laneChipCount).toBeGreaterThan(0)

  // --- Zero-paid single-search UI check: fresh load, form still offers
  //     both modes with Single as the default; no submit, no paid call ---
  await page.goto('/')
  await page.getByTestId('new-review-trigger').click()
  await expect(page.getByTestId('new-review-mode-single')).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByTestId('new-review-mode-lanes')).toHaveAttribute('aria-pressed', 'false')
  await expect(page.getByTestId('new-review-start')).toHaveText('Start')

  // --- Delete the disposable session ---
  const cleanupResult = await page.evaluate(async (sid) => {
    const r = await fetch(`http://localhost:8001/curation/${sid}`, { method: 'DELETE' })
    return { status: r.status, body: await r.json().catch(() => null) }
  }, sessionId)

  writeFileSync('e2e/.rl6-live-evidence.json', JSON.stringify({
    sessionId,
    draftLabels,
    posts: posts.map((p) => ({ url: p.url.replace('http://localhost:8001', ''), body: p.body })),
    suggestResponseLaneCount: (responses.suggest as { lanes?: unknown[] } | null)?.lanes?.length ?? null,
    startResponse: responses.start,
    stateLanes: state.lanes,
    stateLaneResultCounts: state.lane_result_counts,
    statePaperLaneIds: state.paper_lane_ids,
    statePendingBatchIds: (state.pending_batch ?? []).map((p: { paper_id: string }) => p.paper_id),
    turnHistoryFrozen: (state.turn_history ?? []).map((t: { turn_number: number; paper_lane_ids: unknown }) => ({ turn_number: t.turn_number, paper_lane_ids: t.paper_lane_ids })),
    laneChipCount,
    consoleErrors,
    cleanupResult,
  }, null, 2))

  expect(consoleErrors).toEqual([])
  expect(cleanupResult.status).toBe(200)
})
