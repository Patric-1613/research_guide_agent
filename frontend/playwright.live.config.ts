import { defineConfig, devices } from '@playwright/test'

// RL6 Part D: ONE approved live Research Lanes journey. Servers are
// started externally (backend :8001 with RESEARCH_LANES_ENABLED=true,
// Vite :5174 -> :8001). Bounded to one Suggest + one Start, two enabled
// lanes.
export default defineConfig({
  testDir: './e2e',
  testMatch: /research-lanes-live\.spec\.ts/,
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  timeout: 300_000,
  use: { baseURL: 'http://localhost:5174', trace: 'retain-on-failure' },
  projects: [{ name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } } }],
})
