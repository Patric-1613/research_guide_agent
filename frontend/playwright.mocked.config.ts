import { defineConfig, devices } from '@playwright/test'

// RL6 Part B: a fully MOCKED browser journey for Research Lanes -- every
// API response is intercepted with page.route, so this makes ZERO network
// or provider calls and needs no FastAPI backend. It starts its own Vite
// dev server. Kept separate from playwright.config.ts (which drives the
// real live flow).
export default defineConfig({
  testDir: './e2e',
  testMatch: /research-lanes-mocked\.spec\.ts/,
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5174',
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } } },
    { name: 'mobile', use: { ...devices['Desktop Chrome'], viewport: { width: 375, height: 667 } } },
  ],
  webServer: {
    command: 'npm run dev -- --port 5174 --strictPort',
    url: 'http://localhost:5174',
    reuseExistingServer: false,
    timeout: 60_000,
  },
})
