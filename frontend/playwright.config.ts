import { defineConfig } from '@playwright/test'

// e2e tests drive a REAL dev server against a REAL FastAPI backend
// (matching this project's own established standard of pairing mocked
// unit tests with at least one real, live run per feature) -- neither
// server is started here; see frontend/e2e/README.md for how to run
// this locally, and the Phase 6c/6d test files themselves for what each
// one assumes is already running.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:5173',
    trace: 'retain-on-failure',
  },
})
