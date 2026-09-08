import { render, screen, waitFor, act } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

// --- firebase fakes (same shape as FirebaseAuthProvider.test.tsx) --------

const h = vi.hoisted(() => {
  const state: {
    currentUser: { uid: string; getIdToken: () => Promise<string> } | null
    listeners: Set<(u: unknown) => void>
    initCalls: number
  } = { currentUser: null, listeners: new Set(), initCalls: 0 }
  function setUser(uid: string | null) {
    state.currentUser = uid ? { uid, getIdToken: async () => 'tok' } : null
    for (const l of state.listeners) l(state.currentUser)
  }
  return { state, setUser }
})

vi.mock('./lib/auth/firebase', () => ({
  getFirebaseAuth: () => {
    h.state.initCalls++
    return h.state
  },
  googleProvider: () => ({}),
  __resetFirebaseForTests: () => {},
}))
vi.mock('firebase/auth', () => ({
  onIdTokenChanged: (_a: unknown, cb: (u: unknown) => void) => {
    h.state.listeners.add(cb)
    return () => h.state.listeners.delete(cb)
  },
  signInWithPopup: vi.fn(async () => h.setUser('user-a')),
  signOut: vi.fn(async () => h.setUser(null)),
}))

const fetchMe = vi.fn()
vi.mock('./lib/api/me', () => ({ fetchMe: (...a: unknown[]) => fetchMe(...a) }))

// useCurationSession must never run for a non-approved user; spy on it.
const useCurationSessionSpy = vi.fn()
vi.mock('./hooks/useCurationSession', () => ({
  useCurationSession: () => {
    useCurationSessionSpy()
    return {
      sessionId: null, state: null, loading: false, error: null, turnEvents: [],
      lastChatSearchMeta: null, reportPossiblyStale: false, lastAddToReportResult: null,
      dismissReportStaleWarning: vi.fn(), curationAction: null, startingReviewVisible: false,
      researchLanesAvailable: false, laneSuggestions: null, laneSuggestionLoading: false,
      laneSuggestionError: null, suggestResearchLanes: vi.fn(), resetLaneSuggestions: vi.fn(),
      openReview: vi.fn(), startReview: vi.fn(), submitPicks: vi.fn(), activateReportVersion: vi.fn(),
      chatStreamActive: false, chatStreamPhase: null, chatStreamText: '', chatStreamSyncFailed: false,
      sendChatMessageStreaming: vi.fn(), cancelChatStream: vi.fn(),
      reportStreamActive: false, reportStreamOperation: null, reportStreamPhase: null,
      reportStreamPhaseHistory: [], reportStreamStopping: false, reportStreamError: null,
      reportStreamSyncFailed: false, reportStreamCompletionNotice: null,
      generateReportStreaming: vi.fn(), regenerateReportStreaming: vi.fn(), cancelReportStream: vi.fn(),
      generateReport: vi.fn(), regenerateReport: vi.fn(), sendChatMessage: vi.fn(),
      deleteExchanges: vi.fn(), addExchangesToReport: vi.fn(), editExchange: vi.fn(),
      deleteReview: vi.fn(), selectFromHistory: vi.fn(), reopenReview: vi.fn(), refresh: vi.fn(),
    }
  },
}))
vi.mock('./lib/api/client', () => ({
  curationApi: { listReviews: vi.fn().mockResolvedValue([]), getReportExportUrl: vi.fn().mockReturnValue('x') },
  downloadReportExport: vi.fn(),
}))

const me = (approved: boolean) => ({
  user_id: 'u', email: 'me@example.com', display_name: 'Me', approved, disabled: false,
})

beforeEach(() => {
  h.state.currentUser = null
  h.state.listeners.clear()
  h.state.initCalls = 0
  fetchMe.mockReset()
  useCurationSessionSpy.mockClear()
})
afterEach(() => {
  vi.unstubAllEnvs()
  vi.clearAllMocks()
})

describe('App auth (disabled mode)', () => {
  it('renders the application directly and never initializes Firebase', async () => {
    vi.stubEnv('VITE_AUTH_MODE', '')
    render(<App />)
    expect(await screen.findByText('Research Helper Agent')).toBeInTheDocument()
    expect(h.state.initCalls).toBe(0)
    expect(screen.queryByTestId('account-menu-trigger')).not.toBeInTheDocument()
  })
})

describe('App auth (firebase mode)', () => {
  beforeEach(() => {
    vi.stubEnv('VITE_AUTH_MODE', 'firebase')
  })

  it('a signed-out user sees the Google sign-in action, not the app', async () => {
    render(<App />)
    expect(await screen.findByTestId('sign-in-google')).toBeInTheDocument()
    expect(screen.queryByTestId('review-continue')).not.toBeInTheDocument()
    expect(useCurationSessionSpy).not.toHaveBeenCalled()
  })

  it('an approved user enters the app, with an account menu in the header', async () => {
    fetchMe.mockResolvedValue(me(true))
    render(<App />)
    await screen.findByTestId('sign-in-google')

    await act(async () => {
      h.setUser('user-a')
    })

    await waitFor(() => expect(screen.getByTestId('account-menu-trigger')).toBeInTheDocument())
    expect(screen.getByTestId('account-menu-trigger')).toHaveTextContent('me@example.com')
    expect(useCurationSessionSpy).toHaveBeenCalled()
  })

  it('an unapproved user sees only the approval screen; useCurationSession never runs', async () => {
    fetchMe.mockResolvedValue(me(false))
    render(<App />)
    await screen.findByTestId('sign-in-google')

    await act(async () => {
      h.setUser('user-a')
    })

    await waitFor(() => expect(screen.getByText('Awaiting approval')).toBeInTheDocument())
    expect(useCurationSessionSpy).not.toHaveBeenCalled()
    expect(screen.queryByTestId('account-menu-trigger')).not.toBeInTheDocument()
  })
})
