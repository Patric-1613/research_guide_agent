import { render, screen, waitFor, act } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'
import { useAuth } from './authContext'
import { getAuthHeaders, notifyUnauthorized } from './authBridge'
import { ApiError } from '../../types'

// --- fakes ----------------------------------------------------------------

const h = vi.hoisted(() => {
  type Listener = (user: { uid: string } | null) => void
  const state: {
    currentUser: { uid: string; getIdToken: () => Promise<string> } | null
    listeners: Set<Listener>
    getIdTokenImpl: () => Promise<string>
  } = {
    currentUser: null,
    listeners: new Set(),
    getIdTokenImpl: async () => 'id-token-default',
  }
  function emit() {
    for (const l of state.listeners) l(state.currentUser)
  }
  function setUser(uid: string | null) {
    state.currentUser = uid
      ? { uid, getIdToken: () => state.getIdTokenImpl() }
      : null
    emit()
  }
  return { state, setUser }
})

vi.mock('./firebase', () => ({
  getFirebaseAuth: () => h.state,
  googleProvider: () => ({ __google: true }),
  __resetFirebaseForTests: () => {},
}))

vi.mock('firebase/auth', () => ({
  onIdTokenChanged: (_auth: unknown, cb: (u: unknown) => void) => {
    h.state.listeners.add(cb as never)
    return () => h.state.listeners.delete(cb as never)
  },
  signInWithPopup: vi.fn(async () => {
    h.setUser('user-a')
  }),
  signOut: vi.fn(async () => {
    h.setUser(null)
  }),
}))

const fetchMe = vi.fn()
vi.mock('../api/me', () => ({ fetchMe: (...args: unknown[]) => fetchMe(...args) }))

// import AFTER the mocks are declared
import { FirebaseAuthProvider } from './FirebaseAuthProvider'
import { signInWithPopup } from 'firebase/auth'

function Probe() {
  const { status, account, errorMessage } = useAuth()
  return (
    <div>
      <span data-testid="status">{status}</span>
      <span data-testid="email">{account?.email ?? ''}</span>
      <span data-testid="err">{errorMessage ?? ''}</span>
    </div>
  )
}

function renderProvider(children: ReactNode = <Probe />) {
  return render(<FirebaseAuthProvider>{children}</FirebaseAuthProvider>)
}

const me = (over: Partial<{ approved: boolean; disabled: boolean; email: string }> = {}) => ({
  user_id: 'internal-uuid',
  email: over.email ?? 'a@example.com',
  display_name: 'A',
  approved: over.approved ?? true,
  disabled: over.disabled ?? false,
})

beforeEach(() => {
  h.state.currentUser = null
  h.state.listeners.clear()
  h.state.getIdTokenImpl = async () => 'id-token-default'
  fetchMe.mockReset()
  vi.mocked(signInWithPopup).mockClear()
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('FirebaseAuthProvider', () => {
  it('resolves from initializing to signed-out when there is no user', async () => {
    renderProvider()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))
  })

  it('sign-in -> checking -> approved, and the account email is exposed', async () => {
    fetchMe.mockResolvedValue(me({ approved: true }))
    renderProvider()
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))

    await act(async () => {
      await signInWithPopup(h.state as never, {} as never)
    })

    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))
    expect(screen.getByTestId('email')).toHaveTextContent('a@example.com')
  })

  it('an unapproved account resolves to awaiting-approval', async () => {
    fetchMe.mockResolvedValue(me({ approved: false }))
    renderProvider()
    await act(async () => {
      await signInWithPopup(h.state as never, {} as never)
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('awaiting-approval'))
  })

  it('a 401 from /me (disabled account / project mismatch) resolves to denied', async () => {
    fetchMe.mockRejectedValue(new ApiError(401, { detail: { reason_code: 'unauthorized', message: 'x' } }))
    renderProvider()
    await act(async () => {
      await signInWithPopup(h.state as never, {} as never)
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('denied'))
  })

  it('a network failure reaching /me resolves to error', async () => {
    fetchMe.mockRejectedValue(new TypeError('Failed to fetch'))
    renderProvider()
    await act(async () => {
      await signInWithPopup(h.state as never, {} as never)
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
  })

  it('a stale /me response cannot approve after the user signed out', async () => {
    // First check hangs; we sign out before it resolves, then resolve it.
    let resolveFirst!: (v: unknown) => void
    fetchMe.mockImplementationOnce(() => new Promise((res) => { resolveFirst = res }))
    renderProvider()
    await act(async () => {
      await signInWithPopup(h.state as never, {} as never)
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('checking'))

    await act(async () => {
      h.setUser(null)
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))

    await act(async () => {
      resolveFirst(me({ approved: true }))
      await Promise.resolve()
    })
    // The late result for the now-signed-out user must be ignored.
    expect(screen.getByTestId('status')).toHaveTextContent('signed-out')
  })

  it('a stale /me response for user A cannot approve after user B signed in', async () => {
    let resolveA!: (v: unknown) => void
    fetchMe
      .mockImplementationOnce(() => new Promise((res) => { resolveA = res }))
      .mockResolvedValueOnce(me({ approved: false, email: 'b@example.com' }))

    renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('checking'))

    // user B signs in before A's /me resolves
    await act(async () => { h.setUser('user-b') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('awaiting-approval'))
    expect(screen.getByTestId('email')).toHaveTextContent('b@example.com')

    // A's stale response arrives: it says approved, but must be dropped.
    await act(async () => {
      resolveA(me({ approved: true, email: 'a@example.com' }))
      await Promise.resolve()
    })
    expect(screen.getByTestId('status')).toHaveTextContent('awaiting-approval')
    expect(screen.getByTestId('email')).toHaveTextContent('b@example.com')
  })

  it('checkAgain re-runs /me (awaiting-approval -> approved after an admin approves)', async () => {
    fetchMe
      .mockResolvedValueOnce(me({ approved: false }))
      .mockResolvedValueOnce(me({ approved: true }))

    function WithButton() {
      const { status, checkAgain } = useAuth()
      return (
        <div>
          <span data-testid="status">{status}</span>
          <button onClick={checkAgain}>check</button>
        </div>
      )
    }
    renderProvider(<WithButton />)
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('awaiting-approval'))

    await act(async () => {
      screen.getByText('check').click()
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))
    expect(fetchMe).toHaveBeenCalledTimes(2)
  })

  it('sign-out clears the account and returns to signed-out', async () => {
    fetchMe.mockResolvedValue(me({ approved: true }))
    function WithSignOut() {
      const { status, signOut } = useAuth()
      return (
        <div>
          <span data-testid="status">{status}</span>
          <button onClick={() => void signOut()}>out</button>
        </div>
      )
    }
    renderProvider(<WithSignOut />)
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))

    await act(async () => {
      screen.getByText('out').click()
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))
  })

  it('a cancelled sign-in popup is not an error', async () => {
    vi.mocked(signInWithPopup).mockRejectedValueOnce(
      Object.assign(new Error('closed'), { code: 'auth/popup-closed-by-user' }),
    )
    function WithSignIn() {
      const { status, signIn } = useAuth()
      return (
        <div>
          <span data-testid="status">{status}</span>
          <button onClick={() => void signIn()}>in</button>
        </div>
      )
    }
    renderProvider(<WithSignIn />)
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))
    await act(async () => {
      screen.getByText('in').click()
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))
    expect(screen.queryByTestId('err')).toBeNull()
  })

  it('a real sign-in error resolves to error', async () => {
    vi.mocked(signInWithPopup).mockRejectedValueOnce(
      Object.assign(new Error('network'), { code: 'auth/network-request-failed' }),
    )
    function WithSignIn() {
      const { status, signIn } = useAuth()
      return (
        <div>
          <span data-testid="status">{status}</span>
          <button onClick={() => void signIn()}>in</button>
        </div>
      )
    }
    renderProvider(<WithSignIn />)
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('signed-out'))
    await act(async () => {
      screen.getByText('in').click()
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('error'))
  })

  it('registers the token getter on the bridge so requests carry a Bearer header', async () => {
    fetchMe.mockResolvedValue(me({ approved: true }))
    h.state.getIdTokenImpl = async () => 'live-token-xyz'
    renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))

    expect(await getAuthHeaders()).toEqual({ Authorization: 'Bearer live-token-xyz' })
  })

  it('a 401 on a product request (notifyUnauthorized) re-verifies the session', async () => {
    fetchMe
      .mockResolvedValueOnce(me({ approved: true }))
      .mockRejectedValueOnce(new ApiError(401, { detail: 'nope' }))
    renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))

    await act(async () => {
      notifyUnauthorized()
      await Promise.resolve()
    })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('denied'))
  })

  it('clears the auth bridge on unmount, so later requests carry no token', async () => {
    fetchMe.mockResolvedValue(me({ approved: true }))
    h.state.getIdTokenImpl = async () => 'tok-while-mounted'
    const { unmount } = renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))
    expect(await getAuthHeaders()).toEqual({ Authorization: 'Bearer tok-while-mounted' })

    unmount()

    expect(await getAuthHeaders()).toEqual({})
    // and a stray 401 after unmount does not throw or resurrect anything
    expect(() => notifyUnauthorized()).not.toThrow()
  })

  it('a remount re-registers a working bridge (register -> clear -> register is sequential and consistent)', async () => {
    fetchMe.mockResolvedValue(me({ approved: true }))
    const first = renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))
    first.unmount()
    expect(await getAuthHeaders()).toEqual({})

    h.state.currentUser = null
    h.state.listeners.clear()
    h.state.getIdTokenImpl = async () => 'tok-after-remount'
    renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))
    expect(await getAuthHeaders()).toEqual({ Authorization: 'Bearer tok-after-remount' })
  })

  it('never writes the ID token to localStorage, sessionStorage, or the DOM', async () => {
    fetchMe.mockResolvedValue(me({ approved: true }))
    h.state.getIdTokenImpl = async () => 'SECRET-ID-TOKEN-VALUE'
    const { container } = renderProvider()
    await act(async () => { h.setUser('user-a') })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('approved'))
    // force a request through the bridge
    await getAuthHeaders()

    expect(JSON.stringify(localStorage)).not.toContain('SECRET-ID-TOKEN-VALUE')
    expect(JSON.stringify(sessionStorage)).not.toContain('SECRET-ID-TOKEN-VALUE')
    expect(container.innerHTML).not.toContain('SECRET-ID-TOKEN-VALUE')
    expect(window.location.href).not.toContain('SECRET-ID-TOKEN-VALUE')
  })
})
