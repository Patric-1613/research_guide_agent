import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useEffect } from 'react'
import { AuthGate } from './AuthGate'
import { AuthContext, type AuthContextValue } from '../../lib/auth/authContext'

function ctx(over: Partial<AuthContextValue>): AuthContextValue {
  return {
    status: 'approved',
    account: null,
    errorMessage: null,
    signIn: vi.fn(async () => {}),
    signOut: vi.fn(async () => {}),
    checkAgain: vi.fn(),
    ...over,
  }
}

function renderGate(value: AuthContextValue, child = <div data-testid="app">THE APP</div>) {
  return render(
    <AuthContext.Provider value={value}>
      <AuthGate>{child}</AuthGate>
    </AuthContext.Provider>,
  )
}

afterEach(() => vi.clearAllMocks())

describe('AuthGate', () => {
  it('renders the app for status "not-required" (disabled/basic mode)', () => {
    renderGate(ctx({ status: 'not-required' }))
    expect(screen.getByTestId('app')).toBeInTheDocument()
  })

  it('renders the app for status "approved"', () => {
    renderGate(ctx({ status: 'approved' }))
    expect(screen.getByTestId('app')).toBeInTheDocument()
  })

  it('shows a stable loading state for "initializing" and "checking", never the app', () => {
    const { rerender } = renderGate(ctx({ status: 'initializing' }))
    expect(screen.getByRole('status')).toHaveTextContent('Loading…')
    expect(screen.queryByTestId('app')).not.toBeInTheDocument()

    rerender(
      <AuthContext.Provider value={ctx({ status: 'checking' })}>
        <AuthGate><div data-testid="app">THE APP</div></AuthGate>
      </AuthContext.Provider>,
    )
    expect(screen.getByRole('status')).toHaveTextContent('Checking your account…')
    expect(screen.queryByTestId('app')).not.toBeInTheDocument()
  })

  it('shows "Continue with Google" when signed out', () => {
    renderGate(ctx({ status: 'signed-out' }))
    expect(screen.getByTestId('sign-in-google')).toHaveTextContent('Continue with Google')
    expect(screen.queryByTestId('app')).not.toBeInTheDocument()
  })

  it('signed-out screen calls signIn on click', async () => {
    const signIn = vi.fn(async () => {})
    renderGate(ctx({ status: 'signed-out', signIn }))
    await userEvent.click(screen.getByTestId('sign-in-google'))
    expect(signIn).toHaveBeenCalledTimes(1)
  })

  it('awaiting-approval shows only safe account info + Check again + Sign out', () => {
    renderGate(
      ctx({
        status: 'awaiting-approval',
        account: { user_id: 'u', email: 'me@example.com', display_name: 'Me', approved: false, disabled: false },
      }),
    )
    expect(screen.getByText('Awaiting approval')).toBeInTheDocument()
    expect(screen.getByText('me@example.com')).toBeInTheDocument()
    expect(screen.getByText('Me')).toBeInTheDocument()
    expect(screen.getByTestId('approval-check-again')).toBeInTheDocument()
    expect(screen.getByTestId('sign-out')).toBeInTheDocument()
    // the internal user_id is never rendered
    expect(screen.queryByText('u')).not.toBeInTheDocument()
    expect(screen.queryByTestId('app')).not.toBeInTheDocument()
  })

  it('awaiting-approval renders a long email without truncating it away, and lets it wrap', () => {
    const longEmail = 'a-really-quite-long-local-part.with.dots@a-long-subdomain.example-corp.co.uk'
    renderGate(
      ctx({
        status: 'awaiting-approval',
        account: { user_id: 'u', email: longEmail, display_name: null, approved: false, disabled: false },
      }),
    )
    const dd = screen.getByText(longEmail)
    expect(dd.tagName).toBe('DD')
    // it must be allowed to wrap inside the fixed-width card, not overflow it
    expect(dd.className).toContain('break-all')
  })

  it('awaiting-approval "Check again" calls checkAgain', async () => {
    const checkAgain = vi.fn()
    renderGate(ctx({ status: 'awaiting-approval', checkAgain }))
    await userEvent.click(screen.getByTestId('approval-check-again'))
    expect(checkAgain).toHaveBeenCalledTimes(1)
  })

  it('denied shows a non-technical message + Sign out', () => {
    renderGate(ctx({ status: 'denied' }))
    expect(screen.getByText(/Can’t access this app/)).toBeInTheDocument()
    expect(screen.getByTestId('sign-out')).toBeInTheDocument()
    expect(screen.queryByTestId('app')).not.toBeInTheDocument()
  })

  it('error shows the message + Sign out', () => {
    renderGate(ctx({ status: 'error', errorMessage: 'Sign-in is not configured correctly.' }))
    expect(screen.getByText('Sign-in is not configured correctly.')).toBeInTheDocument()
    expect(screen.getByTestId('sign-out')).toBeInTheDocument()
  })

  it('Sign out (any gated screen) calls signOut', async () => {
    const signOut = vi.fn(async () => {})
    renderGate(ctx({ status: 'awaiting-approval', signOut }))
    await userEvent.click(screen.getByTestId('sign-out'))
    expect(signOut).toHaveBeenCalledTimes(1)
  })

  it('does NOT mount its children (so no product hook runs) for an unapproved user', () => {
    const onMount = vi.fn()
    function ProductProbe() {
      useEffect(() => {
        onMount()
      }, [])
      return <div data-testid="app">THE APP</div>
    }
    renderGate(ctx({ status: 'awaiting-approval' }), <ProductProbe />)
    expect(onMount).not.toHaveBeenCalled()
    expect(screen.queryByTestId('app')).not.toBeInTheDocument()
  })
})
