// Day 5: gates the application on the auth status.
//
//   not-required / approved  -> render the app
//   initializing / checking  -> a stable loading state (no content flash)
//   signed-out               -> "Continue with Google"
//   awaiting-approval        -> the approval-pending screen (safe account
//                               info only) with "Check again" + "Sign out"
//   denied                   -> a plain "can't access" message + "Sign out"
//   error                    -> a plain error message + "Sign out"
//
// The app is never mounted until status is `approved` (or auth is not
// required), so no product hook runs and no product API call fires for a
// signed-out or unapproved user.

import type { ReactNode } from 'react'
import { LogIn, Loader2 } from 'lucide-react'
import { useAuth } from '../../lib/auth/authContext'

function AuthShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen w-screen items-center justify-center bg-bg px-4 text-text">
      <div className="w-full max-w-sm rounded-lg border border-border bg-panel p-6 shadow-lg">
        <div className="mb-4 text-sm font-semibold tracking-tight text-text">Research Helper Agent</div>
        {children}
      </div>
    </div>
  )
}

function AuthLoading({ label }: { label: string }) {
  return (
    <div className="flex h-screen w-screen items-center justify-center bg-bg text-text">
      <div role="status" aria-live="polite" className="flex items-center gap-2 text-sm text-text-secondary">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        {label}
      </div>
    </div>
  )
}

const primaryButton =
  'inline-flex w-full items-center justify-center gap-2 rounded-md bg-accent px-3 py-2 text-sm font-medium text-accent-fg ' +
  'outline-none hover:bg-accent-hover focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 ' +
  'focus-visible:ring-offset-panel disabled:cursor-not-allowed disabled:opacity-50'

const secondaryButton =
  'inline-flex w-full items-center justify-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-text-secondary ' +
  'outline-none hover:text-text focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-panel'

function SignedOutScreen() {
  const { signIn, errorMessage } = useAuth()
  return (
    <AuthShell>
      <p className="mb-4 text-sm text-text-secondary">Sign in with your Google account to continue.</p>
      {errorMessage && (
        <p role="alert" className="mb-3 rounded-md bg-danger-soft px-3 py-2 text-xs text-danger">
          {errorMessage}
        </p>
      )}
      <button type="button" data-testid="sign-in-google" onClick={() => void signIn()} className={primaryButton}>
        <LogIn className="h-4 w-4" aria-hidden="true" />
        Continue with Google
      </button>
    </AuthShell>
  )
}

function AccountLines({ email, displayName }: { email: string | null; displayName: string | null }) {
  return (
    <dl className="mb-4 space-y-1 text-xs">
      {displayName && (
        <div className="flex gap-2">
          <dt className="text-text-muted">Name</dt>
          <dd className="text-text-secondary">{displayName}</dd>
        </div>
      )}
      {email && (
        <div className="flex gap-2">
          <dt className="text-text-muted">Email</dt>
          <dd className="text-text-secondary">{email}</dd>
        </div>
      )}
    </dl>
  )
}

function AwaitingApprovalScreen() {
  const { account, signOut, checkAgain, status } = useAuth()
  return (
    <AuthShell>
      <h1 className="mb-2 text-base font-semibold text-text">Awaiting approval</h1>
      <p className="mb-4 text-sm text-text-secondary">
        Your account has been created but is not approved yet. You’ll be able to use the app once an
        administrator approves it.
      </p>
      <AccountLines email={account?.email ?? null} displayName={account?.display_name ?? null} />
      <div className="space-y-2">
        <button
          type="button"
          data-testid="approval-check-again"
          onClick={checkAgain}
          disabled={status === 'checking'}
          className={primaryButton}
        >
          {status === 'checking' && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
          Check again
        </button>
        <button type="button" data-testid="sign-out" onClick={() => void signOut()} className={secondaryButton}>
          Sign out
        </button>
      </div>
    </AuthShell>
  )
}

function DeniedScreen() {
  const { signOut } = useAuth()
  return (
    <AuthShell>
      <h1 className="mb-2 text-base font-semibold text-text">Can’t access this app</h1>
      <p className="mb-4 text-sm text-text-secondary">
        This account isn’t able to use the app right now. If you think this is a mistake, contact the
        person who invited you.
      </p>
      <button type="button" data-testid="sign-out" onClick={() => void signOut()} className={secondaryButton}>
        Sign out
      </button>
    </AuthShell>
  )
}

function AuthErrorScreen() {
  const { signOut, errorMessage } = useAuth()
  return (
    <AuthShell>
      <h1 className="mb-2 text-base font-semibold text-text">Something went wrong</h1>
      <p className="mb-4 text-sm text-text-secondary">{errorMessage ?? 'Please try again in a moment.'}</p>
      <button type="button" data-testid="sign-out" onClick={() => void signOut()} className={secondaryButton}>
        Sign out
      </button>
    </AuthShell>
  )
}

export function AuthGate({ children }: { children: ReactNode }) {
  const { status } = useAuth()

  switch (status) {
    case 'not-required':
    case 'approved':
      return <>{children}</>
    case 'initializing':
      return <AuthLoading label="Loading…" />
    case 'checking':
      return <AuthLoading label="Checking your account…" />
    case 'signed-out':
      return <SignedOutScreen />
    case 'awaiting-approval':
      return <AwaitingApprovalScreen />
    case 'denied':
      return <DeniedScreen />
    case 'error':
      return <AuthErrorScreen />
  }
}
