// Day 5: the mode switch for frontend authentication.
//
//   - `disabled` / `basic`: provide a `not-required` context and render
//     children straight through. `firebase/*` is never imported, so it
//     is not in the running bundle for these deployments.
//   - `firebase`: lazy-load the real provider (its own chunk, the only
//     place `firebase/app` + `firebase/auth` are pulled in).
//   - an invalid `VITE_AUTH_MODE`: an `error` context, no Firebase.
//
// The shared context type + `useAuth()` live in ./authContext so
// consumers never import the firebase-only module.

import { lazy, Suspense, type ReactNode } from 'react'
import { AuthConfigError, getAuthMode } from './config'
import { AuthContext, type AuthContextValue } from './authContext'

const NOT_REQUIRED_VALUE: AuthContextValue = {
  status: 'not-required',
  account: null,
  errorMessage: null,
  signIn: async () => {},
  signOut: async () => {},
  checkAgain: () => {},
}

const INITIALIZING_VALUE: AuthContextValue = { ...NOT_REQUIRED_VALUE, status: 'initializing' }

function errorValue(message: string): AuthContextValue {
  return { ...NOT_REQUIRED_VALUE, status: 'error', errorMessage: message }
}

const LazyFirebaseAuthProvider = lazy(() =>
  import('./FirebaseAuthProvider').then((m) => ({ default: m.FirebaseAuthProvider })),
)

export function AuthProvider({ children }: { children: ReactNode }) {
  let mode: ReturnType<typeof getAuthMode>
  try {
    mode = getAuthMode()
  } catch (err) {
    return (
      <AuthContext.Provider
        value={errorValue(err instanceof AuthConfigError ? err.message : 'Authentication is misconfigured.')}
      >
        {children}
      </AuthContext.Provider>
    )
  }

  if (mode !== 'firebase') {
    return <AuthContext.Provider value={NOT_REQUIRED_VALUE}>{children}</AuthContext.Provider>
  }

  return (
    <Suspense fallback={<AuthContext.Provider value={INITIALIZING_VALUE}>{children}</AuthContext.Provider>}>
      <LazyFirebaseAuthProvider>{children}</LazyFirebaseAuthProvider>
    </Suspense>
  )
}
