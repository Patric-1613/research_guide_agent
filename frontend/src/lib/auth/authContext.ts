// Day 5: the auth context + hook, kept in a component-free module so
// `AuthProvider.tsx` / `FirebaseAuthProvider.tsx` stay component-only
// (Fast Refresh) and consumers never import the firebase-only module.

import { createContext, useContext } from 'react'
import type { MeResponse } from '../../types'

export type AuthStatus =
  | 'not-required' // disabled / basic mode -- the app renders directly
  | 'initializing' // firebase mode, before the first token-state resolves
  | 'signed-out'
  | 'checking' // signed in; GET /me in flight
  | 'awaiting-approval' // signed in, account exists, approved === false
  | 'approved'
  | 'denied' // account disabled, or a valid Google sign-in the backend still rejects
  | 'error' // config error, or a non-401 failure reaching /me

export interface AuthContextValue {
  status: AuthStatus
  account: MeResponse | null
  errorMessage: string | null
  signIn: () => Promise<void>
  signOut: () => Promise<void>
  checkAgain: () => void
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within <AuthProvider>')
  return ctx
}
