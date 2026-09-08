// Day 5: the Firebase-mode auth state owner. This module is the ONLY
// place `firebase/app` / `firebase/auth` are imported, and it is loaded
// (as its own chunk) only when `VITE_AUTH_MODE=firebase`.
//
// Responsibilities:
//   - initialize Firebase once (via ./firebase, in-memory persistence)
//   - subscribe to `onIdTokenChanged`
//   - run GET /me on sign-in / on demand, with stale-result guards so a
//     slow response for one user can never publish for another
//   - expose sign-in (Google popup), sign-out, check-again, status
//   - register the token getter + 401 handler on ./authBridge so the
//     transport layer attaches `Authorization: Bearer <id token>`
//
// No ID token is stored: it is read fresh from Firebase for each request.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import type { Auth } from 'firebase/auth'
import {
  onIdTokenChanged,
  signInWithPopup,
  signOut as firebaseSignOut,
} from 'firebase/auth'
import type { MeResponse } from '../../types'
import { ApiError } from '../../types'
import { fetchMe } from '../api/me'
import { AuthContext, type AuthContextValue, type AuthStatus } from './authContext'
import { clearAuthBridge, registerAuthBridge } from './authBridge'
import { AuthConfigError } from './config'
import { getFirebaseAuth, googleProvider } from './firebase'

const POPUP_CANCEL_CODES = new Set([
  'auth/popup-closed-by-user',
  'auth/cancelled-popup-request',
  'auth/user-cancelled',
])

function firebaseErrorCode(err: unknown): string | null {
  if (err && typeof err === 'object' && 'code' in err) {
    const code = (err as { code: unknown }).code
    if (typeof code === 'string') return code
  }
  return null
}

export function FirebaseAuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('initializing')
  const [account, setAccount] = useState<MeResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const authRef = useRef<Auth | null>(null)
  // Monotonic guard: only the newest /me check may publish.
  const checkSeq = useRef(0)
  // The uid the newest check belongs to.
  const checkUid = useRef<string | null>(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const runMeCheck = useCallback(async (opts?: { showChecking?: boolean }) => {
    const auth = authRef.current
    const user = auth?.currentUser ?? null
    if (!auth || !user) {
      if (mounted.current) {
        setAccount(null)
        setStatus('signed-out')
      }
      return
    }
    const seq = ++checkSeq.current
    checkUid.current = user.uid
    if (opts?.showChecking !== false && mounted.current) setStatus('checking')

    const isStale = () =>
      !mounted.current || seq !== checkSeq.current || authRef.current?.currentUser?.uid !== checkUid.current

    let token: string
    try {
      token = await user.getIdToken()
    } catch {
      if (!isStale()) {
        setStatus('error')
        setErrorMessage('Could not obtain a session token. Please sign in again.')
      }
      return
    }

    try {
      const me = await fetchMe(token)
      if (isStale()) return
      setErrorMessage(null)
      setAccount(me)
      setStatus(me.disabled ? 'denied' : me.approved ? 'approved' : 'awaiting-approval')
    } catch (err) {
      if (isStale()) return
      if (err instanceof ApiError && err.status === 401) {
        // A valid Google sign-in the backend still refuses: a disabled
        // account, or a frontend/backend Firebase-project mismatch.
        setAccount(null)
        setStatus('denied')
      } else if (err instanceof ApiError && err.status === 403) {
        setAccount(null)
        setStatus('awaiting-approval')
      } else {
        setStatus('error')
        setErrorMessage('Could not reach the server to check your account. Please try again.')
      }
    }
  }, [])

  // Init Firebase + subscribe, once.
  useEffect(() => {
    let auth: Auth
    try {
      auth = getFirebaseAuth()
    } catch (err) {
      if (mounted.current) {
        setStatus('error')
        setErrorMessage(
          err instanceof AuthConfigError ? err.message : 'Sign-in is not configured correctly.',
        )
      }
      return
    }
    authRef.current = auth

    let lastUid: string | null = auth.currentUser?.uid ?? null
    const unsubscribe = onIdTokenChanged(auth, (user) => {
      if (!user) {
        lastUid = null
        checkSeq.current++ // invalidate any in-flight check
        checkUid.current = null
        if (mounted.current) {
          setAccount(null)
          setStatus('signed-out')
        }
        return
      }
      if (user.uid === lastUid) {
        // A silent token refresh for the same user -- keep current state.
        return
      }
      lastUid = user.uid
      void runMeCheck()
    })

    if (!auth.currentUser && mounted.current) setStatus('signed-out')

    return unsubscribe
  }, [runMeCheck])

  // Bridge for the transport layer. Firebase mode only; torn down on unmount.
  useEffect(() => {
    registerAuthBridge({
      getToken: async () => {
        const user = authRef.current?.currentUser
        return user ? user.getIdToken() : null
      },
      onUnauthorized: () => {
        // A product request came back 401 mid-session. Re-verify; the
        // originating request already failed visibly (no silent retry).
        if (authRef.current?.currentUser) void runMeCheck({ showChecking: true })
        else if (mounted.current) setStatus('signed-out')
      },
    })
    return () => clearAuthBridge()
  }, [runMeCheck])

  const signIn = useCallback(async () => {
    const auth = authRef.current
    if (!auth) {
      setStatus('error')
      setErrorMessage('Sign-in is unavailable right now.')
      return
    }
    try {
      await signInWithPopup(auth, googleProvider())
      // onIdTokenChanged fires -> runMeCheck runs.
    } catch (err) {
      const code = firebaseErrorCode(err)
      if (code && POPUP_CANCEL_CODES.has(code)) {
        if (mounted.current) setStatus('signed-out')
        return
      }
      if (mounted.current) {
        setStatus('error')
        setErrorMessage('Sign-in did not complete. Please try again.')
      }
    }
  }, [])

  const signOut = useCallback(async () => {
    checkSeq.current++
    checkUid.current = null
    setAccount(null)
    const auth = authRef.current
    if (auth) {
      try {
        await firebaseSignOut(auth)
      } catch {
        // Clear local state regardless.
      }
    }
    if (mounted.current) {
      setErrorMessage(null)
      setStatus('signed-out')
    }
  }, [])

  const checkAgain = useCallback(() => {
    void runMeCheck({ showChecking: true })
  }, [runMeCheck])

  const value = useMemo<AuthContextValue>(
    () => ({ status, account, errorMessage, signIn, signOut, checkAgain }),
    [status, account, errorMessage, signIn, signOut, checkAgain],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
