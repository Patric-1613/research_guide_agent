// Day 5: the single Firebase app + Auth instance, created lazily and
// exactly once, only in `firebase` mode.
//
// Persistence is **in-memory only** (`inMemoryPersistence` passed to
// `initializeAuth`, not the default `getAuth` which uses IndexedDB): the
// ID token and auth state live only in this tab's memory and are gone on
// reload or tab close. This is deliberate -- Day 5's contract is that no
// ID token is ever written to localStorage, sessionStorage, IndexedDB,
// or any other persistent browser store. The cost is that a page reload
// returns the user to the sign-in screen (Google's popup normally
// re-establishes the session in one click without re-prompting).
//
// `disabled`/`basic` mode never calls anything here, so `firebase/app`
// and `firebase/auth` are only pulled into the running app when they are
// actually needed.

import { initializeApp, type FirebaseApp } from 'firebase/app'
import {
  browserPopupRedirectResolver,
  GoogleAuthProvider,
  inMemoryPersistence,
  initializeAuth,
  type Auth,
} from 'firebase/auth'
import { getFirebaseConfig } from './config'

let app: FirebaseApp | null = null
let auth: Auth | null = null

export function getFirebaseAuth(): Auth {
  if (auth) return auth
  const config = getFirebaseConfig()
  app = initializeApp({
    apiKey: config.apiKey,
    authDomain: config.authDomain,
    projectId: config.projectId,
    ...(config.appId ? { appId: config.appId } : {}),
  })
  auth = initializeAuth(app, {
    persistence: inMemoryPersistence,
    popupRedirectResolver: browserPopupRedirectResolver,
  })
  return auth
}

export function googleProvider(): GoogleAuthProvider {
  const provider = new GoogleAuthProvider()
  // Always let the user pick an account rather than silently reusing a
  // single Google session -- clearer in a shared-machine beta.
  provider.setCustomParameters({ prompt: 'select_account' })
  return provider
}

// Test-only: drop the memoized instances so a fresh `getFirebaseAuth()`
// re-reads config. Never called by application code.
export function __resetFirebaseForTests(): void {
  app = null
  auth = null
}
