// Day 5: the frontend auth-mode contract, mirroring the backend's
// AUTH_MODE (research_agent/config/settings.py). One place reads and
// validates the relevant `import.meta.env` values, so no component or
// transport re-parses them.
//
//   - "disabled" (the default when VITE_AUTH_MODE is unset/empty): no
//     sign-in, no Firebase SDK loaded -- the exact current local-dev
//     behaviour.
//   - "basic": no sign-in UI; the browser replays HTTP Basic-Auth
//     credentials on its own (credentials: "include"). No Firebase SDK.
//   - "firebase": Google sign-in required; the Firebase public web
//     config below must all be present, or the app fails loudly at
//     startup rather than rendering a half-working state.
//
// Read at call time (not module load) via `import.meta.env`, matching
// lib/api/client.ts's own convention, so tests can stub the mode per
// case with `vi.stubEnv`.

export type AuthMode = 'disabled' | 'basic' | 'firebase'

const VALID_MODES: readonly AuthMode[] = ['disabled', 'basic', 'firebase']

export class AuthConfigError extends Error {}

export function getAuthMode(): AuthMode {
  const raw = (import.meta.env.VITE_AUTH_MODE ?? '').trim().toLowerCase()
  if (raw === '') return 'disabled'
  if ((VALID_MODES as readonly string[]).includes(raw)) return raw as AuthMode
  throw new AuthConfigError(
    `VITE_AUTH_MODE="${import.meta.env.VITE_AUTH_MODE}" is not valid. ` +
      `Use one of: ${VALID_MODES.join(', ')} (or leave it unset for "disabled").`,
  )
}

export function isFirebaseMode(): boolean {
  return getAuthMode() === 'firebase'
}

export interface FirebasePublicConfig {
  apiKey: string
  authDomain: string
  projectId: string
  appId?: string
}

// The env keys that must be present (and non-empty) in firebase mode.
const REQUIRED_FIREBASE_KEYS = [
  'VITE_FIREBASE_API_KEY',
  'VITE_FIREBASE_AUTH_DOMAIN',
  'VITE_FIREBASE_PROJECT_ID',
] as const

/**
 * The validated Firebase public web config. Throws `AuthConfigError`
 * (listing every missing key at once) when `VITE_AUTH_MODE=firebase` but
 * a required public value is missing or blank -- a misconfiguration must
 * be an obvious, immediate startup failure, never a silent one.
 */
export function getFirebaseConfig(): FirebasePublicConfig {
  const env = import.meta.env
  const missing = REQUIRED_FIREBASE_KEYS.filter((key) => !(env[key] ?? '').trim())
  if (missing.length > 0) {
    throw new AuthConfigError(
      `VITE_AUTH_MODE=firebase requires these public Firebase values, which are missing or empty: ` +
        `${missing.join(', ')}. These are public client-side values (not secrets) from the Firebase ` +
        `console's Web app config -- see .env.example. VITE_FIREBASE_PROJECT_ID must match the backend's ` +
        `FIREBASE_PROJECT_ID.`,
    )
  }
  return {
    apiKey: env.VITE_FIREBASE_API_KEY!.trim(),
    authDomain: env.VITE_FIREBASE_AUTH_DOMAIN!.trim(),
    projectId: env.VITE_FIREBASE_PROJECT_ID!.trim(),
    appId: (env.VITE_FIREBASE_APP_ID ?? '').trim() || undefined,
  }
}
