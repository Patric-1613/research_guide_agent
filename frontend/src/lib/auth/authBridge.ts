// Day 5: the seam between the React auth layer and the framework-
// agnostic transport layer (lib/api/client.ts, chatStream.ts,
// reportStream.ts, the report-export download).
//
// In `firebase` mode, `AuthProvider` registers a token getter and a
// 401 handler here on mount and unregisters on unmount. In
// `disabled`/`basic` mode nothing is ever registered, so `getAuthHeaders()`
// returns `{}` and no `Authorization` header is ever added -- the
// existing single-user / Basic-Auth behaviour is byte-identical.
//
// The token getter must call Firebase's `getIdToken()` each time so the
// SDK can transparently refresh an expiring token. This module never
// stores the returned token.

type TokenGetter = () => Promise<string | null>

let tokenGetter: TokenGetter | null = null
let unauthorizedHandler: (() => void) | null = null

export function registerAuthBridge(opts: { getToken: TokenGetter; onUnauthorized: () => void }): void {
  tokenGetter = opts.getToken
  unauthorizedHandler = opts.onUnauthorized
}

export function clearAuthBridge(): void {
  tokenGetter = null
  unauthorizedHandler = null
}

/**
 * `{ Authorization: 'Bearer <id token>' }` in firebase mode when a user
 * is signed in, otherwise `{}`. Fetches a fresh token every call (via
 * the registered getter, which calls Firebase's own `getIdToken()`), so
 * an about-to-expire token is refreshed before the request goes out.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
  if (!tokenGetter) return {}
  let token: string | null = null
  try {
    token = await tokenGetter()
  } catch {
    // A token-retrieval failure is surfaced by the request itself
    // failing (and the backend's 401), not by throwing here.
    token = null
  }
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/**
 * Called by the transport layer on any `401`. `AuthProvider` uses this
 * to move to an honest "session expired / please sign in again" state.
 * The originating request still rejects with its `ApiError` -- this is
 * never a silent retry.
 */
export function notifyUnauthorized(): void {
  unauthorizedHandler?.()
}
