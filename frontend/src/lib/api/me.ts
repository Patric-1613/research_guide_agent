// Day 5: GET /me -- the only call the auth layer makes directly. Takes
// the ID token as an explicit argument (the caller, AuthProvider, just
// obtained it from Firebase's getIdToken()) rather than going through
// authBridge, so it works during sign-in before the bridge is wired and
// never triggers the shared 401 "session expired" handler.

import { baseUrl, throwApiErrorIfNotOk } from './client'
import type { MeResponse } from '../../types'

export async function fetchMe(idToken: string, signal?: AbortSignal): Promise<MeResponse> {
  const response = await fetch(`${baseUrl()}/me`, {
    method: 'GET',
    credentials: 'include',
    headers: { Authorization: `Bearer ${idToken}` },
    signal,
  })
  await throwApiErrorIfNotOk(response)
  return response.json() as Promise<MeResponse>
}
