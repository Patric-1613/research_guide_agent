import { afterEach, describe, expect, it, vi } from 'vitest'
import { AuthConfigError, getAuthMode, getFirebaseConfig, isFirebaseMode } from './config'

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('getAuthMode', () => {
  it('defaults to "disabled" when VITE_AUTH_MODE is unset or empty', () => {
    vi.stubEnv('VITE_AUTH_MODE', '')
    expect(getAuthMode()).toBe('disabled')
  })

  it('accepts disabled / basic / firebase, case-insensitively and trimmed', () => {
    vi.stubEnv('VITE_AUTH_MODE', 'basic')
    expect(getAuthMode()).toBe('basic')
    vi.stubEnv('VITE_AUTH_MODE', '  FireBase ')
    expect(getAuthMode()).toBe('firebase')
    expect(isFirebaseMode()).toBe(true)
  })

  it('throws AuthConfigError on an unrecognised value', () => {
    vi.stubEnv('VITE_AUTH_MODE', 'oauth')
    expect(() => getAuthMode()).toThrow(AuthConfigError)
    expect(() => getAuthMode()).toThrow(/not valid/)
  })
})

describe('getFirebaseConfig', () => {
  it('returns the trimmed public config when every required value is present', () => {
    vi.stubEnv('VITE_FIREBASE_API_KEY', ' key-123 ')
    vi.stubEnv('VITE_FIREBASE_AUTH_DOMAIN', 'my-app.firebaseapp.com')
    vi.stubEnv('VITE_FIREBASE_PROJECT_ID', 'my-app-12345')
    vi.stubEnv('VITE_FIREBASE_APP_ID', '1:2:web:3')
    expect(getFirebaseConfig()).toEqual({
      apiKey: 'key-123',
      authDomain: 'my-app.firebaseapp.com',
      projectId: 'my-app-12345',
      appId: '1:2:web:3',
    })
  })

  it('omits appId when it is not set', () => {
    vi.stubEnv('VITE_FIREBASE_API_KEY', 'k')
    vi.stubEnv('VITE_FIREBASE_AUTH_DOMAIN', 'd')
    vi.stubEnv('VITE_FIREBASE_PROJECT_ID', 'p')
    vi.stubEnv('VITE_FIREBASE_APP_ID', '')
    expect(getFirebaseConfig().appId).toBeUndefined()
  })

  it('throws AuthConfigError naming EVERY missing value at once', () => {
    vi.stubEnv('VITE_FIREBASE_API_KEY', '')
    vi.stubEnv('VITE_FIREBASE_AUTH_DOMAIN', '   ')
    vi.stubEnv('VITE_FIREBASE_PROJECT_ID', 'p')
    let caught: unknown
    try {
      getFirebaseConfig()
    } catch (err) {
      caught = err
    }
    expect(caught).toBeInstanceOf(AuthConfigError)
    const message = (caught as Error).message
    expect(message).toContain('VITE_FIREBASE_API_KEY')
    expect(message).toContain('VITE_FIREBASE_AUTH_DOMAIN')
    expect(message).not.toContain('VITE_FIREBASE_PROJECT_ID,') // present, so not listed
  })
})
