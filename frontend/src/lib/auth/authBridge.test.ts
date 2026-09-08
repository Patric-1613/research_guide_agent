import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearAuthBridge, getAuthHeaders, notifyUnauthorized, registerAuthBridge } from './authBridge'

afterEach(() => {
  clearAuthBridge()
})

describe('authBridge', () => {
  it('returns no headers when nothing is registered (disabled/basic mode)', async () => {
    expect(await getAuthHeaders()).toEqual({})
  })

  it('returns a Bearer header from the registered token getter', async () => {
    registerAuthBridge({ getToken: async () => 'id-token-abc', onUnauthorized: () => {} })
    expect(await getAuthHeaders()).toEqual({ Authorization: 'Bearer id-token-abc' })
  })

  it('returns no header when the getter yields null (signed out)', async () => {
    registerAuthBridge({ getToken: async () => null, onUnauthorized: () => {} })
    expect(await getAuthHeaders()).toEqual({})
  })

  it('swallows a token-getter rejection and returns no header', async () => {
    registerAuthBridge({ getToken: async () => { throw new Error('offline') }, onUnauthorized: () => {} })
    expect(await getAuthHeaders()).toEqual({})
  })

  it('clearAuthBridge stops adding headers and calling the handler', async () => {
    const onUnauthorized = vi.fn()
    registerAuthBridge({ getToken: async () => 't', onUnauthorized })
    clearAuthBridge()
    expect(await getAuthHeaders()).toEqual({})
    notifyUnauthorized()
    expect(onUnauthorized).not.toHaveBeenCalled()
  })

  it('notifyUnauthorized invokes the registered handler', () => {
    const onUnauthorized = vi.fn()
    registerAuthBridge({ getToken: async () => 't', onUnauthorized })
    notifyUnauthorized()
    expect(onUnauthorized).toHaveBeenCalledTimes(1)
  })
})
