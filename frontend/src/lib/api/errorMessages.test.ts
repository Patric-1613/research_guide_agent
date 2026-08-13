import { describe, expect, it } from 'vitest'
import { ApiError } from '../../types'
import { formatRetryHint, getUserFacingErrorMessage } from './errorMessages'

function guardError(reasonCode: string, message: string, retryAfterHeader?: string) {
  return new ApiError(409, { detail: { reason_code: reasonCode, message } }, retryAfterHeader)
}

describe('getUserFacingErrorMessage', () => {
  it('maps action_in_progress to the exact required message', () => {
    const err = guardError('action_in_progress', 'Another request is already in progress for this session. Please wait for it to finish.')
    expect(getUserFacingErrorMessage(err)).toBe(
      'Another action is already running for this review. Please wait and try again.',
    )
  })

  it('maps selected_paper_limit_reached to a message explaining removal is required', () => {
    const err = guardError('selected_paper_limit_reached', 'x')
    const message = getUserFacingErrorMessage(err)
    expect(message).toMatch(/limit/i)
    expect(message).toMatch(/remove/i)
  })

  it('maps chat_turn_limit_reached to a message preserving access to review/report/export', () => {
    const err = guardError('chat_turn_limit_reached', 'x')
    const message = getUserFacingErrorMessage(err)
    expect(message).toMatch(/turn limit/i)
    expect(message).toMatch(/report/i)
  })

  it('maps request_body_too_large to a message about the request being too large', () => {
    const err = guardError('request_body_too_large', 'x')
    expect(getUserFacingErrorMessage(err)).toBe('The submitted request is too large. Please shorten it and try again.')
  })

  it('maps usage_protection_unavailable to a message noting no paid action started', () => {
    const err = guardError('usage_protection_unavailable', 'x')
    const message = getUserFacingErrorMessage(err)
    expect(message).toMatch(/temporarily unavailable/i)
    expect(message).toMatch(/no paid action/i)
  })

  it.each(['session_hourly_limit_reached', 'session_daily_limit_reached', 'global_window_limit_reached'])(
    'maps %s to a usage-limit message with no retry hint when Retry-After is absent',
    (code) => {
      const err = guardError(code, 'x')
      const message = getUserFacingErrorMessage(err)
      expect(message.toLowerCase()).toMatch(/limit|capacity/)
      expect(message).not.toMatch(/try again in/i)
    },
  )

  it('appends a seconds-based retry hint when Retry-After is under 60 seconds', () => {
    const err = guardError('session_hourly_limit_reached', 'x', '45')
    expect(getUserFacingErrorMessage(err)).toContain('Try again in about 45 seconds.')
  })

  it('appends a singular "second" hint for exactly 1 second', () => {
    const err = guardError('global_window_limit_reached', 'x', '1')
    expect(getUserFacingErrorMessage(err)).toContain('Try again in about 1 second.')
  })

  it('appends a minutes-based retry hint, rounded up, for 60 seconds or more', () => {
    const err = guardError('session_daily_limit_reached', 'x', '181')
    // 181s -> 3.01... minutes -> rounds up to 4
    expect(getUserFacingErrorMessage(err)).toContain('Try again in about 4 minutes.')
  })

  it('formats exactly 60 seconds as 1 minute, not 60 seconds', () => {
    expect(formatRetryHint(60)).toBe('Try again in about 1 minute.')
  })

  it('formats exactly 180 seconds as 3 minutes', () => {
    expect(formatRetryHint(180)).toBe('Try again in about 3 minutes.')
  })

  it('formats 59 seconds as seconds, not minutes', () => {
    expect(formatRetryHint(59)).toBe('Try again in about 59 seconds.')
  })

  it('falls back to the backend-authored safe message for an unrecognized reason code', () => {
    const err = guardError('some_future_reason_code', 'A safe backend-authored explanation.')
    expect(getUserFacingErrorMessage(err)).toBe('A safe backend-authored explanation.')
  })

  it('never shows the raw reason_code string to the user', () => {
    for (const code of [
      'action_in_progress', 'selected_paper_limit_reached', 'chat_turn_limit_reached',
      'request_body_too_large', 'session_hourly_limit_reached', 'usage_protection_unavailable',
    ]) {
      const message = getUserFacingErrorMessage(guardError(code, 'x'))
      expect(message).not.toContain(code)
    }
  })

  it('maps a 422 with validation errors to a concise field-level message', () => {
    const err = new ApiError(422, {
      detail: [{ type: 'string_too_long', loc: ['body', 'topic'], msg: 'String should have at most 2000 characters' }],
    })
    expect(getUserFacingErrorMessage(err)).toBe('body.topic: String should have at most 2000 characters')
  })

  it('maps a 422 with no derivable validation detail to the generic fallback', () => {
    const err = new ApiError(422, { detail: [] })
    expect(getUserFacingErrorMessage(err)).toBe('Please check the submitted information and try again.')
  })

  it('maps a plain-string-detail ApiError (e.g. 404 session not found) to that safe message', () => {
    const err = new ApiError(404, { detail: 'session_id not found' })
    expect(getUserFacingErrorMessage(err)).toBe('session_id not found')
  })

  it('falls back to a generic message for an ApiError with no derivable detail at all', () => {
    const err = new ApiError(500, null)
    expect(getUserFacingErrorMessage(err)).toBe('Something went wrong (500). Please try again.')
  })

  it('UXH.3: an unexpected Error never exposes its raw message text', () => {
    const err = new Error('network down')
    const message = getUserFacingErrorMessage(err)
    expect(message).toBe('Something went wrong. Please try again.')
    expect(message).not.toContain('network down')
  })

  it('UXH.3: a TypeError (e.g. a thrown coding defect) still returns the safe generic fallback', () => {
    const err = new TypeError("Cannot read properties of undefined (reading 'foo')")
    expect(getUserFacingErrorMessage(err)).toBe('Something went wrong. Please try again.')
  })

  it('UXH.3: a raw thrown string never renders verbatim', () => {
    expect(getUserFacingErrorMessage('/Users/someone/secret/path.ts:42')).toBe('Something went wrong. Please try again.')
  })

  it('UXH.3: a plain thrown object never stringifies into the message', () => {
    expect(getUserFacingErrorMessage({ some: 'internal', detail: 'object' })).toBe('Something went wrong. Please try again.')
  })

  it('UXH.3: null and undefined both return the safe generic fallback', () => {
    expect(getUserFacingErrorMessage(null)).toBe('Something went wrong. Please try again.')
    expect(getUserFacingErrorMessage(undefined)).toBe('Something went wrong. Please try again.')
  })

  it('UXH.3: an AbortError that reaches this function is not presented as a scary failure', () => {
    const err = new DOMException('The operation was aborted.', 'AbortError')
    const message = getUserFacingErrorMessage(err)
    expect(message).toBe('Something went wrong. Please try again.')
    expect(message).not.toMatch(/abort/i)
  })

  it('UXH.3: a raw error message containing a URL or file path is never leaked', () => {
    const err = new Error('fetch failed for https://internal.example.com/api/v9/secret?key=abc123')
    const message = getUserFacingErrorMessage(err)
    expect(message).not.toMatch(/https?:\/\//)
    expect(message).not.toContain('secret')
  })

  it('never includes a raw response object/body in the returned message for any case above', () => {
    const cases = [
      guardError('action_in_progress', 'x'),
      new ApiError(422, { detail: [{ type: 't', loc: ['body', 'x'], msg: 'bad' }] }),
      new ApiError(404, { detail: 'not found' }),
      new ApiError(500, null),
    ]
    for (const err of cases) {
      const message = getUserFacingErrorMessage(err)
      expect(message).not.toMatch(/^\{.*\}$/)
      expect(message).not.toContain('"detail"')
    }
  })
})
