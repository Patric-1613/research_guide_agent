import { describe, expect, it } from 'vitest'
import { ApiError } from './index'

describe('ApiError', () => {
  it('parses the M2 guard-error shape: reason_code + message', () => {
    const err = new ApiError(409, {
      detail: { reason_code: 'action_in_progress', message: 'Another request is already in progress for this session. Please wait for it to finish.' },
    })

    expect(err.reasonCode).toBe('action_in_progress')
    expect(err.safeMessage).toBe('Another request is already in progress for this session. Please wait for it to finish.')
    expect(err.validationErrors).toBeNull()
  })

  it('parses every known reason code from the {detail: {reason_code, message}} shape', () => {
    const codes = [
      'action_in_progress',
      'selected_paper_limit_reached',
      'chat_turn_limit_reached',
      'request_body_too_large',
      'session_hourly_limit_reached',
      'session_daily_limit_reached',
      'global_window_limit_reached',
      'usage_protection_unavailable',
    ]
    for (const code of codes) {
      const err = new ApiError(409, { detail: { reason_code: code, message: 'safe text' } })
      expect(err.reasonCode).toBe(code)
      expect(err.safeMessage).toBe('safe text')
    }
  })

  it('parses a plain-string detail (e.g. ServiceError 400/404) as safeMessage, keeping .message backward compatible', () => {
    const err = new ApiError(404, { detail: 'session_id not found' })

    expect(err.safeMessage).toBe('session_id not found')
    expect(err.message).toContain('session_id not found')
    expect(err.reasonCode).toBeNull()
  })

  it('parses a FastAPI 422 detail array into structured validationErrors, WITHOUT the raw echoed input', () => {
    const rawInput = 'x'.repeat(3000)
    const err = new ApiError(422, {
      detail: [
        {
          type: 'string_too_long', loc: ['body', 'topic'], msg: 'String should have at most 2000 characters',
          ctx: { max_length: 2000 }, input: rawInput,
        },
      ],
    })

    expect(err.validationErrors).toEqual([{ loc: 'body.topic', msg: 'String should have at most 2000 characters' }])
    // The raw oversized input must never leak into any of the SAFE,
    // derived fields this class computes for display -- `.body` itself
    // still retains the original response (kept for completeness/
    // debugging, never rendered directly; see lib/api/errorMessages.ts,
    // which only ever reads reasonCode/safeMessage/validationErrors).
    expect(err.message).not.toContain(rawInput)
    expect(JSON.stringify(err.validationErrors)).not.toContain(rawInput)
  })

  it('never puts a serialized/stringified response object into .message for a structured or array detail', () => {
    const objErr = new ApiError(409, { detail: { reason_code: 'action_in_progress', message: 'safe' } })
    expect(objErr.message).not.toMatch(/[{}[\]]/)

    const arrErr = new ApiError(422, { detail: [{ type: 't', loc: ['body', 'x'], msg: 'bad' }] })
    expect(arrErr.message).not.toMatch(/[{}[\]]/)
  })

  it('handles a null body (malformed JSON / empty response) safely, with a generic message', () => {
    const err = new ApiError(500, null)

    expect(err.reasonCode).toBeNull()
    expect(err.safeMessage).toBeNull()
    expect(err.validationErrors).toBeNull()
    expect(err.message).toBe('API request failed (500).')
  })

  it('parses a valid positive-integer Retry-After header', () => {
    const err = new ApiError(429, { detail: { reason_code: 'global_window_limit_reached', message: 'x' } }, '180')
    expect(err.retryAfterSeconds).toBe(180)
  })

  it('ignores a malformed Retry-After header (non-numeric)', () => {
    const err = new ApiError(429, { detail: {} }, 'not-a-number')
    expect(err.retryAfterSeconds).toBeNull()
  })

  it('ignores an HTTP-date-format Retry-After header -- only the integer-seconds format is supported', () => {
    const err = new ApiError(429, { detail: {} }, 'Wed, 21 Oct 2026 07:28:00 GMT')
    expect(err.retryAfterSeconds).toBeNull()
  })

  it('ignores a zero or negative Retry-After value', () => {
    expect(new ApiError(429, { detail: {} }, '0').retryAfterSeconds).toBeNull()
    expect(new ApiError(429, { detail: {} }, '-5').retryAfterSeconds).toBeNull()
  })

  it('treats a missing Retry-After header as null', () => {
    const err = new ApiError(429, { detail: {} })
    expect(err.retryAfterSeconds).toBeNull()
  })

  it('treats an empty-string Retry-After header as null', () => {
    const err = new ApiError(429, { detail: {} }, '')
    expect(err.retryAfterSeconds).toBeNull()
  })
})
