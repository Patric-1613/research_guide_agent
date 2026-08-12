// Usage Protection M2.3 Part B: the one place that turns a structured
// API failure (ApiError, see ../../types) into concise, user-facing
// text. Every caller (currently: useCurationSession's shared error
// state) goes through this instead of re-deriving a message itself, so
// wording for a given reason_code only ever needs to change in one
// place.
//
// Never shows a raw reason_code, a raw HTTP status alone, or anything
// from ApiError.body directly -- only ApiError's own already-safe
// fields (reasonCode/safeMessage/validationErrors/retryAfterSeconds).
import { ApiError } from '../../types'

// seconds < 60: "about N seconds"; seconds >= 60: rounded UP to whole
// minutes ("about N minutes") -- never a live countdown, just one
// static hint computed at render time from the Retry-After value the
// response carried.
export function formatRetryHint(seconds: number): string {
  if (seconds < 60) {
    return `Try again in about ${seconds} second${seconds === 1 ? '' : 's'}.`
  }
  const minutes = Math.ceil(seconds / 60)
  return `Try again in about ${minutes} minute${minutes === 1 ? '' : 's'}.`
}

function withRetryHint(base: string, err: ApiError): string {
  return err.retryAfterSeconds != null ? `${base} ${formatRetryHint(err.retryAfterSeconds)}` : base
}

// research_agent/usage_guard.py's GuardReasonCode + research_agent/
// session_limits.py's SessionCapacityReasonCode + research_agent/
// request_limits.py's "request_body_too_large" -- every reason code
// this app's backend can currently send, mirrored field-for-field
// (same convention as src/types/index.ts's own "mirrors the backend's
// Pydantic models" header comment).
const REASON_CODE_MESSAGES: Record<string, (err: ApiError) => string> = {
  action_in_progress: () => 'Another action is already running for this review. Please wait and try again.',
  selected_paper_limit_reached: () =>
    'This review has reached its selected-paper limit. Remove some papers before adding more.',
  chat_turn_limit_reached: () =>
    'This chat has reached its turn limit. You can still review the papers, generate or export the report, and read the existing chat history.',
  request_body_too_large: () => 'The submitted request is too large. Please shorten it and try again.',
  session_hourly_limit_reached: (err) => withRetryHint('This session has reached its usage limit.', err),
  session_daily_limit_reached: (err) => withRetryHint('This session has reached its usage limit.', err),
  global_window_limit_reached: (err) => withRetryHint('The service is at capacity right now.', err),
  usage_protection_unavailable: () =>
    'Usage protection is temporarily unavailable, so no paid action was started. Please try again shortly.',
}

// Usage Protection M2.3: the one function every caller uses to turn a
// caught error (an ApiError, some other Error, or anything else a
// try/catch might see) into text safe to render directly.
export function getUserFacingErrorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 422) {
      const first = err.validationErrors?.[0]
      if (first) return first.loc ? `${first.loc}: ${first.msg}` : first.msg
      return 'Please check the submitted information and try again.'
    }
    if (err.reasonCode) {
      const known = REASON_CODE_MESSAGES[err.reasonCode]
      if (known) return known(err)
      // An unrecognized-but-present reason code still carries the
      // backend's own already-safe message (GUARD_REASON_MESSAGE/
      // SessionCapacityError.message are both deliberately generic,
      // user-safe text) -- falling back to it, rather than a raw
      // reason_code string, keeps this forward-compatible with a
      // reason code added to the backend before this map is updated.
      if (err.safeMessage) return err.safeMessage
    }
    if (err.safeMessage) return err.safeMessage
    return `Something went wrong (${err.status}). Please try again.`
  }
  // Unknown/network/parse failure: preserve the app's existing generic
  // fallback behavior exactly (String(err), the same expression
  // useCurationSession's own runAction already used) -- not something
  // this phase changes.
  return String(err)
}
