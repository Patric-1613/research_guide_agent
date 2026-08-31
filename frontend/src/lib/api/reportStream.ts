// Usage Protection M4.3B: the fetch-based adapter that turns POST
// /curation/{session_id}/report/stream and .../report/regenerate/
// stream's raw SSE response body into a typed, ordered async iterable of
// ReportStreamServerEvent, for research_agent's own frozen M4.3A event
// vocabulary (research_agent/report_streaming.py). Reuses the M4.1
// SSEDecoder (./sseDecoder.ts) for wire-level framing ONLY -- mirrors
// lib/api/chatStream.ts's own shape closely, but stays entirely
// independent of it: M4.2 is closed, and this file does not import from
// or modify chatStream.ts.
//
// One shared internal generator (streamReportOperation) plus two thin,
// exported wrappers (streamGenerateReport/streamRegenerateReport) -- the
// only difference between the two operations is which URL suffix to
// POST to; fetch/decode/validate is otherwise identical.
//
// Deliberately does NOT use EventSource: EventSource only ever issues a
// GET with no request body, and both these endpoints are a POST carrying
// a JSON body -- fetch() with a manually-read ReadableStream is the only
// way to stream a POST response.
//
// This module enforces ONLY framing-level validity (a frame decodes as
// valid SSE, its event name is one this protocol actually defines, its
// data is a JSON object, and -- specifically for `completed`, whose
// payload is a full ReportOut -- that the object at least carries
// ReportOut's own always-required fields) -- NOT event-SEQUENCING
// semantics (e.g. "no phase before started", "completed at most once").
// Sequencing is a report-domain state-machine concern, owned by
// useCurationSession.ts, not this transport adapter.

import { baseUrl, throwApiErrorIfNotOk } from './client'
import { SSEDecoder } from './sseDecoder'
import type {
  CurationGenerateReportRequest,
  CurationRegenerateReportRequest,
  ReportStreamEventType,
  ReportStreamServerEvent,
} from '../../types'

const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set<ReportStreamEventType>([
  'started', 'phase', 'completed', 'error', 'done',
])

// ReportOut's own always-required fields (research_agent/api_app/
// schemas.py's ReportOut -- findings/limitations/future_scope/
// skipped_paper_ids have no default; every other field is optional).
// Checked structurally on a `completed` event's own payload specifically
// -- that's the one payload shape complex enough, and important enough,
// to be worth a fail-safe against a corrupted/truncated report body
// reaching the UI as though it were a real, usable ReportOut.
const REQUIRED_REPORT_OUT_KEYS = ['findings', 'limitations', 'future_scope', 'skipped_paper_ids'] as const

// The ONE transport-layer failure this module raises -- a frame that
// failed to decode as SSE (including malformed JSON -- SSEDecoder's own
// parseFrame already catches JSON.parse failures), an unrecognized event
// name/non-object payload, an invalid/incomplete `completed` payload, a
// response with no body, or a stream that ended without its last frame's
// own blank-line terminator (SSEDecoder.finish()'s own signal). Always a
// small, fixed, safe message -- never raw exception text. Callers
// (useCurationSession.ts) treat this the same way regardless of which
// specific cause produced it: stop the stream, surface a safe retryable
// message, reload canonical state.
export class ReportStreamTransportError extends Error {}

function toReportStreamEvent(eventName: string, data: unknown): ReportStreamServerEvent {
  if (!KNOWN_EVENT_TYPES.has(eventName)) {
    throw new ReportStreamTransportError('Received an unrecognized message from the server.')
  }
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new ReportStreamTransportError('Received a malformed message from the server.')
  }
  if (eventName === 'completed') {
    const record = data as Record<string, unknown>
    const missingRequiredField = REQUIRED_REPORT_OUT_KEYS.some((key) => !(key in record))
    if (missingRequiredField) {
      throw new ReportStreamTransportError('Received an incomplete report from the server.')
    }
  }
  // A minimal structural check, not a full schema validation -- same
  // level of trust this codebase's other API calls already place in a
  // same-origin, already-tested backend (e.g. client.ts's own `response.
  // json() as Promise<T>`). Event-sequencing/deeper payload correctness
  // is the caller's (useCurationSession.ts's) job, not this transport
  // layer's.
  return { type: eventName, data } as ReportStreamServerEvent
}

async function* streamReportOperation(
  path: string,
  req: CurationGenerateReportRequest | CurationRegenerateReportRequest,
  options: { signal: AbortSignal },
): AsyncGenerator<ReportStreamServerEvent> {
  const response = await fetch(`${baseUrl()}${path}`, {
    method: 'POST',
    // Same credentialed-CORS contract as lib/api/client.ts's request().
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: options.signal,
  })
  // Every preflight rejection (404/400/409/422/429/503) is a plain JSON
  // response from research_agent/services/curation_report_service.py's
  // own synchronous checks, which all run BEFORE any StreamingResponse
  // is constructed -- so this is the exact same non-2xx-to-ApiError
  // mapping every other endpoint in this app already gets, never a
  // special case for these two.
  await throwApiErrorIfNotOk(response)

  if (!response.body) {
    throw new ReportStreamTransportError('The server response had no body to stream.')
  }

  const reader = response.body.getReader()
  const decoder = new SSEDecoder()
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      for (const result of decoder.pushBytes(value)) {
        if (!result.ok) {
          throw new ReportStreamTransportError('Received a malformed message from the server.')
        }
        yield toReportStreamEvent(result.event, result.data)
      }
    }
  } finally {
    reader.releaseLock()
  }

  const { danglingBuffer } = decoder.finish()
  if (danglingBuffer !== null) {
    throw new ReportStreamTransportError('The connection ended before the response finished.')
  }
}

export function streamGenerateReport(
  sessionId: string,
  req: CurationGenerateReportRequest,
  options: { signal: AbortSignal },
): AsyncGenerator<ReportStreamServerEvent> {
  return streamReportOperation(`/curation/${sessionId}/report/stream`, req, options)
}

export function streamRegenerateReport(
  sessionId: string,
  req: CurationRegenerateReportRequest,
  options: { signal: AbortSignal },
): AsyncGenerator<ReportStreamServerEvent> {
  return streamReportOperation(`/curation/${sessionId}/report/regenerate/stream`, req, options)
}
