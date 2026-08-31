// Usage Protection M4.2B: the fetch-based adapter that turns POST
// /curation/{session_id}/chat/stream's raw SSE response body into a
// typed, ordered async iterable of ChatStreamServerEvent, for research_
// agent's own frozen M4.1/M4.2A event vocabulary (research_agent/
// chat_streaming.py). Reuses the M4.1 SSEDecoder (./sseDecoder.ts) for
// wire-level framing ONLY -- this module is the one place that knows the
// chat-specific event names and payload shapes; sseDecoder.ts itself
// stays completely protocol-agnostic, unchanged, same separation of
// concerns it already documents for itself.
//
// Deliberately does NOT use EventSource: EventSource only ever issues a
// GET with no request body, and this endpoint is a POST carrying a JSON
// message -- fetch() with a manually-read ReadableStream is the only way
// to stream a POST response.
//
// This module enforces ONLY framing-level validity (a frame decodes as
// valid SSE, its event name is one this protocol actually defines, its
// data is a JSON object) -- NOT event-SEQUENCING semantics (e.g. "no
// delta before started", "completed at most once"). Sequencing is a
// chat-domain state-machine concern, owned by useCurationSession.ts, not
// this transport adapter.

import { baseUrl, throwApiErrorIfNotOk } from './client'
import { SSEDecoder } from './sseDecoder'
import type { ChatStreamEventType, ChatStreamServerEvent, CurationChatRequest } from '../../types'

const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set<ChatStreamEventType>([
  'started', 'phase', 'delta', 'completed', 'error', 'done',
])

// The ONE transport-layer failure this module raises -- a frame that
// failed to decode as SSE, an unrecognized event name/non-object
// payload, a response with no body, or a stream that ended without its
// last frame's own blank-line terminator (SSEDecoder.finish()'s own
// signal). Always a small, fixed, safe message -- never raw exception
// text. Callers (useCurationSession.ts) treat this the same way
// regardless of which specific cause produced it: stop the stream,
// surface a safe retryable message, reload canonical state.
export class ChatStreamTransportError extends Error {}

function toChatStreamEvent(eventName: string, data: unknown): ChatStreamServerEvent {
  if (!KNOWN_EVENT_TYPES.has(eventName)) {
    throw new ChatStreamTransportError('Received an unrecognized message from the server.')
  }
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new ChatStreamTransportError('Received a malformed message from the server.')
  }
  // A minimal structural check, not a full schema validation -- same
  // level of trust this codebase's other API calls already place in a
  // same-origin, already-tested backend (e.g. client.ts's own `response.
  // json() as Promise<T>`). Event-sequencing/payload-field correctness is
  // the caller's (useCurationSession.ts's) job, not this transport layer's.
  return { type: eventName, data } as ChatStreamServerEvent
}

export async function* streamCurationChat(
  sessionId: string,
  req: CurationChatRequest,
  options: { signal: AbortSignal },
): AsyncGenerator<ChatStreamServerEvent> {
  const response = await fetch(`${baseUrl()}/curation/${sessionId}/chat/stream`, {
    method: 'POST',
    // Same credentialed-CORS contract as lib/api/client.ts's request().
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: options.signal,
  })
  // Every preflight rejection (404/400/409/422/429/503) is a plain JSON
  // response from research_agent/services/curation_chat_service.py::
  // stream_answer_curation_chat's own synchronous checks, which all run
  // BEFORE any StreamingResponse is constructed -- so this is the exact
  // same non-2xx-to-ApiError mapping every other endpoint in this app
  // already gets, never a special case for this one.
  await throwApiErrorIfNotOk(response)

  if (!response.body) {
    throw new ChatStreamTransportError('The server response had no body to stream.')
  }

  const reader = response.body.getReader()
  const decoder = new SSEDecoder()
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      for (const result of decoder.pushBytes(value)) {
        if (!result.ok) {
          throw new ChatStreamTransportError('Received a malformed message from the server.')
        }
        yield toChatStreamEvent(result.event, result.data)
      }
    }
  } finally {
    reader.releaseLock()
  }

  const { danglingBuffer } = decoder.finish()
  if (danglingBuffer !== null) {
    throw new ChatStreamTransportError('The connection ended before the response finished.')
  }
}
