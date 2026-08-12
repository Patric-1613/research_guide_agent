// Usage Protection M4.1 Part A: an incremental Server-Sent-Events frame
// decoder for a `fetch()`-consumed `ReadableStream<Uint8Array>` response
// body. Deliberately generic/protocol-agnostic -- this file knows only
// SSE wire syntax (an `event:` line, a `data:` line, a blank-line
// terminator), never any chat/report-specific event vocabulary or
// payload shape. NOT wired into any React component or API call yet --
// see the backend's own research_agent/chat_streaming.py module
// docstring for the matching server-side protocol this is designed to
// consume, once a real streaming endpoint exists (M4.2).
//
// Never uses a naive `buffer.split('\n\n')` one-shot approach: a real
// stream can, and will, split a frame across multiple `push()` calls
// (a chunk boundary has no relationship to a frame boundary), so this
// decoder keeps whatever trailing, not-yet-terminated text it has seen
// across calls, only ever emitting a frame once its own blank-line
// terminator has actually arrived.

export interface DecodedSSEFrame {
  ok: true
  event: string
  data: unknown
}

export interface MalformedSSEFrame {
  ok: false
  raw: string
}

export type SSEDecodeResult = DecodedSSEFrame | MalformedSSEFrame

// A single frame is `event: <type>\ndata: <json>\n\n` -- this decoder
// also tolerates a frame with no `event:` line (data-only, valid per
// the SSE spec) by falling back to event: "" -- callers with a fixed,
// known event vocabulary (a future chat/report protocol layer) are
// expected to reject an empty/unrecognized event name themselves; this
// decoder's own job stops at "did this parse as valid SSE framing."
function parseFrame(raw: string): SSEDecodeResult {
  const lines = raw.split('\n')
  let event = ''
  const dataLines: string[] = []
  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      // Preserve everything after "data:" verbatim except exactly one
      // leading space (the SSE spec's own convention: "data: X" carries
      // value "X", not " X") -- never trim trailing whitespace, which
      // could be significant inside the JSON payload itself.
      const value = line.slice('data:'.length)
      dataLines.push(value.startsWith(' ') ? value.slice(1) : value)
    } else if (line.trim() === '') {
      // A blank line inside the frame body itself (shouldn't happen
      // for a well-formed single-JSON-line payload, but tolerated as a
      // no-op rather than treated as a second terminator) -- ignored.
      continue
    } else {
      // A line that's neither `event:`, `data:`, nor blank -- not
      // valid SSE framing this decoder understands. Reject predictably
      // rather than guessing at a partial interpretation.
      return { ok: false, raw }
    }
  }
  if (dataLines.length === 0) {
    return { ok: false, raw }
  }
  const joined = dataLines.join('\n')
  try {
    const data: unknown = JSON.parse(joined)
    return { ok: true, event, data }
  } catch {
    return { ok: false, raw }
  }
}

export class SSEDecoder {
  private decoder = new TextDecoder('utf-8')
  private buffer = ''

  /**
   * Feeds one more chunk of raw bytes into the decoder. Returns every
   * COMPLETE frame the accumulated buffer now contains (zero, one, or
   * many -- multiple frames can legitimately arrive in a single
   * chunk); any trailing, not-yet-terminated text is kept internally
   * for the next call. `{stream: true}` is what makes a multibyte
   * UTF-8 codepoint split across two chunks decode correctly -- the
   * browser's own TextDecoder buffers an incomplete trailing byte
   * sequence internally between calls, exactly mirroring what this
   * decoder itself does one level up for incomplete FRAMES.
   */
  pushBytes(chunk: Uint8Array): SSEDecodeResult[] {
    return this.pushText(this.decoder.decode(chunk, { stream: true }))
  }

  /** Same incremental-buffering contract as pushBytes, for a caller
   * that already has decoded text (e.g. a test, or a text-mode
   * stream reader). */
  pushText(text: string): SSEDecodeResult[] {
    this.buffer += text
    const results: SSEDecodeResult[] = []
    let terminatorIndex = this.buffer.indexOf('\n\n')
    while (terminatorIndex !== -1) {
      const rawFrame = this.buffer.slice(0, terminatorIndex)
      this.buffer = this.buffer.slice(terminatorIndex + 2)
      if (rawFrame.length > 0) {
        results.push(parseFrame(rawFrame))
      }
      terminatorIndex = this.buffer.indexOf('\n\n')
    }
    return results
  }

  /**
   * Call once the underlying stream has genuinely ended (the fetch
   * body reader reports `done: true`). Flushes the TextDecoder's own
   * trailing state and reports whether an incomplete, never-terminated
   * frame was left sitting in the buffer -- a real signal of a
   * truncated/malformed stream (a clean stream's last frame always
   * ends with its own blank-line terminator, already consumed by
   * pushBytes/pushText above), never silently discarded.
   */
  finish(): { danglingBuffer: string | null } {
    this.buffer += this.decoder.decode()
    const dangling = this.buffer.trim().length > 0 ? this.buffer : null
    this.buffer = ''
    return { danglingBuffer: dangling }
  }
}
