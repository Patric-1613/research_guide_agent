// Usage Protection M4.1 Part A: tests for the incremental SSE decoder.
// No network/fetch calls anywhere here -- every test feeds raw
// text/bytes directly into SSEDecoder.
import { describe, expect, it } from 'vitest'
import { SSEDecoder } from './sseDecoder'

function encode(text: string): Uint8Array {
  return new TextEncoder().encode(text)
}

describe('SSEDecoder', () => {
  it('parses one complete frame in a single push', () => {
    const decoder = new SSEDecoder()
    const results = decoder.pushText('event: started\ndata: {}\n\n')
    expect(results).toEqual([{ ok: true, event: 'started', data: {} }])
  })

  it('parses a frame split across several chunks', () => {
    const decoder = new SSEDecoder()
    expect(decoder.pushText('event: del')).toEqual([])
    expect(decoder.pushText('ta\ndata: {"tex')).toEqual([])
    expect(decoder.pushText('t":"hello"}')).toEqual([])
    const results = decoder.pushText('\n\n')
    expect(results).toEqual([{ ok: true, event: 'delta', data: { text: 'hello' } }])
  })

  it('parses multiple frames delivered in one chunk', () => {
    const decoder = new SSEDecoder()
    const results = decoder.pushText(
      'event: started\ndata: {}\n\nevent: delta\ndata: {"text":"a"}\n\nevent: delta\ndata: {"text":"b"}\n\n',
    )
    expect(results).toEqual([
      { ok: true, event: 'started', data: {} },
      { ok: true, event: 'delta', data: { text: 'a' } },
      { ok: true, event: 'delta', data: { text: 'b' } },
    ])
  })

  it('preserves a partial trailing frame across pushes, never losing it', () => {
    const decoder = new SSEDecoder()
    const first = decoder.pushText('event: started\ndata: {}\n\nevent: delta\ndata: {"tex')
    expect(first).toEqual([{ ok: true, event: 'started', data: {} }])
    const second = decoder.pushText('t":"finished"}\n\n')
    expect(second).toEqual([{ ok: true, event: 'delta', data: { text: 'finished' } }])
  })

  it('decodes a multibyte UTF-8 character split across raw byte chunks', () => {
    const decoder = new SSEDecoder()
    const full = encode('event: delta\ndata: {"text":"café 论文"}\n\n')
    // Split at an arbitrary byte offset that lands INSIDE a multibyte
    // UTF-8 sequence (é is 2 bytes, 论 and 文 are each 3 bytes) --
    // picking the midpoint of the whole buffer virtually guarantees
    // landing mid-codepoint somewhere in this string.
    const mid = Math.floor(full.length / 2)
    const first = decoder.pushBytes(full.slice(0, mid))
    const second = decoder.pushBytes(full.slice(mid))
    expect(first).toEqual([])
    expect(second).toEqual([{ ok: true, event: 'delta', data: { text: 'café 论文' } }])
  })

  it('splits every possible byte boundary of a multibyte payload and always reconstructs correctly', () => {
    const full = encode('event: delta\ndata: {"text":"café 论文 😀"}\n\n')
    for (let cut = 1; cut < full.length; cut++) {
      const decoder = new SSEDecoder()
      const results = [...decoder.pushBytes(full.slice(0, cut)), ...decoder.pushBytes(full.slice(cut))]
      expect(results).toEqual([{ ok: true, event: 'delta', data: { text: 'café 论文 😀' } }])
    }
  })

  it('rejects malformed JSON in the data field predictably, without throwing', () => {
    const decoder = new SSEDecoder()
    const results = decoder.pushText('event: delta\ndata: {not valid json\n\n')
    expect(results).toHaveLength(1)
    expect(results[0].ok).toBe(false)
    expect((results[0] as { raw: string }).raw).toContain('not valid json')
  })

  it('rejects a frame with no data line at all', () => {
    const decoder = new SSEDecoder()
    const results = decoder.pushText('event: started\n\n')
    expect(results).toEqual([{ ok: false, raw: 'event: started' }])
  })

  it('accepts and surfaces an unknown event type unchanged (decoder is protocol-agnostic)', () => {
    const decoder = new SSEDecoder()
    const results = decoder.pushText('event: some_future_event\ndata: {"x":1}\n\n')
    expect(results).toEqual([{ ok: true, event: 'some_future_event', data: { x: 1 } }])
  })

  it('a malformed frame does not corrupt parsing of subsequent well-formed frames', () => {
    const decoder = new SSEDecoder()
    const results = decoder.pushText('event: bad\ndata: {broken\n\nevent: delta\ndata: {"text":"ok"}\n\n')
    expect(results).toEqual([
      { ok: false, raw: 'event: bad\ndata: {broken' },
      { ok: true, event: 'delta', data: { text: 'ok' } },
    ])
  })

  it('finish() reports no dangling buffer for a cleanly terminated stream', () => {
    const decoder = new SSEDecoder()
    decoder.pushText('event: done\ndata: {}\n\n')
    expect(decoder.finish()).toEqual({ danglingBuffer: null })
  })

  it('finish() reports a dangling buffer when the stream ends mid-frame (truncated/malformed)', () => {
    const decoder = new SSEDecoder()
    decoder.pushText('event: delta\ndata: {"text":"incomplete')
    const result = decoder.finish()
    expect(result.danglingBuffer).not.toBeNull()
    expect(result.danglingBuffer).toContain('incomplete')
  })

  it('finish() flushes a pending partial multibyte codepoint from the underlying TextDecoder safely', () => {
    const decoder = new SSEDecoder()
    const full = encode('event: delta\ndata: {"text":"café"}')
    // Push everything except the final byte of the multibyte 'é' sequence.
    decoder.pushBytes(full.slice(0, full.length - 1))
    // Must not throw even though a codepoint was left incomplete.
    expect(() => decoder.finish()).not.toThrow()
  })

  it('ignores an empty push (no frames, no error)', () => {
    const decoder = new SSEDecoder()
    expect(decoder.pushText('')).toEqual([])
  })

  it('never uses a naive one-shot split -- consecutive pushes each only return their own newly-completed frames', () => {
    const decoder = new SSEDecoder()
    const first = decoder.pushText('event: a\ndata: {}\n\n')
    const second = decoder.pushText('event: b\ndata: {}\n\n')
    expect(first).toEqual([{ ok: true, event: 'a', data: {} }])
    expect(second).toEqual([{ ok: true, event: 'b', data: {} }])
  })
})
