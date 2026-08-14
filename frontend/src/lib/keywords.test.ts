import { describe, expect, it } from 'vitest'
import { aggregateKeywords, canonicalKeywordKey } from './keywords'
import type { PaperOut } from '../types'

function paperWithKeywords(keywords: string[]): Pick<PaperOut, 'keywords'> {
  return { keywords }
}

describe('canonicalKeywordKey', () => {
  it('folds case, Unicode-compatibly', () => {
    expect(canonicalKeywordKey('Neural Networks')).toBe(canonicalKeywordKey('neural networks'))
    expect(canonicalKeywordKey('RAG')).toBe(canonicalKeywordKey('rag'))
  })

  it('treats ASCII hyphens and spaces as equivalent', () => {
    expect(canonicalKeywordKey('Retrieval-Augmented Generation')).toBe(
      canonicalKeywordKey('Retrieval Augmented Generation'),
    )
  })

  it('treats Unicode dash variants (en dash, em dash) the same as ASCII hyphens', () => {
    expect(canonicalKeywordKey('Retrieval–Augmented Generation')).toBe(
      canonicalKeywordKey('Retrieval-Augmented Generation'),
    )
    expect(canonicalKeywordKey('Retrieval—Augmented Generation')).toBe(
      canonicalKeywordKey('Retrieval-Augmented Generation'),
    )
  })

  it('collapses whitespace and trims', () => {
    expect(canonicalKeywordKey('  Graph   Neural Networks  ')).toBe(canonicalKeywordKey('Graph Neural Networks'))
  })

  it('does NOT perform semantic or acronym/full-form merging', () => {
    expect(canonicalKeywordKey('RAG')).not.toBe(canonicalKeywordKey('Retrieval-Augmented Generation'))
  })
})

describe('aggregateKeywords', () => {
  it('groups hyphen/space/case variants under one canonical option', () => {
    const options = aggregateKeywords([
      paperWithKeywords(['Retrieval-Augmented Generation']),
      paperWithKeywords(['retrieval augmented generation']),
    ])

    expect(options).toHaveLength(1)
    expect(options[0].count).toBe(2)
  })

  it('counts each canonical keyword at most once per paper, even with duplicate surface variants within that paper', () => {
    const options = aggregateKeywords([paperWithKeywords(['RAG', 'rag', 'Rag'])])

    expect(options).toHaveLength(1)
    expect(options[0].count).toBe(1)
  })

  it('never inflates counts across papers beyond the number of distinct papers', () => {
    const options = aggregateKeywords([
      paperWithKeywords(['Neural Networks', 'neural networks']),
      paperWithKeywords(['Neural Networks']),
      paperWithKeywords(['NEURAL NETWORKS']),
    ])

    expect(options).toHaveLength(1)
    expect(options[0].count).toBe(3)
  })

  it('prefers the surface label occurring on the most papers', () => {
    const options = aggregateKeywords([
      paperWithKeywords(['neural networks']),
      paperWithKeywords(['neural networks']),
      paperWithKeywords(['Neural Networks']),
    ])

    expect(options[0].label).toBe('neural networks')
  })

  it('resolves display-label ties deterministically by first-seen order', () => {
    const options = aggregateKeywords([
      paperWithKeywords(['Neural Networks']),
      paperWithKeywords(['neural networks']),
    ])

    // Both labels occur exactly once (across distinct papers) -- the
    // FIRST one encountered in batch order wins, not an arbitrary map
    // iteration order.
    expect(options[0].label).toBe('Neural Networks')
  })

  it('does not merge distinct keywords (RAG vs Retrieval-Augmented Generation)', () => {
    const options = aggregateKeywords([
      paperWithKeywords(['RAG', 'Retrieval-Augmented Generation']),
    ])

    expect(options).toHaveLength(2)
    const labels = options.map((o) => o.label).sort()
    expect(labels).toEqual(['RAG', 'Retrieval-Augmented Generation'])
  })

  it('ignores keywords that canonicalize to an empty string', () => {
    const options = aggregateKeywords([paperWithKeywords(['   ', '-'])])
    expect(options).toHaveLength(0)
  })

  it('produces deterministic option ordering (first-seen canonical key order) across repeated calls', () => {
    const papers = [
      paperWithKeywords(['zebra topic', 'apple topic']),
      paperWithKeywords(['common topic']),
    ]
    const first = aggregateKeywords(papers)
    const second = aggregateKeywords(papers)

    expect(first.map((o) => o.key)).toEqual(second.map((o) => o.key))
    expect(first.map((o) => o.key)).toEqual([
      canonicalKeywordKey('zebra topic'),
      canonicalKeywordKey('apple topic'),
      canonicalKeywordKey('common topic'),
    ])
  })
})
