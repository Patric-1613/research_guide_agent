import type { PaperOut } from '../types'

// Paper Keywords and Filtering, K4.2: the one canonical comparison key used
// by BOTH keyword aggregation (grouping hyphen/space/case surface variants
// of the same keyword into one option) and filtering/search (so a search
// with or without hyphens, or in any casing, matches the same options) --
// deliberately the single source of truth so the two can never disagree
// with each other.
//
// - `toLocaleLowerCase()` (not `toLowerCase()`) for a Unicode-aware,
//   locale-independent case fold -- handles characters plain ASCII
//   lowercasing misses (e.g. the Turkish dotless i, or accented Latin
//   letters), matching the intent (not the exact mechanism -- JS has no
//   built-in Unicode "casefold" primitive) of the backend's Python
//   `casefold()` counterpart (research_agent/keywords.py's
//   `_canonical_tokens`).
// - A small dash-variant character class (hyphen-minus plus the common
//   Unicode dash block) is replaced with a plain space before whitespace
//   collapse, so "Retrieval-Augmented Generation" and "Retrieval Augmented
//   Generation" canonicalize identically -- mirroring the backend's own
//   `_DASH_VARIANTS_RE`.
// - Deliberately NOT semantic/acronym merging: "RAG" and
//   "Retrieval-Augmented Generation" canonicalize to two different keys
//   and are never merged -- only literal hyphen/space/case variants of the
//   SAME surface phrase are.
const _DASH_VARIANTS_RE = /[-‐‑‒–—―−]/g

export function canonicalKeywordKey(keyword: string): string {
  return keyword
    .toLocaleLowerCase()
    .replace(_DASH_VARIANTS_RE, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export interface KeywordOption {
  key: string
  label: string
  count: number
}

// Aggregates every paper's keywords into one option per canonical key,
// across `papers` in their given (batch) order.
//
// - Each paper contributes a canonical key AT MOST ONCE, even if that
//   paper's own keyword list happens to contain two surface variants of
//   the same phrase (e.g. both "RAG" and "rag") -- `count` is a
//   distinct-PAPER count, never a raw occurrence count, so one paper can
//   never inflate a keyword's count on its own.
// - The display label is the surface form seen on the MOST papers (ties
//   broken by first-seen order across `papers`, i.e. whichever surface
//   variant this function encountered first while scanning in batch
//   order -- deterministic because `papers` itself is always iterated in
//   one fixed order); a further tie (equal occurrence count AND
//   simultaneous first sighting, only possible if a single paper lists
//   two variants that both canonicalize the same way) breaks by label,
//   alphabetically.
// - Returned in the SAME "first-seen canonical key" order every time for
//   the same input, so callers that need a stable base ordering (before
//   their own count/label sort) get one for free.
export function aggregateKeywords(papers: Pick<PaperOut, 'keywords'>[]): KeywordOption[] {
  interface Accumulator {
    key: string
    paperCount: number
    labelOccurrences: Map<string, number>
    labelFirstSeenIndex: Map<string, number>
  }
  const byKey = new Map<string, Accumulator>()
  let labelSightingIndex = 0

  for (const paper of papers) {
    const seenKeysThisPaper = new Set<string>()
    for (const keyword of paper.keywords) {
      const key = canonicalKeywordKey(keyword)
      if (!key) continue
      let acc = byKey.get(key)
      if (!acc) {
        acc = { key, paperCount: 0, labelOccurrences: new Map(), labelFirstSeenIndex: new Map() }
        byKey.set(key, acc)
      }
      if (!seenKeysThisPaper.has(key)) {
        seenKeysThisPaper.add(key)
        acc.paperCount += 1
      }
      acc.labelOccurrences.set(keyword, (acc.labelOccurrences.get(keyword) ?? 0) + 1)
      if (!acc.labelFirstSeenIndex.has(keyword)) {
        acc.labelFirstSeenIndex.set(keyword, labelSightingIndex)
        labelSightingIndex += 1
      }
    }
  }

  const options: KeywordOption[] = []
  for (const acc of byKey.values()) {
    let bestLabel = ''
    let bestOccurrences = -1
    let bestFirstSeen = Infinity
    for (const [label, occurrences] of acc.labelOccurrences.entries()) {
      const firstSeen = acc.labelFirstSeenIndex.get(label)!
      const better =
        occurrences > bestOccurrences ||
        (occurrences === bestOccurrences &&
          (firstSeen < bestFirstSeen || (firstSeen === bestFirstSeen && label.localeCompare(bestLabel) < 0)))
      if (better) {
        bestLabel = label
        bestOccurrences = occurrences
        bestFirstSeen = firstSeen
      }
    }
    options.push({ key: acc.key, label: bestLabel, count: acc.paperCount })
  }
  return options
}
