import { describe, expect, it } from 'vitest'
import { mergeSelectedPaperIds } from './selection'

describe('mergeSelectedPaperIds', () => {
  it('returns just the persisted ids when nothing is staged', () => {
    expect(mergeSelectedPaperIds(['p1', 'p2'], [])).toEqual(['p1', 'p2'])
  })

  it('returns just the staged ids when nothing is persisted yet', () => {
    expect(mergeSelectedPaperIds([], ['p1', 'p2'])).toEqual(['p1', 'p2'])
  })

  it('unions persisted and staged ids, persisted first', () => {
    expect(mergeSelectedPaperIds(['p1', 'p2'], ['p3'])).toEqual(['p1', 'p2', 'p3'])
  })

  it('counts an id present in both arrays only once', () => {
    const merged = mergeSelectedPaperIds(['p1', 'p2'], ['p2', 'p3'])
    expect(merged).toEqual(['p1', 'p2', 'p3'])
    expect(merged).toHaveLength(3)
  })

  it('does not mutate either input array', () => {
    const persisted = ['p1']
    const staged = ['p2']
    mergeSelectedPaperIds(persisted, staged)
    expect(persisted).toEqual(['p1'])
    expect(staged).toEqual(['p2'])
  })

  it('an empty result for two empty inputs', () => {
    expect(mergeSelectedPaperIds([], [])).toEqual([])
  })
})
