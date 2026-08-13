// UXH.1 (UX-01): the one place "how many/which papers are effectively
// selected" is computed for display purposes -- persisted picks
// (state.selected_paper_ids, confirmed server-side) plus locally staged
// picks (stagedPickIds -- e.g. added from Browse Past Turns while curation
// is still active, not yet submitted via Continue). Every visible counter
// that means to answer "how many has the user picked so far" should derive
// from this, not from state.selected_paper_ids alone or from
// `a.length + b.length` (the previous, dedup-unsafe formula) -- a paper id
// present in both arrays (structurally prevented today, but not something
// a display-only counter should assume stays true forever) must still only
// count once.
//
// Order is preserved (persisted ids first, in their existing order, then
// any staged id not already present, in its own order) purely so a future
// consumer of the actual id list -- not just its length -- gets a stable,
// deterministic result; nothing currently reads anything but `.length`.
export function mergeSelectedPaperIds(persistedIds: string[], stagedIds: string[]): string[] {
  const seen = new Set(persistedIds)
  const merged = [...persistedIds]
  for (const id of stagedIds) {
    if (!seen.has(id)) {
      seen.add(id)
      merged.push(id)
    }
  }
  return merged
}
