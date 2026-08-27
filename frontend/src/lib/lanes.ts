// Research Lanes (RL5): presentation-only helpers for mapping a paper's
// stored discovery provenance (lane_ids) to human lane labels. Never
// derives or persists lane identity -- it only reads what the server
// already sent (state.lanes, state.paper_lane_ids, or a frozen turn
// entry's paper_lane_ids).
import type { ResearchLaneOut } from '../types'

export function buildLaneLabelMap(lanes: ResearchLaneOut[] | null | undefined): Map<string, string> {
  const map = new Map<string, string>()
  for (const lane of lanes ?? []) map.set(lane.lane_id, lane.label)
  return map
}

// The ordered lane labels a paper was discovered through, per the given
// provenance map. Dangling lane_ids (not in `laneLabels`) are skipped.
// Returns [] when the paper has no provenance -- callers must render
// nothing (no "Found via" row, no placeholder) in that case.
export function laneLabelsForPaper(
  paperId: string,
  provenance: Record<string, string[]> | null | undefined,
  laneLabels: Map<string, string>,
): string[] {
  const ids = provenance?.[paperId] ?? []
  const labels: string[] = []
  for (const id of ids) {
    const label = laneLabels.get(id)
    if (label !== undefined) labels.push(label)
  }
  return labels
}
