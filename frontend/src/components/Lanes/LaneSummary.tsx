import { useState } from 'react'
import type { ResearchLaneOut } from '../../types'

interface LaneSummaryProps {
  lanes: ResearchLaneOut[]
  laneResultCounts: Record<string, number>
}

// Research Lanes (RL5): a compact, read-only disclosure shown near the
// turn context of an active lane session. Lane definitions are frozen
// once curation starts, so this never edits -- it just discloses each
// lane's label / research question / enabled state / cumulative count.
// Single-query sessions never render this at all (the parent gates on
// state.lanes.length).
const PANEL_ID = 'lane-summary-panel'

export function LaneSummary({ lanes, laneResultCounts }: LaneSummaryProps) {
  const [open, setOpen] = useState(false)
  const activeCount = lanes.filter((lane) => lane.enabled).length

  return (
    <div className="border-b border-border bg-panel px-4 py-1.5">
      <button
        type="button"
        data-testid="lane-summary-toggle"
        aria-expanded={open}
        aria-controls={PANEL_ID}
        onClick={() => setOpen((prev) => !prev)}
        className="text-xs text-text-secondary underline decoration-dotted hover:text-accent"
      >
        Research lanes · {activeCount} active
      </button>
      {open && (
        <ul
          id={PANEL_ID}
          data-testid="lane-summary-panel"
          className="mt-1.5 flex max-h-40 flex-col gap-1.5 overflow-y-auto"
        >
          {lanes.map((lane) => (
            <li
              key={lane.lane_id}
              data-testid={`lane-summary-item-${lane.lane_id}`}
              className="flex flex-col gap-0.5 text-xs"
            >
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <span className="font-medium text-text">{lane.label}</span>
                <span
                  className={`text-[10px] uppercase tracking-wide ${
                    lane.enabled ? 'text-accent' : 'text-text-muted'
                  }`}
                >
                  {lane.enabled ? 'active' : 'off'}
                </span>
                <span className="text-text-muted">
                  · {laneResultCounts[lane.lane_id] ?? 0} found
                </span>
              </div>
              {lane.question && <span className="break-words text-text-muted">{lane.question}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
