import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { LaneSummary } from './LaneSummary'
import type { ResearchLaneOut } from '../../types'

function lane(overrides: Partial<ResearchLaneOut> = {}): ResearchLaneOut {
  return {
    lane_id: 'l1', label: 'Retrieval', question: 'How does retrieval help?',
    query: 'retrieval augmented generation', enabled: true, origin: 'user', generation_version: 1,
    ...overrides,
  }
}

describe('LaneSummary', () => {
  const lanes = [
    lane({ lane_id: 'l1', label: 'Retrieval', enabled: true }),
    lane({ lane_id: 'l2', label: 'Evaluation', enabled: true }),
    lane({ lane_id: 'l3', label: 'Failure modes', enabled: false }),
  ]

  it('shows "Research lanes · N active" counting only enabled lanes, collapsed by default', () => {
    render(<LaneSummary lanes={lanes} laneResultCounts={{}} />)
    const toggle = screen.getByTestId('lane-summary-toggle')
    expect(toggle).toHaveTextContent('Research lanes · 2 active')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByTestId('lane-summary-panel')).not.toBeInTheDocument()
  })

  it('expands to a bounded list of every lane with its label, enabled state, and cumulative count', async () => {
    const user = userEvent.setup()
    render(<LaneSummary lanes={lanes} laneResultCounts={{ l1: 7, l2: 3 }} />)

    await user.click(screen.getByTestId('lane-summary-toggle'))

    const panel = screen.getByTestId('lane-summary-panel')
    expect(screen.getByTestId('lane-summary-toggle')).toHaveAttribute('aria-expanded', 'true')
    expect(panel).toHaveAttribute('id')
    expect(screen.getByTestId('lane-summary-item-l1')).toHaveTextContent('Retrieval')
    expect(screen.getByTestId('lane-summary-item-l1')).toHaveTextContent('7 found')
    expect(screen.getByTestId('lane-summary-item-l3')).toHaveTextContent('off')
    // A lane with no recorded count shows 0, never a crash or blank.
    expect(screen.getByTestId('lane-summary-item-l3')).toHaveTextContent('0 found')
  })

  it('aria-controls points at the panel id', async () => {
    const user = userEvent.setup()
    render(<LaneSummary lanes={lanes} laneResultCounts={{}} />)
    const toggle = screen.getByTestId('lane-summary-toggle')
    await user.click(toggle)
    expect(toggle.getAttribute('aria-controls')).toBe(screen.getByTestId('lane-summary-panel').id)
  })
})
