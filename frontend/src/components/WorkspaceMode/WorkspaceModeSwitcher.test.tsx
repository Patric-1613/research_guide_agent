import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { WorkspaceModeSwitcher } from './WorkspaceModeSwitcher'

describe('WorkspaceModeSwitcher', () => {
  it('locked: Chat and Report are disabled and show a lock, Review stays clickable', () => {
    render(<WorkspaceModeSwitcher mode="review" unlocked={false} onChange={vi.fn()} />)

    expect(screen.getByTestId('workspace-mode-review')).toBeEnabled()
    expect(screen.getByTestId('workspace-mode-chat')).toBeDisabled()
    expect(screen.getByTestId('workspace-mode-report')).toBeDisabled()
    expect(screen.getByText(/Finish curation to unlock/)).toBeInTheDocument()
  })

  it('unlocked: all three tabs are clickable and the hint disappears', () => {
    render(<WorkspaceModeSwitcher mode="review" unlocked onChange={vi.fn()} />)

    expect(screen.getByTestId('workspace-mode-chat')).toBeEnabled()
    expect(screen.getByTestId('workspace-mode-report')).toBeEnabled()
    expect(screen.queryByText(/Finish curation to unlock/)).not.toBeInTheDocument()
  })

  it('clicking an unlocked tab calls onChange with that mode', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<WorkspaceModeSwitcher mode="review" unlocked onChange={onChange} />)

    await user.click(screen.getByTestId('workspace-mode-report'))

    expect(onChange).toHaveBeenCalledWith('report')
  })

  it('clicking a locked tab does not call onChange', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<WorkspaceModeSwitcher mode="review" unlocked={false} onChange={onChange} />)

    await user.click(screen.getByTestId('workspace-mode-chat'))

    expect(onChange).not.toHaveBeenCalled()
  })
})
