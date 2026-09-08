import { AccountMenu } from '../Auth/AccountMenu'

export function AppHeader() {
  return (
    <header className="flex h-11 shrink-0 items-center justify-between border-b border-border bg-panel px-4">
      <span className="text-sm font-semibold tracking-tight text-text">Research Helper Agent</span>
      {/* Day 5: renders only for an approved firebase account; nothing in
          disabled/basic mode. */}
      <AccountMenu />
    </header>
  )
}
