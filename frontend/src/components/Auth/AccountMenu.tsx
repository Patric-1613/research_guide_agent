// Day 5: the small account / sign-out control in the header. Renders
// only when auth resolved to an approved firebase account -- in
// disabled/basic mode `useAuth()` reports `not-required` and this
// renders nothing, so the header is unchanged.

import { useEffect, useRef, useState } from 'react'
import { LogOut, UserRound } from 'lucide-react'
import { useAuth } from '../../lib/auth/authContext'

export function AccountMenu() {
  const { status, account, signOut } = useAuth()
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) setOpen(false)
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onOutside)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onOutside)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (status !== 'approved') return null

  const label = account?.email ?? account?.display_name ?? 'Account'

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        data-testid="account-menu-trigger"
        aria-label={`Account: ${label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex max-w-[14rem] items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-text-secondary outline-none hover:text-text focus-visible:ring-2 focus-visible:ring-accent"
      >
        <UserRound className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <span className="truncate">{label}</span>
      </button>
      {open && (
        <div
          data-testid="account-menu"
          role="menu"
          className="absolute right-0 z-10 mt-1 w-44 rounded-md border border-border bg-panel py-1 text-xs shadow-lg"
        >
          {account?.email && (
            <p className="truncate px-3 py-1.5 text-text-muted" title={account.email}>
              {account.email}
            </p>
          )}
          <button
            type="button"
            role="menuitem"
            data-testid="account-menu-sign-out"
            onClick={() => {
              setOpen(false)
              void signOut()
            }}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-text-secondary hover:bg-panel-alt"
          >
            <LogOut className="h-3.5 w-3.5" aria-hidden="true" />
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}
