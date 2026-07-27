export type WorkspaceMode = 'review' | 'chat' | 'report'

interface WorkspaceModeSwitcherProps {
  mode: WorkspaceMode
  // Chat/Report unlock together once curation finishes (stage ===
  // "synthesize") -- chat_turn() itself has no dependency on a report
  // existing yet, so gating both on curation-finished (not on
  // has_report specifically) is what actually matches backend
  // capability, confirmed by reading chat_turn()'s own guard.
  unlocked: boolean
  onChange: (mode: WorkspaceMode) => void
}

const TABS: { mode: WorkspaceMode; label: string }[] = [
  { mode: 'review', label: 'Review' },
  { mode: 'chat', label: 'Chat' },
  { mode: 'report', label: 'Report' },
]

export function WorkspaceModeSwitcher({ mode, unlocked, onChange }: WorkspaceModeSwitcherProps) {
  return (
    <div className="-mx-3 flex shrink-0 flex-col gap-1.5 border-t border-border px-3 pt-3">
      <h3 className="px-1 text-[11px] font-semibold uppercase tracking-wide text-text-muted">Workspace mode</h3>
      <div className="flex gap-1 rounded-lg border border-border bg-panel-alt p-1">
        {TABS.map((tab) => {
          const locked = tab.mode !== 'review' && !unlocked
          return (
            <button
              key={tab.mode}
              type="button"
              data-testid={`workspace-mode-${tab.mode}`}
              disabled={locked}
              title={locked ? 'Finish curation to unlock' : undefined}
              onClick={() => onChange(tab.mode)}
              className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
                mode === tab.mode
                  ? 'bg-accent text-accent-fg'
                  : locked
                    ? 'cursor-not-allowed text-text-muted opacity-50'
                    : 'text-text-secondary hover:text-text'
              }`}
            >
              {locked ? `🔒 ${tab.label}` : tab.label}
            </button>
          )
        })}
      </div>
      {!unlocked && <p className="px-1 text-[11px] text-text-muted">Finish curation to unlock Chat &amp; Report.</p>}
    </div>
  )
}
