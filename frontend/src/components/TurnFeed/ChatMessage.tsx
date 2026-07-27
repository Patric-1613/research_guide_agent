import type { ChatTurn } from '../../api/types'

export function ChatMessage({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
          isUser ? 'bg-accent text-accent-fg' : 'border border-border bg-panel-alt text-text'
        }`}
      >
        {turn.content}
      </div>
    </div>
  )
}
