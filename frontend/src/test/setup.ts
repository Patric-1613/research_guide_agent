import '@testing-library/jest-dom/vitest'

// jsdom doesn't implement scrollIntoView -- TurnFeed calls it on mount to
// keep the feed pinned to the latest turn/message.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
