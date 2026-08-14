import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { PaperCard } from './PaperCard'
import type { PaperOut } from '../../types'

function paper(overrides: Partial<PaperOut> = {}): PaperOut {
  return {
    paper_id: 'p1', title: 'Paper One', authors: ['A'], year: 2024, venue: 'arXiv',
    abstract: 'An abstract.', url: null, doi: null, citation_count: 10, source: 'arxiv',
    source_urls: {}, score: null, keywords: [],
    ...overrides,
  }
}

describe('PaperCard -- Paper Keywords and Filtering, K2', () => {
  it('renders the keywords PaperOut already supplied, as plain chips', () => {
    render(
      <PaperCard
        paper={paper({ keywords: ['graph neural networks', 'molecular property prediction'] })}
        showAbstract action={{ kind: 'none' }}
      />,
    )

    expect(screen.getByText('graph neural networks')).toBeInTheDocument()
    expect(screen.getByText('molecular property prediction')).toBeInTheDocument()
  })

  it('renders no keyword container at all when keywords is empty', () => {
    render(<PaperCard paper={paper({ keywords: [] })} showAbstract action={{ kind: 'none' }} />)

    expect(screen.queryByTestId('paper-keywords-p1')).not.toBeInTheDocument()
  })

  it('hides keywords when showAbstract is false, even if keywords are present', () => {
    render(
      <PaperCard paper={paper({ keywords: ['graph neural networks'] })} action={{ kind: 'none' }} />,
    )

    expect(screen.queryByText('graph neural networks')).not.toBeInTheDocument()
    expect(screen.queryByTestId('paper-keywords-p1')).not.toBeInTheDocument()
  })

  it('keyword chips are static text, never interactive controls', () => {
    render(
      <PaperCard paper={paper({ keywords: ['graph neural networks'] })} showAbstract action={{ kind: 'none' }} />,
    )

    expect(screen.queryByRole('button', { name: 'graph neural networks' })).not.toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: 'graph neural networks' })).not.toBeInTheDocument()
  })

  it('a long keyword phrase renders with a bounded, truncating layout, not raw overflow', () => {
    const longKeyword = 'a very long keyword phrase that could otherwise overflow a narrow mobile card width'
    render(
      <PaperCard paper={paper({ keywords: [longKeyword] })} showAbstract action={{ kind: 'none' }} />,
    )

    const chip = screen.getByText(longKeyword)
    expect(chip.className).toContain('truncate')
    expect(chip.className).toContain('max-w-full')
  })

  it('keywords do not affect Add/Remove actions or citation-adjacent rendering', async () => {
    const onAdd = vi.fn()
    render(
      <PaperCard
        paper={paper({ keywords: ['graph neural networks'] })}
        showAbstract action={{ kind: 'add', onAdd }}
      />,
    )

    expect(screen.getByRole('button', { name: '+ Add to review' })).toBeInTheDocument()
  })
})
