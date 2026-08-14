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

describe('PaperCard -- Paper Keywords and Filtering, K4.2', () => {
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

  it('keyword chips use the readable accent treatment, not K2\'s muted 10px styling', () => {
    render(
      <PaperCard paper={paper({ keywords: ['graph neural networks'] })} showAbstract action={{ kind: 'none' }} />,
    )

    const chip = screen.getByText('graph neural networks')
    expect(chip.className).toContain('text-xs')
    expect(chip.className).toContain('font-medium')
    expect(chip.className).toContain('text-accent')
    expect(chip.className).toContain('bg-accent-soft')
    expect(chip.className).toContain('border-accent/30')
    expect(chip.className).toContain('rounded-md')
    expect(chip.className).not.toContain('text-[10px]')
    expect(chip.className).not.toContain('text-text-muted')
  })

  it('a long keyword phrase wraps safely instead of truncating or overflowing', () => {
    const longKeyword = 'a very long keyword phrase that could otherwise overflow a narrow mobile card width'
    render(
      <PaperCard paper={paper({ keywords: [longKeyword] })} showAbstract action={{ kind: 'none' }} />,
    )

    const chip = screen.getByText(longKeyword)
    expect(chip.className).toContain('max-w-full')
    expect(chip.className).toContain('whitespace-normal')
    expect(chip.className).toContain('break-words')
    expect(chip.className).not.toContain('truncate')
  })

  it('keywords render after the title and before source/year/citation metadata', () => {
    render(
      <PaperCard
        paper={paper({ keywords: ['graph neural networks'], venue: 'arXiv', year: 2024, citation_count: 5 })}
        showAbstract action={{ kind: 'none' }}
      />,
    )

    const card = screen.getByTestId('paper-card-p1')
    const title = screen.getByText('Paper One')
    const keywordsContainer = screen.getByTestId('paper-keywords-p1')
    const metadata = screen.getByText(/arXiv/)

    // DOM order check: title, then keywords, then metadata.
    // eslint-disable-next-line no-bitwise
    expect(title.compareDocumentPosition(keywordsContainer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    // eslint-disable-next-line no-bitwise
    expect(keywordsContainer.compareDocumentPosition(metadata) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(card).toContainElement(keywordsContainer)
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
