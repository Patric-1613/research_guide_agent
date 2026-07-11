"""Deterministic tests for APA/BibTeX formatting — pure string logic, no LLM."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.citations import (
    format_apa_citation,
    format_authors_apa,
    format_authors_harvard,
    format_bibtex_citation,
    format_harvard_citation,
    select_citation,
)
from research_agent.schema import Paper


def _paper(**overrides) -> Paper:
    defaults = dict(
        title="A Great Paper About Testing",
        authors=["Jane Q. Smith", "Bob Lee"],
        year=2023,
        venue="NeurIPS",
        abstract="abstract text",
        url="http://arxiv.org/abs/1234.5678",
        doi=None,
        citation_count=10,
        source="arxiv",
        paper_id="1234.5678",
    )
    defaults.update(overrides)
    return Paper(**defaults)


def test_author_formatting_one_two_and_many():
    assert format_authors_apa(["Jane Smith"]) == "Smith, J."
    assert format_authors_apa(["Jane Smith", "Bob Lee"]) == "Smith, J., & Lee, B."
    assert format_authors_apa(["A One", "B Two", "C Three"]) == "One, A., Two, B., & Three, C."


def test_author_formatting_empty():
    assert format_authors_apa([]) == "Unknown Author"


def test_apa_citation_includes_doi_over_url_when_present():
    p = _paper(doi="10.1234/abc")
    citation = format_apa_citation(p)
    assert "https://doi.org/10.1234/abc" in citation
    assert p.url not in citation


def test_apa_citation_falls_back_to_url_without_doi():
    p = _paper(doi=None)
    citation = format_apa_citation(p)
    assert p.url in citation


def test_apa_citation_handles_missing_year():
    p = _paper(year=None)
    assert "(n.d.)" in format_apa_citation(p)


def test_bibtex_key_is_lastname_year_firstword():
    p = _paper()
    bibtex = format_bibtex_citation(p)
    assert bibtex.startswith("@article{smith2023great,")
    assert "title     = {A Great Paper About Testing}" in bibtex


def test_bibtex_uses_misc_for_arxiv_preprint():
    p = _paper(venue="arXiv preprint")
    bibtex = format_bibtex_citation(p)
    assert bibtex.startswith("@misc{")


def test_harvard_author_formatting_one_two_three_and_many():
    assert format_authors_harvard(["Jane Smith"]) == "Smith, J."
    assert format_authors_harvard(["Jane Smith", "Bob Lee"]) == "Smith, J. and Lee, B."
    assert format_authors_harvard(["A One", "B Two", "C Three"]) == "One, A., Two, B. and Three, C."
    # 4+ authors collapse to first-author + "et al." — a different rule from
    # APA 7's 21-author cutoff, and no ampersand anywhere in Harvard style.
    assert format_authors_harvard(["A One", "B Two", "C Three", "D Four"]) == "One, A. et al."


def test_harvard_author_initials_have_no_space_unlike_apa():
    # The actual distinguishing formatting detail between the two styles.
    assert format_authors_apa(["Jane Q Smith"]) == "Smith, J. Q."
    assert format_authors_harvard(["Jane Q Smith"]) == "Smith, J.Q."


def test_harvard_author_formatting_empty():
    assert format_authors_harvard([]) == "Unknown Author"


def test_harvard_citation_uses_single_quoted_title_and_available_at():
    p = _paper(doi="10.1234/abc")
    citation = format_harvard_citation(p)
    assert citation.startswith("Smith, J.Q. and Lee, B. (2023)")
    assert "'A Great Paper About Testing'" in citation
    assert "Available at: https://doi.org/10.1234/abc" in citation


def test_harvard_citation_falls_back_to_url_without_doi():
    p = _paper(doi=None)
    citation = format_harvard_citation(p)
    assert f"Available at: {p.url}" in citation


def test_harvard_citation_handles_missing_year():
    p = _paper(year=None)
    assert "(n.d.)" in format_harvard_citation(p)


def test_harvard_and_apa_citations_are_genuinely_different_formats():
    p = _paper(doi="10.1234/abc")
    apa = format_apa_citation(p)
    harvard = format_harvard_citation(p)
    assert apa != harvard
    assert "&" in apa and "&" not in harvard
    assert "'A Great Paper About Testing'" in harvard
    assert "'A Great Paper About Testing'" not in apa


def test_select_citation_picks_requested_style():
    assert select_citation("APA_TEXT", "HARVARD_TEXT", "BIBTEX_TEXT", "apa") == "APA_TEXT"
    assert select_citation("APA_TEXT", "HARVARD_TEXT", "BIBTEX_TEXT", "harvard") == "HARVARD_TEXT"
    assert select_citation("APA_TEXT", "HARVARD_TEXT", "BIBTEX_TEXT", "bibtex") == "BIBTEX_TEXT"


if __name__ == "__main__":
    test_author_formatting_one_two_and_many()
    test_author_formatting_empty()
    test_apa_citation_includes_doi_over_url_when_present()
    test_apa_citation_falls_back_to_url_without_doi()
    test_apa_citation_handles_missing_year()
    test_bibtex_key_is_lastname_year_firstword()
    test_bibtex_uses_misc_for_arxiv_preprint()
    test_harvard_author_formatting_one_two_three_and_many()
    test_harvard_author_initials_have_no_space_unlike_apa()
    test_harvard_author_formatting_empty()
    test_harvard_citation_uses_single_quoted_title_and_available_at()
    test_harvard_citation_falls_back_to_url_without_doi()
    test_harvard_citation_handles_missing_year()
    test_harvard_and_apa_citations_are_genuinely_different_formats()
    test_select_citation_picks_requested_style()
    print("All citation tests passed.")
