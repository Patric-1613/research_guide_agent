"""Deterministic tests for APA/BibTeX formatting — pure string logic, no LLM."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.citations import format_apa_citation, format_authors_apa, format_bibtex_citation
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


if __name__ == "__main__":
    test_author_formatting_one_two_and_many()
    test_author_formatting_empty()
    test_apa_citation_includes_doi_over_url_when_present()
    test_apa_citation_falls_back_to_url_without_doi()
    test_apa_citation_handles_missing_year()
    test_bibtex_key_is_lastname_year_firstword()
    test_bibtex_uses_misc_for_arxiv_preprint()
    print("All citation tests passed.")
