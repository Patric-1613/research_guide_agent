"""Deterministic tests for ingestion helpers that don't need a live API call."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.ingestion import _clean_abstract, _clean_venue


def test_clean_venue_accepts_plausible_venue_names():
    assert _clean_venue("Neural Information Processing Systems") == "Neural Information Processing Systems"
    assert _clean_venue(None) is None


def test_clean_venue_rejects_citation_dumps():
    dump = (
        "Afzal, Z.R., Esmaeilbeig, T., Soltanalian, M. and Ohannessian, M.I., 2025. "
        "Linearization Explains Fine-Tuning in Large Language Models. In The "
        "Thirty-ninth Annual Conference on Neural Information Processing Systems"
    )
    assert _clean_venue(dump) is None


def test_clean_abstract_collapses_whitespace_and_normalizes_empty():
    assert _clean_abstract("line one\nline   two  ") == "line one line two"
    assert _clean_abstract("   ") is None
    assert _clean_abstract(None) is None


if __name__ == "__main__":
    test_clean_venue_accepts_plausible_venue_names()
    test_clean_venue_rejects_citation_dumps()
    test_clean_abstract_collapses_whitespace_and_normalizes_empty()
    print("All ingestion tests passed.")
