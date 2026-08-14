"""Paper Keywords and Filtering, K1: tests for research_agent/keywords.py's
extract_keywords() -- pure, deterministic, offline (no LLM/embeddings/
network call anywhere in this module or these tests).

Deliberately does NOT assert on an exact full keyword list for a real
abstract (YAKE is not version-pinned in a way this suite enforces, and a
future YAKE point release could reorder/reweight candidates) -- assertions
instead check the STRUCTURAL contract (count, dedup, noise exclusion,
determinism, title/abstract influence) that must hold regardless of the
installed YAKE version.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.keywords import MAX_KEYWORDS, extract_keywords

_REAL_ABSTRACT = (
    "We present a comprehensive study of graph neural networks for molecular "
    "property prediction. Our method combines message passing with attention "
    "mechanisms to capture long-range dependencies between atoms. Across five "
    "benchmark datasets, our approach consistently outperforms prior graph "
    "learning baselines, improving mean absolute error by a substantial margin "
    "while remaining computationally efficient at inference time."
)


def test_missing_abstract_returns_empty_list():
    assert extract_keywords("Some Paper Title", None) == []


def test_blank_abstract_returns_empty_list():
    assert extract_keywords("Some Paper Title", "") == []
    assert extract_keywords("Some Paper Title", "   ") == []


def test_very_short_abstract_returns_empty_list():
    # Real words, but nowhere near enough content to extract meaningful
    # keywords from -- must not be treated as "an abstract exists."
    assert extract_keywords("Some Paper Title", "A short note.") == []
    assert extract_keywords("Some Paper Title", "TBD.") == []


def test_abstract_that_is_only_a_url_or_citation_returns_empty_list():
    # After noise-stripping, nothing meaningful is left -- must fall
    # through the same short-abstract floor as an actually-short abstract,
    # not crash or return junk.
    assert extract_keywords("Title", "https://example.com/paper.pdf") == []
    assert extract_keywords("Title", "[1] Smith et al., 2020.") == []


def test_deterministic_for_repeated_identical_input():
    first = extract_keywords("Graph Neural Networks", _REAL_ABSTRACT)
    second = extract_keywords("Graph Neural Networks", _REAL_ABSTRACT)
    assert first == second
    assert len(first) > 0


def test_title_and_abstract_both_influence_extraction():
    same_abstract_a = extract_keywords("Graph Neural Networks for Chemistry", _REAL_ABSTRACT)
    same_abstract_b = extract_keywords("Reinforcement Learning for Robotics", _REAL_ABSTRACT)
    assert same_abstract_a != same_abstract_b

    same_title_a = extract_keywords(
        "A Study of Molecular Property Prediction",
        _REAL_ABSTRACT,
    )
    same_title_b = extract_keywords(
        "A Study of Molecular Property Prediction",
        "We present a comprehensive study of reinforcement learning for robotic "
        "manipulation. Our method combines policy gradients with curiosity-driven "
        "exploration to accelerate training. Across five simulated benchmark "
        "environments, our approach consistently outperforms prior baselines, "
        "improving sample efficiency by a substantial margin.",
    )
    assert same_title_a != same_title_b


def test_never_more_than_max_keywords():
    long_abstract = _REAL_ABSTRACT + " " + _REAL_ABSTRACT.replace("graph", "network").replace("molecular", "chemical")
    result = extract_keywords("A Very Long And Detailed Title About Many Different Research Topics", long_abstract)
    assert len(result) <= MAX_KEYWORDS


def test_urls_dois_and_citation_markers_are_excluded_from_output():
    noisy_abstract = (
        "We present a comprehensive study of graph neural networks, see "
        "https://arxiv.org/abs/2101.00001 and DOI: 10.1234/abcd.5678 for the "
        "full dataset and code. Prior work [1][2][3] (Smith et al., 2020) "
        "reported strong baseline results on molecular property prediction "
        "using message passing and attention mechanisms across benchmarks."
    )
    result = extract_keywords("Graph Neural Networks", noisy_abstract)
    joined = " ".join(result).lower()
    assert "http" not in joined
    assert "doi" not in joined
    assert "arxiv.org" not in joined
    for kw in result:
        assert not kw.strip().startswith("[")
        assert not kw.strip().startswith("(")


def test_numeric_only_and_single_character_candidates_are_excluded():
    numeric_heavy_abstract = (
        "Our method achieves 99.2% accuracy on 3 benchmark datasets, a 12% "
        "improvement over 1,234 prior baseline evaluations. Across five graph "
        "neural network configurations, message passing with attention "
        "mechanisms consistently improves molecular property prediction "
        "accuracy over strong baseline methods on held-out test data."
    )
    result = extract_keywords("Benchmarking Study", numeric_heavy_abstract)
    for kw in result:
        cleaned = kw.strip()
        assert len(cleaned) > 1
        assert not cleaned.replace(".", "").replace(",", "").replace("%", "").isdigit()


def test_case_insensitive_duplicate_phrases_are_not_both_returned():
    abstract = (
        "Neural Networks have transformed machine learning research. In this "
        "paper we study neural networks applied to graph-structured data, "
        "showing that neural networks combined with attention mechanisms "
        "improve molecular property prediction accuracy across five widely "
        "used benchmark datasets compared to prior baseline approaches."
    )
    result = extract_keywords("Neural Networks", abstract)
    lowered = [kw.lower() for kw in result]
    assert len(lowered) == len(set(lowered))


def test_normal_technical_abstract_produces_useful_nonempty_keywords():
    result = extract_keywords("Graph Neural Networks for Molecular Property Prediction", _REAL_ABSTRACT)
    assert len(result) > 0
    assert len(result) <= MAX_KEYWORDS
    for kw in result:
        assert isinstance(kw, str)
        assert kw.strip() == kw
        assert len(kw) > 1
