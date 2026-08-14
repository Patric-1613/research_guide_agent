"""Paper Keywords and Filtering, K4.1: tests for research_agent/keywords.py's
extract_keywords() -- pure, deterministic, offline (no LLM/embeddings/
network call anywhere in this module or these tests).

Deliberately does NOT assert on an exact full keyword list for a real
abstract (YAKE is not version-pinned in a way this suite enforces, and a
future YAKE point release could reorder/reweight candidates) -- assertions
instead check the STRUCTURAL contract (count, dedup, noise exclusion,
determinism, title/abstract influence, redundancy resolution, canonical
normalization) that must hold regardless of the installed YAKE version.

Two layers of test:
- Direct unit tests against the small pure helper functions
  (`_canonical_tokens`, `_is_acronym`, `_is_contiguous_subsequence`,
  `_dedup_canonical`, `_resolve_redundancy`) with hand-built candidate
  lists -- deterministic and independent of YAKE's own scoring/windowing
  quirks for any particular sentence.
- Integration tests against the public `extract_keywords()` for the
  higher-level, black-box guarantees (abstract-mandatory, title
  contribution capped at one, comma-rejection, complete-phrase
  preservation, max count, determinism).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.keywords import (
    MAX_KEYWORDS,
    _canonical_tokens,
    _dedup_canonical,
    _is_acronym,
    _is_contiguous_subsequence,
    _is_organization_candidate,
    _resolve_redundancy,
    extract_keywords,
)

_REAL_ABSTRACT = (
    "We present a comprehensive study of graph neural networks for molecular "
    "property prediction. Our method combines message passing with attention "
    "mechanisms to capture long-range dependencies between atoms. Across five "
    "benchmark datasets, our approach consistently outperforms prior graph "
    "learning baselines, improving mean absolute error by a substantial margin "
    "while remaining computationally efficient at inference time."
)


# ---------------------------------------------------------------------------
# Abstract-mandatory (retained from K1)
# ---------------------------------------------------------------------------


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


def test_title_alone_never_generates_keywords():
    # No abstract at all -- a substantial, keyword-rich title must still
    # yield nothing. Title only ever supplements a genuinely present
    # abstract, never substitutes for one.
    assert extract_keywords("Agentic Retrieval-Augmented Generation for Enterprise Workflows", None) == []
    assert extract_keywords("Agentic Retrieval-Augmented Generation for Enterprise Workflows", "") == []


# ---------------------------------------------------------------------------
# Determinism / count (retained + adjusted from K1)
# ---------------------------------------------------------------------------


def test_deterministic_for_repeated_identical_input():
    first = extract_keywords("Graph Neural Networks", _REAL_ABSTRACT)
    second = extract_keywords("Graph Neural Networks", _REAL_ABSTRACT)
    assert first == second
    assert len(first) > 0


def test_deterministic_order_across_many_runs():
    runs = [extract_keywords("Graph Neural Networks", _REAL_ABSTRACT) for _ in range(5)]
    assert all(run == runs[0] for run in runs)


def test_never_more_than_max_keywords():
    long_abstract = _REAL_ABSTRACT + " " + _REAL_ABSTRACT.replace("graph", "network").replace("molecular", "chemical")
    result = extract_keywords("A Very Long And Detailed Title About Many Different Research Topics", long_abstract)
    assert len(result) <= MAX_KEYWORDS


def test_no_duplicates_under_canonical_comparison_key():
    abstract = (
        "Neural Networks have transformed machine learning research. In this "
        "paper we study neural networks applied to graph-structured data, "
        "showing that neural networks combined with attention mechanisms "
        "improve molecular property prediction accuracy across five widely "
        "used benchmark datasets compared to prior baseline approaches."
    )
    result = extract_keywords("Neural Networks", abstract)
    canonical_keys = [" ".join(_canonical_tokens(kw)) for kw in result]
    assert len(canonical_keys) == len(set(canonical_keys))


# ---------------------------------------------------------------------------
# Noise / malformed-candidate exclusion (retained + extended)
# ---------------------------------------------------------------------------


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


def test_normal_technical_abstract_produces_useful_nonempty_keywords():
    result = extract_keywords("Graph Neural Networks for Molecular Property Prediction", _REAL_ABSTRACT)
    assert len(result) > 0
    assert len(result) <= MAX_KEYWORDS
    for kw in result:
        assert isinstance(kw, str)
        assert kw.strip() == kw
        assert len(kw) > 1


def test_malformed_clause_join_candidate_is_rejected():
    # Reproduces a real failure found in production data: a missing space
    # after a comma lets YAKE emit a candidate spanning a clause boundary
    # (e.g. "Agentic AI,this paper proposes..."). No output keyword may
    # contain a comma or semicolon.
    abstract = (
        "In this work we build an enterprise Agentic AI,this paper proposes a new "
        "orchestration framework for coordinating specialized autonomous agents across "
        "complex multi-step business workflows in regulated industries with strict "
        "compliance and audit requirements throughout the deployment lifecycle."
    )
    result = extract_keywords("Enterprise Agentic AI Orchestration", abstract)
    for kw in result:
        assert "," not in kw
        assert ";" not in kw


def test_unicode_punctuation_variants_do_not_produce_malformed_candidates():
    # A full-width / smart-punctuation comma should be caught by NFKC
    # normalization the same way a plain ASCII comma is.
    abstract = (
        "In this work we build an enterprise Agentic AI，this paper proposes a new "
        "orchestration framework for coordinating specialized autonomous agents across "
        "complex multi-step business workflows in regulated industries with strict "
        "compliance and audit requirements throughout the deployment lifecycle."
    )
    result = extract_keywords("Enterprise Agentic AI Orchestration", abstract)
    for kw in result:
        assert "," not in kw
        assert "，" not in kw


# ---------------------------------------------------------------------------
# Title domination / title-contributes-at-most-one (K4.1 core requirement)
# ---------------------------------------------------------------------------


def test_title_and_abstract_both_influence_extraction():
    # With an abstract too repetitive to fill all six slots on its own,
    # the title's single allowed contribution should differ when the
    # title differs.
    repetitive_abstract = (
        "This paper studies protein folding stability. Protein folding stability protein "
        "folding stability protein folding stability protein folding stability protein "
        "folding."
    )
    result_a = extract_keywords("Graph Neural Networks for Chemistry", repetitive_abstract)
    result_b = extract_keywords("Reinforcement Learning for Robotics", repetitive_abstract)
    assert result_a != result_b
    # Everything but the title-derived slot must be identical -- the
    # abstract-derived portion of the output must not itself change with
    # the title.
    common_prefix_len = min(len(result_a), len(result_b))
    differing = sum(1 for a, b in zip(result_a[:common_prefix_len], result_b[:common_prefix_len]) if a != b)
    assert differing <= 1


def test_title_domination_regression_with_substantial_abstract():
    # Regression for the observed production failure: a title-heavy paper
    # ("MMU-RAG Competition Winning System") whose substantial, distinct
    # abstract content must not be entirely crowded out by repeated title
    # fragments in the final six keywords.
    title = "MMU-RAG Competition Winning System"
    abstract = (
        "We describe a modular pipeline for multilingual multi-hop question answering "
        "that separates document retrieval from answer synthesis. Our approach introduces "
        "a lightweight reranking stage trained on weak supervision signals, and a citation "
        "verification module that checks generated answers against retrieved evidence spans. "
        "Across four multilingual benchmarks our pipeline improves answer faithfulness while "
        "reducing end-to-end latency compared to a strong retrieval-augmented baseline."
    )
    result = extract_keywords(title, abstract)
    title_only_terms = {"mmu-rag", "competition", "winning", "system"}
    title_fragment_count = sum(
        1 for kw in result if any(term in kw.lower() for term in title_only_terms)
    )
    # At most one of the six keywords may be a bare title fragment; the
    # rest must come from the abstract's own distinct content.
    assert title_fragment_count <= 1
    assert len(result) >= 4


def test_title_never_contributes_more_than_one_keyword():
    # Direct check on the mechanism itself: build a title deliberately
    # full of distinct, extractable multi-word phrases and an abstract
    # that only weakly overlaps it -- still, at most one final keyword
    # may trace to title-only vocabulary absent from the abstract.
    title = "Byzantine Fault Tolerant Consensus For Federated Edge Devices"
    abstract = (
        "We propose a lightweight voting protocol for coordinating distributed nodes under "
        "unreliable network conditions. Our protocol tolerates a bounded number of faulty "
        "participants while maintaining low message overhead, and we evaluate it on a testbed "
        "of resource-constrained embedded devices communicating over lossy wireless links."
    )
    result = extract_keywords(title, abstract)
    title_only_vocab = {"byzantine", "fault", "tolerant", "federated"}
    matches = sum(1 for kw in result if any(term in kw.lower() for term in title_only_vocab))
    assert matches <= 1


# ---------------------------------------------------------------------------
# Complete-phrase preservation (n=3)
# ---------------------------------------------------------------------------


def test_three_word_compound_preserved_as_complete_phrase():
    # A non-RAG synthetic example: the compound "natural language
    # processing" must survive as one complete candidate, never as two
    # separate incomplete fragments ("natural language" / "language
    # processing") in the final output.
    abstract = (
        "Recent advances in natural language processing have enabled large models to "
        "perform diverse tasks with minimal supervision. We study how natural language "
        "processing techniques can be combined with structured knowledge bases to improve "
        "factual consistency, evaluating our method on a suite of open-domain benchmarks "
        "spanning multiple languages and domains."
    )
    result = extract_keywords("Improving Factual Consistency", abstract)
    lowered = [kw.lower() for kw in result]
    assert "natural language processing" in lowered
    assert "natural language" not in lowered
    assert "language processing" not in lowered


# ---------------------------------------------------------------------------
# Redundancy resolution: pure unit tests on the helper functions
# ---------------------------------------------------------------------------


def test_canonical_tokens_treats_hyphen_and_space_as_equivalent():
    assert _canonical_tokens("Retrieval-Augmented Generation") == _canonical_tokens("Retrieval Augmented Generation")


def test_canonical_tokens_is_case_insensitive():
    assert _canonical_tokens("Neural Networks") == _canonical_tokens("neural networks")


def test_is_contiguous_subsequence_true_for_contained_phrase():
    assert _is_contiguous_subsequence(["dynamic"], ["dynamic", "workflow", "scheduler"])
    assert _is_contiguous_subsequence(["workflow", "scheduler"], ["dynamic", "workflow", "scheduler"])


def test_is_contiguous_subsequence_false_for_non_contiguous_or_equal():
    # Shares words but is not a contiguous run.
    assert not _is_contiguous_subsequence(["dynamic", "scheduler"], ["dynamic", "workflow", "scheduler"])
    # Equal-length lists are never "contained" -- that is exact-duplicate
    # territory, handled separately by `_dedup_canonical`.
    assert not _is_contiguous_subsequence(["dynamic", "workflow"], ["dynamic", "workflow"])


def test_is_acronym():
    assert _is_acronym("RAG")
    assert _is_acronym("LLM")
    assert _is_acronym("GPT4")
    assert not _is_acronym("Rag")
    assert not _is_acronym("agentic")
    assert not _is_acronym("A")
    assert not _is_acronym("VERYLONGACRONYM")


def test_dedup_canonical_collapses_hyphen_variant_of_same_phrase():
    result = _dedup_canonical(["Retrieval-Augmented Generation", "Retrieval Augmented Generation"])
    assert len(result) == 1
    assert result[0] == "Retrieval-Augmented Generation"


def test_resolve_redundancy_drops_shorter_contained_fragment():
    # Structural containment removes a generic single word when the
    # longer, more informative phrase containing it is also present --
    # no hand-maintained generic-word list involved.
    candidates = ["Dynamic", "Dynamic Workflow", "Leveraging", "Leveraging Automated", "Generation", "Generation Systems"]
    result = _resolve_redundancy(candidates)
    assert result == ["Dynamic Workflow", "Leveraging Automated", "Generation Systems"]


def test_resolve_redundancy_prefers_complete_phrase_regardless_of_input_order():
    # The shorter fragment ranks BEFORE its more informative longer form
    # in this input order -- a naive single-pass "drop only if already
    # kept" algorithm would keep the fragment and miss the later, longer
    # phrase's redundancy with it. Bidirectional resolution must still
    # drop the fragment.
    candidates = ["language processing", "natural language processing"]
    result = _resolve_redundancy(candidates)
    assert result == ["natural language processing"]


def test_resolve_redundancy_preserves_standalone_acronym():
    candidates = ["RAG", "Agentic RAG"]
    result = _resolve_redundancy(candidates)
    assert "RAG" in result
    assert "Agentic RAG" in result


def test_resolve_redundancy_does_not_unsafely_collapse_distinct_overlapping_phrases():
    # "natural language" and "language processing" share only a boundary
    # word and neither is a contiguous subsequence of the other, with no
    # third, longer candidate covering both present -- both must survive.
    candidates = ["natural language", "language processing"]
    result = _resolve_redundancy(candidates)
    assert set(result) == {"natural language", "language processing"}


def test_resolve_redundancy_is_bidirectional_not_last_word_first_word_rule():
    # "language model" and "model architecture" share a boundary word
    # ("model") but are genuinely distinct phrases -- a naive "last word
    # of A equals first word of B" rule would wrongly treat them as
    # adjacent fragments of one compound. Neither is a subsequence of the
    # other, so both must survive.
    candidates = ["language model", "model architecture"]
    result = _resolve_redundancy(candidates)
    assert set(result) == {"language model", "model architecture"}


# ---------------------------------------------------------------------------
# Topic-agnosticism (non-RAG synthetic, generic-word containment)
# ---------------------------------------------------------------------------


def test_generic_word_containment_is_topic_agnostic():
    # Same structural pattern as the RAG-domain noise the user observed
    # ("Dynamic", "Leveraging", "Generation" surviving alone), reproduced
    # in an unrelated domain (distributed systems) to confirm the rule is
    # structural, not a hand-written RAG-specific mapping.
    abstract = (
        "This paper introduces a dynamic workflow scheduler for leveraging automated "
        "resource allocation across heterogeneous compute clusters. Our scheduler adapts "
        "task placement decisions using live utilization signals, and we evaluate report "
        "generation systems that summarize scheduling decisions for operators managing "
        "large-scale distributed infrastructure deployments."
    )
    result = extract_keywords("Dynamic Workflow Scheduling", abstract)
    lowered = [kw.lower() for kw in result]
    assert "dynamic" not in lowered
    assert "leveraging" not in lowered
    assert "generation" not in lowered


# ---------------------------------------------------------------------------
# K4.1b: organization/affiliation exclusion
# ---------------------------------------------------------------------------


def test_organization_candidate_university_affiliation_excluded():
    assert _is_organization_candidate("Hai Phong University")
    assert _is_organization_candidate("Stanford University")
    assert _is_organization_candidate("University")  # bare organizational label


def test_organization_candidate_department_institute_laboratory_variants_excluded():
    assert _is_organization_candidate("Department of Computer Science")
    assert _is_organization_candidate("Max Planck Institute")
    # Real production example (session 8fa9857f21fb4a2dbd103ca771e54e7b's
    # own local sample): the affiliation this rule was written for.
    assert _is_organization_candidate("National Accelerator Laboratory")
    assert _is_organization_candidate("Accelerator Laboratory rely")
    assert _is_organization_candidate("Lab")
    assert _is_organization_candidate("XYZ College")
    assert _is_organization_candidate("Acme Corporation")
    assert _is_organization_candidate("Acme Corp")
    assert _is_organization_candidate("Research Consortium")


def test_organization_candidate_complete_token_matching_avoids_substring_false_positives():
    # Real production candidates (same local sample) that a naive
    # substring check on "corp"/"lab" would wrongly reject.
    assert not _is_organization_candidate("annotated scientific corpora")
    assert not _is_organization_candidate("large textual corpora")
    assert not _is_organization_candidate("scientific corpus distillation")
    assert not _is_organization_candidate("conversation remains labor-intensive")
    assert not _is_organization_candidate("collaborative filtering")
    assert not _is_organization_candidate("incorporating external knowledge")


def test_organization_candidate_retains_valid_application_domains_and_research_tasks():
    # Explicit product examples this rule must never touch.
    assert not _is_organization_candidate("Student Support")
    assert not _is_organization_candidate("Question Answering")
    assert not _is_organization_candidate("Question Answering Model")
    assert not _is_organization_candidate("multi-hop question answering")
    assert not _is_organization_candidate("open-domain question answering")


def test_organization_candidate_preserves_technical_acronyms_and_system_names():
    assert not _is_organization_candidate("RAG")
    assert not _is_organization_candidate("BERT")
    assert not _is_organization_candidate("Agentic RAG Chatbot")
    assert not _is_organization_candidate("SLAC National Accelerator")


def test_organization_exclusion_removes_university_affiliation_from_real_extraction():
    # A non-RAG synthetic abstract genuinely mentioning a university
    # affiliation, structurally mirroring the real production case this
    # rule was written for (a course-support system built and evaluated
    # at a named university) -- confirms the rule fires through the real
    # extract_keywords() pipeline, not just the isolated helper.
    abstract = (
        "We present a novel gradient compression technique for distributed deep "
        "learning training, developed and evaluated by researchers at Westbrook "
        "University. Our method reduces communication overhead between workers by "
        "adaptively quantizing gradient updates before each synchronization step, "
        "achieving comparable convergence to full-precision baselines while cutting "
        "network bandwidth usage substantially across large-scale training clusters."
    )
    result = extract_keywords("Gradient Compression for Distributed Training", abstract)
    lowered = [kw.lower() for kw in result]
    assert not any("university" in kw for kw in lowered)
    assert not any("westbrook" in kw and "university" in kw for kw in lowered)
    # The genuine topic still comes through.
    assert any("gradient" in kw or "compression" in kw or "distributed" in kw for kw in lowered)


def test_organization_exclusion_preserves_technical_phrase_when_university_mentioned_nearby():
    # The SAME sentence names a university AND a genuine technical
    # phrase -- only the organizational candidate must be rejected; nothing
    # else from the same text is discarded because of it.
    abstract = (
        "Researchers at Blackwood University introduce a federated anomaly "
        "detection framework for industrial sensor networks. The framework "
        "combines lightweight autoencoders with a federated averaging protocol "
        "so that individual sensor sites never share raw measurements, only "
        "locally trained model updates, reducing bandwidth cost while preserving "
        "detection accuracy across a large deployment of heterogeneous devices."
    )
    result = extract_keywords("Federated Anomaly Detection for Sensor Networks", abstract)
    lowered = [kw.lower() for kw in result]
    assert not any("university" in kw for kw in lowered)
    assert any("federated" in kw or "anomaly" in kw or "autoencoder" in kw for kw in lowered)
