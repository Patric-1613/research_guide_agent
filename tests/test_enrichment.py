"""Deterministic tests for round-2 enhancement 4's abstract recovery.
requests.get is mocked throughout — live Unpaywall/CrossRef calls are
covered by scripts/test_enrichment.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.enrichment import (
    _fetch_crossref_abstract,
    _fetch_unpaywall_abstract,
    _init_cache_db,
    _strip_jats_tags,
    enrich_missing_abstracts,
    recover_abstract,
)
from research_agent.schema import Paper


def _paper(paper_id: str, doi: str | None, abstract: str | None) -> Paper:
    return Paper(
        title=f"Paper {paper_id}", authors=["A"], year=2024, venue="X",
        abstract=abstract, url=None, doi=doi, citation_count=None,
        source="semantic_scholar", paper_id=paper_id,
    )


def _fake_response(status_code: int, json_data=None, raise_on_json: bool = False) -> MagicMock:
    resp = MagicMock(status_code=status_code)
    if raise_on_json:
        resp.json.side_effect = ValueError("malformed")
    else:
        resp.json.return_value = json_data
    return resp


def test_strip_jats_tags_collapses_whitespace_and_removes_markup():
    assert _strip_jats_tags("<jats:p>Hello   world</jats:p>") == "Hello world"
    assert _strip_jats_tags("<tag></tag>") is None


def test_unpaywall_skips_lookup_entirely_when_no_email_configured():
    # Unpaywall 422s on placeholder/fake emails (confirmed against the live
    # API), so unlike CrossRef there's no safe generic default — if
    # UNPAYWALL_EMAIL isn't set, this must skip the network call rather than
    # burn a request guaranteed to fail.
    with patch("research_agent.enrichment._unpaywall_email", return_value=None), \
         patch("research_agent.enrichment.requests.get") as mock_get:
        assert _fetch_unpaywall_abstract("10.1/x") is None
        mock_get.assert_not_called()


def test_unpaywall_returns_none_when_no_abstract_field():
    with patch("research_agent.enrichment._unpaywall_email", return_value="test@example.org"), \
         patch("research_agent.enrichment.requests.get", return_value=_fake_response(200, {"doi": "10.1/x"})):
        assert _fetch_unpaywall_abstract("10.1/x") is None


def test_unpaywall_extracts_abstract_when_present():
    with patch("research_agent.enrichment._unpaywall_email", return_value="test@example.org"), \
         patch("research_agent.enrichment.requests.get", return_value=_fake_response(200, {"abstract": "<p>Real abstract</p>"})):
        assert _fetch_unpaywall_abstract("10.1/x") == "Real abstract"


def test_crossref_extracts_jats_abstract():
    payload = {"message": {"abstract": "<jats:p>Findings show X.</jats:p>"}}
    with patch("research_agent.enrichment.requests.get", return_value=_fake_response(200, payload)):
        assert _fetch_crossref_abstract("10.1/y") == "Findings show X."


def test_crossref_handles_zero_results_gracefully():
    with patch("research_agent.enrichment.requests.get", return_value=_fake_response(404)):
        assert _fetch_crossref_abstract("10.1/missing") is None


def test_crossref_handles_rate_limit_gracefully():
    with patch("research_agent.enrichment.requests.get", return_value=_fake_response(429)):
        assert _fetch_crossref_abstract("10.1/rl") is None


def test_crossref_handles_malformed_json_gracefully():
    with patch("research_agent.enrichment.requests.get", return_value=_fake_response(200, raise_on_json=True)):
        assert _fetch_crossref_abstract("10.1/bad") is None


def test_unpaywall_handles_network_error_gracefully():
    import requests as requests_module
    with patch("research_agent.enrichment._unpaywall_email", return_value="test@example.org"), \
         patch("research_agent.enrichment.requests.get", side_effect=requests_module.ConnectionError("boom")):
        assert _fetch_unpaywall_abstract("10.1/neterr") is None


def test_recover_abstract_falls_through_unpaywall_to_crossref():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _init_cache_db(Path(tmp) / "cache.sqlite")
        with patch("research_agent.enrichment._fetch_unpaywall_abstract", return_value=None), \
             patch("research_agent.enrichment._fetch_crossref_abstract", return_value="Recovered via CrossRef"):
            result = recover_abstract("10.1/fallthrough", conn=conn)
        assert result == "Recovered via CrossRef"
        conn.close()


def test_recover_abstract_caches_result_and_skips_network_on_second_call():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _init_cache_db(Path(tmp) / "cache.sqlite")
        with patch("research_agent.enrichment._fetch_unpaywall_abstract", return_value="From Unpaywall") as mock_up, \
             patch("research_agent.enrichment._fetch_crossref_abstract", return_value=None) as mock_cr:
            first = recover_abstract("10.1/cached", conn=conn)
            second = recover_abstract("10.1/cached", conn=conn)
        assert first == second == "From Unpaywall"
        mock_up.assert_called_once()
        mock_cr.assert_not_called()
        conn.close()


def test_recover_abstract_caches_unrecoverable_result_too():
    """A DOI genuinely unrecoverable from either source must not be
    retried on every future search that surfaces it — the negative result
    is cached too."""
    with tempfile.TemporaryDirectory() as tmp:
        conn = _init_cache_db(Path(tmp) / "cache.sqlite")
        with patch("research_agent.enrichment._fetch_unpaywall_abstract", return_value=None) as mock_up, \
             patch("research_agent.enrichment._fetch_crossref_abstract", return_value=None) as mock_cr:
            first = recover_abstract("10.1/unrecoverable", conn=conn)
            second = recover_abstract("10.1/unrecoverable", conn=conn)
        assert first is None and second is None
        mock_up.assert_called_once()
        mock_cr.assert_called_once()
        conn.close()


def test_enrich_missing_abstracts_only_targets_doi_present_abstract_missing():
    has_both = _paper("a", doi="10.1/a", abstract="already have one")
    no_doi = _paper("b", doi=None, abstract=None)
    needs_recovery = _paper("c", doi="10.1/c", abstract=None)

    with patch("research_agent.enrichment.recover_abstract", return_value="Recovered!") as mock_recover:
        recovered_count = enrich_missing_abstracts([has_both, no_doi, needs_recovery])

    assert recovered_count == 1
    mock_recover.assert_called_once()
    assert mock_recover.call_args[0][0] == "10.1/c"
    assert needs_recovery.abstract == "Recovered!"
    assert has_both.abstract == "already have one"  # untouched
    assert no_doi.abstract is None  # untouched, nothing to look up by


def test_enrich_missing_abstracts_degrades_gracefully_when_unrecoverable():
    unrecoverable = _paper("d", doi="10.1/d", abstract=None)
    with patch("research_agent.enrichment.recover_abstract", return_value=None):
        recovered_count = enrich_missing_abstracts([unrecoverable])
    assert recovered_count == 0
    assert unrecoverable.abstract is None  # falls through to embeddings.py's title fallback, no crash


def test_enrich_missing_abstracts_empty_input_returns_zero():
    assert enrich_missing_abstracts([]) == 0


if __name__ == "__main__":
    test_strip_jats_tags_collapses_whitespace_and_removes_markup()
    test_unpaywall_skips_lookup_entirely_when_no_email_configured()
    test_unpaywall_returns_none_when_no_abstract_field()
    test_unpaywall_extracts_abstract_when_present()
    test_crossref_extracts_jats_abstract()
    test_crossref_handles_zero_results_gracefully()
    test_crossref_handles_rate_limit_gracefully()
    test_crossref_handles_malformed_json_gracefully()
    test_unpaywall_handles_network_error_gracefully()
    test_recover_abstract_falls_through_unpaywall_to_crossref()
    test_recover_abstract_caches_result_and_skips_network_on_second_call()
    test_recover_abstract_caches_unrecoverable_result_too()
    test_enrich_missing_abstracts_only_targets_doi_present_abstract_missing()
    test_enrich_missing_abstracts_degrades_gracefully_when_unrecoverable()
    test_enrich_missing_abstracts_empty_input_returns_zero()
    print("All enrichment tests passed.")
