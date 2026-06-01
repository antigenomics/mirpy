"""Tests for mir.common.alleles — genes_match and allele resolution utilities."""

from __future__ import annotations

import pytest

from mir.common.alleles import (
    allele_to_major,
    allele_with_default,
    genes_match,
    strip_allele,
)


# ── genes_match ──────────────────────────────────────────────────────────────

class TestGenesMatch:
    """Bare gene = wildcard; specific allele = exact match."""

    def test_both_bare_same_gene(self):
        assert genes_match("TRAV1", "TRAV1")

    def test_both_bare_different_gene(self):
        assert not genes_match("TRAV1", "TRAV2")

    def test_bare_matches_any_allele(self):
        assert genes_match("TRAV1", "TRAV1*01")
        assert genes_match("TRAV1", "TRAV1*02")
        assert genes_match("TRAV1", "TRAV1*07")

    def test_specific_allele_matches_same_allele(self):
        assert genes_match("TRAV1*02", "TRAV1*02")

    def test_specific_allele_does_not_match_different_allele(self):
        assert not genes_match("TRAV1*01", "TRAV1*02")
        assert not genes_match("TRAV1*02", "TRAV1*03")

    def test_specific_allele_matches_bare(self):
        # Bare gene is a wildcard on either side.
        assert genes_match("TRAV1*02", "TRAV1")
        assert genes_match("TRAV1", "TRAV1*02")

    def test_different_base_genes_never_match(self):
        assert not genes_match("TRAV1*01", "TRAV2*01")
        assert not genes_match("TRAV1", "TRAV2")
        assert not genes_match("TRAV1*01", "TRAV2")

    def test_empty_matches_empty(self):
        assert genes_match("", "")
        assert genes_match(None, None)

    def test_empty_does_not_match_gene(self):
        assert not genes_match("", "TRAV1")
        assert not genes_match("TRAV1", "")
        assert not genes_match(None, "TRAV1")

    def test_symmetry(self):
        pairs = [
            ("TRAV1", "TRAV1*01"),
            ("TRAV1*01", "TRAV1*02"),
            ("TRBV5-1", "TRBV5-1*07"),
        ]
        for g1, g2 in pairs:
            assert genes_match(g1, g2) == genes_match(g2, g1)


# ── allele utility functions ──────────────────────────────────────────────────

def test_strip_allele_removes_suffix():
    assert strip_allele("TRBV6-5*02") == "TRBV6-5"
    assert strip_allele("TRBV6-5") == "TRBV6-5"
    assert strip_allele(None) == ""
    assert strip_allele("") == ""


def test_allele_with_default_preserves_explicit():
    assert allele_with_default("TRBV6-5*02") == "TRBV6-5*02"
    assert allele_with_default("TRBV6-5") == "TRBV6-5*01"
    assert allele_with_default(None) == ""


def test_allele_to_major_normalises_to_01():
    assert allele_to_major("TRBV6-5*02") == "TRBV6-5*01"
    assert allele_to_major("TRBV6-5") == "TRBV6-5*01"
    assert allele_to_major("TRBV6-5*01") == "TRBV6-5*01"
    assert allele_to_major(None) == ""
