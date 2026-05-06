"""Cross-graph integration test: k-mer RS cluster vs. Hamming edit-distance graph.

Validates that the set of RS-bearing rearrangements identified by the k-mer
token graph overlaps substantially with the largest connected component of the
Hamming edit-distance graph built from the same GILGFVFTL CDR3 sequences.

Both graphs capture proximity in sequence space, but via different lenses:
* **k-mer graph** — shared annotated 3-mers; the RS-filtered cluster contains
  sequences that share at least one RS-containing k-mer.
* **Hamming graph** — pairwise single-substitution distance; the largest CC
  groups sequences that are mutation-neighbours of each other.

The overlap between the two largest CCs must be ≥50 % of the k-mer RS cluster,
confirming that RS-bearing sequences are mutation-adjacent in CDR3 space.

Run::

    RUN_BENCHMARK=1 pytest tests/test_rs_graph_overlap.py -v -s
"""

from __future__ import annotations

import gzip
import unittest
from pathlib import Path

from tests.conftest import skip_benchmarks
from mir.basic.token_tables import (
    filter_token_table,
    tokenize_rearrangements,
)
from mir.common.clonotype import Clonotype
from mir.graph.edit_distance_graph import build_edit_distance_graph
from mir.graph.token_graph import build_token_graph

# ---------------------------------------------------------------------------
# Asset path
# ---------------------------------------------------------------------------

GILG_FILE = Path(__file__).parent / "assets" / "gilgfvftl_trb_cdr3.txt.gz"

_K = 3
_HAMMING_THRESHOLD = 1

# Minimum fraction of k-mer RS sequences that must appear in Hamming largest CC.
_MIN_OVERLAP_FRACTION = 0.50

# Minimum fraction of total RS sequences that must be in k-mer RS cluster.
_MIN_RS_KMER_COVERAGE = 0.80


# ---------------------------------------------------------------------------
# Shared fixture loader
# ---------------------------------------------------------------------------

def _load_rearrangements() -> list[Clonotype]:
    with gzip.open(GILG_FILE, "rt", encoding="utf-8") as fh:
        seqs = [line.strip() for line in fh if line.strip()]
    return [
        Clonotype(
            sequence_id=str(i),
            locus="TRB",
            v_gene="TRB",
            junction_aa=seq,
            duplicate_count=1,
        )
        for i, seq in enumerate(seqs)
    ]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    GILG_FILE.exists(),
    "VDJdb GILG asset missing — run python tests/prepare_airr_benchmark_data.py",
)
@skip_benchmarks
class TestRSGraphOverlap(unittest.TestCase):
    """Overlap between k-mer RS cluster and Hamming graph largest CC."""

    @classmethod
    def setUpClass(cls) -> None:
        rearrangements = _load_rearrangements()
        cls.n_total = len(rearrangements)
        cls.n_rs_total = sum(1 for r in rearrangements if "RS" in r.junction_aa)

        # --- k-mer RS cluster ------------------------------------------------
        table    = tokenize_rearrangements(rearrangements, k=_K)
        rs_table = filter_token_table(table, kmer_pattern="RS")
        g_kmer   = build_token_graph(rearrangements, rs_table)
        kmer_cc  = g_kmer.components().giant()
        cls.kmer_rs_seqs: set[str] = {
            v["name"]
            for v in kmer_cc.vs
            if v["node_type"] == "rearrangement"
        }

        # --- Hamming graph largest CC ----------------------------------------
        g_hamming    = build_edit_distance_graph(
            rearrangements,
            metric="hamming",
            threshold=_HAMMING_THRESHOLD,
            nproc=1,
        )
        hamming_cc   = g_hamming.components().giant()
        cls.hamming_seqs: set[str] = set(hamming_cc.vs["name"])

        # Pre-compute overlap
        cls.overlap: set[str] = cls.kmer_rs_seqs & cls.hamming_seqs

        print(
            f"\n  Total CDR3s         : {cls.n_total}"
            f"\n  RS-bearing          : {cls.n_rs_total}"
            f"\n  k-mer RS cluster    : {len(cls.kmer_rs_seqs)} sequences"
            f"\n  Hamming largest CC  : {len(cls.hamming_seqs)} sequences"
            f"\n  Overlap             : {len(cls.overlap)} "
            f"({len(cls.overlap) / len(cls.kmer_rs_seqs):.1%} of k-mer RS cluster)"
        )

    # -- k-mer RS cluster sanity checks -------------------------------------

    def test_kmer_rs_cluster_covers_all_rs_sequences(self) -> None:
        """The k-mer RS graph's giant CC captures ≥80 % of all RS-bearing sequences."""
        coverage = len(self.kmer_rs_seqs) / self.n_rs_total
        self.assertGreaterEqual(
            coverage,
            _MIN_RS_KMER_COVERAGE,
            f"Expected ≥{_MIN_RS_KMER_COVERAGE:.0%} RS coverage in k-mer cluster, "
            f"got {coverage:.1%} ({len(self.kmer_rs_seqs)}/{self.n_rs_total})",
        )

    def test_kmer_rs_cluster_contains_only_rs_sequences(self) -> None:
        """Every rearrangement in the k-mer RS cluster contains 'RS'."""
        non_rs = [s for s in self.kmer_rs_seqs if "RS" not in s]
        self.assertEqual(
            non_rs,
            [],
            f"Found {len(non_rs)} non-RS sequences in k-mer RS cluster: {non_rs[:5]}",
        )

    # -- Hamming graph sanity checks ----------------------------------------

    def test_hamming_cc_is_non_trivial(self) -> None:
        """Hamming largest CC must contain a meaningful number of sequences."""
        self.assertGreater(
            len(self.hamming_seqs),
            50,
            "Hamming largest CC is too small — graph may be disconnected",
        )

    def test_hamming_cc_contains_rs_sequences(self) -> None:
        """Hamming largest CC must include some RS-bearing sequences."""
        rs_in_hamming = {s for s in self.hamming_seqs if "RS" in s}
        self.assertGreater(
            len(rs_in_hamming),
            0,
            "Expected RS sequences in Hamming largest CC",
        )

    # -- Cross-graph overlap ------------------------------------------------

    def test_overlap_fraction_above_threshold(self) -> None:
        """≥50 % of k-mer RS sequences must appear in the Hamming largest CC."""
        overlap_fraction = len(self.overlap) / len(self.kmer_rs_seqs)
        self.assertGreaterEqual(
            overlap_fraction,
            _MIN_OVERLAP_FRACTION,
            f"Expected ≥{_MIN_OVERLAP_FRACTION:.0%} overlap between k-mer RS cluster "
            f"and Hamming largest CC, "
            f"got {overlap_fraction:.1%} ({len(self.overlap)}/{len(self.kmer_rs_seqs)})",
        )

    def test_overlap_sequences_have_rs_motif(self) -> None:
        """Every sequence in the overlap must contain 'RS'."""
        non_rs = [s for s in self.overlap if "RS" not in s]
        self.assertEqual(
            non_rs,
            [],
            f"Found {len(non_rs)} non-RS sequences in overlap: {non_rs[:5]}",
        )

    def test_hamming_cc_contains_majority_of_rs_sequences(self) -> None:
        """Hamming largest CC contains ≥50 % of all RS-bearing sequences."""
        rs_in_hamming = {s for s in self.hamming_seqs if "RS" in s}
        fraction = len(rs_in_hamming) / self.n_rs_total
        self.assertGreaterEqual(
            fraction,
            0.50,
            f"Expected ≥50% of RS sequences in Hamming CC, "
            f"got {fraction:.1%} ({len(rs_in_hamming)}/{self.n_rs_total})",
        )
