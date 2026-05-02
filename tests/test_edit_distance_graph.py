"""Unit and benchmark tests for :mod:`mir.graph.edit_distance_graph`.

Unit tests use hand-crafted Clonotypes whose pairwise distances are
known exactly, verifying all combinations of metric, threshold, and
gene-match filters.

The benchmark suite (``RUN_BENCHMARK=1``) loads the real GILGFVFTL VDJdb
dataset and validates biological plausibility of the resulting graphs.

Run unit tests only::

    pytest tests/test_edit_distance_graph.py -v

Run benchmark tests::

    RUN_BENCHMARK=1 pytest tests/test_edit_distance_graph.py -v -s
"""

from __future__ import annotations

import gzip
import time
import unittest
from pathlib import Path

import pytest

from tests.conftest import skip_benchmarks
from tests.conftest import benchmark_repertoire_workers
from mir.common.clonotype import Clonotype
from mir.graph.edit_distance_graph import build_edit_distance_graph

# ---------------------------------------------------------------------------
# Shared mock sequences
# ---------------------------------------------------------------------------
#
# SEQ_A  "CASSRSGYTF"  — reference (10 AA)
# SEQ_B  "XASSRSGYTF"  — Hamming 1 from A  (C→X at pos 0)
# SEQ_C  "XXSSRSGYTF"  — Hamming 2 from A  (C→X, A→X at pos 0-1)
# SEQ_D  "CASSRSGYTFF" — length 11, levenshtein 1 from A (insertion)
#
# Pairwise Hamming (equal-length only):
#   ham(A,B)=1  ham(A,C)=2  ham(B,C)=1  ham(B,B)=0
#
# Pairwise Levenshtein:
#   lev(A,B)=1  lev(A,C)=2  lev(A,D)=1  lev(B,C)=1

SEQ_A = "CASSRSGYTF"
SEQ_B = "XASSRSGYTF"
SEQ_C = "XXSSRSGYTF"
SEQ_D = "CASSRSGYTFF"

_LOCUS = "TRB"
_V1 = "TRBV19"
_V2 = "TRBV6"
_C1 = "TRBC1"
_C2 = "TRBC2"


def _r(idx: int, seq: str, v: str = _V1, c: str = _C1) -> Clonotype:
    return Clonotype(sequence_id=str(idx), locus=_LOCUS, v_gene=v, c_gene=c, junction_aa=seq, duplicate_count=1, _validate=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _edge_set(g) -> set[frozenset[int]]:
    return {frozenset(e.tuple) for e in g.es}


# ---------------------------------------------------------------------------
# Hamming graph tests
# ---------------------------------------------------------------------------

class TestHammingGraph(unittest.TestCase):

    def test_single_edge_ham1(self):
        """Two sequences differing by 1 → connected."""
        ra, rb = _r(0, SEQ_A), _r(1, SEQ_B)
        g = build_edit_distance_graph([ra, rb], metric="hamming", threshold=1, nproc=1)
        self.assertEqual(g.vcount(), 2)
        self.assertEqual(g.ecount(), 1)

    def test_threshold_filters_ham2(self):
        """Hamming distance 2 > threshold 1 → no edge."""
        ra, rc = _r(0, SEQ_A), _r(1, SEQ_C)
        g = build_edit_distance_graph([ra, rc], metric="hamming", threshold=1, nproc=1)
        self.assertEqual(g.ecount(), 0)

    def test_threshold_two_includes_ham2(self):
        """With threshold=2, Hamming 2 sequences are connected."""
        ra, rc = _r(0, SEQ_A), _r(1, SEQ_C)
        g = build_edit_distance_graph([ra, rc], metric="hamming", threshold=2, nproc=1)
        self.assertEqual(g.ecount(), 1)

    def test_identical_seq_zero_distance(self):
        """Identical sequences have Hamming 0 ≤ threshold → edge."""
        ra = _r(0, SEQ_A)
        ra2 = _r(1, SEQ_A)
        g = build_edit_distance_graph([ra, ra2], metric="hamming", threshold=1, nproc=1)
        self.assertEqual(g.ecount(), 1)

    def test_diff_length_no_edge(self):
        """Different-length sequences → no Hamming edge (undefined)."""
        ra, rd = _r(0, SEQ_A), _r(1, SEQ_D)
        g = build_edit_distance_graph([ra, rd], metric="hamming", threshold=1, nproc=1)
        self.assertEqual(g.ecount(), 0)

    def test_v_gene_match_filters_cross_v(self):
        """v_gene_match=True skips pairs with different v_gene."""
        ra = _r(0, SEQ_A, v=_V1)
        rb_diff = _r(1, SEQ_B, v=_V2)  # ham 1 from ra but diff v_gene
        # Without filter: 1 edge
        g_no = build_edit_distance_graph([ra, rb_diff], metric="hamming",
                                         threshold=1, nproc=1)
        self.assertEqual(g_no.ecount(), 1)
        # With filter: 0 edges
        g_yes = build_edit_distance_graph([ra, rb_diff], metric="hamming",
                                          threshold=1, v_gene_match=True, nproc=1)
        self.assertEqual(g_yes.ecount(), 0)

    def test_v_gene_match_keeps_same_v(self):
        """v_gene_match=True still connects pairs with the same v_gene."""
        ra = _r(0, SEQ_A, v=_V1)
        rb = _r(1, SEQ_B, v=_V1)
        g = build_edit_distance_graph([ra, rb], metric="hamming",
                                      threshold=1, v_gene_match=True, nproc=1)
        self.assertEqual(g.ecount(), 1)

    def test_v_gene_match_multi(self):
        """
        Four rearrangements: two V1 (ham 1 apart) and one V2 (ham 1 from A).
        Without v_gene_match: 5 edges.
        With v_gene_match:    2 edges (only V1 pairs at ham ≤ 1).
        """
        # V1: ra(A), rb(B), rc(C)   V2: rb_v2(B)
        ra = _r(0, SEQ_A, v=_V1)
        rb = _r(1, SEQ_B, v=_V1)
        rc = _r(2, SEQ_C, v=_V1)
        rb_v2 = _r(3, SEQ_B, v=_V2)

        # Without filter:
        # (A,B)=1✓ (A,C)=2✗ (A,B_v2)=1✓ (B,C)=1✓ (B,B_v2)=0✓ (C,B_v2)=1✓ → 5 edges
        g_all = build_edit_distance_graph([ra, rb, rc, rb_v2], metric="hamming",
                                          threshold=1, nproc=1)
        self.assertEqual(g_all.ecount(), 5)

        # With v_gene_match: only V1×V1 pairs
        # (A,B)=1✓ (A,C)=2✗ (B,C)=1✓ → 2 edges
        g_v = build_edit_distance_graph([ra, rb, rc, rb_v2], metric="hamming",
                                        threshold=1, v_gene_match=True, nproc=1)
        self.assertEqual(g_v.ecount(), 2)

    def test_c_gene_match_filters_cross_c(self):
        """c_gene_match=True skips pairs with different c_gene."""
        ra = _r(0, SEQ_A, c=_C1)
        rb_diff_c = _r(1, SEQ_B, c=_C2)
        g_no = build_edit_distance_graph([ra, rb_diff_c], metric="hamming",
                                         threshold=1, nproc=1)
        self.assertEqual(g_no.ecount(), 1)
        g_yes = build_edit_distance_graph([ra, rb_diff_c], metric="hamming",
                                          threshold=1, c_gene_match=True, nproc=1)
        self.assertEqual(g_yes.ecount(), 0)

    def test_both_gene_match(self):
        """v_gene_match AND c_gene_match both must pass for an edge."""
        ra = _r(0, SEQ_A, v=_V1, c=_C1)
        rb_same = _r(1, SEQ_B, v=_V1, c=_C1)     # both match → edge
        rb_diff_v = _r(2, SEQ_B, v=_V2, c=_C1)   # v differs → no edge
        rb_diff_c = _r(3, SEQ_B, v=_V1, c=_C2)   # c differs → no edge
        g = build_edit_distance_graph(
            [ra, rb_same, rb_diff_v, rb_diff_c],
            metric="hamming", threshold=1,
            v_gene_match=True, c_gene_match=True, nproc=1,
        )
        # Only (ra, rb_same) qualifies
        self.assertEqual(g.ecount(), 1)
        edge = list(g.es)[0]
        vi, vj = edge.tuple
        names = {g.vs[vi]["name"], g.vs[vj]["name"]}
        self.assertEqual(names, {SEQ_A, SEQ_B})

    def test_vertex_attributes(self):
        """Graph carries name, v_gene, c_gene vertex attributes."""
        ra = _r(0, SEQ_A, v=_V1, c=_C1)
        rb = _r(1, SEQ_B, v=_V2, c=_C2)
        g = build_edit_distance_graph([ra, rb], metric="hamming", threshold=1, nproc=1)
        self.assertEqual(g.vs["name"], [SEQ_A, SEQ_B])
        self.assertEqual(g.vs["v_gene"], [_V1, _V2])
        self.assertEqual(g.vs["c_gene"], [_C1, _C2])

    def test_invalid_metric_raises(self):
        """Unknown metric raises ValueError."""
        with self.assertRaises(ValueError):
            build_edit_distance_graph([_r(0, SEQ_A)], metric="jaccard", nproc=1)

    def test_empty_list(self):
        """Empty input produces an empty graph."""
        g = build_edit_distance_graph([], metric="hamming", nproc=1)
        self.assertEqual(g.vcount(), 0)
        self.assertEqual(g.ecount(), 0)

    def test_single_rearrangement(self):
        """Single rearrangement → 1 vertex, 0 edges."""
        g = build_edit_distance_graph([_r(0, SEQ_A)], metric="hamming", nproc=1)
        self.assertEqual(g.vcount(), 1)
        self.assertEqual(g.ecount(), 0)

    def test_n_jobs_matches_nproc(self):
        """`n_jobs` and backward-compat `nproc` yield identical graphs."""
        rearrangements = [_r(0, SEQ_A), _r(1, SEQ_B), _r(2, SEQ_C), _r(3, SEQ_D)]
        g_nproc = build_edit_distance_graph(
            rearrangements,
            metric="levenshtein",
            threshold=1,
            nproc=1,
        )
        g_njobs = build_edit_distance_graph(
            rearrangements,
            metric="levenshtein",
            threshold=1,
            n_jobs=1,
        )
        self.assertEqual(_edge_set(g_nproc), _edge_set(g_njobs))

    def test_parallel_matches_single_worker(self):
        """Threaded trie search must match serial edge set."""
        seqs = [SEQ_A, SEQ_B, SEQ_C, SEQ_D, "CASSRSGYTH", "CASSRSGYTY", "CASSPSGYTF"]
        rearrangements = [_r(i, s) for i, s in enumerate(seqs)]
        serial = build_edit_distance_graph(
            rearrangements,
            metric="hamming",
            threshold=1,
            n_jobs=1,
        )
        parallel = build_edit_distance_graph(
            rearrangements,
            metric="hamming",
            threshold=1,
            n_jobs=4,
        )
        self.assertEqual(_edge_set(serial), _edge_set(parallel))


# ---------------------------------------------------------------------------
# Levenshtein graph tests
# ---------------------------------------------------------------------------

class TestLevenshteinGraph(unittest.TestCase):

    def test_connects_diff_length_lev1(self):
        """Levenshtein distance 1 for sequences of different lengths → edge."""
        ra, rd = _r(0, SEQ_A), _r(1, SEQ_D)
        g = build_edit_distance_graph([ra, rd], metric="levenshtein", threshold=1, nproc=1)
        self.assertEqual(g.ecount(), 1)

    def test_same_result_equal_length(self):
        """For equal-length sequences, Levenshtein and Hamming agree at d≤1."""
        ra, rb = _r(0, SEQ_A), _r(1, SEQ_B)
        g_h = build_edit_distance_graph([ra, rb], metric="hamming", threshold=1, nproc=1)
        g_l = build_edit_distance_graph([ra, rb], metric="levenshtein", threshold=1, nproc=1)
        self.assertEqual(g_h.ecount(), g_l.ecount())

    def test_threshold_filters_lev2(self):
        """Levenshtein distance 2 > threshold 1 → no edge."""
        ra, rc = _r(0, SEQ_A), _r(1, SEQ_C)
        g = build_edit_distance_graph([ra, rc], metric="levenshtein", threshold=1, nproc=1)
        self.assertEqual(g.ecount(), 0)

    def test_lev_more_edges_than_ham(self):
        """Levenshtein connects diff-length pairs that Hamming cannot."""
        ra, rb, rd = _r(0, SEQ_A), _r(1, SEQ_B), _r(2, SEQ_D)
        g_h = build_edit_distance_graph([ra, rb, rd], metric="hamming", threshold=1, nproc=1)
        g_l = build_edit_distance_graph([ra, rb, rd], metric="levenshtein", threshold=1, nproc=1)
        # Hamming: (A,B)=1 edge; (A,D)=skip diff length; (B,D)=skip diff length → 1 edge
        # Lev:     (A,B)=1 edge; (A,D)=1 edge; (B,D)=lev("XASSRSGYTF","CASSRSGYTFF")
        #           lev(B,D) = min substitutions + 1 insertion from XASSRSGYTF → CASSRSGYTFF
        #                    = at least 2 (C→X change + final F insertion) → no edge
        self.assertEqual(g_h.ecount(), 1)
        self.assertEqual(g_l.ecount(), 2)

    def test_v_gene_match_levenshtein(self):
        """v_gene_match works the same way for levenshtein."""
        ra = _r(0, SEQ_A, v=_V1)
        rd_same_v = _r(1, SEQ_D, v=_V1)  # lev 1, same v
        rd_diff_v = _r(2, SEQ_D, v=_V2)  # lev 1, diff v
        # (A,D_v1)=1✓, (A,D_v2)=1✓, (D_v1,D_v2)=0✓ → 3 edges
        g_no = build_edit_distance_graph([ra, rd_same_v, rd_diff_v],
                                         metric="levenshtein", threshold=1, nproc=1)
        self.assertEqual(g_no.ecount(), 3)
        # v_gene_match: only same-v pairs → (A,D_v1)=1✓ → 1 edge
        g_yes = build_edit_distance_graph([ra, rd_same_v, rd_diff_v],
                                          metric="levenshtein", threshold=1,
                                          v_gene_match=True, nproc=1)
        self.assertEqual(g_yes.ecount(), 1)  # only (A,D_v1)

    def test_connected_component(self):
        """Three sequences forming a chain A-B-C are in the same CC."""
        # A=(ham1)=B=(ham1)=C  all equal-length, threshold=1
        ra, rb, rc = _r(0, SEQ_A), _r(1, SEQ_B), _r(2, SEQ_C)
        g = build_edit_distance_graph([ra, rb, rc], metric="levenshtein", threshold=1, nproc=1)
        # edges: (A,B)=1✓, (A,C)=2✗, (B,C)=1✓ → 2 edges, 1 CC
        self.assertEqual(g.ecount(), 2)
        self.assertEqual(len(g.components()), 1)


# ---------------------------------------------------------------------------
# Benchmark tests (real GILGFVFTL data)
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent / "assets"
GILG_FILE = ASSETS / "gilgfvftl_trb_cdr3.txt.gz"


def _load_gilg_rearrangements() -> list[Clonotype]:
    with gzip.open(GILG_FILE, "rt", encoding="utf-8") as f:
        seqs = [l.strip() for l in f if l.strip()]
    return [Clonotype(sequence_id=str(i), locus="TRB", v_gene="TRB", junction_aa=seq, duplicate_count=1) for i, seq in enumerate(seqs)]


@unittest.skipUnless(GILG_FILE.exists(), "VDJdb asset missing — run tests/assets/fetch_vdjdb_gilgfvftl.sh")
@skip_benchmarks
class TestEditDistanceGraphBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rearrangements = _load_gilg_rearrangements()
        print(f"\n  Loaded {len(cls.rearrangements)} GILGFVFTL CDR3 rearrangements")

    def _largest_cc(self, g):
        return g.components().giant()

    def test_hamming_runtime_serial_vs_parallel(self):
        """Benchmark runtime consistency for n_jobs matrix on GIL data."""
        workers = benchmark_repertoire_workers(default="1,4")
        runtimes: dict[int, float] = {}
        baseline_edges: set[frozenset[int]] | None = None

        for w in workers:
            t0 = time.perf_counter()
            g = build_edit_distance_graph(
                self.rearrangements,
                metric="hamming",
                threshold=1,
                n_jobs=w,
            )
            elapsed = time.perf_counter() - t0
            runtimes[w] = elapsed
            edges = _edge_set(g)
            if baseline_edges is None:
                baseline_edges = edges
            else:
                self.assertEqual(edges, baseline_edges)

        print("\n  Hamming runtime matrix:")
        for w in workers:
            print(f"    n_jobs={w}: {runtimes[w]:.2f}s")
        if 1 in runtimes:
            for w in workers:
                if w == 1:
                    continue
                if runtimes[w] > 0:
                    print(f"    speedup 1->{w}: {runtimes[1] / runtimes[w]:.2f}x")

    # -- Hamming threshold=1 --------------------------------------------------

    def test_hamming_graph_basic(self):
        """Hamming graph (threshold=1) builds without error and has correct vertex count."""
        t0 = time.perf_counter()
        g = build_edit_distance_graph(self.rearrangements, metric="hamming",
                                      threshold=1, nproc=1)
        elapsed = time.perf_counter() - t0
        print(f"\n  Hamming (threshold=1): {g.vcount()} vertices, "
              f"{g.ecount()} edges in {elapsed:.2f}s")
        self.assertEqual(g.vcount(), len(self.rearrangements))
        self.assertGreater(g.ecount(), 0)

    def test_hamming_largest_cc_has_rs_motif(self):
        """Largest CC of Hamming graph contains sequences with 'RS' motif."""
        g = build_edit_distance_graph(self.rearrangements, metric="hamming",
                                      threshold=1, nproc=1)
        largest = self._largest_cc(g)
        seqs = largest.vs["name"]
        print(f"\n  Largest CC (Hamming): {largest.vcount()} vertices")
        print(f"  Sample sequences:\n    " + "\n    ".join(seqs[:5]))
        rs_seqs = [s for s in seqs if "RS" in s]
        print(f"  Sequences with RS motif: {len(rs_seqs)}")
        self.assertGreater(len(rs_seqs), 0,
                           "Largest CC should contain sequences with 'RS' motif")

    def test_hamming_with_v_gene_match(self):
        """Hamming graph with v_gene_match=True runs correctly.

        Since all rearrangements carry the same placeholder v_gene, the result
        equals the unconstrained graph (edge count unchanged).
        """
        g_plain = build_edit_distance_graph(self.rearrangements, metric="hamming",
                                            threshold=1, nproc=1)
        g_vmatch = build_edit_distance_graph(self.rearrangements, metric="hamming",
                                             threshold=1, v_gene_match=True, nproc=1)
        print(f"\n  Hamming v_gene_match edges: {g_vmatch.ecount()} "
              f"(plain: {g_plain.ecount()})")
        # All have same placeholder v_gene → identical edge sets
        self.assertEqual(g_plain.ecount(), g_vmatch.ecount())

    # -- Levenshtein threshold=1 ----------------------------------------------

    def test_levenshtein_graph_basic(self):
        """Levenshtein graph (threshold=1) has at least as many edges as Hamming."""
        g_h = build_edit_distance_graph(self.rearrangements, metric="hamming",
                                        threshold=1, nproc=1)
        t0 = time.perf_counter()
        g_l = build_edit_distance_graph(self.rearrangements, metric="levenshtein",
                                        threshold=1, nproc=1)
        elapsed = time.perf_counter() - t0
        print(f"\n  Levenshtein (threshold=1): {g_l.vcount()} vertices, "
              f"{g_l.ecount()} edges in {elapsed:.2f}s")
        self.assertGreaterEqual(g_l.ecount(), g_h.ecount())

    def test_levenshtein_largest_cc_has_rs_motif(self):
        """Largest CC of Levenshtein graph contains 'RS' motif sequences."""
        g = build_edit_distance_graph(self.rearrangements, metric="levenshtein",
                                      threshold=1, nproc=1)
        largest = self._largest_cc(g)
        seqs = largest.vs["name"]
        print(f"\n  Largest CC (Levenshtein): {largest.vcount()} vertices")
        rs_seqs = [s for s in seqs if "RS" in s]
        print(f"  Sequences with RS motif: {len(rs_seqs)}")
        self.assertGreater(len(rs_seqs), 0,
                           "Largest CC should contain sequences with 'RS' motif")

    def test_levenshtein_with_v_gene_match(self):
        """Levenshtein graph with v_gene_match=True runs correctly."""
        g_plain = build_edit_distance_graph(self.rearrangements, metric="levenshtein",
                                            threshold=1, nproc=1)
        g_vmatch = build_edit_distance_graph(self.rearrangements, metric="levenshtein",
                                             threshold=1, v_gene_match=True, nproc=1)
        print(f"\n  Levenshtein v_gene_match edges: {g_vmatch.ecount()} "
              f"(plain: {g_plain.ecount()})")
        self.assertEqual(g_plain.ecount(), g_vmatch.ecount())


if __name__ == "__main__":
    unittest.main()
