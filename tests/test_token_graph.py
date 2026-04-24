"""Unit and benchmark tests for token graph construction.

Mock sequences
--------------
Three rearrangements (k=3) with known kmer overlap:

  r_rs1  "CASRS"   → kmers: CAS, ASR, SRS          (SRS contains "RS")
  r_rs2  "CASSRS"  → kmers: CAS, ASS, SSR, SRS      (SRS contains "RS")
  r_none "CASTLG"  → kmers: CAS, AST, STL, TLG      (none contains "RS")

Shared kmer "CAS" links all three rearrangements in the full graph.
After RS-filter only kmer "SRS" remains → r_rs1 and r_rs2 stay connected,
r_none becomes isolated.

Run unit tests::

    pytest tests/test_token_graph.py -v

Run benchmark::

    RUN_BENCHMARK=1 pytest tests/test_token_graph.py -v -s
"""

from __future__ import annotations

import gzip
import time
import unittest
from pathlib import Path

import pytest

from tests.conftest import skip_benchmarks
from mir.basic.token_tables import (
    Kmer,
    Rearrangement,
    filter_token_table,
    tokenize_rearrangements,
)
from mir.graph.token_graph import build_token_graph

# RS retention threshold for count-based filtering benchmarks
_RS_RETENTION_THRESHOLD = 0.80  # ≥80% of RS rearrangements must stay in filtered CC

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
#
# k=3 kmer breakdown:
#   "CASRS"  → CAS(0), ASR(1), SRS(2)
#   "CASSRS" → CAS(0), ASS(1), SSR(2), SRS(3)
#   "CASTLG" → CAS(0), AST(1), STL(2), TLG(3)
#
# Unique kmers (all share locus/v_gene/c_gene so one Kmer object each):
#   CAS, ASR, SRS, ASS, SSR, AST, STL, TLG  →  8 unique Kmer objects
#
# Full graph: 3 rearrangement + 8 kmer = 11 vertices, 11 edges, 1 component
# RS-filtered: only SRS kmer → 3+1=4 vertices, 2 edges, components [3, 1]

_LOCUS = "TRB"
_V = "TRBV1"
_C = "TRBC1"

SEQ_RS1  = "CASRS"
SEQ_RS2  = "CASSRS"
SEQ_NONE = "CASTLG"

_K = 3


def _make_mock():
    r_rs1  = Rearrangement(sequence_id="0", locus=_LOCUS, v_gene=_V, c_gene=_C, junction_aa=SEQ_RS1,  duplicate_count=1)
    r_rs2  = Rearrangement(sequence_id="1", locus=_LOCUS, v_gene=_V, c_gene=_C, junction_aa=SEQ_RS2,  duplicate_count=1)
    r_none = Rearrangement(sequence_id="2", locus=_LOCUS, v_gene=_V, c_gene=_C, junction_aa=SEQ_NONE, duplicate_count=1)
    rearrangements = [r_rs1, r_rs2, r_none]
    table = tokenize_rearrangements(rearrangements, k=_K)
    return rearrangements, table


# ---------------------------------------------------------------------------
# filter_token_table
# ---------------------------------------------------------------------------

class TestFilterTokenTable(unittest.TestCase):

    def setUp(self):
        self.rearrangements, self.table = _make_mock()

    def test_no_filter_returns_same_table(self):
        """None pattern returns the original table object unchanged."""
        result = filter_token_table(self.table)
        self.assertIs(result, self.table)

    def test_rs_filter_keeps_only_srs(self):
        """'RS' pattern retains only the SRS kmer."""
        result = filter_token_table(self.table, kmer_pattern="RS")
        seqs = {k.seq.decode("ascii") for k in result}
        self.assertEqual(seqs, {"SRS"})

    def test_rs_filter_kmer_count(self):
        result = filter_token_table(self.table, kmer_pattern="RS")
        self.assertEqual(len(result), 1)

    def test_no_match_returns_empty(self):
        """Pattern that matches nothing returns an empty dict."""
        result = filter_token_table(self.table, kmer_pattern="XYZ")
        self.assertEqual(len(result), 0)

    def test_regex_anchor_start(self):
        """'^CAS' keeps only the CAS kmer (which starts with CAS)."""
        result = filter_token_table(self.table, kmer_pattern="^CAS")
        seqs = {k.seq.decode("ascii") for k in result}
        self.assertEqual(seqs, {"CAS"})

    def test_regex_multi_match(self):
        """'S' matches all kmers that contain 'S' — a superset check."""
        result = filter_token_table(self.table, kmer_pattern="S")
        seqs = {k.seq.decode("ascii") for k in result}
        # All 8 mock kmers contain at least one S
        for seq in ("CAS", "ASR", "SRS", "ASS", "SSR", "AST", "STL"):
            self.assertIn(seq, seqs)

    def test_original_table_not_modified(self):
        """Filtering never changes the original table."""
        original_keys = set(self.table.keys())
        filter_token_table(self.table, kmer_pattern="RS")
        self.assertEqual(set(self.table.keys()), original_keys)

    def test_kmer_objects_in_filtered_table_match_seq(self):
        """Each retained Kmer's seq decodes to a string matching the pattern."""
        import re
        result = filter_token_table(self.table, kmer_pattern="RS")
        rx = re.compile("RS")
        for kmer in result:
            self.assertTrue(rx.search(kmer.seq.decode("ascii")))


# ---------------------------------------------------------------------------
# filter_token_table — count-based filter
# ---------------------------------------------------------------------------

class TestFilterTokenTableByCount(unittest.TestCase):
    """Tests for min_rearrangement_count filtering.

    Mock table (k=3):
      CAS  → r_rs1, r_rs2, r_none  (3 rearrangements)
      ASR  → r_rs1                  (1 rearrangement)
      SRS  → r_rs1, r_rs2           (2 rearrangements)
      ASS  → r_rs2                  (1 rearrangement)
      SSR  → r_rs2                  (1 rearrangement)
      AST  → r_none                 (1 rearrangement)
      STL  → r_none                 (1 rearrangement)
      TLG  → r_none                 (1 rearrangement)
    """

    def setUp(self):
        self.rearrangements, self.table = _make_mock()

    def test_count_1_keeps_all(self):
        """min_rearrangement_count=1 keeps every kmer."""
        result = filter_token_table(self.table, min_rearrangement_count=1)
        self.assertEqual(len(result), len(self.table))

    def test_count_2_keeps_cas_and_srs(self):
        """min_rearrangement_count=2 retains only CAS (×3) and SRS (×2)."""
        result = filter_token_table(self.table, min_rearrangement_count=2)
        seqs = {k.seq.decode("ascii") for k in result}
        self.assertEqual(seqs, {"CAS", "SRS"})

    def test_count_3_keeps_only_cas(self):
        """min_rearrangement_count=3 retains only CAS (seen in all 3 rearrangements)."""
        result = filter_token_table(self.table, min_rearrangement_count=3)
        seqs = {k.seq.decode("ascii") for k in result}
        self.assertEqual(seqs, {"CAS"})

    def test_count_4_returns_empty(self):
        """min_rearrangement_count=4 exceeds any kmer count → empty dict."""
        result = filter_token_table(self.table, min_rearrangement_count=4)
        self.assertEqual(len(result), 0)

    def test_none_count_returns_same_table(self):
        """min_rearrangement_count=None and no pattern → original table unchanged."""
        result = filter_token_table(self.table)
        self.assertIs(result, self.table)

    def test_combined_pattern_and_count(self):
        """RS pattern + min_count=2 keeps only SRS (RS-containing, ≥2 rearrangements)."""
        result = filter_token_table(self.table, kmer_pattern="RS", min_rearrangement_count=2)
        seqs = {k.seq.decode("ascii") for k in result}
        self.assertEqual(seqs, {"SRS"})

    def test_combined_pattern_and_count_too_strict(self):
        """RS pattern + min_count=3 → empty (SRS only appears in 2 rearrangements)."""
        result = filter_token_table(self.table, kmer_pattern="RS", min_rearrangement_count=3)
        self.assertEqual(len(result), 0)

    def test_original_table_not_modified_by_count_filter(self):
        """Count filtering never mutates the original table."""
        original_keys = set(self.table.keys())
        filter_token_table(self.table, min_rearrangement_count=2)
        self.assertEqual(set(self.table.keys()), original_keys)

    def test_count_matches_distinct_ids(self):
        """Count reflects *distinct* rearrangement IDs, not raw KmerMatch count."""
        # CAS matches r_rs1 at pos 0, r_rs2 at pos 0, r_none at pos 0 → 3 distinct IDs.
        # Even if we had duplicate matches, the count should still be 3.
        result = filter_token_table(self.table, min_rearrangement_count=3)
        cas_keys = [k for k in result if k.seq == b"CAS"]
        self.assertEqual(len(cas_keys), 1)


# ---------------------------------------------------------------------------
# build_token_graph — r_id attribute
# ---------------------------------------------------------------------------

class TestTokenGraphRId(unittest.TestCase):

    def setUp(self):
        self.rearrangements, self.table = _make_mock()
        self.g = build_token_graph(self.rearrangements, self.table)

    def test_rearrangement_vertices_have_correct_r_id(self):
        """Each rearrangement vertex r_id matches Rearrangement.id."""
        for r in self.rearrangements:
            v = next(v for v in self.g.vs
                     if v["node_type"] == "rearrangement" and v["name"] == r.junction_aa)
            self.assertEqual(v["r_id"], r.id)

    def test_kmer_vertices_have_sentinel_r_id(self):
        """Kmer vertices carry r_id=-1 (sentinel)."""
        for v in self.g.vs:
            if v["node_type"] == "kmer":
                self.assertEqual(v["r_id"], -1)

    def test_r_id_attribute_exists_on_all_vertices(self):
        """r_id is set on every vertex (not None)."""
        self.assertTrue(all(v["r_id"] is not None for v in self.g.vs))


# ---------------------------------------------------------------------------
# build_token_graph — full (unfiltered) table
# ---------------------------------------------------------------------------

class TestTokenGraphFull(unittest.TestCase):

    def setUp(self):
        self.rearrangements, self.table = _make_mock()
        self.g = build_token_graph(self.rearrangements, self.table)

    def test_vertex_count(self):
        """3 rearrangements + 8 unique kmers = 11 vertices."""
        self.assertEqual(self.g.vcount(), 11)

    def test_edge_count(self):
        """
        r_rs1  contributes 3 edges (CAS, ASR, SRS)
        r_rs2  contributes 4 edges (CAS, ASS, SSR, SRS)
        r_none contributes 4 edges (CAS, AST, STL, TLG)
        Total = 11.
        """
        self.assertEqual(self.g.ecount(), 11)

    def test_node_type_counts(self):
        types = self.g.vs["node_type"]
        self.assertEqual(types.count("rearrangement"), 3)
        self.assertEqual(types.count("kmer"), 8)

    def test_rearrangement_vertex_names(self):
        r_names = {v["name"] for v in self.g.vs if v["node_type"] == "rearrangement"}
        self.assertEqual(r_names, {SEQ_RS1, SEQ_RS2, SEQ_NONE})

    def test_kmer_vertex_names(self):
        k_names = {v["name"] for v in self.g.vs if v["node_type"] == "kmer"}
        self.assertEqual(k_names, {"CAS", "ASR", "SRS", "ASS", "SSR", "AST", "STL", "TLG"})

    def test_bipartite_edges(self):
        """Every edge connects a rearrangement vertex to a kmer vertex."""
        for e in self.g.es:
            t0 = self.g.vs[e.tuple[0]]["node_type"]
            t1 = self.g.vs[e.tuple[1]]["node_type"]
            self.assertNotEqual(t0, t1, f"Edge between two {t0} vertices")

    def test_single_connected_component(self):
        """All vertices connected through the shared CAS kmer."""
        self.assertEqual(len(self.g.components()), 1)

    def test_vertex_attributes_set(self):
        """v_gene, c_gene, locus are set on all vertices."""
        for attr in ("v_gene", "c_gene", "locus"):
            vals = self.g.vs[attr]
            self.assertTrue(all(v is not None for v in vals))

    def test_r_rs1_degree(self):
        """r_rs1 ('CASRS', 3 unique kmers) has degree 3."""
        v = next(v for v in self.g.vs if v["name"] == SEQ_RS1)
        self.assertEqual(self.g.degree(v.index), 3)

    def test_r_rs2_degree(self):
        """r_rs2 ('CASSRS', 4 unique kmers) has degree 4."""
        v = next(v for v in self.g.vs if v["name"] == SEQ_RS2)
        self.assertEqual(self.g.degree(v.index), 4)

    def test_cas_kmer_degree(self):
        """CAS kmer is shared by all 3 rearrangements → degree 3."""
        v = next(v for v in self.g.vs if v["name"] == "CAS" and v["node_type"] == "kmer")
        self.assertEqual(self.g.degree(v.index), 3)

    def test_srs_kmer_degree(self):
        """SRS kmer appears in r_rs1 and r_rs2 → degree 2."""
        v = next(v for v in self.g.vs if v["name"] == "SRS" and v["node_type"] == "kmer")
        self.assertEqual(self.g.degree(v.index), 2)


# ---------------------------------------------------------------------------
# build_token_graph — RS-filtered table
# ---------------------------------------------------------------------------

class TestTokenGraphFiltered(unittest.TestCase):

    def setUp(self):
        self.rearrangements, table = _make_mock()
        self.ft = filter_token_table(table, kmer_pattern="RS")
        self.g = build_token_graph(self.rearrangements, self.ft)

    def test_vertex_count(self):
        """3 rearrangements + 1 RS kmer = 4 vertices."""
        self.assertEqual(self.g.vcount(), 4)

    def test_edge_count(self):
        """r_rs1 ↔ SRS and r_rs2 ↔ SRS → 2 edges."""
        self.assertEqual(self.g.ecount(), 2)

    def test_kmer_vertex_is_srs(self):
        k_names = {v["name"] for v in self.g.vs if v["node_type"] == "kmer"}
        self.assertEqual(k_names, {"SRS"})

    def test_isolated_no_rs_rearrangement(self):
        """r_none (CASTLG, no RS kmer) is isolated — degree 0."""
        v = next(v for v in self.g.vs if v["name"] == SEQ_NONE)
        self.assertEqual(self.g.degree(v.index), 0)

    def test_component_sizes(self):
        """One component of size 3 (r_rs1, r_rs2, SRS) and one isolate."""
        sizes = sorted(self.g.components().sizes(), reverse=True)
        self.assertEqual(sizes, [3, 1])

    def test_largest_cc_kmer_has_rs(self):
        """The kmer in the largest CC contains 'RS'."""
        largest = self.g.components().giant()
        kmer_names = [v["name"] for v in largest.vs if v["node_type"] == "kmer"]
        self.assertTrue(all("RS" in name for name in kmer_names))

    def test_largest_cc_rearrangements_have_rs(self):
        """All rearrangements in the largest CC contain 'RS' in junction_aa."""
        largest = self.g.components().giant()
        r_names = [v["name"] for v in largest.vs if v["node_type"] == "rearrangement"]
        self.assertTrue(all("RS" in name for name in r_names))
        self.assertEqual(set(r_names), {SEQ_RS1, SEQ_RS2})

    def test_no_parallel_edges(self):
        """SRS appears twice in r_rs2 junction_aa but only one edge is created."""
        # SRS matches r_rs2 at position 3 (and r_rs1 at position 2);
        # even if duplicated internally, the graph has no parallel edges.
        srs_v = next(v.index for v in self.g.vs if v["name"] == "SRS")
        self.assertEqual(self.g.degree(srs_v), 2)

    def test_diff_v_gene_makes_separate_kmer_vertices(self):
        """Two rearrangements with different v_gene produce separate Kmer nodes."""
        r_v1 = Rearrangement(sequence_id="10", locus=_LOCUS, v_gene="TRBV1", c_gene=_C, junction_aa=SEQ_RS1, duplicate_count=1)
        r_v2 = Rearrangement(sequence_id="11", locus=_LOCUS, v_gene="TRBV2", c_gene=_C, junction_aa=SEQ_RS1, duplicate_count=1)
        table = tokenize_rearrangements([r_v1, r_v2], k=_K)
        # SRS appears in both but under different Kmer keys (different v_gene)
        srs_kmers = [k for k in table if k.seq == b"SRS"]
        self.assertEqual(len(srs_kmers), 2)
        g = build_token_graph([r_v1, r_v2], table)
        # 2 rearrangements + 2×3 unique-per-vgene kmers = 8 (CAS×2, ASR×2, SRS×2)
        self.assertEqual(g.vcount(), 8)
        srs_vertices = [v for v in g.vs if v["name"] == "SRS"]
        self.assertEqual(len(srs_vertices), 2)


# ---------------------------------------------------------------------------
# Benchmark tests (real GILGFVFTL data)
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent / "assets"
GILG_FILE = ASSETS / "gilgfvftl_trb_cdr3.txt.gz"
_BENCH_K = 3
_NONRS_LOSS_THRESHOLD = 0.90  # filtering must remove ≥90% of non-RS rearrangements from the major CC


def _load_gilg():
    with gzip.open(GILG_FILE, "rt", encoding="utf-8") as f:
        seqs = [l.strip() for l in f if l.strip()]
    return [Rearrangement(sequence_id=str(i), locus="TRB", v_gene="TRB", junction_aa=seq, duplicate_count=1) for i, seq in enumerate(seqs)]


@unittest.skipUnless(GILG_FILE.exists(),
                     "VDJdb asset missing — run tests/assets/fetch_vdjdb_gilgfvftl.sh")
@skip_benchmarks
class TestTokenGraphBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rearrangements = _load_gilg()
        cls.table = tokenize_rearrangements(cls.rearrangements, k=_BENCH_K)
        cls.rs_table = filter_token_table(cls.table, kmer_pattern="RS")
        print(f"\n  Loaded {len(cls.rearrangements)} CDR3s; "
              f"{len(cls.table)} total kmers; "
              f"{len(cls.rs_table)} RS kmers")

    # -- full graph -----------------------------------------------------------

    def test_full_graph_structure(self):
        t0 = time.perf_counter()
        g = build_token_graph(self.rearrangements, self.table)
        elapsed = time.perf_counter() - t0
        print(f"\n  Full graph: {g.vcount()} vertices, {g.ecount()} edges in {elapsed:.2f}s")
        self.assertEqual(g.vcount(), len(self.rearrangements) + len(self.table))
        self.assertGreater(g.ecount(), 0)
        # bipartite check (sample)
        for e in list(g.es)[:500]:
            t0v = g.vs[e.tuple[0]]["node_type"]
            t1v = g.vs[e.tuple[1]]["node_type"]
            self.assertNotEqual(t0v, t1v)

    # -- RS-filtered graph ----------------------------------------------------

    def test_filtered_graph_has_rs_kmers(self):
        """Every kmer vertex in the filtered graph contains 'RS'."""
        g = build_token_graph(self.rearrangements, self.rs_table)
        kmer_names = [v["name"] for v in g.vs if v["node_type"] == "kmer"]
        self.assertTrue(len(kmer_names) > 0)
        self.assertTrue(all("RS" in name for name in kmer_names),
                        f"Non-RS kmer vertices found: "
                        f"{[n for n in kmer_names if 'RS' not in n]}")

    def test_filtered_largest_cc_has_rs_kmers(self):
        """Largest CC of filtered graph contains kmer vertices with 'RS'."""
        g = build_token_graph(self.rearrangements, self.rs_table)
        largest = g.components().giant()
        kmer_names = [v["name"] for v in largest.vs if v["node_type"] == "kmer"]
        rs_kmer_names = [n for n in kmer_names if "RS" in n]
        print(f"\n  RS kmers in largest filtered CC: {sorted(rs_kmer_names)}")
        self.assertGreater(len(rs_kmer_names), 0)

    def test_filtered_largest_cc_has_rs_rearrangements(self):
        """Rearrangement nodes in the largest filtered CC contain 'RS' in junction_aa."""
        g = build_token_graph(self.rearrangements, self.rs_table)
        largest = g.components().giant()
        r_seqs = [v["name"] for v in largest.vs if v["node_type"] == "rearrangement"]
        rs_r_seqs = [s for s in r_seqs if "RS" in s]
        print(f"\n  Rearrangements in largest filtered CC: {len(r_seqs)}")
        print(f"  Of which contain 'RS': {len(rs_r_seqs)}")
        self.assertGreater(len(rs_r_seqs), 0)

    def _rs_retention_for_min_count(self, min_count: int) -> float:
        """Return the fraction of RS rearrangements retained in the largest CC."""
        filtered = filter_token_table(self.table, min_rearrangement_count=min_count)
        if not filtered:
            return 0.0
        g = build_token_graph(self.rearrangements, filtered)
        largest = g.components().giant()
        rs_in_largest = sum(
            1 for v in largest.vs
            if v["node_type"] == "rearrangement" and "RS" in v["name"]
        )
        total_rs = sum(1 for r in self.rearrangements if "RS" in r.junction_aa)
        return rs_in_largest / total_rs if total_rs else 0.0

    def test_count_filter_rs_retention_n2(self):
        """min_count=2: ≥80% of RS rearrangements remain in the largest CC."""
        rate = self._rs_retention_for_min_count(2)
        print(f"\n  RS retention (min_count=2): {rate:.1%}")
        self.assertGreaterEqual(rate, _RS_RETENTION_THRESHOLD)

    def test_count_filter_rs_retention_n3(self):
        """min_count=3: ≥80% of RS rearrangements remain in the largest CC."""
        rate = self._rs_retention_for_min_count(3)
        print(f"\n  RS retention (min_count=3): {rate:.1%}")
        self.assertGreaterEqual(rate, _RS_RETENTION_THRESHOLD)

    def test_count_filter_rs_retention_n5(self):
        """min_count=5: ≥80% of RS rearrangements remain in the largest CC."""
        rate = self._rs_retention_for_min_count(5)
        print(f"\n  RS retention (min_count=5): {rate:.1%}")
        self.assertGreaterEqual(rate, _RS_RETENTION_THRESHOLD)

    def test_count_filter_rs_retention_n10(self):
        """min_count=10: ≥80% of RS rearrangements remain in the largest CC."""
        rate = self._rs_retention_for_min_count(10)
        print(f"\n  RS retention (min_count=10): {rate:.1%}")
        self.assertGreaterEqual(rate, _RS_RETENTION_THRESHOLD)

    def test_nonrs_rearrangements_lost_after_filtering(self):
        """Non-RS rearrangements are expelled from the major CC by RS filtering.

        The full token graph is one large CC because all CDR3 sequences share
        common k-mers (e.g. "CAS").  After filtering to RS-only k-mers, every
        rearrangement without "RS" in its junction_aa has no edges and becomes
        isolated.  We verify that ≥90 % of non-RS rearrangements present in the
        full CC are absent from the filtered CC.
        """
        g_full = build_token_graph(self.rearrangements, self.table)
        g_rs   = build_token_graph(self.rearrangements, self.rs_table)

        full_cc_nonrs = {
            v["name"] for v in g_full.components().giant().vs
            if v["node_type"] == "rearrangement" and "RS" not in v["name"]
        }
        filt_cc_nonrs = {
            v["name"] for v in g_rs.components().giant().vs
            if v["node_type"] == "rearrangement" and "RS" not in v["name"]
        }
        full_cc_r_total = sum(
            1 for v in g_full.components().giant().vs
            if v["node_type"] == "rearrangement"
        )
        full_cc_k_nonrs = sum(
            1 for v in g_full.components().giant().vs
            if v["node_type"] == "kmer" and "RS" not in v["name"]
        )

        n_full   = len(full_cc_nonrs)
        n_lost   = n_full - len(filt_cc_nonrs & full_cc_nonrs)
        loss_rate = n_lost / n_full if n_full else 1.0

        print(f"\n  Full CC total rearrangements       : {full_cc_r_total}")
        print(f"  Non-RS rearrangements in full CC   : {n_full}")
        print(f"  Non-RS k-mers in full CC           : {full_cc_k_nonrs}")
        print(f"  Non-RS rearrangements after filter : {n_full - n_lost}")
        print(f"  Loss rate                          : {n_lost}/{n_full} = {loss_rate:.1%}")

        self.assertGreater(n_full, 0,
                           "Expected non-RS rearrangements in the full CC")
        self.assertGreaterEqual(
            loss_rate, _NONRS_LOSS_THRESHOLD,
            f"Expected ≥{_NONRS_LOSS_THRESHOLD:.0%} non-RS rearrangements lost, "
            f"got {loss_rate:.1%}",
        )


if __name__ == "__main__":
    unittest.main()
