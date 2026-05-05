"""Tests and benchmarks for token_tables: Clonotype / Kmer indexing."""

from __future__ import annotations

import time

import pytest

from tests.conftest import skip_benchmarks, skip_integration
from mir.basic.token_tables import (
    Kmer,
    KmerAnnotation,
    KmerSeq,
    KmerStats,
    summarize_annotations,
    summarize_rearrangements,
    tokenize_rearrangements,
)
from mir.common.clonotype import Clonotype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rearrangement(
    junction_aa: str = "CASSLAPGATNEKLFF",
    *,
    locus: str = "TRB",
    id: int = 1,
    v_gene: str = "TRBV5-1",
    c_gene: str = "TRBC1",
    duplicate_count: int = 10,
) -> Clonotype:
    return Clonotype(
        sequence_id=str(id), locus=locus, v_gene=v_gene, c_gene=c_gene,
        junction_aa=junction_aa, duplicate_count=duplicate_count,
        _validate=False,
    )


# ---------------------------------------------------------------------------
# Unit tests — types
# ---------------------------------------------------------------------------

class TestClonotype:
    def test_slots(self):
        r = _make_rearrangement()
        assert r.locus == "TRB"
        assert r.junction_aa == "CASSLAPGATNEKLFF"
        assert r.duplicate_count == 10

    def test_fields(self):
        r = _make_rearrangement()
        assert r.id == "1"
        assert r.v_gene == "TRBV5-1"
        assert r.c_gene == "TRBC1"


class TestKmer:
    def test_hashable(self):
        k = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        d = {k: 1}
        assert d[k] == 1

    def test_equal_by_value(self):
        a = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        b = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        assert a == b
        assert hash(a) == hash(b)

    def test_not_equal_different_seq(self):
        a = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        b = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASX")
        assert a != b

    def test_not_equal_different_gene(self):
        a = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        b = Kmer("TRB", "TRBV5-2", "TRBC1", b"CASS")
        assert a != b


# ---------------------------------------------------------------------------
# Unit tests — tokenize_rearrangements
# ---------------------------------------------------------------------------

class TestTokenizeClonotypes:
    def test_basic_indexing(self):
        r = _make_rearrangement("CASSLAP")
        idx = tokenize_rearrangements([r], k=4)
        # 7 - 4 + 1 = 4 k-mers: CASS, ASSL, SSLA, SLAP
        assert len(idx) == 4
        assert all(len(v) == 1 and v[0].rearrangement is r for v in idx.values())

    def test_true_lookup(self):
        """K-mer known to exist can be found."""
        r = _make_rearrangement("CASSLAPGATNEKLFF")
        idx = tokenize_rearrangements([r], k=5)
        key = Kmer("TRB", "TRBV5-1", "TRBC1", b"LAPGA")
        assert key in idx
        assert idx[key][0].rearrangement is r

    def test_false_lookup(self):
        """K-mer not present is absent from the index."""
        r = _make_rearrangement("CASSLAPGATNEKLFF")
        idx = tokenize_rearrangements([r], k=5)
        missing = Kmer("TRB", "TRBV5-1", "TRBC1", b"ZZZZZ")
        assert missing not in idx

    def test_false_lookup_wrong_gene(self):
        """Same sequence but different gene annotation → not found."""
        r = _make_rearrangement("CASSLAPGATNEKLFF")
        idx = tokenize_rearrangements([r], k=5)
        wrong_gene = Kmer("TRB", "TRBV99", "TRBC1", b"LAPGA")
        assert wrong_gene not in idx

    def test_multiple_rearrangements_shared_kmer(self):
        """Two rearrangements sharing a k-mer both appear in the list."""
        r1 = _make_rearrangement("CASSLA", id=1)
        r2 = _make_rearrangement("CASSXY", id=2)
        idx = tokenize_rearrangements([r1, r2], k=4)
        shared = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        assert shared in idx
        rearrangements = [m.rearrangement for m in idx[shared]]
        assert r1 in rearrangements
        assert r2 in rearrangements

    def test_skip_short_junction(self):
        """Clonotype with junction shorter than k is silently skipped."""
        r = _make_rearrangement("CA")
        idx = tokenize_rearrangements([r], k=5)
        assert len(idx) == 0

    def test_empty_input(self):
        idx = tokenize_rearrangements([], k=3)
        assert idx == {}

    def test_different_loci(self):
        r_trb = Clonotype(sequence_id="1", locus="TRB", v_gene="TRBV5-1", c_gene="TRBC1", junction_aa="CASSLA", duplicate_count=1)
        r_tra = Clonotype(sequence_id="2", locus="TRA", v_gene="TRAV12", c_gene="TRAC", junction_aa="CASSLA", duplicate_count=1)
        idx = tokenize_rearrangements([r_trb, r_tra], k=4)
        key_trb = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        key_tra = Kmer("TRA", "TRAV12", "TRAC", b"CASS")
        assert key_trb in idx and idx[key_trb][0].rearrangement is r_trb
        assert key_tra in idx and idx[key_tra][0].rearrangement is r_tra

    def test_positions_plain(self):
        """Plain k-mers record correct extraction positions."""
        r = _make_rearrangement("CASSLAP")
        idx = tokenize_rearrangements([r], k=4)
        # CASS@0, ASSL@1, SSLA@2, SLAP@3
        assert idx[Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")][0].position == 0
        assert idx[Kmer("TRB", "TRBV5-1", "TRBC1", b"ASSL")][0].position == 1
        assert idx[Kmer("TRB", "TRBV5-1", "TRBC1", b"SSLA")][0].position == 2
        assert idx[Kmer("TRB", "TRBV5-1", "TRBC1", b"SLAP")][0].position == 3

    def test_positions_gapped(self):
        """Gapped k-mers from the same window share the window position."""
        r = _make_rearrangement("CASSLA")
        idx = tokenize_rearrangements([r], k=4, mask_byte=ord("X"))
        # Window 0 (CASS) → XASS, CXSS, CAXS, CASX all at position 0
        for seq in [b"XASS", b"CXSS", b"CAXS", b"CASX"]:
            key = Kmer("TRB", "TRBV5-1", "TRBC1", seq)
            if key in idx:
                assert idx[key][0].position == 0
        # Window 1 (ASSL) → position 1
        key1 = Kmer("TRB", "TRBV5-1", "TRBC1", b"XSSL")
        assert key1 in idx and idx[key1][0].position == 1


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestTokenizeClonotypesBenchmark:
    N = 100_000
    K = 5
    JUNCTION = "CASSLAPGATNEKLFF"  # 16 aa → 12 k-mers per rearrangement

    @pytest.fixture(scope="class")
    def rearrangements(self):
        return [
            Clonotype(sequence_id=str(i), locus="TRB", v_gene="TRBV5-1", c_gene="TRBC1",
                          junction_aa=self.JUNCTION, duplicate_count=10)
            for i in range(self.N)
        ]

    def test_benchmark_tokenize(self, rearrangements):
        # Warm-up
        tokenize_rearrangements(rearrangements[:1000], self.K)

        t0 = time.perf_counter()
        idx = tokenize_rearrangements(rearrangements, self.K)
        elapsed = time.perf_counter() - t0

        n_kmers = len(self.JUNCTION) - self.K + 1
        print(
            f"\ntokenize_rearrangements: {self.N:,} rearrangements, "
            f"k={self.K}, {n_kmers} kmers/seq → "
            f"{len(idx):,} unique Kmer keys, "
            f"{elapsed:.3f}s "
            f"({self.N / elapsed:,.0f} rearrangements/s)"
        )

    def test_benchmark_lookup(self, rearrangements):
        idx = tokenize_rearrangements(rearrangements, self.K)
        key_hit = Kmer("TRB", "TRBV5-1", "TRBC1", b"LAPGA")
        key_miss = Kmer("TRB", "TRBV5-1", "TRBC1", b"ZZZZZ")

        n_lookups = 1_000_000
        t0 = time.perf_counter()
        for _ in range(n_lookups):
            _ = key_hit in idx
        hit_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_lookups):
            _ = key_miss in idx
        miss_elapsed = time.perf_counter() - t0

        print(
            f"\nlookup: {n_lookups:,} hits in {hit_elapsed:.3f}s "
            f"({n_lookups / hit_elapsed:,.0f} ops/s), "
            f"{n_lookups:,} misses in {miss_elapsed:.3f}s "
            f"({n_lookups / miss_elapsed:,.0f} ops/s)"
        )


# ---------------------------------------------------------------------------
# Unit tests — gapped k-mers
# ---------------------------------------------------------------------------

MASK = ord("X")


class TestTokenizeClonotypesGapped:
    def test_gapped_kmer_count(self):
        """Each window produces k gapped variants."""
        r = _make_rearrangement("CASSLAP")  # 7 aa, k=4 → 4 windows × 4 = 16
        idx = tokenize_rearrangements([r], k=4, mask_byte=MASK)
        # All keys must contain exactly one X
        for key in idx:
            assert key.seq.count(MASK) == 1
        # Total unique gapped k-mers ≤ 16 (some may collide)
        total_refs = sum(len(v) for v in idx.values())
        assert total_refs == 16

    def test_gapped_true_lookup(self):
        r = _make_rearrangement("CASSLA")
        idx = tokenize_rearrangements([r], k=4, mask_byte=MASK)
        # Window CASS → gapped: XASS, CXSS, CAXS, CASX
        assert Kmer("TRB", "TRBV5-1", "TRBC1", b"XASS") in idx
        assert Kmer("TRB", "TRBV5-1", "TRBC1", b"CXSS") in idx
        assert Kmer("TRB", "TRBV5-1", "TRBC1", b"CASX") in idx

    def test_gapped_false_lookup(self):
        r = _make_rearrangement("CASSLA")
        idx = tokenize_rearrangements([r], k=4, mask_byte=MASK)
        assert Kmer("TRB", "TRBV5-1", "TRBC1", b"XXSS") not in idx

    def test_gapped_no_mask_is_plain(self):
        """mask_byte=None gives identical result to plain call."""
        r = _make_rearrangement("CASSLA")
        plain = tokenize_rearrangements([r], k=4)
        explicit_none = tokenize_rearrangements([r], k=4, mask_byte=None)
        assert plain.keys() == explicit_none.keys()


# ---------------------------------------------------------------------------
# Unit tests — summarize_rearrangements
# ---------------------------------------------------------------------------

class TestSummarizeClonotypes:
    def test_single_rearrangement(self):
        r = _make_rearrangement("CASSLA", duplicate_count=5)
        stats = summarize_rearrangements([r], k=4)
        # 3 k-mers: CASS, ASSL, SSLA
        assert len(stats) == 3
        for v in stats.values():
            assert v.rearrangement_count == 1
            assert v.duplicate_count == 5

    def test_two_rearrangements_shared_kmer(self):
        r1 = _make_rearrangement("CASSLA", id=1, duplicate_count=3)
        r2 = _make_rearrangement("CASSXY", id=2, duplicate_count=7)
        stats = summarize_rearrangements([r1, r2], k=4)
        shared = Kmer("TRB", "TRBV5-1", "TRBC1", b"CASS")
        assert shared in stats
        assert stats[shared].rearrangement_count == 2
        assert stats[shared].duplicate_count == 10  # 3 + 7

    def test_unique_kmers(self):
        r1 = _make_rearrangement("CASSLA", id=1, duplicate_count=2)
        r2 = _make_rearrangement("CASSXY", id=2, duplicate_count=8)
        stats = summarize_rearrangements([r1, r2], k=4)
        unique_r1 = Kmer("TRB", "TRBV5-1", "TRBC1", b"SSLA")
        unique_r2 = Kmer("TRB", "TRBV5-1", "TRBC1", b"SSXY")
        assert stats[unique_r1] == KmerStats(1, 2)
        assert stats[unique_r2] == KmerStats(1, 8)

    def test_empty(self):
        assert summarize_rearrangements([], k=3) == {}

    def test_skip_short(self):
        r = _make_rearrangement("CA", duplicate_count=99)
        assert summarize_rearrangements([r], k=5) == {}

    def test_different_loci_separate(self):
        r1 = Clonotype(sequence_id="1", locus="TRB", v_gene="V1", c_gene="C1", junction_aa="CASSLA", duplicate_count=1)
        r2 = Clonotype(sequence_id="2", locus="TRA", v_gene="V2", c_gene="C2", junction_aa="CASSLA", duplicate_count=4)
        stats = summarize_rearrangements([r1, r2], k=4)
        k_trb = Kmer("TRB", "V1", "C1", b"CASS")
        k_tra = Kmer("TRA", "V2", "C2", b"CASS")
        assert stats[k_trb] == KmerStats(1, 1)
        assert stats[k_tra] == KmerStats(1, 4)

    def test_gapped_summary(self):
        r = _make_rearrangement("CASSLA", duplicate_count=6)
        stats = summarize_rearrangements([r], k=4, mask_byte=MASK)
        # Gapped: 3 windows × 4 positions = 12 total k-mer emissions
        # Each maps to 1 rearrangement with dup_count 6
        for v in stats.values():
            assert v.duplicate_count % 6 == 0
            assert v.rearrangement_count >= 1
        # All keys contain exactly one mask
        for key in stats:
            assert key.seq.count(MASK) == 1

    def test_gapped_shared_summary(self):
        r1 = _make_rearrangement("CASSLA", id=1, duplicate_count=2)
        r2 = _make_rearrangement("CASSXY", id=2, duplicate_count=3)
        stats = summarize_rearrangements([r1, r2], k=4, mask_byte=MASK)
        # Both produce gapped XASS from window CASS
        shared = Kmer("TRB", "TRBV5-1", "TRBC1", b"XASS")
        assert shared in stats
        assert stats[shared].rearrangement_count == 2
        assert stats[shared].duplicate_count == 5

    def test_repeated_kmer_counts_duplicate_once_per_rearrangement(self):
        r = _make_rearrangement("CASSCAS", duplicate_count=11)
        stats = summarize_rearrangements([r], k=3)
        assert stats[Kmer("TRB", "TRBV5-1", "TRBC1", b"CAS")] == KmerStats(1, 11)


# ---------------------------------------------------------------------------
# Benchmark — summarize
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestSummarizeClonotypesBenchmark:
    N = 100_000
    K = 5
    JUNCTION = "CASSLAPGATNEKLFF"

    @pytest.fixture(scope="class")
    def rearrangements(self):
        return [
            Clonotype(sequence_id=str(i), locus="TRB", v_gene="TRBV5-1", c_gene="TRBC1",
                          junction_aa=self.JUNCTION, duplicate_count=10)
            for i in range(self.N)
        ]

    def test_benchmark_summarize_plain(self, rearrangements):
        summarize_rearrangements(rearrangements[:1000], self.K)
        t0 = time.perf_counter()
        stats = summarize_rearrangements(rearrangements, self.K)
        elapsed = time.perf_counter() - t0
        print(
            f"\nsummarize (plain): {self.N:,} rearrangements, k={self.K} → "
            f"{len(stats):,} unique keys, {elapsed:.3f}s "
            f"({self.N / elapsed:,.0f} rearrangements/s)"
        )

    def test_benchmark_summarize_gapped(self, rearrangements):
        summarize_rearrangements(rearrangements[:1000], self.K, mask_byte=MASK)
        t0 = time.perf_counter()
        stats = summarize_rearrangements(rearrangements, self.K, mask_byte=MASK)
        elapsed = time.perf_counter() - t0
        print(
            f"\nsummarize (gapped): {self.N:,} rearrangements, k={self.K} → "
            f"{len(stats):,} unique keys, {elapsed:.3f}s "
            f"({self.N / elapsed:,.0f} rearrangements/s)"
        )


# ---------------------------------------------------------------------------
# Unit tests — summarize_annotations
# ---------------------------------------------------------------------------

class TestSummarizeAnnotations:
    def test_single_rearrangement_positions(self):
        """Each k-mer gets a separate position annotation."""
        r = _make_rearrangement("CASSLA", duplicate_count=5)
        ann = summarize_annotations([r], k=4)
        # 3 plain k-mers at positions 0, 1, 2
        assert len(ann) == 3
        ks_cass = KmerSeq("TRB", b"CASS")
        assert ks_cass in ann
        inner = ann[ks_cass]
        assert KmerAnnotation("TRBV5-1", "TRBC1", 0) in inner
        assert inner[KmerAnnotation("TRBV5-1", "TRBC1", 0)] == KmerStats(1, 5)

    def test_different_genes_merge_under_same_kmer_seq(self):
        """Same locus+seq but different v_gene → single KmerSeq key,
        two KmerAnnotation entries."""
        r1 = Clonotype(sequence_id="1", locus="TRB", v_gene="TRBV5-1", c_gene="TRBC1", junction_aa="CASSLA", duplicate_count=3)
        r2 = Clonotype(sequence_id="2", locus="TRB", v_gene="TRBV6-2", c_gene="TRBC2", junction_aa="CASSLA", duplicate_count=7)
        ann = summarize_annotations([r1, r2], k=4)
        ks = KmerSeq("TRB", b"CASS")
        assert ks in ann
        inner = ann[ks]
        a1 = KmerAnnotation("TRBV5-1", "TRBC1", 0)
        a2 = KmerAnnotation("TRBV6-2", "TRBC2", 0)
        assert a1 in inner and inner[a1] == KmerStats(1, 3)
        assert a2 in inner and inner[a2] == KmerStats(1, 7)

    def test_different_loci_separate(self):
        r_trb = Clonotype(sequence_id="1", locus="TRB", v_gene="V1", c_gene="C1", junction_aa="CASSLA", duplicate_count=1)
        r_tra = Clonotype(sequence_id="2", locus="TRA", v_gene="V2", c_gene="C2", junction_aa="CASSLA", duplicate_count=4)
        ann = summarize_annotations([r_trb, r_tra], k=4)
        ks_trb = KmerSeq("TRB", b"CASS")
        ks_tra = KmerSeq("TRA", b"CASS")
        assert ks_trb in ann and ks_tra in ann
        assert len(ann[ks_trb]) == 1
        assert len(ann[ks_tra]) == 1

    def test_shared_kmer_same_gene_accumulates(self):
        """Two rearrangements with identical gene annotations at same position
        accumulate counts."""
        r1 = _make_rearrangement("CASSLA", id=1, duplicate_count=2)
        r2 = _make_rearrangement("CASSXY", id=2, duplicate_count=8)
        ann = summarize_annotations([r1, r2], k=4)
        ks = KmerSeq("TRB", b"CASS")
        a = KmerAnnotation("TRBV5-1", "TRBC1", 0)
        assert ann[ks][a] == KmerStats(2, 10)

    def test_position_distinguishes_annotations(self):
        """Same k-mer at different positions → separate KmerAnnotation entries."""
        # ACASS has "AS" at position 2 (from ACAS→no, let's be precise)
        # Use junction where same 3-mer appears twice: CASCA → k=3: CAS@0, ASC@1, SCA@2
        # No repeated k-mer there. Use CASSCAS → CAS@0, ASS@1, SSC@2, SCA@3, CAS@4
        r = _make_rearrangement("CASSCAS", duplicate_count=1)
        ann = summarize_annotations([r], k=3)
        ks_cas = KmerSeq("TRB", b"CAS")
        assert ks_cas in ann
        inner = ann[ks_cas]
        # CAS appears at position 0 and position 4
        a0 = KmerAnnotation("TRBV5-1", "TRBC1", 0)
        a4 = KmerAnnotation("TRBV5-1", "TRBC1", 4)
        assert a0 in inner and inner[a0] == KmerStats(1, 1)
        assert a4 in inner and inner[a4] == KmerStats(1, 1)

    def test_repeated_annotation_counts_duplicate_once_per_rearrangement(self):
        r = _make_rearrangement("CASSCAS", duplicate_count=9)
        ann = summarize_annotations([r], k=3)
        ks = KmerSeq("TRB", b"CAS")
        assert ann[ks][KmerAnnotation("TRBV5-1", "TRBC1", 0)] == KmerStats(1, 9)
        assert ann[ks][KmerAnnotation("TRBV5-1", "TRBC1", 4)] == KmerStats(1, 9)

    def test_gapped_annotations(self):
        r = _make_rearrangement("CASSLA", duplicate_count=6)
        ann = summarize_annotations([r], k=4, mask_byte=MASK)
        # All outer keys should have locus only
        for ks in ann:
            assert isinstance(ks, KmerSeq)
            assert ks.locus == "TRB"
        # Gapped k-mers from window 0 (CASS) should have position 0
        ks_xass = KmerSeq("TRB", b"XASS")
        assert ks_xass in ann
        inner = ann[ks_xass]
        assert KmerAnnotation("TRBV5-1", "TRBC1", 0) in inner

    def test_empty(self):
        assert summarize_annotations([], k=3) == {}

    def test_skip_short(self):
        r = _make_rearrangement("CA", duplicate_count=99)
        assert summarize_annotations([r], k=5) == {}

    def test_gapped_different_genes_merge(self):
        """Gapped: different v_gene rearrangements with same locus+seq
        merge under one KmerSeq."""
        r1 = Clonotype(sequence_id="1", locus="TRB", v_gene="TRBV5-1", c_gene="TRBC1", junction_aa="CASSLA", duplicate_count=2)
        r2 = Clonotype(sequence_id="2", locus="TRB", v_gene="TRBV6-2", c_gene="TRBC2", junction_aa="CASSLA", duplicate_count=3)
        ann = summarize_annotations([r1, r2], k=4, mask_byte=MASK)
        ks = KmerSeq("TRB", b"XASS")
        assert ks in ann
        inner = ann[ks]
        a1 = KmerAnnotation("TRBV5-1", "TRBC1", 0)
        a2 = KmerAnnotation("TRBV6-2", "TRBC2", 0)
        assert a1 in inner and inner[a1] == KmerStats(1, 2)
        assert a2 in inner and inner[a2] == KmerStats(1, 3)


# ---------------------------------------------------------------------------
# Benchmark — summarize_annotations
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestSummarizeAnnotationsBenchmark:
    N = 100_000
    K = 5
    JUNCTION = "CASSLAPGATNEKLFF"

    @pytest.fixture(scope="class")
    def rearrangements(self):
        return [
            Clonotype(sequence_id=str(i), locus="TRB", v_gene="TRBV5-1", c_gene="TRBC1",
                          junction_aa=self.JUNCTION, duplicate_count=10)
            for i in range(self.N)
        ]

    def test_benchmark_annotations_plain(self, rearrangements):
        summarize_annotations(rearrangements[:1000], self.K)
        t0 = time.perf_counter()
        ann = summarize_annotations(rearrangements, self.K)
        elapsed = time.perf_counter() - t0
        total_annotations = sum(len(v) for v in ann.values())
        print(
            f"\nsummarize_annotations (plain): {self.N:,} rearrangements, "
            f"k={self.K} → {len(ann):,} KmerSeq keys, "
            f"{total_annotations:,} annotations, "
            f"{elapsed:.3f}s ({self.N / elapsed:,.0f} rearrangements/s)"
        )


# ---------------------------------------------------------------------------
# OLGA-based realistic benchmark
# ---------------------------------------------------------------------------

@skip_integration
@pytest.mark.integration
class TestOlgaKmerSummary:
    """Generate 10,000 human TCR-beta rearrangements via OLGA and validate
    biological expectations on k-mer incidence."""

    N = 10_000
    K = 3

    @pytest.fixture(scope="class")
    def olga_rearrangements(self):
        from mir.basic.pgen import OlgaModel

        model = OlgaModel(locus="TRB")
        seqs = model.generate_sequences_with_meta(self.N, pgens=False)
        return [
            Clonotype(
                sequence_id=str(i),
                locus="TRB",
                v_gene=rec["v_gene"].split("*")[0],  # strip allele
                c_gene="",
                junction_aa=rec["junction_aa"],
                duplicate_count=1,
            )
            for i, rec in enumerate(seqs)
        ]

    @pytest.fixture(scope="class")
    def annotations(self, olga_rearrangements):
        return summarize_annotations(olga_rearrangements, self.K)

    # -- CSA: V-gene–specific, beginning of junction -----------------------

    def test_csa_present(self, annotations):
        """CSA should be a common k-mer (most CDR3s start with C)."""
        ks = KmerSeq("TRB", b"CSA")
        assert ks in annotations
        total = sum(st.rearrangement_count for st in annotations[ks].values())
        # ~6% of 10k rearrangements start with CSA → expect 400-900
        assert 300 <= total <= 1200, f"CSA total count {total} outside [300, 1200]"

    def test_csa_linked_to_trbv20_1(self, annotations):
        """CSA at position 0 should be predominantly from TRBV20-1."""
        ks = KmerSeq("TRB", b"CSA")
        inner = annotations[ks]
        # Collect annotations at position 0
        pos0 = {ka: st for ka, st in inner.items() if ka.position == 0}
        assert len(pos0) > 0, "CSA should appear at position 0"
        # TRBV20-1 should dominate among pos-0 annotations
        total = sum(st.rearrangement_count for st in pos0.values())
        trbv20_count = sum(
            st.rearrangement_count
            for ka, st in pos0.items()
            if ka.v_gene == "TRBV20-1"
        )
        fraction = trbv20_count / total
        print(
            f"\nCSA@pos0: {trbv20_count}/{total} from TRBV20-1 "
            f"({fraction:.1%})"
        )
        # Observed ~96.6%; TRBV20-1 encodes the CSA motif
        assert 0.85 <= fraction <= 1.0, (
            f"Expected TRBV20-1 fraction 0.85–1.0 for CSA@pos0, got {fraction:.3f}"
        )

    def test_csa_at_beginning(self, annotations):
        """CSA occurrences should overwhelmingly be at position 0."""
        ks = KmerSeq("TRB", b"CSA")
        inner = annotations[ks]
        total = sum(st.rearrangement_count for st in inner.values())
        at_pos0 = sum(
            st.rearrangement_count
            for ka, st in inner.items()
            if ka.position == 0
        )
        fraction = at_pos0 / total
        print(f"\nCSA: {at_pos0}/{total} at position 0 ({fraction:.1%})")
        # Observed ~100%; CSA is a V-gene–encoded motif at CDR3 start
        assert 0.95 <= fraction <= 1.0, (
            f"Expected ≥95% CSA at position 0, got {fraction:.3f}"
        )

    # -- GGG: V-gene–agnostic, middle of junction -------------------------

    def test_ggg_present(self, annotations):
        """GGG should appear in the repertoire."""
        ks = KmerSeq("TRB", b"GGG")
        assert ks in annotations, (
            "GGG not found — unlikely for 10k TRB sequences"
        )
        total = sum(st.rearrangement_count for st in annotations[ks].values())
        # ~307 observed; GGG arises from random N/D insertions
        assert 100 <= total <= 800, f"GGG total count {total} outside [100, 800]"

    def test_ggg_v_gene_agnostic(self, annotations):
        """GGG should come from multiple V genes, not just one."""
        ks = KmerSeq("TRB", b"GGG")
        inner = annotations[ks]
        v_genes = {ka.v_gene for ka in inner}
        print(f"\nGGG: {len(v_genes)} distinct V genes — {sorted(v_genes)}")
        # Observed ~45; GGG arises from N/D insertions, not V-gene–encoded
        assert 20 <= len(v_genes) <= 60, (
            f"Expected 20–60 V genes for GGG, got {len(v_genes)}"
        )

    def test_ggg_middle_position(self, annotations, olga_rearrangements):
        """GGG should predominantly come from the middle portion of
        junction_aa, not the very start or end."""
        ks = KmerSeq("TRB", b"GGG")
        inner = annotations[ks]
        # Compute median junction length for context
        lengths = [len(r.junction_aa) for r in olga_rearrangements]
        median_len = sorted(lengths)[len(lengths) // 2]
        # Count how many GGG hits are at interior positions (not 0, not last)
        total = sum(st.rearrangement_count for st in inner.values())
        interior = sum(
            st.rearrangement_count
            for ka, st in inner.items()
            if 1 <= ka.position <= median_len - self.K - 1
        )
        fraction = interior / total
        print(
            f"\nGGG: {interior}/{total} at interior positions ({fraction:.1%}), "
            f"median junction length={median_len}"
        )
        # Observed ~97.7%; GGG comes from N/D insertions in the junction core
        assert 0.90 <= fraction <= 1.0, (
            f"Expected ≥90% GGG in middle, got {fraction:.3f}"
        )
        # Median junction length for human TRB is typically 14-16 aa
        assert 12 <= median_len <= 18, (
            f"Median junction length {median_len} outside expected [12, 18]"
        )

    # -- Timing ------------------------------------------------------------

    def test_benchmark_olga_summarize(self, olga_rearrangements):
        """Time the full summarize_annotations pipeline on OLGA data."""
        # Warm-up
        summarize_annotations(olga_rearrangements[:500], self.K)

        t0 = time.perf_counter()
        ann = summarize_annotations(olga_rearrangements, self.K)
        elapsed = time.perf_counter() - t0
        total_kmer_seqs = len(ann)
        total_annotations = sum(len(v) for v in ann.values())
        print(
            f"\nOLGA summarize_annotations: {self.N:,} rearrangements, "
            f"k={self.K} → {total_kmer_seqs:,} KmerSeq keys, "
            f"{total_annotations:,} annotations, "
            f"{elapsed:.3f}s ({self.N / elapsed:,.0f} rearrangements/s)"
        )
        # Observed ~6,191 unique 3-mers, ~76,215 annotations for 10k seqs
        assert 4_000 <= total_kmer_seqs <= 9_000, (
            f"KmerSeq count {total_kmer_seqs} outside [4000, 9000]"
        )
        assert 50_000 <= total_annotations <= 120_000, (
            f"Annotation count {total_annotations} outside [50000, 120000]"
        )
