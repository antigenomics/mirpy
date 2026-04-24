"""Tests for Polars k-mer implementation and benchmarks comparing
Polars vs the naive object-based implementation in token_tables.py.

Provides memory and time measurements for both approaches.
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import polars as pl
import pytest

from tests.conftest import skip_benchmarks
from mir.basic import token_tables_pl as plmod
from mir.basic.token_tables import (
    Kmer,
    Rearrangement,
    summarize_annotations,
    summarize_rearrangements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pl_df(rows: list[dict]) -> pl.DataFrame:
    """Build a Polars rearrangement DataFrame from a list of dicts."""
    return pl.DataFrame(rows).cast({
        "id": pl.Int64,
        "duplicate_count": pl.Int64,
    })


def _row(
    junction_aa: str = "CASSLAPGATNEKLFF",
    *,
    locus: str = "TRB",
    id: int = 1,
    v_gene: str = "TRBV5-1",
    c_gene: str = "TRBC1",
    duplicate_count: int = 10,
) -> dict:
    return dict(
        id=id, locus=locus, v_gene=v_gene, c_gene=c_gene,
        junction_aa=junction_aa, duplicate_count=duplicate_count,
    )


def _rows_to_rearrangements(rows: list[dict]) -> list[Rearrangement]:
    return [
        Rearrangement(
            sequence_id=str(d["id"]),
            locus=d["locus"],
            v_gene=d["v_gene"],
            c_gene=d["c_gene"],
            junction_aa=d["junction_aa"],
            duplicate_count=d["duplicate_count"],
        )
        for d in rows
    ]


# ===================================================================
# Polars unit tests
# ===================================================================


class TestExpandKmersPl:
    def test_basic(self):
        df = _make_pl_df([_row("CASSLAP")])
        ex = plmod.expand_kmers(df, k=4)
        assert ex.height == 4
        assert set(ex["kmer_seq"].to_list()) == {"CASS", "ASSL", "SSLA", "SLAP"}
        assert set(ex["kmer_pos"].to_list()) == {0, 1, 2, 3}

    def test_skip_short(self):
        df = _make_pl_df([_row("CA")])
        assert plmod.expand_kmers(df, k=5).height == 0

    def test_empty(self):
        df = _make_pl_df([_row("CASSLA")])
        assert plmod.expand_kmers(df.head(0), k=3).height == 0

    def test_multiple_rows(self):
        df = _make_pl_df([
            _row("CASSLA", id=1, duplicate_count=3),
            _row("CASSXY", id=2, duplicate_count=7),
        ])
        assert plmod.expand_kmers(df, k=4).height == 6


class TestSummarizeByGenePl:
    def test_single(self):
        df = _make_pl_df([_row("CASSLA", duplicate_count=5)])
        ex = plmod.expand_kmers(df, k=4)
        s = plmod.summarize_by_gene(ex)
        assert s.height == 3
        for row in s.iter_rows(named=True):
            assert row["rearrangement_count"] == 1
            assert row["duplicate_count"] == 5

    def test_shared_kmer(self):
        df = _make_pl_df([
            _row("CASSLA", id=1, duplicate_count=3),
            _row("CASSXY", id=2, duplicate_count=7),
        ])
        s = plmod.summarize_by_gene(plmod.expand_kmers(df, k=4))
        cass = s.filter(pl.col("kmer_seq") == "CASS")
        assert cass["rearrangement_count"][0] == 2
        assert cass["duplicate_count"][0] == 10

    def test_different_loci(self):
        df = _make_pl_df([
            _row("CASSLA", id=1, locus="TRB", v_gene="V1", c_gene="C1", duplicate_count=1),
            _row("CASSLA", id=2, locus="TRA", v_gene="V2", c_gene="C2", duplicate_count=4),
        ])
        s = plmod.summarize_by_gene(plmod.expand_kmers(df, k=4))
        trb = s.filter((pl.col("locus") == "TRB") & (pl.col("kmer_seq") == "CASS"))
        tra = s.filter((pl.col("locus") == "TRA") & (pl.col("kmer_seq") == "CASS"))
        assert trb["rearrangement_count"][0] == 1 and trb["duplicate_count"][0] == 1
        assert tra["rearrangement_count"][0] == 1 and tra["duplicate_count"][0] == 4


class TestSummarizeByPosPl:
    def test_positions(self):
        df = _make_pl_df([_row("CASSLA", duplicate_count=5)])
        s = plmod.summarize_by_pos(plmod.expand_kmers(df, k=4))
        assert s.height == 3
        for row in s.iter_rows(named=True):
            assert row["rearrangement_count"] == 1
            assert row["duplicate_count"] == 5


class TestSummarizeByVPl:
    def test_different_v_genes(self):
        df = _make_pl_df([
            _row("CASSLA", id=1, v_gene="TRBV5-1", duplicate_count=3),
            _row("CASSLA", id=2, v_gene="TRBV6-2", duplicate_count=7),
        ])
        s = plmod.summarize_by_v(plmod.expand_kmers(df, k=4))
        cass_v5 = s.filter((pl.col("kmer_seq") == "CASS") & (pl.col("v_gene") == "TRBV5-1"))
        cass_v6 = s.filter((pl.col("kmer_seq") == "CASS") & (pl.col("v_gene") == "TRBV6-2"))
        assert cass_v5["rearrangement_count"][0] == 1 and cass_v5["duplicate_count"][0] == 3
        assert cass_v6["rearrangement_count"][0] == 1 and cass_v6["duplicate_count"][0] == 7


class TestSummarizeByCPl:
    def test_different_c_genes(self):
        df = _make_pl_df([
            _row("CASSLA", id=1, c_gene="TRBC1", duplicate_count=2),
            _row("CASSLA", id=2, c_gene="TRBC2", duplicate_count=8),
        ])
        s = plmod.summarize_by_c(plmod.expand_kmers(df, k=4))
        cass_c1 = s.filter((pl.col("kmer_seq") == "CASS") & (pl.col("c_gene") == "TRBC1"))
        cass_c2 = s.filter((pl.col("kmer_seq") == "CASS") & (pl.col("c_gene") == "TRBC2"))
        assert cass_c1["rearrangement_count"][0] == 1 and cass_c1["duplicate_count"][0] == 2
        assert cass_c2["rearrangement_count"][0] == 1 and cass_c2["duplicate_count"][0] == 8


class TestFetchPl:
    @pytest.fixture()
    def data(self):
        df = _make_pl_df([
            _row("CASSLA", id=1, v_gene="TRBV5-1", c_gene="TRBC1", duplicate_count=3),
            _row("CASSXY", id=2, v_gene="TRBV5-1", c_gene="TRBC1", duplicate_count=7),
            _row("TTTXYZ", id=3, locus="TRA", v_gene="TRAV12", c_gene="TRAC", duplicate_count=1),
        ])
        ex = plmod.expand_kmers(df, k=4)
        return df, ex

    def test_fetch_by_kmer(self, data):
        df, ex = data
        assert set(plmod.fetch_by_kmer(df, ex, "TRB", "CASS")["id"].to_list()) == {1, 2}

    def test_fetch_by_kmer_miss(self, data):
        df, ex = data
        assert plmod.fetch_by_kmer(df, ex, "TRB", "ZZZZ").height == 0

    def test_fetch_by_annotated_kmer(self, data):
        df, ex = data
        result = plmod.fetch_by_annotated_kmer(df, ex, "TRB", "TRBV5-1", "TRBC1", "CASS")
        assert set(result["id"].to_list()) == {1, 2}

    def test_fetch_by_annotated_kmer_wrong_gene(self, data):
        df, ex = data
        assert plmod.fetch_by_annotated_kmer(df, ex, "TRB", "TRBV99", "TRBC1", "CASS").height == 0

    def test_fetch_different_locus(self, data):
        df, ex = data
        assert set(plmod.fetch_by_kmer(df, ex, "TRA", "TTXY")["id"].to_list()) == {3}

    def test_fetch_original_columns(self, data):
        df, ex = data
        result = plmod.fetch_by_kmer(df, ex, "TRB", "CASS")
        assert set(result.columns) == set(df.columns)


# ===================================================================
# Cross-implementation: Polars vs naive (object-based)
# ===================================================================


class TestCrossImplementation:
    """Verify Polars and object-based (naive) produce consistent results."""

    @pytest.fixture()
    def shared_input(self):
        dicts = [
            _row("CASSLA", id=1, duplicate_count=3, v_gene="TRBV5-1", c_gene="TRBC1"),
            _row("CASSXY", id=2, duplicate_count=7, v_gene="TRBV5-1", c_gene="TRBC1"),
            _row("CASSLA", id=3, duplicate_count=2, v_gene="TRBV6-2", c_gene="TRBC2"),
        ]
        objs = _rows_to_rearrangements(dicts)
        pl_df = _make_pl_df(dicts)
        return objs, pl_df

    def test_expand_row_count(self, shared_input):
        objs, pl_df = shared_input
        k = 4
        ex_pl = plmod.expand_kmers(pl_df, k)
        # naive: each rearrangement with len >= k produces len-k+1 k-mers
        naive_count = sum(max(0, len(r.junction_aa) - k + 1) for r in objs)
        assert ex_pl.height == naive_count == 9

    def test_summarize_by_gene_matches_naive(self, shared_input):
        objs, pl_df = shared_input
        k = 4
        # Polars
        s_pl = plmod.summarize_by_gene(plmod.expand_kmers(pl_df, k)).sort(
            ["locus", "v_gene", "c_gene", "kmer_seq"]
        )
        # Naive (object-based)
        s_obj = summarize_rearrangements(objs, k)

        # For each Polars summary row, verify it matches the naive result
        for row in s_pl.iter_rows(named=True):
            key = Kmer(row["locus"], row["v_gene"], row["c_gene"],
                       row["kmer_seq"].encode("ascii"))
            assert key in s_obj
            assert row["rearrangement_count"] == s_obj[key].rearrangement_count
            assert row["duplicate_count"] == s_obj[key].duplicate_count

        # Same total number of groups
        assert s_pl.height == len(s_obj)


# ===================================================================
# Benchmark: naive (object-based) vs Polars — time and memory
# ===================================================================


def _measure(func, label: str) -> dict:
    """Run *func*, returning wall time (s) and peak memory (bytes)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"label": label, "elapsed": elapsed, "peak_mem": peak, "result": result}


@skip_benchmarks
@pytest.mark.benchmark
class TestBenchmarkImplementations:
    """Compare time and memory: naive (token_tables.py) vs Polars on
    10,000 OLGA-generated TCR-beta rearrangements."""

    N = 10_000
    K = 3

    @pytest.fixture(scope="class")
    def olga_data(self):
        from mir.basic.pgen import OlgaModel

        model = OlgaModel(chain="TRB")
        seqs = model.generate_sequences_with_meta(self.N, pgens=False)
        dicts = [
            _row(
                rec["cdr3"],
                id=i,
                locus="TRB",
                v_gene=rec["v_gene"].split("*")[0],
                c_gene="",
                duplicate_count=1,
            )
            for i, rec in enumerate(seqs)
        ]
        objs = _rows_to_rearrangements(dicts)
        pl_df = _make_pl_df(dicts)
        return objs, pl_df

    def test_benchmark_naive_summarize(self, olga_data):
        objs, _ = olga_data
        # warm-up
        summarize_rearrangements(objs[:500], self.K)
        m = _measure(lambda: summarize_rearrangements(objs, self.K), "naive")
        print(
            f"\n[naive] summarize_rearrangements: {self.N:,} seqs, k={self.K} → "
            f"{len(m['result']):,} keys, "
            f"time={m['elapsed']:.3f}s, peak_mem={m['peak_mem'] / 1024:.0f} KiB"
        )

    def test_benchmark_polars_summarize(self, olga_data):
        _, pl_df = olga_data
        # warm-up
        plmod.summarize_by_gene(plmod.expand_kmers(pl_df.head(500), self.K))

        def run():
            ex = plmod.expand_kmers(pl_df, self.K)
            return plmod.summarize_by_gene(ex)

        m = _measure(run, "polars")
        print(
            f"\n[polars] expand + summarize_by_gene: {self.N:,} seqs, k={self.K} → "
            f"{m['result'].height:,} summary rows, "
            f"time={m['elapsed']:.3f}s, peak_mem={m['peak_mem'] / 1024:.0f} KiB"
        )

    def test_benchmark_naive_annotations(self, olga_data):
        objs, _ = olga_data
        summarize_annotations(objs[:500], self.K)
        m = _measure(lambda: summarize_annotations(objs, self.K), "naive_ann")
        total = sum(len(v) for v in m["result"].values())
        print(
            f"\n[naive] summarize_annotations: {self.N:,} seqs, k={self.K} → "
            f"{len(m['result']):,} KmerSeq, {total:,} annotations, "
            f"time={m['elapsed']:.3f}s, peak_mem={m['peak_mem'] / 1024:.0f} KiB"
        )

    def test_benchmark_polars_all_summaries(self, olga_data):
        _, pl_df = olga_data
        plmod.expand_kmers(pl_df.head(500), self.K)

        def run():
            ex = plmod.expand_kmers(pl_df, self.K)
            return {
                "by_gene": plmod.summarize_by_gene(ex),
                "by_pos": plmod.summarize_by_pos(ex),
                "by_v": plmod.summarize_by_v(ex),
                "by_c": plmod.summarize_by_c(ex),
            }

        m = _measure(run, "polars_all")
        r = m["result"]
        print(
            f"\n[polars] expand + 4 summaries: {self.N:,} seqs, k={self.K} → "
            f"by_gene={r['by_gene'].height:,}, by_pos={r['by_pos'].height:,}, "
            f"by_v={r['by_v'].height:,}, by_c={r['by_c'].height:,}, "
            f"time={m['elapsed']:.3f}s, peak_mem={m['peak_mem'] / 1024:.0f} KiB"
        )

    def test_benchmark_fetch(self, olga_data):
        _, pl_df = olga_data
        ex = plmod.expand_kmers(pl_df, self.K)
        top = ex.group_by("kmer_seq").len().sort("len", descending=True).head(1)
        kmer_seq = top["kmer_seq"][0]

        n_lookups = 1000
        t0 = time.perf_counter()
        for _ in range(n_lookups):
            plmod.fetch_by_kmer(pl_df, ex, "TRB", kmer_seq)
        elapsed = time.perf_counter() - t0
        print(
            f"\n[polars] fetch_by_kmer '{kmer_seq}': "
            f"{n_lookups:,} lookups in {elapsed:.3f}s "
            f"({n_lookups / elapsed:,.0f} ops/s)"
        )
