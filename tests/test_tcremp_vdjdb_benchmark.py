"""VDJdb-backed TCREmp benchmark tests with clustering-quality assertions.

Run with:
    env RUN_BENCHMARK=1 python -m pytest -s tests/test_tcremp_vdjdb_benchmark.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.parser import VDJdbFullPairedParser, VDJdbSlimParser
from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
from mir.common.single_cell_repair import impute_missing_chains
from mir.embedding.tcremp import PairedTCREmp, TCREmp
from mir.utils.embedding_diagnostics import analyze_embedding_dbscan

skip_benchmarks = pytest.mark.skipif(
    not os.getenv("RUN_BENCHMARK"), reason="set RUN_BENCHMARK=1 to run"
)

ASSETS = Path(__file__).parent / "assets"
_VDJDB_SLIM_FILE = ASSETS / "vdjdb.slim.txt.gz"
_VDJDB_FULL_FILE = ASSETS / "vdjdb_full.txt.gz"

SEED = 42


def _seeded_sample(df: pl.DataFrame, n: int) -> pl.DataFrame:
    if df.height <= n:
        return df
    return df.sample(n=n, with_replacement=False, shuffle=True, seed=SEED)


def _categorize_epitopes(df: pl.DataFrame, focal_epitopes: list[str]) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("epitope").is_in(focal_epitopes))
        .then(pl.col("epitope"))
        .otherwise(pl.lit("other"))
        .alias("epitope_cat")
    )


def _balanced_subset(
    df: pl.DataFrame,
    *,
    label_col: str,
    focal_epitopes: list[str],
    sample_per_epitope: int,
    other_sample: int,
) -> pl.DataFrame:
    selected_parts: list[pl.DataFrame] = []
    for ep in focal_epitopes:
        selected_parts.append(
            _seeded_sample(df.filter(pl.col(label_col) == ep), sample_per_epitope)
        )
    selected_parts.append(
        _seeded_sample(df.filter(pl.col(label_col) == "other"), other_sample)
    )
    return pl.concat([x for x in selected_parts if x.height > 0], how="vertical")


def _cluster_metrics(X_raw: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    metrics = analyze_embedding_dbscan(X_raw, labels, seed=SEED)
    return {
        "n_comp": int(metrics["n_comp"]),
        "eps": float(metrics["eps"]),
        "n_clusters": int(metrics["n_clusters"]),
        "retention": float(metrics["retention"]),
        "purity": float(metrics["purity"]),
        "consistency": float(metrics["consistency"]),
        "median_4nn": float(metrics["median_4nn"]),
    }


@skip_benchmarks
@pytest.mark.benchmark
class TestSingleChainVDJdbTCREmpQuality:
    """Single-chain VDJdb slim benchmark with quality and speed assertions."""

    N_PROTOTYPES = 1000

    def test_single_chain_metrics_and_speed(self):
        sample = VDJdbSlimParser().parse_file(_VDJDB_SLIM_FILE, species="HomoSapiens")
        trb = sample["TRB"].clonotypes
        rows = []
        for c in trb:
            ep = c.clone_metadata.get("antigen.epitope", "")
            if ep:
                rows.append(
                    {
                        "sequence_id": c.sequence_id,
                        "epitope": ep,
                        "v_gene": c.v_gene,
                        "j_gene": c.j_gene,
                        "junction_aa": c.junction_aa,
                    }
                )
        df = pl.DataFrame(rows)
        focal = (
            df.group_by("epitope")
            .len()
            .sort("len", descending=True)
            .head(10)
            .get_column("epitope")
            .to_list()
        )
        df = _categorize_epitopes(df, focal)
        subset = _balanced_subset(
            df,
            label_col="epitope_cat",
            focal_epitopes=focal,
            sample_per_epitope=250,
            other_sample=500,
        )

        clonos = [
            Clonotype(v_gene=r["v_gene"], j_gene=r["j_gene"], junction_aa=r["junction_aa"])
            for r in subset.iter_rows(named=True)
        ]
        labels = subset.get_column("epitope_cat").to_numpy()

        model = TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES)
        t0 = time.perf_counter()
        X = model.embed(clonos)
        elapsed = time.perf_counter() - t0
        metrics = _cluster_metrics(X, labels)

        print("\nSingle-chain VDJdb TRB benchmark")
        print(f"records={len(clonos)} prototypes={self.N_PROTOTYPES} time={elapsed:.3f}s")
        print(
            f"n_comp={metrics['n_comp']} eps={metrics['eps']:.4f} "
            f"clusters={metrics['n_clusters']} retention={metrics['retention']:.3f} "
            f"purity={metrics['purity']:.3f} consistency={metrics['consistency']:.3f}"
        )

        assert X.shape == (len(clonos), 3 * self.N_PROTOTYPES)
        assert elapsed / max(len(clonos), 1) < 0.0006
        assert metrics["n_clusters"] >= 1
        # Bounded-kneedle DBSCAN keeps this benchmark in a moderate-retention regime.
        assert metrics["retention"] >= 0.55
        assert metrics["purity"] >= 0.35
        assert metrics["consistency"] >= 0.10


@skip_benchmarks
@pytest.mark.benchmark
class TestPairedVDJdbTCREmpQuality:
    """Paired VDJdb full benchmark with strict vs imputed quality assertions."""

    N_PROTOTYPES = 1000

    def _paired_table(self, sample) -> pl.DataFrame:
        rows = []
        for pair in sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes:
            chains = {
                pair.clonotype1.locus: pair.clonotype1,
                pair.clonotype2.locus: pair.clonotype2,
            }
            barcode = pair.pair_id.split("_", 1)[0]
            meta = sample.single_cell_repertoire.barcode_metadata.get(barcode, {})
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "epitope": meta.get("antigen.epitope", ""),
                    "tra_v": chains["TRA"].v_gene,
                    "tra_j": chains["TRA"].j_gene,
                    "tra_junction": chains["TRA"].junction_aa,
                    "trb_v": chains["TRB"].v_gene,
                    "trb_j": chains["TRB"].j_gene,
                    "trb_junction": chains["TRB"].junction_aa,
                }
            )
        return pl.DataFrame(rows)

    def _pairs_from_subset(self, subset: pl.DataFrame, pair_map: dict):
        return [pair_map[pair_id] for pair_id in subset.get_column("pair_id").to_list()]

    def test_paired_metrics_and_performance(self):
        parser = VDJdbFullPairedParser()
        strict_cell_df, strict_meta = parser.parse_cell_clonotypes_file(
            _VDJDB_FULL_FILE,
            sample_id="vdjdb_full_human_strict",
            species="HomoSapiens",
            include_incomplete=False,
        )
        strict_sample = build_tenx_sample_from_cell_clonotypes(
            strict_cell_df,
            sample_id="vdjdb_full_human_strict",
            barcode_metadata=strict_meta,
        )

        impute_input_df, impute_meta = parser.parse_cell_clonotypes_file(
            _VDJDB_FULL_FILE,
            sample_id="vdjdb_full_human_impute",
            species="HomoSapiens",
            include_incomplete=True,
        )
        imputed_cell_df = impute_missing_chains(impute_input_df)
        imputed_sample = build_tenx_sample_from_cell_clonotypes(
            imputed_cell_df,
            sample_id="vdjdb_full_human_impute",
            barcode_metadata=impute_meta,
        )

        strict_df = self._paired_table(strict_sample)
        imputed_df = self._paired_table(imputed_sample)

        focal = (
            strict_df.group_by("epitope")
            .len()
            .sort("len", descending=True)
            .head(10)
            .get_column("epitope")
            .to_list()
        )
        strict_df = _categorize_epitopes(strict_df, focal)
        imputed_df = _categorize_epitopes(imputed_df, focal)

        strict_subset = _balanced_subset(
            strict_df,
            label_col="epitope_cat",
            focal_epitopes=focal,
            sample_per_epitope=250,
            other_sample=500,
        )
        imputed_subset = _balanced_subset(
            imputed_df,
            label_col="epitope_cat",
            focal_epitopes=focal,
            sample_per_epitope=250,
            other_sample=500,
        )

        strict_map = {
            p.pair_id: p
            for p in strict_sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
        }
        imputed_map = {
            p.pair_id: p
            for p in imputed_sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
        }

        strict_pairs = self._pairs_from_subset(strict_subset, strict_map)
        imputed_pairs = self._pairs_from_subset(imputed_subset, imputed_map)
        strict_labels = strict_subset.get_column("epitope_cat").to_numpy()
        imputed_labels = imputed_subset.get_column("epitope_cat").to_numpy()

        model = PairedTCREmp.from_defaults(
            species="human",
            locus_pair="TRA_TRB",
            n_prototypes=self.N_PROTOTYPES,
        )

        t0 = time.perf_counter()
        X_strict = model.embed(strict_pairs)
        t_strict = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_imputed = model.embed(imputed_pairs)
        t_imputed = time.perf_counter() - t0

        tra_only = [p.clonotype1 if p.clonotype1.locus == "TRA" else p.clonotype2 for p in strict_pairs]
        trb_only = [p.clonotype1 if p.clonotype1.locus == "TRB" else p.clonotype2 for p in strict_pairs]
        t0 = time.perf_counter()
        _ = model.chain1_model.embed(tra_only)
        t_tra = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = model.chain2_model.embed(trb_only)
        t_trb = time.perf_counter() - t0

        strict_metrics = _cluster_metrics(X_strict, strict_labels)
        imputed_metrics = _cluster_metrics(X_imputed, imputed_labels)

        print("\nPaired VDJdb benchmark")
        print(
            f"strict: n={len(strict_pairs)} t={t_strict:.3f}s "
            f"purity={strict_metrics['purity']:.3f} retention={strict_metrics['retention']:.3f}"
        )
        print(
            f"imputed: n={len(imputed_pairs)} t={t_imputed:.3f}s "
            f"purity={imputed_metrics['purity']:.3f} retention={imputed_metrics['retention']:.3f}"
        )
        print(f"paired/(TRA+TRB)={t_strict / max(t_tra + t_trb, 1e-9):.3f}x")

        assert X_strict.shape == (len(strict_pairs), model.embedding_dim)
        assert X_imputed.shape == (len(imputed_pairs), model.embedding_dim)
        assert t_strict / max(len(strict_pairs), 1) < 0.0008
        assert t_imputed / max(len(imputed_pairs), 1) < 0.0008
        assert t_strict / max(t_tra + t_trb, 1e-9) < 1.6

        assert strict_metrics["n_clusters"] >= 1
        assert imputed_metrics["n_clusters"] >= 1
        assert strict_metrics["retention"] >= 0.50
        assert imputed_metrics["retention"] >= 0.60
        assert strict_metrics["purity"] >= 0.50
        assert imputed_metrics["purity"] >= 0.40
        assert strict_metrics["consistency"] >= 0.20
        assert imputed_metrics["consistency"] >= 0.08

        # Imputation should not catastrophically degrade cluster purity.
        assert imputed_metrics["purity"] + 0.15 >= strict_metrics["purity"]


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
class TestSingleChainVDJdbMixedLargeBootstrap:
    """Stress benchmark for bootstrap-stable eps selection on mixed noisy embeddings."""

    N_PROTOTYPES = 1000
    N_RANDOM = 30_000

    def test_single_chain_metrics_with_mixed_random_30k(self):
        sample = VDJdbSlimParser().parse_file(_VDJDB_SLIM_FILE, species="HomoSapiens")
        trb = sample["TRB"].clonotypes

        rows = []
        for c in trb:
            ep = c.clone_metadata.get("antigen.epitope", "")
            if ep:
                rows.append(
                    {
                        "sequence_id": c.sequence_id,
                        "epitope": ep,
                        "v_gene": c.v_gene,
                        "j_gene": c.j_gene,
                        "junction_aa": c.junction_aa,
                    }
                )
        df = pl.DataFrame(rows)
        focal = (
            df.group_by("epitope")
            .len()
            .sort("len", descending=True)
            .head(10)
            .get_column("epitope")
            .to_list()
        )
        df = _categorize_epitopes(df, focal)
        subset = _balanced_subset(
            df,
            label_col="epitope_cat",
            focal_epitopes=focal,
            sample_per_epitope=250,
            other_sample=500,
        )

        real_clonos = [
            Clonotype(v_gene=r["v_gene"], j_gene=r["j_gene"], junction_aa=r["junction_aa"])
            for r in subset.iter_rows(named=True)
        ]
        real_labels = subset.get_column("epitope_cat").to_list()

        rng = np.random.default_rng(SEED)
        v_genes = df.get_column("v_gene").drop_nulls().unique().to_list()
        j_genes = df.get_column("j_gene").drop_nulls().unique().to_list()
        olga = OlgaModel(locus="TRB", species="human", seed=SEED)
        random_junctions = olga.generate_sequences_parallel(
            self.N_RANDOM,
            n_jobs=8,
            seed=SEED,
        )
        olga.close()

        random_clonos = [
            Clonotype(
                v_gene=v_genes[int(rng.integers(0, len(v_genes)))],
                j_gene=j_genes[int(rng.integers(0, len(j_genes)))],
                junction_aa=j,
            )
            for j in random_junctions
        ]
        all_clonos = real_clonos + random_clonos
        labels = np.asarray(real_labels + ["random"] * len(random_clonos), dtype=object)

        model = TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES)
        t0 = time.perf_counter()
        X = model.embed(all_clonos)
        t_embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = analyze_embedding_dbscan(X, labels, seed=SEED)
        t_diag = time.perf_counter() - t0

        selector_meta = metrics["eps_selector_meta"]
        print("\nSingle-chain VDJdb mixed random benchmark")
        print(
            f"n={len(all_clonos)} random={len(random_clonos)} "
            f"embed={t_embed:.3f}s diag={t_diag:.3f}s "
            f"eps={metrics['eps']:.4f} retention={metrics['retention']:.3f}"
        )
        print(f"selector_meta={selector_meta}")

        assert X.shape == (len(all_clonos), 3 * self.N_PROTOTYPES)
        assert metrics["eps_selection_mode"] == "stable_kneedle"
        assert int(selector_meta["n_bootstrap"]) >= 10
        assert int(selector_meta["n_bootstrap"]) <= 100
        assert int(selector_meta["subset_size"]) <= 35_000
        assert int(selector_meta["n_candidates"]) >= int(selector_meta["n_bootstrap"])
        assert metrics["n_clusters"] >= 1
        assert metrics["retention"] >= 0.40
