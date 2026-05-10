"""Benchmark and cohort comparison for repertoire pooling on aging data.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_pool_benchmark.py -s
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from mir.common.parser import VDJtoolsParser
from mir.common.pool import pool_samples
from mir.common.repertoire import SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset
from tests.conftest import benchmark_repertoire_workers, skip_benchmarks

REAL_REPS = Path(__file__).parent / "assets" / "real_repertoires"
META_PATH = REAL_REPS / "metadata_aging.txt"


@skip_benchmarks
def test_pooling_convergence_younger_vs_aged(capsys) -> None:
    """Pool cohorts and compare convergence distributions with a K-S test.

    Primary target cohorts:
    - cord blood: ``age == 0``
    - aged: ``age >= 80``

    Fallback (when no age==0 is present in current metadata):
    - youngest available subset: ``age <= 10``
    """
    meta = pd.read_csv(META_PATH, sep="\t")

    aged_ids = set(meta.loc[meta["age"] >= 80, "sample_id"].astype(str))
    cord_ids = set(meta.loc[meta["age"] == 0, "sample_id"].astype(str))
    if not cord_ids:
        cord_ids = set(meta.loc[meta["age"] <= 10, "sample_id"].astype(str))

    if len(cord_ids) < 2:
        pytest.skip("No sufficient cord/young cohort samples (age==0 or <=10) in metadata_aging.txt")
    if len(aged_ids) < 2:
        pytest.skip("No sufficient aged cohort samples (age>=80) in metadata_aging.txt")

    workers = benchmark_repertoire_workers(default="4")[0]

    t0 = time.perf_counter()
    ds = RepertoireDataset.from_folder_polars(
        REAL_REPS,
        parser=VDJtoolsParser(),
        metadata_file="metadata_aging.txt",
        file_name_column="file_name",
        sample_id_column="sample_id",
        metadata_sep="\t",
        skip_missing_files=True,
        n_workers=workers,
        progress=False,
    )
    load_s = time.perf_counter() - t0

    cord_samples = [s for sid, s in ds.samples.items() if str(sid) in cord_ids]
    aged_samples = [s for sid, s in ds.samples.items() if str(sid) in aged_ids]

    cord_pool = pool_samples(cord_samples, rule="ntvj", weighted=True, include_sample_ids=False)
    aged_pool = pool_samples(aged_samples, rule="ntvj", weighted=True, include_sample_ids=False)

    def _as_locus_pool(pooled) -> tuple[str, list]:
        if isinstance(pooled, SampleRepertoire):
            if "TRB" in pooled.loci:
                return "TRB", pooled.loci["TRB"].clonotypes
            first_locus = next(iter(pooled.loci.keys()))
            return first_locus, pooled.loci[first_locus].clonotypes
        return pooled.locus, pooled.clonotypes

    cord_locus, cord_clones = _as_locus_pool(cord_pool)
    aged_locus, aged_clones = _as_locus_pool(aged_pool)
    cord_total_clonotypes = len(cord_clones)
    aged_total_clonotypes = len(aged_clones)

    # Compare normalized incidence (fraction of cohort samples carrying a clonotype key).
    cord_n = max(len(cord_samples), 1)
    aged_n = max(len(aged_samples), 1)
    cord_conv = [c.clone_metadata.get("incidence", 0) / cord_n for c in cord_clones]
    aged_conv = [c.clone_metadata.get("incidence", 0) / aged_n for c in aged_clones]

    if len(cord_conv) < 20 or len(aged_conv) < 20:
        raise AssertionError(
            "Insufficient pooled clonotypes for robust KS comparison: "
            f"cord={len(cord_conv)}, aged={len(aged_conv)}"
        )

    # Higher convergence in cord implies a right-shifted distribution, i.e.
    # CDF_cord(x) < CDF_aged(x) for many x.
    ks_stat, ks_p = stats.ks_2samp(cord_conv, aged_conv, alternative="less")

    cord_med = float(pd.Series(cord_conv).median())
    aged_med = float(pd.Series(aged_conv).median())

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Pooling benchmark on aging cohort")
        print(f"Loaded in {load_s:.2f}s using {workers} worker(s)")
        print(f"Cord/young cohort size: {len(cord_samples)} | locus used: {cord_locus}")
        print(f"Aged cohort size:       {len(aged_samples)} | locus used: {aged_locus}")
        print(
            "Pooled clonotypes total: "
            f"cord={cord_total_clonotypes}, aged={aged_total_clonotypes}"
        )
        print(f"Pooled clonotypes used in KS: cord={len(cord_conv)}, aged={len(aged_conv)}")
        print(f"Median normalized incidence: cord={cord_med:.4f}, aged={aged_med:.4f}")
        print(f"K-S (cord > aged, one-sided): statistic={ks_stat:.4f}, p={ks_p:.6g}")
        print("=" * 76)

    assert cord_med > aged_med, (
        f"Expected higher convergence in cord/young cohort, got medians "
        f"cord={cord_med:.4f}, aged={aged_med:.4f}"
    )
    assert ks_p < 0.05, (
        f"Expected significant K-S separation (cord>aged), got p={ks_p:.6g}"
    )
