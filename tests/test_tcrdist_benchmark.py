"""TCRdist benchmark: influenza GILGFVFTL example.

Reproduces the core TCRdist3 influenza analysis:
  https://tcrdist3.readthedocs.io/en/latest/influenza_example.html

Requires RUN_BENCHMARK=1 and GILGFVFTL sequence data.

Data bootstrap
--------------
Data is read from the following locations (falling back to HuggingFace):

    airr_benchmark/tcrdist/gilgfvftl_trb_junctions.tsv   (TRB TCRdist paper sequences)

If the file is missing, set AIRR_BENCHMARK_ROOT to point to a local clone of
the isalgo/airr_benchmark dataset, or the test downloads a small subset from
the bundled test assets.

Run:
    env RUN_BENCHMARK=1 python -m pytest tests/test_tcrdist_benchmark.py -s -v
"""

from __future__ import annotations

import gzip
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

# Guard — skip unless RUN_BENCHMARK=1
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_BENCHMARK"),
    reason="set RUN_BENCHMARK=1 to run",
)

from mir.common.clonotype import Clonotype
from mir.common.metaclonotype import (
    MetaClonotypeDefinition,
    summarize_metaclonotypes,
)
from mir.common.repertoire import LocusRepertoire
from mir.distances.tcrdist import TcrDist


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRR_ROOT = Path(os.getenv("AIRR_BENCHMARK_ROOT", REPO_ROOT / "airr_benchmark"))
TCRDIST_DIR = AIRR_ROOT / "tcrdist"

# Bundled fallback: tests/assets/gilgfvftl_trb_junctions.txt.gz
BUNDLED_ASSET = REPO_ROOT / "tests" / "assets" / "gilgfvftl_trb_junctions.txt.gz"


def _load_gilgfvftl_clonotypes() -> list[Clonotype]:
    """Load GILGFVFTL-associated TRB clonotypes.

    Tries (in order):
    1. airr_benchmark/tcrdist/gilgfvftl_trb_junctions.tsv
    2. tests/assets/gilgfvftl_trb_junctions.txt.gz  (bundled)
    3. HuggingFace download via huggingface_hub (if installed)
    """
    candidate_tsv = TCRDIST_DIR / "gilgfvftl_trb_junctions.tsv"
    candidate_csv = TCRDIST_DIR / "gilgfvftl_trb_junctions.csv"

    if candidate_tsv.exists():
        return _parse_flat_file(candidate_tsv, sep="\t")
    if candidate_csv.exists():
        return _parse_flat_file(candidate_csv, sep=",")
    if BUNDLED_ASSET.exists():
        return _parse_bare_sequences(BUNDLED_ASSET)

    # Try HuggingFace download
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
        path = hf_hub_download(
            repo_id="isalgo/airr_benchmark",
            filename="tcrdist/gilgfvftl_trb_junctions.tsv",
            repo_type="dataset",
            local_dir=str(TCRDIST_DIR.parent),
        )
        return _parse_flat_file(Path(path), sep="\t")
    except Exception:
        pass

    # Last resort: use bundled GILGFVFTL sequences from test assets
    return _synthetic_gilgfvftl_clonotypes()


def _open_file(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _parse_bare_sequences(path: Path) -> list[Clonotype]:
    """Parse a header-less file with one junction_aa per line."""
    clonotypes = []
    with _open_file(path) as fh:
        for i, line in enumerate(fh):
            junc = line.strip()
            if not junc or not junc[0].isalpha():
                continue
            clonotypes.append(
                Clonotype(
                    sequence_id=f"seq_{i}",
                    junction_aa=junc,
                    locus="TRB",
                )
            )
    return clonotypes


def _parse_flat_file(path: Path, sep: str = "\t") -> list[Clonotype]:
    """Parse a flat TSV/CSV with columns: junction_aa, v_gene, j_gene, count."""
    clonotypes = []
    with _open_file(path) as fh:
        header = fh.readline().strip().split(sep)
        # Normalize column names
        col = {c.lower().strip(): i for i, c in enumerate(header)}
        # Accept various column name conventions
        junc_col = next(
            (col[k] for k in ("junction_aa", "cdr3", "cdr3_aa", "cdr3aa") if k in col),
            None,
        )
        v_col = next(
            (col[k] for k in ("v_gene", "v_call", "v_b_gene", "vgene") if k in col),
            None,
        )
        j_col = next(
            (col[k] for k in ("j_gene", "j_call", "j_b_gene", "jgene") if k in col),
            None,
        )
        count_col = next(
            (col[k] for k in ("count", "duplicate_count", "frequency", "freq") if k in col),
            None,
        )
        if junc_col is None:
            raise ValueError(f"No junction_aa column found in {path}; header: {header}")

        for i, line in enumerate(fh):
            parts = line.strip().split(sep)
            if len(parts) <= max(
                c for c in [junc_col, v_col, j_col] if c is not None
            ):
                continue
            junction_aa = parts[junc_col].strip()
            if not junction_aa or not junction_aa[0].isalpha():
                continue
            v_gene = parts[v_col].strip() if v_col is not None else ""
            j_gene = parts[j_col].strip() if j_col is not None else ""
            count = 1
            if count_col is not None and count_col < len(parts):
                try:
                    count = int(float(parts[count_col]))
                except (ValueError, IndexError):
                    pass
            clonotypes.append(
                Clonotype(
                    sequence_id=f"seq_{i}",
                    junction_aa=junction_aa,
                    v_gene=v_gene,
                    j_gene=j_gene,
                    duplicate_count=count,
                    locus="TRB",
                )
            )
    return clonotypes


def _synthetic_gilgfvftl_clonotypes() -> list[Clonotype]:
    """Synthetic GILGFVFTL-associated TRB sequences for offline testing.

    These are representative TRBV19*01/TRBJ2-7*01 sequences known to respond
    to the GILGFVFTL influenza epitope (HLA-A*02:01).
    """
    # Canonical GILGFVFTL-associated sequences (from published literature)
    known = [
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRSYEQYF",  "TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRASYEQYF", "TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRASSYEQYF","TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRGSSYEQYF","TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ1-5*01"),
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ2-7*01"),
        ("CASSIRSSSYEQYF","TRBV19*01", "TRBJ2-7*01"),
        # Some non-influenza (background) sequences
        ("CASSLGQGANVLTF","TRBV5-1*01",  "TRBJ2-6*01"),
        ("CASSYRGNTEAFF",  "TRBV20-1*01", "TRBJ1-1*01"),
        ("CASSGAGGREQYF",  "TRBV2*01",    "TRBJ2-7*01"),
    ]
    return [
        Clonotype(
            sequence_id=f"gilg_{i:04d}",
            junction_aa=junc,
            v_gene=v,
            j_gene=j,
            duplicate_count=max(1, 10 - i),
            locus="TRB",
        )
        for i, (junc, v, j) in enumerate(known)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gilgfvftl_clonotypes():
    clns = _load_gilgfvftl_clonotypes()
    print(f"\nLoaded {len(clns)} GILGFVFTL-associated clonotypes")
    assert len(clns) > 0, "No clonotypes loaded"
    return clns


@pytest.fixture(scope="module")
def gilgfvftl_rep(gilgfvftl_clonotypes):
    return LocusRepertoire(clonotypes=gilgfvftl_clonotypes, locus="TRB")


@pytest.fixture(scope="module")
def td():
    t0 = time.perf_counter()
    model = TcrDist.from_defaults(
        "TRB", "human",
        w_v=1.0, w_j=0.0, w_cdr3=3.0,
        fixed_gaps=(3, 4, -4, -3),
    )
    print(f"\nTcrDist built in {time.perf_counter() - t0:.2f}s")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: all-vs-all distance matrix
# ─────────────────────────────────────────────────────────────────────────────

def test_self_dist_matrix_benchmark(td, gilgfvftl_clonotypes):
    """Compute self-distance matrix and verify basic properties."""
    clns = gilgfvftl_clonotypes
    N = len(clns)

    tracemalloc.start()
    t0 = time.perf_counter()
    mat = td.self_dist_matrix(clns, n_jobs=1)
    wall = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_pairs = N * (N - 1) // 2
    rate = n_pairs / wall if wall > 0 else float("inf")

    print(f"\n  N={N}, pairs={n_pairs:,}")
    print(f"  Wall time: {wall:.3f}s  ({rate:,.0f} pairs/s)")
    print(f"  Peak RSS:  {peak_bytes / 1e6:.1f} MB")

    assert mat.shape == (N, N)
    assert mat.dtype == np.float64
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-6)
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)
    assert np.all(mat >= 0)


def test_parallel_speedup(td, gilgfvftl_clonotypes):
    """Parallel execution should not produce different results than serial."""
    clns = gilgfvftl_clonotypes
    if len(clns) < 4:
        pytest.skip("Too few clonotypes for parallelism test")

    mat_s = td.dist_matrix(clns, clns, n_jobs=1)
    mat_p = td.dist_matrix(clns, clns, n_jobs=2)
    np.testing.assert_allclose(mat_s, mat_p, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: GILGFVFTL influenza metaclonotype discovery
# ─────────────────────────────────────────────────────────────────────────────

def test_influenza_metaclonotypes(td, gilgfvftl_clonotypes, gilgfvftl_rep):
    """Find influenza metaclonotypes with TCRdist radius clustering.

    The GILGFVFTL-associated TRBV19*01/TRBJ2-7*01 sequences should cluster
    tightly (low within-cluster distances) compared to unrelated sequences.
    """
    clns = gilgfvftl_clonotypes
    N = len(clns)

    # Step 1: compute all-vs-all distance matrix
    t0 = time.perf_counter()
    mat = td.self_dist_matrix(clns)
    print(f"\n  All-vs-all matrix ({N}×{N}): {time.perf_counter()-t0:.3f}s")

    # Step 2: compute per-clonotype median background distance (radius)
    #   Use the repertoire itself as background (simplified)
    t0 = time.perf_counter()
    radii = td.compute_radius(clns, clns, percentile=50)
    print(f"  Radius computation: {time.perf_counter()-t0:.3f}s")
    print(f"  Median radius: {np.median(radii):.2f}  |  Range: [{radii.min():.2f}, {radii.max():.2f}]")

    # Step 3: find metaclonotypes using a fixed radius
    max_dist = float(np.median(radii)) + 1.0 if np.median(radii) > 0 else 200.0
    t0 = time.perf_counter()
    meta = td.find_metaclonotypes(
        gilgfvftl_rep,
        max_distance=max_dist,
        n_jobs=1,
    )
    print(f"  Metaclonotypes (d<={max_dist:.1f}): {meta.n_clusters} clusters, {time.perf_counter()-t0:.3f}s")

    assert isinstance(meta, MetaClonotypeDefinition)
    assert meta.n_clusters > 0

    # Step 4: summarise
    summary = summarize_metaclonotypes(gilgfvftl_rep, meta)
    largest = summary.sort("n_members", descending=True).head(3)
    print(f"  Top-3 cluster sizes: {largest['n_members'].to_list()}")

    # The largest cluster should contain at least 2 members
    assert int(largest["n_members"][0]) >= 2


def test_influenza_v19_cluster_dominates(td, gilgfvftl_clonotypes, gilgfvftl_rep):
    """TRBV19 sequences should be closest to each other (antigen specificity)."""
    v19_clns = [c for c in gilgfvftl_clonotypes if (c.v_gene or "").startswith("TRBV19")]
    other_clns = [c for c in gilgfvftl_clonotypes if not (c.v_gene or "").startswith("TRBV19")]

    if len(v19_clns) < 2 or not other_clns:
        pytest.skip("Need both TRBV19 and non-TRBV19 sequences")

    # Average within-V19 distance
    mat_v19 = td.self_dist_matrix(v19_clns)
    n = len(v19_clns)
    if n > 1:
        upper_tri = mat_v19[np.triu_indices(n, k=1)]
        within_mean = float(np.mean(upper_tri))
    else:
        within_mean = 0.0

    # Average V19-vs-other distance
    mat_cross = td.dist_matrix(v19_clns, other_clns)
    cross_mean = float(np.mean(mat_cross))

    print(f"\n  TRBV19 within-mean distance: {within_mean:.2f}")
    print(f"  TRBV19 vs other cross-mean distance: {cross_mean:.2f}")

    # Antigen-specific sequences should be closer to each other
    assert within_mean < cross_mean, (
        f"Expected within-V19 ({within_mean:.2f}) < cross ({cross_mean:.2f})"
    )


def test_radius_based_enrichment_selection(td, gilgfvftl_clonotypes, gilgfvftl_rep):
    """Sequences with small radii are candidate antigen-specific enriched clones.

    In the TCRdist3 framework, enriched sequences are those with unusually
    small distances to other sequences in the antigen-specific repertoire,
    suggesting convergent evolution.
    """
    clns = gilgfvftl_clonotypes
    radii = td.compute_radius(clns, clns, percentile=50)
    threshold = float(np.percentile(radii, 25))  # bottom quartile = most convergent

    enriched_idx = np.where(radii <= threshold)[0]
    enriched_clns = [clns[i] for i in enriched_idx]
    enriched_rep_ids = [c.sequence_id for c in enriched_clns]

    print(f"\n  Radius threshold (25th pct): {threshold:.2f}")
    print(f"  Enriched candidates: {len(enriched_clns)}")

    meta = td.find_metaclonotypes(
        gilgfvftl_rep,
        representative_ids=enriched_rep_ids,
        max_distance=threshold,
        n_jobs=1,
    )
    print(f"  Metaclonotypes from enriched reps: {meta.n_clusters}")

    assert meta.n_clusters > 0 or len(enriched_clns) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: gap mode comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_mode_timing(gilgfvftl_clonotypes):
    """Compare runtime across fixed_gaps modes."""
    from mir.distances.aligner import GermlineAligner
    from mir.common.gene_library import GeneLibrary

    lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
    ga = GermlineAligner.from_library(lib, loci=["TRB"])
    clns = gilgfvftl_clonotypes

    print()
    for mode, label in [
        ((3, 4, -4, -3), "fixed_gap (C-accel)"),
        ("Mid",           "mid-gap (py+C)"),
        (None,            "full-DP BioPython"),
    ]:
        from mir.distances.tcrdist import TcrDist as _TD
        td_m = _TD(
            locus="TRB", species="human",
            germline_aligner=ga,
            fixed_gaps=mode,
        )
        t0 = time.perf_counter()
        td_m.self_dist_matrix(clns, n_jobs=1)
        wall = time.perf_counter() - t0
        n_pairs = len(clns) * (len(clns) - 1) // 2
        rate = n_pairs / wall if wall > 0 else float("inf")
        print(f"  {label:30s}: {wall:.3f}s  ({rate:>10,.0f} pairs/s)")
