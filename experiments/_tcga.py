# 2026-07-14
# Shared loader + clinical helpers for the airr_tcga cohort (TCGA tumor-infiltrating repertoires).
# LOCAL git-LFS checkout at ~/hf/airr_tcga (HF `isalgo/airr_tcga` auto-fallback): one TSV per sample inside
# samples.tar.gz with a `locus` column carrying all 7 chains, metadata.tsv (clinical + OS survival + read
# depth), metadata.hla.tsv (class-I, donor-level). IG-dominant (~97% IG); TR chains are shallow (~tens/sample).

from __future__ import annotations

import os
import tarfile

import numpy as np
import polars as pl

from _hf import fetch

REPO = "isalgo/airr_tcga"
BASE = os.path.expanduser("~/hf/airr_tcga")
CHAINS = ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
_CANON = r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$"
_STAGE = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}                 # cancer_stage -> ordinal


def tcga_path(fname: str) -> str:
    """Resolve a cohort file local-first (the git-LFS checkout), else HF ``hf_hub_download`` (cached)."""
    local = f"{BASE}/{fname}"
    return local if os.path.exists(local) else fetch(REPO, fname)


def _samples_dir() -> str:
    """Extract samples.tar.gz once into ``<BASE>/samples/`` (random per-sample access); return the dir."""
    d = f"{BASE}/samples"
    if os.path.isdir(d) and any(f.endswith(".tsv") for f in os.listdir(d)):
        return d
    tar = tcga_path("samples.tar.gz")
    with tarfile.open(tar, "r:gz") as t:
        t.extractall(BASE, filter="data")                    # writes <BASE>/samples/<sample_id>.tsv
    return d


def load_metadata(cancer_types=None, *, require_os: bool = False) -> pl.DataFrame:
    """Sample-level clinical table (all Utf8). Filters to usable rows; adds ``stage`` ordinal + ``log_reads``."""
    m = pl.read_csv(tcga_path("metadata.tsv"), separator="\t", infer_schema_length=0)
    m = m.with_columns(
        pl.col("cancer_stage").replace_strict(_STAGE, default=None, return_dtype=pl.Int64).alias("stage"),
        pl.col("age").cast(pl.Float64, strict=False),
        pl.col("total_reads").cast(pl.Float64, strict=False).log1p().alias("log_reads"),
        pl.col("OS").cast(pl.Float64, strict=False),
        pl.col("OS_event").cast(pl.Float64, strict=False),
    )
    if cancer_types is not None:
        m = m.filter(pl.col("study_id").is_in(list(cancer_types)))
    if require_os:
        m = m.filter(pl.col("OS").is_not_null() & pl.col("OS_event").is_in([0.0, 1.0]) & (pl.col("OS") > 0))
    return m


def _canon(path: str, chain: str) -> pl.DataFrame:
    """Read one sample TSV, keep locus==chain, collapse to canonical junction_aa/v_call/j_call/duplicate_count."""
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    return (df.filter(pl.col("locus") == chain)
            .select(pl.col("junction_aa"), pl.col("v_call"), pl.col("j_call"),
                    pl.col("duplicate_count").cast(pl.Float64, strict=False).fill_null(1.0))
            .filter(pl.col("junction_aa").str.contains(_CANON))
            .group_by(["junction_aa", "v_call", "j_call"]).agg(pl.col("duplicate_count").sum())
            .sort(["junction_aa", "v_call", "j_call"]))


def load_tcga(chain: str, cancer_types=None, *, downsample_to: int | None = None,
              min_clonotypes: int = 5, cap: int | None = None, seed: int = 0):
    """Per-sample frames for one chain. Returns ``(rows, frames)`` aligned; samples with
    ``< min_clonotypes`` for that chain are dropped (TR chains are sparse in TCGA)."""
    from vdjtools.io.schema import recompute_frequency
    from vdjtools.preprocess import downsample

    sdir = _samples_dir()
    meta = load_metadata(cancer_types, require_os=False)
    rows_all = meta.to_dicts()
    if cap:
        rows_all = rows_all[:cap]
    rows, frames = [], []
    for r in rows_all:
        p = f"{sdir}/{r['sample_id']}.tsv"
        if not os.path.exists(p):
            continue
        df = _canon(p, chain)
        if df.height < min_clonotypes:
            continue
        if downsample_to and int(df["duplicate_count"].sum()) > downsample_to:
            df = downsample(df, downsample_to, by="reads", seed=seed)
        else:
            df = recompute_frequency(df)
        rows.append(r)
        frames.append(df)
    return rows, frames


def load_hla() -> dict:
    """subject_id -> set of class-I alleles (e.g. {'HLA-A*02:01', ...})."""
    h = pl.read_csv(tcga_path("metadata.hla.tsv"), separator="\t", infer_schema_length=0)
    cols = [c for c in h.columns if c.startswith("HLA-")]
    out = {}
    for r in h.to_dicts():
        out[r["subject_id"]] = {r[c] for c in cols if r[c]}
    return out


def clinical_matrix(rows) -> tuple[np.ndarray, list[str]]:
    """Base clinical covariates for the Cox model: age, sex (male=1), stage ordinal, log(total_reads).

    Rows with any missing covariate get column-median (age/stage/reads) / mode (sex) imputation so the
    survival regression keeps every sample; returns ``(X, names)``.
    """
    age = np.array([r["age"] if r["age"] not in (None, "") else np.nan for r in rows], dtype=float)
    sex = np.array([1.0 if (r["sex"] or "").lower() == "male" else 0.0 for r in rows])
    stage = np.array([_STAGE.get(r["cancer_stage"], np.nan) for r in rows], dtype=float)
    reads = np.array([np.log1p(float(r["total_reads"])) if r["total_reads"] not in (None, "") else np.nan
                      for r in rows])
    X = np.column_stack([age, sex, stage, reads])
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        col[~finite] = np.median(col[finite]) if finite.any() else 0.0
    return X, ["age", "sex_male", "stage", "log_reads"]
