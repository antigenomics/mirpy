# 2026-07-14
# Biologically-grounded per-sample AIRR features for the TCGA survival analysis — the axes the raw
# clonotype embedding misses. Three axes (from the internal BostonGene AIRR-tissue EDA): (1) ISOTYPE
# composition (IgG/IgA/IgM from IGH c_call — B-cell class switching / mucosal responses); (2) INFILTRATION
# magnitude / hot-vs-cold (receptor-read load, T-vs-B balance); (3) ATYPICALITY (how far a sample's gene
# usage is from what is typical for its tumour type). Plus clonal expansion. Read counts weighted by
# duplicate_count (abundance), matching the read-based metrics of the source pipeline.

from __future__ import annotations

import numpy as np
import polars as pl

T_LOCI = ("TRA", "TRB", "TRG", "TRD")
B_LOCI = ("IGH", "IGK", "IGL")


def _isotype(c: str | None) -> str | None:
    """IGH constant-gene call -> isotype class. IGHG*→IgG, IGHA*→IgA, IGHM→IgM, IGHD→IgD, IGHE→IgE.
    IGHC (ambiguous) / IGHGP (pseudogene) / null -> None (excluded from isotype fractions)."""
    if not c or not c.startswith("IGH"):
        return None
    if c.startswith("IGHG") and c != "IGHGP":
        return "IgG"
    if c.startswith("IGHA"):
        return "IgA"
    if c == "IGHM":
        return "IgM"
    if c == "IGHD":
        return "IgD"
    if c == "IGHE":
        return "IgE"
    return None


def _clonality(counts: np.ndarray) -> tuple[float, float]:
    """(1 − Pielou evenness, top-clone fraction) from a clone-size vector — expansion, not diversity."""
    c = counts[counts > 0]
    if c.size <= 1:
        return 0.0, 1.0 if c.size == 1 else 0.0
    p = c / c.sum()
    shannon = -np.sum(p * np.log(p))
    evenness = shannon / np.log(c.size)
    return float(1.0 - evenness), float(p.max())


def sample_airr_features(path: str, total_reads: float | None) -> tuple[dict, dict]:
    """Per-sample AIRR features. Returns (scalars, v_usage_counts).

    scalars: infiltration (log receptor reads; log receptor read-fraction of total_reads), T-vs-B balance,
    isotype fractions (IgG/IgA/IgM of typed IGH) + class-switch fraction, pooled clonality + top-clone frac.
    v_usage_counts: {"<locus>:<v_gene>": read_count} for the atypicality distance (assembled cohort-side).
    """
    df = pl.read_csv(path, separator="\t", infer_schema_length=0).with_columns(
        pl.col("duplicate_count").cast(pl.Float64, strict=False).fill_null(1.0))
    locus = df["locus"].to_numpy()
    cnt = df["duplicate_count"].to_numpy()

    reads_all = float(cnt.sum())
    reads_t = float(cnt[np.isin(locus, T_LOCI)].sum())
    reads_b = float(cnt[np.isin(locus, B_LOCI)].sum())

    # isotype composition (read-weighted, IGH only)
    igh = df.filter(pl.col("locus") == "IGH")
    iso_reads = {"IgG": 0.0, "IgA": 0.0, "IgM": 0.0, "IgD": 0.0, "IgE": 0.0}
    for c, n in zip(igh["c_call"].to_list(), igh["duplicate_count"].to_list()):
        k = _isotype(c)
        if k:
            iso_reads[k] += float(n)
    typed = sum(iso_reads.values())
    frac = {k: (v / typed if typed else 0.0) for k, v in iso_reads.items()}
    switched = iso_reads["IgG"] + iso_reads["IgA"] + iso_reads["IgE"]

    clon, top = _clonality(cnt)

    scalars = {
        "infiltration": np.log1p(reads_all),
        "infiltration_frac": np.log10(reads_all / total_reads) if total_reads else np.nan,
        "tb_balance": np.log1p(reads_t) - np.log1p(reads_b),        # >0 T-hot, <0 B-hot (z-scored cohort-side)
        "igg_frac": frac["IgG"], "iga_frac": frac["IgA"], "igm_frac": frac["IgM"],
        "switch_frac": switched / typed if typed else 0.0,          # class-switched fraction of typed IGH
        "clonality": clon, "top_clone": top,
    }
    # V-gene usage (read-weighted) for atypicality — dominant chains only (deep signal)
    v_usage = {}
    for loc in ("IGH", "TRB"):
        sub = df.filter(pl.col("locus") == loc)
        for v, n in zip(sub["v_call"].to_list(), sub["duplicate_count"].to_list()):
            if v:
                v_usage[f"{loc}:{v.split('*')[0]}"] = v_usage.get(f"{loc}:{v.split('*')[0]}", 0.0) + float(n)
    return scalars, v_usage


def atypicality(v_usages: list[dict], tumor_types: list[str]) -> np.ndarray:
    """Per-sample gene-usage distance from its OWN tumour type's centroid = 'atypical' repertoire.

    Builds a sample×V-gene frequency matrix, L1-normalized per sample, then 1 − cosine to the mean usage
    of the sample's tumour type (leave-one-out-ish via the group mean). High = atypical for its cancer.
    """
    vocab = sorted({k for u in v_usages for k in u})
    idx = {k: i for i, k in enumerate(vocab)}
    M = np.zeros((len(v_usages), len(vocab)))
    for i, u in enumerate(v_usages):
        for k, n in u.items():
            M[i, idx[k]] = n
    M /= (M.sum(1, keepdims=True) + 1e-9)                           # per-sample usage frequencies
    tt = np.array(tumor_types)
    out = np.zeros(len(v_usages))
    for t in set(tumor_types):
        m = tt == t
        centroid = M[m].mean(0)
        cn = np.linalg.norm(centroid) + 1e-9
        for i in np.where(m)[0]:
            si = np.linalg.norm(M[i]) + 1e-9
            out[i] = 1.0 - float(M[i] @ centroid) / (si * cn)      # 1 − cosine similarity to centroid
    return out
