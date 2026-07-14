# 2026-07-14
# Shared loader + HLA/batch helpers for the airr_covid19 cohort (Vlasova et al., Genome Med 2026).
# The dataset is a LOCAL git-LFS checkout at ~/hf/airr_covid19 (HF `isalgo/airr_covid19` auto-fallback), vdjtools-format
# per-sample files with full 4-digit HLA class I+II typing, COVID serostatus, and 9 real FMBA batches.

from __future__ import annotations

import os

import numpy as np
import polars as pl

from _hf import fetch, load_repertoire

REPO = "isalgo/airr_covid19"
BASE = os.path.expanduser("~/hf/airr_covid19")


def covid_path(fname: str) -> str:
    """Resolve a cohort file **local-first, HF fallback**: the local git-LFS checkout if present, else
    ``hf_hub_download`` (cached). Makes every COVID benchmark reproducible on any machine with HF access."""
    local = f"{BASE}/{fname}"
    return local if os.path.exists(local) else fetch(REPO, fname)

# HLA metadata columns by locus (values look like "A*02:05" / "DRB1*11:04" — no "HLA-" prefix in the cell)
HLA_COLS = {
    "A": ["HLA-A_1", "HLA-A_2"], "B": ["HLA-B_1", "HLA-B_2"], "C": ["HLA-C_1", "HLA-C_2"],
    "DRB1": ["HLA-DRB1_1", "HLA-DRB1_2"], "DQB1": ["HLA-DQB1_1", "HLA-DQB1_2"],
    "DPB1": ["HLA-DPB1_1", "HLA-DPB1_2"],
}


def load_covid(n_donors: int | None = None, downsample_to: int | None = 20_000, *,
               chain: str = "TRB", statuses=("COVID", "healthy"), seed: int = 0):
    """One frame per donor (deduped on ``donor_id`` → no leakage), downsampled. Returns (rows, frames)."""
    from vdjtools.io.schema import recompute_frequency
    from vdjtools.preprocess import downsample

    meta = pl.read_csv(covid_path("metadata.tsv"), separator="\t", infer_schema_length=0)
    meta = (meta.filter((pl.col("locus") == chain) & pl.col("COVID_status").is_in(list(statuses)))
            .unique(subset=["donor_id"], keep="first"))
    rows = meta.to_dicts()[:n_donors] if n_donors else meta.to_dicts()
    frames = []
    for r in rows:
        df = load_repertoire(covid_path(r["file_name"]))
        df = (downsample(df, downsample_to, by="reads", seed=seed)
              if downsample_to and int(df["duplicate_count"].sum()) > downsample_to
              else recompute_frequency(df))
        frames.append(df)
    return rows, frames


def load_covid_paired(n_donors: int | None = None, downsample_to: int | None = 20_000, *,
                      statuses=("COVID", "healthy"), seed: int = 0):
    """Per-donor **(TRB, TRA)** frames matched by ``donor_id`` — the cohort is fully paired (1258/1258).

    Returns ``(rows, [(trb_df, tra_df), ...])`` where ``rows`` are the TRB metadata dicts (they carry the
    HLA/COVID/batch labels). Both chains downsampled to ``downsample_to``. TRA carries independent V/J/CDR3
    information (and 87% of the shipped COVID ground-truth clones are α) so α+β concatenation should sharpen
    HLA-imprint and COVID-biomarker signals over β alone.
    """
    from vdjtools.io.schema import recompute_frequency
    from vdjtools.preprocess import downsample

    meta = pl.read_csv(covid_path("metadata.tsv"), separator="\t",
                       infer_schema_length=0).filter(pl.col("COVID_status").is_in(list(statuses)))
    trb = meta.filter(pl.col("locus") == "TRB").unique(subset=["donor_id"], keep="first")
    tra = meta.filter(pl.col("locus") == "TRA").unique(subset=["donor_id"], keep="first")
    tra_file = dict(zip(tra["donor_id"].to_list(), tra["file_name"].to_list()))

    def _load(fname):
        df = load_repertoire(covid_path(fname))
        return (downsample(df, downsample_to, by="reads", seed=seed)
                if downsample_to and int(df["duplicate_count"].sum()) > downsample_to else recompute_frequency(df))

    rows, pairs = [], []
    for r in trb.to_dicts():
        if r["donor_id"] not in tra_file:
            continue
        pairs.append((_load(r["file_name"]), _load(tra_file[r["donor_id"]])))
        rows.append(r)
        if n_donors and len(rows) >= n_donors:
            break
    return rows, pairs


def paired_spaces(pairs, *, n_proto=1000, n_comp=20, n_rff=2048, n_rff_second=256, seed=0):
    """Fit ONE RepertoireSpace per locus (comparability invariant). Returns ``{"TRB": (space, frames),
    "TRA": (space, frames)}`` so callers can build block arrays *and* run per-locus ``class_witness``."""
    from _cohort import pooled_clonotypes
    from mir.embedding.tcremp import TCREmp
    from mir.repertoire import fit_repertoire_space

    out = {}
    for locus, idx in (("TRB", 0), ("TRA", 1)):
        frames = [p[idx] for p in pairs]
        model = TCREmp.from_defaults("human", locus, n_prototypes=n_proto)
        space = fit_repertoire_space(model, pooled_clonotypes([(None, f) for f in frames]),
                                     n_rff=n_rff, n_rff_second=n_rff_second, n_components=n_comp, seed=seed)
        out[locus] = (space, frames)
    return out


def batch_of(rows) -> np.ndarray:
    """Short batch label(s) from ``batch_id`` (e.g. '2020/10_FMBA_NovaSeq5' → 'NovaSeq5')."""
    def one(r):
        return (r["batch_id"] or "").split("_FMBA_")[-1].replace("_DNA", "").replace("Novaseq", "NovaSeq")
    return np.array([one(r) for r in rows])


def carries(row, allele: str) -> bool:
    """True if the donor carries ``allele`` (prefix/substring match, e.g. 'A*02' or 'DRB1*15:01')."""
    return any(allele in (row[c] or "") for c in HLA_COLS.get(allele.split("*")[0], []))


def residualize(X: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Subtract each group's mean vector — removes the first-order batch offset (Prop. ``prop:batch``)."""
    out = X.copy()
    for g in np.unique(group):
        m = group == g
        out[m] -= X[m].mean(axis=0)
    return out
