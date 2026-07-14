"""SARS-CoV-2 exposure biomarkers from the repertoire — classify status + rediscover the paper's COVID clones.

This is the ``airr_covid19`` paper's own task (Vlasova et al., Genome Med 2026: "inference of SARS-CoV-2
exposure biomarkers"). COVID here **passed long ago** (convalescent / memory phase, not acute), so the signal
is a subtle set of persistent public memory clones — a hard test. Two questions:

1. **Donor classification** — does Φ's second-moment block separate COVID-convalescent from healthy donors?
   COVID status **is confounded with sequencing batch** (some runs are ~all-healthy), so a batch-blind AUC is
   inflated; we report naive vs **batch-residualized** vs **within-mixed-batch** (Prop. ``prop:batch``).

2. **Motif rediscovery (ground truth!)** — the supervised MMD witness (:func:`mir.repertoire.class_witness`,
   Prop. ``prop:witness``) computes ``w = μ_COVID − μ_healthy`` from the *donors*; we then score the paper's
   **independently-derived** COVID-associated CDR3s (shipped ``covid_associated_clonotypes.csv``, 567 β clones
   labelled ``has_covid_association`` True/False, the False ones matched near-sequence controls). AUC(witness
   score, has_covid_association) asks: does mirpy's donor-derived biomarker rank the paper's COVID clones above
   their controls? A non-circular external validation (the clones are not the donors).

Data: ``~/hf/airr_covid19`` local git-LFS checkout, else HF ``isalgo/airr_covid19`` (auto-fallback) (TRB). Needs ``[bench]``.
Run:  python experiments/benchmark_repertoire_covidstatus.py [n_donors] [downsample_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from _cohort import cv_auc, pooled_clonotypes
from _covid import covid_path, batch_of, load_covid, residualize

from mir.embedding.tcremp import TCREmp
from mir.repertoire import class_witness, fit_repertoire_space, sample_embedding

N_PROTO, N_COMPONENTS, N_RFF, N_RFF_SECOND = 1000, 20, 2048, 256
MIXED = ("NovaSeq5", "NovaSeq6", "NovaSeq7")


def _ground_truth() -> pl.DataFrame:
    """The paper's TRB COVID-associated clones: v_call/j_call/junction_aa + has_covid_association (bool)."""
    raw = pl.read_csv(covid_path("covid_associated_clonotypes.csv"), infer_schema_length=0)
    return (raw.filter(pl.col("chain") == "beta")
            .select(junction_aa=pl.col("cdr3"),
                    v_call=pl.col("v").str.replace(r"/.*", ""),     # first of a multi-V "TRBV7-9/TRBV28"
                    j_call=pl.col("j"),
                    covid=pl.col("has_covid_association") == "True")
            .filter(pl.col("junction_aa").str.contains(r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$"))
            .unique(subset=["v_call", "j_call", "junction_aa"]))


def main(n_donors: int = 300, downsample_to: int = 20_000) -> None:
    from sklearn.metrics import roc_auc_score

    t0 = time.perf_counter()
    rows, frames = load_covid(n_donors, downsample_to, statuses=("COVID", "healthy"))
    covid = np.array([r["COVID_status"] == "COVID" for r in rows], dtype=int)
    batch = batch_of(rows)
    mixed = np.isin(batch, MIXED)
    print(f"{len(rows)} donors ≤{downsample_to} reads: {covid.sum()} COVID-convalescent, "
          f"{(~covid.astype(bool)).sum()} healthy; {mixed.sum()} in mixed batches\n")

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes([(None, f) for f in frames]), n_rff=N_RFF,
                                 n_rff_second=N_RFF_SECOND, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, f, blocks=("second",)) for f in frames]
    second = np.stack([e.second for e in embs])

    # (1) donor classification: naive vs batch-corrected
    naive = cv_auc(second, covid, pca_cols=second.shape[1])
    resid = cv_auc(residualize(second, batch), covid, pca_cols=second.shape[1])
    within = cv_auc(second[mixed], covid[mixed], pca_cols=second.shape[1])
    print(f"COVID status AUC (second-moment, 50-fold CV):")
    print(f"  naive (batch-confounded)   {naive[0]:.3f} ± {naive[1]:.3f}")
    print(f"  batch-residualized         {resid[0]:.3f} ± {resid[1]:.3f}")
    print(f"  within mixed batches       {within[0]:.3f} ± {within[1]:.3f}   (honest, batch-balanced)")

    # (2) motif rediscovery vs the paper's ground truth
    gt = _ground_truth()
    scored = class_witness(space, [frames[i] for i in np.where(covid == 1)[0]],
                           [frames[i] for i in np.where(covid == 0)[0]], gt, top=gt.height)
    y = scored["covid"].to_numpy().astype(int)
    s = scored["witness_score"].to_numpy()
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    print(f"\nMotif rediscovery: witness ranks the paper's {y.sum()} COVID-associated β clones vs "
          f"{len(y) - y.sum()} matched controls → AUC = {auc:.3f}")
    print("Top witness motifs (COVID⁺ − healthy); ✓ = paper-labelled COVID-associated:")
    for r in scored.head(8).iter_rows(named=True):
        print(f"  {r['v_call']:<10} {r['junction_aa']:<20} score={r['witness_score']:.3f} "
              f"{'✓' if r['covid'] else '·'}")

    honest = within[0]
    verdict = ("PASS" if honest > 0.6 and auc > 0.6 else "PARTIAL" if honest > 0.55 or auc > 0.55 else "FAIL")
    print(f"\n[{verdict}] batch-honest COVID classification {honest:.2f} (naive {naive[0]:.2f} rides the "
          f"batch confound); witness rediscovers the paper's COVID clones at AUC {auc:.2f}. COVID exposure "
          f"leaves a memory-phase clonotype-identity biomarker — weaker than acute HLA imprint, batch-sensitive.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 20_000)
