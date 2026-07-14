"""Yellow-fever vaccination: the sample-level MMD witness recovers the LLW/A*02 response motifs.

Yellow-fever vaccination (Pogorelyy et al. 2018; HF ``isalgo/airr_yfv19``) drives an HLA-A*02-restricted
CD8 response to the NS4b epitope **LLWNGPMAV**. Given each donor's day-0 (baseline) and day-15 (peak
response) repertoire, the supervised MMD witness (:func:`mir.repertoire.class_witness`, Prop.
``prop:witness``) — ``w = μ(day15) − μ(day0)``, clonotype score ``s(σ)=⟨w, ψ(φ(σ))⟩`` — should rank the
LLW-specific clones at the top: a *sample-level* analog of the graph-free density enrichment
(``benchmark_density_yfv.py``), and a ground-truth test of "find the antigen motifs".

Metric: per donor, score every day-15 clonotype by the witness, label it by membership in the LLW/A*02
reference, and compute AUROC (does the differential rank the YF-specific clones high?). Reported vs a
naive day15/day0 fold-change baseline.

Data: HF isalgo/airr_yfv19; LLW reference tests/assets/llwngpmav_trb_a02.tsv.gz. Cached (needs [bench]).
Run:  python experiments/benchmark_repertoire_yfv.py [donors_csv] [downsample_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from _cohort import pooled_clonotypes
from _hf import fetch, load_repertoire

from mir.embedding.tcremp import TCREmp
from mir.repertoire import class_witness, fit_repertoire_space

REPO = "isalgo/airr_yfv19"
DONORS = ("P1", "P2", "Q1", "Q2")
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048


def _llw() -> set:
    return set(pl.read_csv("tests/assets/llwngpmav_trb_a02.tsv.gz", separator="\t")["junction_aa"].to_list())


def _fold_auc(d15: pl.DataFrame, d0: pl.DataFrame, is_llw) -> float:
    """Naive baseline: rank day-15 clonotypes by day15/day0 frequency fold-change."""
    f0 = dict(zip(d0["junction_aa"].to_list(), (d0["duplicate_count"] / d0["duplicate_count"].sum())))
    f15 = d15["duplicate_count"].to_numpy() / d15["duplicate_count"].sum()
    juncs = d15["junction_aa"].to_list()
    fold = np.array([f15[i] / (f0.get(juncs[i], 0.0) + 1e-9) for i in range(len(juncs))])
    return roc_auc_score(is_llw, fold) if is_llw.any() and not is_llw.all() else float("nan")


def main(donors=DONORS, downsample_to: int = 30_000) -> None:
    t0 = time.perf_counter()
    llw = _llw()

    # load day0 + day15 for each donor; downsample to a common depth
    from vdjtools.preprocess import downsample
    samples = {}
    for d in donors:
        try:
            d0 = load_repertoire(fetch(REPO, f"{d}_0_F1.airr.tsv.gz"))
            d15 = load_repertoire(fetch(REPO, f"{d}_15_F1.airr.tsv.gz"))
        except Exception as e:
            print(f"  skip {d}: {e}"); continue
        d0 = downsample(d0, min(downsample_to, int(d0["duplicate_count"].sum())), by="reads", seed=0)
        d15 = downsample(d15, min(downsample_to, int(d15["duplicate_count"].sum())), by="reads", seed=0)
        samples[d] = (d0, d15)

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    pool = pooled_clonotypes([(None, f) for pair in samples.values() for f in pair], per_sample=3000)
    space = fit_repertoire_space(model, pool, n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)

    print(f"{len(samples)} donors, ≤{downsample_to} reads/timepoint, {len(llw)} LLW/A*02 reference CDR3s\n")
    print(f"{'donor':<7}{'LLW@d15':>9}{'witness AUC':>13}{'fold AUC':>10}{'top-LLW@30':>12}")
    wa, fa = [], []
    for d, (d0, d15) in samples.items():
        is_llw = d15["junction_aa"].is_in(llw).to_numpy()
        if is_llw.sum() < 2:
            print(f"{d:<7}{int(is_llw.sum()):>9}{'-- too few LLW --':>35}"); continue
        ranked = class_witness(space, [d15], [d0], d15, top=d15.height)
        y = ranked["junction_aa"].is_in(llw).to_numpy()
        w_auc = roc_auc_score(y, ranked["witness_score"].to_numpy())
        f_auc = _fold_auc(d15, d0, is_llw)
        top_llw = int(ranked.head(30)["junction_aa"].is_in(llw).sum())
        wa.append(w_auc); fa.append(f_auc)
        print(f"{d:<7}{int(is_llw.sum()):>9}{w_auc:>13.3f}{f_auc:>10.3f}{top_llw:>12}")

    mw, mf = float(np.nanmean(wa)), float(np.nanmean(fa))
    verdict = "PASS" if mw > 0.7 else "PARTIAL" if mw > 0.55 else "FAIL"
    print(f"\n[{verdict}] mean witness AUC={mw:.3f} (LLW ranked high, prop:witness) vs fold-change {mf:.3f}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    ds = args[1].split(",") if len(args) > 1 else DONORS
    main(tuple(ds), int(args[2]) if len(args) > 2 else 30_000)
