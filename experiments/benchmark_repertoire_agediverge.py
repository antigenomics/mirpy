"""Aging is directional divergence, not only diversity loss — and only an embedding can see it.

Diversity is a *scalar*: it says a repertoire is "less even", but not *in which direction* it drifts. The
hypothesis (beyond the age↓diversity decline) is that repertoires **diverge in different directions with age** —
older individuals are more *dissimilar to each other*, their private antigen/HLA histories pushing each Φ(S) its
own way. That is a property of the sample-level embedding (a point in ℝ^D, compared by MMD) that a diversity
number structurally cannot represent.

⚠ Confound: older repertoires are less diverse ⇒ lower ``n_eff`` ⇒ noisier Φ₁ ⇒ larger pairwise MMD *by
estimation variance alone* (Prop. ``prop:kme``, ``err ∝ n_eff^{-1/2}``). So the honest test is a **partial**
correlation: does divergence rise with age *after conditioning on diversity/n_eff*? All donors are first
downsampled to a common depth so the n_eff floor is comparable.

Metrics (rank-based): Spearman(age, ¹D) [expected < 0]; Spearman(age, divergence); and the key
**partial Spearman(age, divergence | ¹D)** with a permutation p-value — the directional signal the embedding
adds over the scalar. Divergence = each donor's mean MMD to all others (and distance to the young centroid).

Data: HF isalgo/airr_benchmark aging (79 donors). Cached (needs [bench]).
Run:  python experiments/benchmark_repertoire_agediverge.py [common_depth] [young_age]
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.stats import spearmanr

from _cohort import load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, mmd_matrix, sample_embedding

REPO, META = "isalgo/airr_benchmark", "vdjtools/metadata_aging.txt"
PREFIX, SUFFIX = "vdjtools/", ".gz"
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048


def _rank(a):
    return a.argsort().argsort().astype(float)


def _partial_spearman(x, y, z):
    """Partial Spearman corr(x, y | z): correlation of rank-residuals after removing rank(z)."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    rz1 = np.c_[np.ones_like(rz), rz]
    ex = rx - rz1 @ np.linalg.lstsq(rz1, rx, rcond=None)[0]
    ey = ry - rz1 @ np.linalg.lstsq(rz1, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main(common_depth: int = 15_000, young_age: int = 30) -> None:
    from vdjtools.preprocess import downsample

    t0 = time.perf_counter()
    _, samples = load_cohort(REPO, META, prefix=PREFIX, suffix=SUFFIX, downsample_to=common_depth)
    # keep only donors that actually reach the common depth, so n_eff floor is comparable
    samples = [(r, downsample(df, common_depth, by="reads", seed=0))
               for r, df in samples if int(df["duplicate_count"].sum()) >= common_depth]
    ages = np.array([float(r["age"]) for r, _ in samples])

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("mean", "diversity")) for _, df in samples]
    d1 = np.array([np.exp(e.diversity[1]) for e in embs])          # ¹D (Shannon)
    n_eff = np.array([e.n_eff for e in embs])

    D = mmd_matrix(embs)                                            # pairwise repertoire MMD
    np.fill_diagonal(D, np.nan)
    divergence = np.nanmean(D, axis=1)                             # mean dissimilarity to all others
    young = ages < young_age
    if young.sum() >= 3:
        cy = np.mean([embs[i].mean for i in np.where(young)[0]], axis=0)
        dist_young = np.array([np.linalg.norm(e.mean - cy) for e in embs])
    else:
        dist_young = divergence

    print(f"{len(samples)} donors at common depth {common_depth}; ages {ages.min():.0f}–{ages.max():.0f}, "
          f"{young.sum()} young (<{young_age})\n")
    r_div = spearmanr(ages, d1).correlation
    r_dvg = spearmanr(ages, divergence).correlation
    r_dy = spearmanr(ages, dist_young).correlation
    pr = _partial_spearman(ages, divergence, d1)                   # age vs divergence | diversity
    pr_ne = _partial_spearman(ages, divergence, n_eff)             # | n_eff (estimation-noise control)

    # permutation p-value for the partial correlation (shuffle age)
    rng = np.random.default_rng(0)
    null = np.array([_partial_spearman(rng.permutation(ages), divergence, d1) for _ in range(2000)])
    p = float((np.abs(null) >= abs(pr)).mean())

    print(f"{'Spearman(age, ¹D diversity)':<38}{r_div:>8.3f}   (expected < 0: less diverse with age)")
    print(f"{'Spearman(age, divergence MMD)':<38}{r_dvg:>8.3f}   (hypothesis > 0: more dissimilar)")
    print(f"{'Spearman(age, dist to young centroid)':<38}{r_dy:>8.3f}")
    print(f"{'partial Spearman(age, divergence | ¹D)':<38}{pr:>8.3f}   (adds BEYOND diversity; p={p:.3f})")
    print(f"{'partial Spearman(age, divergence | n_eff)':<38}{pr_ne:>8.3f}   (not just estimation noise)")

    verdict = "PASS" if pr > 0 and p < 0.05 else "PARTIAL" if pr > 0 else "FAIL"
    print(f"\n[{verdict}] repertoires diverge directionally with age beyond the diversity decline: "
          f"partial ρ(age, divergence | diversity)={pr:.3f}, p={p:.3f} — a directional signal the scalar "
          f"diversity cannot encode (the embedding's unique contribution).")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 15_000, int(args[2]) if len(args) > 2 else 30)
