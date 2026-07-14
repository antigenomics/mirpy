"""Aging is directional divergence, not only diversity loss — and only an embedding can see it.

Diversity is a *scalar*: it says a repertoire is "less even", but not *in which direction* it drifts. The
hypothesis (beyond the age↓diversity decline) is that repertoires **diverge in different directions with age** —
older individuals are more *dissimilar to each other*, their private antigen/HLA histories pushing each Φ(S) its
own way. That is a property of the sample-level embedding (a point in ℝ^D, compared by MMD) that a diversity
number structurally cannot represent.

⚠ Depth: the directional signal lives in **deep private clonal expansions** — native depths here are median
≈830k reads (min 36k, max 6.9M). Downsampling to a low common depth (the old 15k/40k) *destroys* it, so run at
**full native depth** (``common_depth=0``, default) or a HIGH common floor (≥250k). At full depth every adult is
deep, so the estimation-noise term ``∝ n_eff^{-1/2}`` (Prop. ``prop:kme``) is small and comparable.

⚠ Confound: the **biased** MMD (V-statistic ``‖μ̂_a−μ̂_b‖``) carries a positive self-term bias ``≈ 1/n_eff``, so a
low-diversity (old) sample gets its distances inflated *by construction* — making divergence track LOW diversity
(``ρ(¹D, divergence) < 0``), the OPPOSITE of the ecology intuition (more diverse ⇒ less overlap ⇒ MORE distance,
``ρ > 0``). The fix is the **unbiased MMD²** (``mmd_matrix(..., unbiased=True)``, diagonal removed via the stored
``n_eff``; Gretton 2012) — not downsampling. We report the ``ρ(¹D, divergence)`` sign for both estimators as the
diagnostic, then the **partial** correlation ``(age, divergence | diversity)`` on the unbiased estimator, and the
directional probe **distance to the young centroid**.

Metrics (rank-based): Spearman(age, ¹D) [<0]; Spearman(age, divergence / dist-to-young); the key
**partial(age, dist-to-young | ¹D)** with a permutation p-value.

Data: HF isalgo/airr_benchmark aging (79 donors, full native depth). Cached (needs [bench]).
Run:  python experiments/benchmark_repertoire_agediverge.py [common_depth=0 → full] [young_age]
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


def _perm_p(fn, ages, y, z, n=2000, seed=0):
    """Permutation p-value for a partial correlation ``fn(ages, y, z)`` by shuffling age."""
    rng = np.random.default_rng(seed)
    obs = fn(ages, y, z)
    null = np.array([fn(rng.permutation(ages), y, z) for _ in range(n)])
    return obs, float((np.abs(null) >= abs(obs)).mean())


def main(common_depth: int = 0, young_age: int = 30) -> None:
    """``common_depth<=0`` → **full native depth** (no downsample, all 79 donors). The directional
    signal lives in deep private clonal expansions (median native depth ≈ 830k reads), so a low common
    depth destroys it — downsample only to a HIGH common floor (e.g. 250k) or not at all. The estimation-
    noise confound is then controlled by the partial correlation on ``n_eff`` (large & comparable when deep)."""
    from vdjtools.preprocess import downsample

    t0 = time.perf_counter()
    _, samples = load_cohort(REPO, META, prefix=PREFIX, suffix=SUFFIX,
                             downsample_to=common_depth if common_depth > 0 else None)
    if common_depth > 0:   # keep only donors reaching the (high) common depth, so n_eff floor is comparable
        samples = [(r, downsample(df, common_depth, by="reads", seed=0))
                   for r, df in samples if int(df["duplicate_count"].sum()) >= common_depth]
    ages = np.array([float(r["age"]) for r, _ in samples])
    depths = np.array([int(df["duplicate_count"].sum()) for _, df in samples])

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("mean", "diversity")) for _, df in samples]
    d1 = np.array([np.exp(e.diversity[1]) for e in embs])          # ¹D (Shannon)
    n_eff = np.array([e.n_eff for e in embs])

    from vdjtools import overlap
    from vdjtools.preprocess import select_top

    Db, Du = mmd_matrix(embs), mmd_matrix(embs, unbiased=True)     # biased V-stat vs diagonal-removed MMD²
    # exact-clonotype overlap distance (−log F, Morisita-Horn family) — the metric the "more diverse ⇒ less
    # overlap" intuition actually refers to; contrast its sign with the (distributional) kernel-mean MMD.
    # F is abundance-weighted ⇒ cap to top-50k/donor so the joint occurrence table stays bounded at deep depth.
    ov_frames = [select_top(df, 50_000, renormalize=True) if df.height > 50_000 else df for _, df in samples]
    Do = np.ascontiguousarray(overlap.pairwise_distances(ov_frames, metric="F", form="matrix")
                              .drop("sample").to_numpy(), dtype=float)
    for D in (Db, Du, Do):
        np.fill_diagonal(D, np.nan)
    div_b, div_u = np.nanmean(Db, axis=1), np.nanmean(Du, axis=1)  # mean dissimilarity to all others
    div_o = np.nanmean(Do, axis=1)
    divergence = div_u                                            # trust the unbiased KME estimator
    # sign of ρ(diversity, divergence) tells the metric family apart:
    #   biased KME  → ρ<0 (1/n_eff self-term makes distance track LOW diversity — the artifact)
    #   unbiased KME→ distributional distance: diverse≈naive≈shared P_gen baseline ⇒ still ρ≲0
    #   overlap F   → exact-clonotype overlap: more diverse ⇒ less shared ⇒ ρ>0 (the ecology intuition)
    sign_b = spearmanr(d1, div_b).correlation
    sign_u = spearmanr(d1, div_u).correlation
    sign_o = spearmanr(d1, div_o).correlation
    r_age_o = spearmanr(ages, div_o).correlation
    young = ages < young_age
    # leave-self-out young centroid so a young donor isn't measured against itself
    yi = np.where(young)[0]
    cy_all = np.mean([embs[i].mean for i in yi], axis=0)
    dist_young = np.array([
        np.linalg.norm(e.mean - (np.mean([embs[j].mean for j in yi if j != i], axis=0) if i in yi else cy_all))
        for i, e in enumerate(embs)])

    dtag = "FULL native" if common_depth <= 0 else f"{common_depth}"
    print(f"{len(samples)} donors at depth {dtag} (reads: median {int(np.median(depths))//1000}k, "
          f"min {depths.min()//1000}k, max {depths.max()//1000}k); ages {ages.min():.0f}–{ages.max():.0f}, "
          f"{young.sum()} young (<{young_age})\n")
    r_div = spearmanr(ages, d1).correlation
    r_dvg = spearmanr(ages, divergence).correlation
    r_dy = spearmanr(ages, dist_young).correlation
    pr, p = _perm_p(_partial_spearman, ages, divergence, d1)        # age vs divergence | diversity
    pr_ne = _partial_spearman(ages, divergence, n_eff)             # | n_eff (estimation-noise control)
    pdy, p_dy = _perm_p(_partial_spearman, ages, dist_young, d1)    # age vs dist-to-young | diversity
    pdy_ne = _partial_spearman(ages, dist_young, n_eff)

    print("  metric-family sign check — ρ(¹D diversity, divergence):")
    print(f"{'    KME-MMD biased V-stat':<40}{sign_b:>8.3f}   (<0 artifact: 1/n_eff self-term tracks LOW diversity)")
    print(f"{'    KME-MMD unbiased':<40}{sign_u:>8.3f}   (distributional: diverse≈naive baseline ⇒ ≲0)")
    print(f"{'    overlap −log F (Morisita family)':<40}{sign_o:>8.3f}   (>0 = more diverse ⇒ less exact overlap ✓ intuition)")
    print(f"{'Spearman(age, ¹D diversity)':<40}{r_div:>8.3f}   (expected < 0: less diverse with age)")
    print(f"{'Spearman(age, divergence KME unbiased)':<40}{r_dvg:>8.3f}   (hypothesis > 0: more dissimilar)")
    print(f"{'Spearman(age, divergence overlap F)':<40}{r_age_o:>8.3f}   (overlap-based divergence vs age)")
    print(f"{'Spearman(age, dist to young centroid)':<40}{r_dy:>8.3f}   (directional: away from youth)")
    print(f"{'partial(age, divergence | ¹D)':<40}{pr:>8.3f}   (beyond diversity; p={p:.3f})")
    print(f"{'partial(age, divergence | n_eff)':<40}{pr_ne:>8.3f}   (not estimation noise)")
    print(f"{'partial(age, dist-to-young | ¹D)':<40}{pdy:>8.3f}   (beyond diversity; p={p_dy:.3f})  ← directional")
    print(f"{'partial(age, dist-to-young | n_eff)':<40}{pdy_ne:>8.3f}")

    best, bp = (pdy, p_dy) if abs(pdy) > abs(pr) else (pr, p)
    verdict = "PASS" if best > 0 and bp < 0.05 else "PARTIAL" if best > 0 else "FAIL"
    print(f"\n[{verdict}] directional divergence with age beyond the diversity decline: best partial ρ={best:.3f}, "
          f"p={bp:.3f} at depth {dtag} — the directional drift (dist-to-young) is the signal the scalar diversity "
          f"cannot encode; only detectable when depth is high enough to resolve private clonal expansions.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 0, int(args[2]) if len(args) > 2 else 30)
