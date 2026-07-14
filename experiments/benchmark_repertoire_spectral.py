"""Spectral compaction of the second-moment (interaction) block — how small can D₂ / top-r get?

The interaction block ``Σ_σ w_σ ψ₂ψ₂ᵀ`` is stored as its full ``D₂(D₂+1)/2`` upper triangle
(32 896 dims at D₂=256). Most of that is noise: the block is a weighted covariance, so its
information lives in a handful of leading **eigenvalues** (a rotation-invariant energy spectrum).
This sweep asks whether the opt-in ``n_eigs`` top-r spectrum retains the block's known signal —
HLA-A*02 carriage (THEORY §T7: second-moment ≈ 0.62, diversity ≈ chance) — at a fraction of the
dimension, and where the knee is.

Grid: D₂ ∈ {128, 256, 512} × r ∈ {16, 32, 64, 128, full-upper-tri}. Reports AUC±std (50-fold CV),
block dimension, and per-D₂ embed time. Pick the smallest (D₂, r) whose interval matches full.

Data: ``~/hf/airr_covid19`` local git-LFS checkout, else HF ``isalgo/airr_covid19`` (TRB). Needs ``[bench]``.
Run:  python experiments/benchmark_repertoire_spectral.py [n_donors] [downsample_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _cohort import cv_auc, pooled_clonotypes
from _covid import carries, load_covid

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048
D2_GRID = (128, 256, 512)
R_GRID = (16, 32, 64, 128)
ALLELE = "A*02"                                  # broad, well-powered class-I label


def _sigma_features(space, frames):
    """Per-sample (top-D₂ descending eigenvalues, full upper-triangle) of Σ_σ w_σ ψ₂ψ₂ᵀ."""
    evs, uppertri = [], []
    iu = np.triu_indices(space.rff2.dim)
    for f in frames:
        Z, w = space.sample_cloud(f)                       # shared PCA coords + log1p freq weights
        psi2 = space.rff2.transform(Z)
        sigma = (psi2 * w[:, None]).T @ psi2               # (D₂, D₂), symmetric PSD
        evs.append(np.linalg.eigvalsh(sigma)[::-1])        # descending eigenvalues
        uppertri.append(sigma[iu])
    return np.stack(evs), np.stack(uppertri)


def main(n_donors: int = 300, downsample_to: int = 20_000) -> None:
    t0 = time.perf_counter()
    rows, frames = load_covid(n_donors, downsample_to, statuses=("COVID", "healthy", "precovid"))
    y = np.array([carries(r, ALLELE) for r in rows], dtype=int)
    print(f"{len(rows)} donors ≤{downsample_to} reads; HLA-{ALLELE} carriers {y.sum()}/{len(y)}\n")

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    pooled = pooled_clonotypes([(None, f) for f in frames])

    # diversity baseline (should be ~chance)
    space0 = fit_repertoire_space(model, pooled, n_rff=N_RFF, n_rff_second=0, n_components=N_COMPONENTS, seed=0)
    div = np.stack([sample_embedding(space0, f, blocks=("diversity",)).diversity for f in frames])
    dm, ds = cv_auc(div, y)
    print(f"{'representation':<22}{'block dim':>11}{'AUC':>14}{'time':>8}")
    print(f"{'diversity (4)':<22}{4:>11}{dm:>9.3f}±{ds:.3f}{'':>8}")

    best = None
    for d2 in D2_GRID:
        t = time.perf_counter()
        space = fit_repertoire_space(model, pooled, n_rff=N_RFF, n_rff_second=d2,
                                     n_components=N_COMPONENTS, seed=0)
        evs, UT = _sigma_features(space, frames)
        dt = time.perf_counter() - t
        for r in R_GRID:
            if r > d2:
                continue
            am, as_ = cv_auc(evs[:, :r], y, pca_cols=r)
            print(f"{'D₂=%d top-%d' % (d2, r):<22}{r:>11}{am:>9.3f}±{as_:.3f}{'':>8}")
            if best is None or am > best[0]:
                best = (am, as_, f"D₂={d2} top-{r}", r)
        um, us = cv_auc(UT, y, pca_cols=UT.shape[1])
        print(f"{'D₂=%d full-tri' % d2:<22}{UT.shape[1]:>11}{um:>9.3f}±{us:.3f}{dt:>7.0f}s")
        if best is None or um > best[0]:
            best = (um, us, f"D₂={d2} full-tri", UT.shape[1])

    print(f"\nbest: {best[2]} (dim {best[3]}) AUC={best[0]:.3f}±{best[1]:.3f} "
          f"vs diversity {dm:.3f} — the top-r spectrum matches the full triangle at a fraction of the dim "
          f"iff a small-r row's interval reaches the full-tri interval.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 20_000)
