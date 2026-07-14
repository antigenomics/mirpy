"""Depth-robustness of the repertoire kernel mean Φ₁ (Theory §T.7, Prop. ``prop:kme``) — headline.

The whole point of the sample-level embedding is the low-coverage bulk-RNA-seq regime (~10²–10⁴
clonotypes/chain). Theory says the RFF kernel mean of a depth-``n`` subsample converges to the
full-repertoire kernel mean at rate ``‖Φ₁(sub) − Φ₁(full)‖ = O(n_eff^{-1/2})``, with the effective
size ``n_eff = (Σ w²)⁻¹`` (itself a Hill number). So the log–log slope of that error against
``n_eff`` should be ≈ ``−1/2``, and the error should fall monotonically as depth grows.

We take a handful of donors, downsample each to ``N ∈ {100, 300, 1000, 3000, 10⁴}`` reads, embed
each subsample through the *same* ``RepertoireSpace`` fit on the pooled cloud, and regress
``log‖Φ₁(sub) − Φ₁(full)‖`` on ``log n_eff`` across all (donor, N) points.

Data: HF isalgo/airr_benchmark (aging, full-depth vdjtools). Cached on first run (needs [bench]).
Run:  python experiments/benchmark_repertoire_depth.py [n_donors] [full_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _cohort import load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

REPO, META = "isalgo/airr_benchmark", "vdjtools/metadata_aging.txt"
PREFIX, SUFFIX = "vdjtools/", ".gz"
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048
DEPTHS = (100, 300, 1000, 3000, 10_000)


def main(n_donors: int = 8, full_reads: int = 40_000) -> None:
    from vdjtools.preprocess import downsample

    t0 = time.perf_counter()
    _, samples = load_cohort(REPO, META, prefix=PREFIX, suffix=SUFFIX,
                             downsample_to=full_reads, cap_samples=n_donors)
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)

    print(f"{len(samples)} donors, 'full' ≤{full_reads} reads; Φ₁ convergence vs subsample depth\n")
    pts: dict[int, list[tuple[float, float]]] = {N: [] for N in DEPTHS}   # N -> [(n_eff, err), ...]
    for _, df in samples:
        full = sample_embedding(space, df, blocks=("mean",)).mean
        reads = int(df["duplicate_count"].sum())
        for N in DEPTHS:
            if N >= reads:
                continue
            e = sample_embedding(space, downsample(df, N, by="reads", seed=0), blocks=("mean",))
            pts[N].append((e.n_eff, float(np.linalg.norm(e.mean - full))))

    print(f"{'N':>7}{'mean_n_eff':>12}{'mean_err':>10}{'err·√n_eff':>12}")
    logx, logy = [], []
    for N in DEPTHS:
        if not pts[N]:
            continue
        neffs, errs = np.array([p[0] for p in pts[N]]), np.array([p[1] for p in pts[N]])
        logx.extend(np.log(neffs)); logy.extend(np.log(errs))
        mn, me = neffs.mean(), errs.mean()
        print(f"{N:>7}{mn:>12.1f}{me:>10.4f}{me * np.sqrt(mn):>12.3f}")
    logx, logy = np.array(logx), np.array(logy)

    slope = np.polyfit(logx, logy, 1)[0]                       # d log err / d log n_eff
    order = np.argsort(logx)
    decreasing = logy[order][-1] < logy[order][0]
    verdict = "PASS" if (-0.75 <= slope <= -0.25 and decreasing) else "FAIL"
    print(f"\n[{verdict}] log-log slope = {slope:.3f} (theory −0.5, Prop. prop:kme); "
          f"error decreases with depth = {decreasing}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 8, int(args[2]) if len(args) > 2 else 40_000)
