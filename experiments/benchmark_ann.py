# 2026-07-14
# Is fast (approximate) nearest-neighbour indexing worth it for the density hot path? Two questions:
#   (1) end-to-end: neighbor_enrichment backend='exact' (BallTree) vs 'ann' (pynndescent) — wall-time
#       and hit-set agreement (Jaccard) as N grows. When does ANN pay off, and does it agree?
#   (2) raw neighbour queries: BallTree vs scipy cKDTree vs pynndescent (kNN + radius count) at the
#       density D (~20-65). cKDTree is exact + multithreaded and needs no new dep — is it the better
#       *exact* backend than BallTree, before reaching for approximate ANN?
#
# The balloon estimator uses per-point variable-radius counts; the ANN path reformulates them as a
# kNN graph thresholded by radius (recall < 1 undercounts -> conservative). See mir.density._ann_neighbors.
#
# Run (needs [bench]):  python experiments/benchmark_ann.py

from __future__ import annotations

import resource
import time
import warnings

import numpy as np
from sklearn.neighbors import BallTree

from mir.density import enriched_mask, neighbor_enrichment


def _rss_gb() -> float:
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if m > 1e7 else m / 1e6  # macOS bytes / Linux KiB


def _data(n_obs: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    bg = rng.standard_normal((5 * n_obs, d))
    obs = np.vstack([rng.standard_normal((n_obs - n_obs // 10, d)),
                     rng.normal(4.0, 0.05, (n_obs // 10, d))])  # 10% injected convergent signal
    return obs, bg


def bench_e2e() -> None:
    print("== end-to-end neighbor_enrichment: exact (BallTree) vs ann (pynndescent), D=20 ==")
    print(f"{'N_obs':>8}{'N_bg':>9}{'exact_s':>9}{'ann_s':>8}{'speedup':>8}{'Jaccard':>9}{'peakGB':>8}",
          flush=True)
    for n in (10_000, 40_000, 100_000):
        obs, bg = _data(n, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.perf_counter(); rx = neighbor_enrichment(obs, bg, backend="exact"); te = time.perf_counter() - t
            t = time.perf_counter(); ra = neighbor_enrichment(obs, bg, backend="ann"); ta = time.perf_counter() - t
        mx, ma = enriched_mask(rx), enriched_mask(ra)
        jac = (mx & ma).sum() / max((mx | ma).sum(), 1)
        print(f"{n:>8}{5 * n:>9}{te:>9.1f}{ta:>8.1f}{te / max(ta, 1e-9):>7.1f}x{jac:>9.3f}{_rss_gb():>8.2f}",
              flush=True)
    print(flush=True)


def bench_raw() -> None:
    from pynndescent import NNDescent
    from scipy.spatial import cKDTree

    print("== raw kNN (k=25) build+query, N=100000: exact backends vs ANN ==")
    print(f"{'D':>4}{'BallTree_s':>12}{'cKDTree_s':>11}{'pynndesc_s':>12}{'ann_recall@25':>15}")
    n = 100_000
    for d in (20, 65):
        X = np.random.default_rng(0).standard_normal((n, d))
        q = X[:5000]
        t = time.perf_counter(); bt = BallTree(X); di_bt = bt.query(q, k=25)[1]; t_bt = time.perf_counter() - t
        t = time.perf_counter(); kd = cKDTree(X); di_kd = kd.query(q, k=25, workers=-1)[1]; t_kd = time.perf_counter() - t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.perf_counter()
            nn = NNDescent(X, n_neighbors=30, random_state=0); di_nn = nn.query(q, k=25)[0]
            t_nn = time.perf_counter() - t
        # recall of the approximate kNN against the exact BallTree neighbours
        recall = np.mean([len(set(a) & set(b)) / 25 for a, b in zip(di_nn, di_bt)])
        print(f"{d:>4}{t_bt:>12.1f}{t_kd:>11.1f}{t_nn:>12.1f}{recall:>15.3f}", flush=True)
    print(flush=True)


def main() -> None:
    print(f"ANN indexing benchmark  (peak RSS budget 20 GB on M3)\n")
    bench_e2e()
    bench_raw()
    print(f"peak RSS {_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
