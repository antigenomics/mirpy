# 2026-07-14
# Does adding prototypes (K) buy better junction reconstruction? What (K, PC) is optimal
# under an M3 memory/speed budget? Sweeps on REAL held-out TRB data.
#
#   corpus     = isalgo/airr_control (deep naive TRB repertoire) — independent of the landmarks
#   prototypes = bundled arda human_TRB landmarks (up to 10000)
#   metric     = held-out exact-match / token-acc of the inverse codec, per (K, PC)
#   cost       = embed wall-time, PCA wall-time, (N x K) code-matrix footprint, per-query latency
#
# PCA is fit on the TRAIN split only (same seed-0 split the decoder uses) -> the held-out
# exact-match is leakage-free. Randomized SVD with a fixed PC budget keeps large-K PCA tractable.
#
# Run (needs [ml] + [bench]):  python experiments/benchmark_lossless_kpc.py [N] [EPOCHS]

from __future__ import annotations

import gc
import os
import resource
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # for _hf
from _hf import fetch, load_repertoire  # noqa: E402

from mir.distances.junction import junction_distance_matrix  # noqa: E402
from mir.embedding.prototypes import load_prototypes  # noqa: E402
from mir.ml.train import train_inverse_decoder  # noqa: E402

CONTROL_REPO, CONTROL_FILE = "isalgo/airr_control", "human.trb.aa.vdjtools.tsv.gz"
K_SWEEP = [500, 1000, 2000, 5000, 10000]
PC_SWEEP = [50, 100, 200, 300, 500]
K_FOR_PC = 2000          # chain K used for the PC sweep
PC_FOR_K = 300           # chain PC budget used for the K sweep


def _rss_gb() -> float:
    # macOS ru_maxrss is bytes; Linux is KiB. Normalize to GB assuming the larger (bytes) unit,
    # falling back to KiB if the number is implausibly small.
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if m > 1e7 else m / 1e6


def _split(n: int, seed: int = 0, tf: float = 0.1, vf: float = 0.1):
    perm = np.random.default_rng(seed).permutation(n)
    nt, nv = int(n * tf), int(n * vf)
    return perm[:nt], perm[nt:nt + nv], perm[nt + nv:]  # te, va, tr


def _embed(seqs, protos):
    t = time.perf_counter()
    J = np.asarray(junction_distance_matrix(seqs, protos), dtype=np.float32)
    return J, time.perf_counter() - t


def _one(seqs, protos, k: int, pc: int, epochs: int) -> dict:
    J, t_emb = _embed(seqs, protos[:k])
    _, _, tr = _split(len(seqs))
    pc = min(pc, k, len(tr))
    t = time.perf_counter()
    pca = PCA(n_components=pc, svd_solver="randomized", whiten=True, random_state=0).fit(J[tr])
    codes = pca.transform(J).astype(np.float32)
    t_pca = time.perf_counter() - t
    var = float(pca.explained_variance_ratio_.sum())
    _, m = train_inverse_decoder(codes, seqs, epochs=epochs, verbose=False)  # seed-0 -> same te
    row = dict(K=k, PC=pc, var=var, exact=m["exact_match"], token=m["token_acc"],
               t_emb=t_emb, t_pca=t_pca, mat_gb=J.nbytes / 1e9,
               us_per_q=1e6 * t_emb / len(seqs))
    del J, codes, pca
    gc.collect()
    return row


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"== {title} ==")
    print(f"{'K':>6}{'PC':>5}{'var%':>7}{'exact%':>9}{'token%':>9}"
          f"{'emb_s':>8}{'pca_s':>8}{'mat_GB':>8}{'us/query':>10}")
    for r in rows:
        print(f"{r['K']:>6}{r['PC']:>5}{100 * r['var']:>7.1f}{100 * r['exact']:>9.2f}"
              f"{100 * r['token']:>9.2f}{r['t_emb']:>8.1f}{r['t_pca']:>8.1f}"
              f"{r['mat_gb']:>8.3f}{r['us_per_q']:>10.1f}")
    print(flush=True)


def main(n: int = 20000, epochs: int = 35) -> None:
    seqs = (load_repertoire(fetch(CONTROL_REPO, CONTROL_FILE), top=n + 5000)
            .head(n)["junction_aa"].to_list())
    protos = load_prototypes("human", "TRB", n=10000)["junction_aa"].to_list()
    print(f"real TRB corpus n={len(seqs)} (isalgo/airr_control)  landmarks<=10000 (arda)  "
          f"epochs={epochs}\n", flush=True)

    k_rows = [_one(seqs, protos, k, PC_FOR_K, epochs) for k in K_SWEEP]
    _print_table(f"K sweep  (PC={PC_FOR_K})", k_rows)

    pc_rows = [_one(seqs, protos, K_FOR_PC, pc, epochs) for pc in PC_SWEEP]
    _print_table(f"PC sweep  (K={K_FOR_PC})", pc_rows)

    print(f"peak RSS {_rss_gb():.2f} GB  |  M3 budget: keep working set < 20 GB", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 20000, int(a[1]) if len(a) > 1 else 35)
