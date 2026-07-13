# 2026-07-14
# How deep must the compact code be to recover the receptor exactly?
#
# Splits the "lossless recovery" question into its three real parts:
#   * V / J  — exact germline lookups carried as metadata (v_call/j_call). Recovered
#     100% by construction if the columns are kept; nothing to reconstruct. (Not swept.)
#   * junction (CDR3) — the ONLY lossy part: reconstructed by the inverse codec from the
#     PCA-compacted junction distance code. This is what we sweep here.
#   * C gene — absent from the embedding entirely; a low-cardinality categorical (isotype)
#     independent of V/J/CDR3, so it must be STORED exactly, never embedded. (Not swept.)
#
# For each chain we report exact-match and token-accuracy of the junction as a function of
# retained variance (=> #PCs), plus the full-precision ceiling (all K dims, no truncation)
# which isolates decoder/injectivity limits from PCA-truncation limits. The "code bits"
# column (n_PCs x 32) vs "seq bits" (len x log2(20)) makes the key point quantitative: the
# distance-to-prototypes map is an *expansion*, not a compression — for archival losslessness
# store the sequence; the codec inverse is inherently lossy and earns its keep for ML/generation.
#
# Run (needs [ml] + [bench]):  python experiments/benchmark_lossless_depth.py [N] [K] [EPOCHS]

from __future__ import annotations

import sys

import numpy as np
from sklearn.decomposition import PCA

from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import load_prototypes
from mir.ml.train import train_inverse_decoder

CHAINS = [("human", "TRB"), ("human", "IGH")]
VAR_TARGETS = [0.95, 0.99, 0.999]


def _seq_bits(seqs) -> float:
    """Mean raw sequence content in bits (len * log2(20))."""
    return float(np.mean([len(s) for s in seqs]) * np.log2(20))


def sweep_chain(species: str, locus: str, n: int, k: int, epochs: int) -> list[tuple]:
    pool = load_prototypes(species, locus, n=min(10000, k + n + 500))
    protos = pool["junction_aa"].to_list()[:k]
    seqs = pool[k:k + n]["junction_aa"].to_list()
    junc = junction_distance_matrix(seqs, protos)  # (n, k) squared-distance code
    sbits = _seq_bits(seqs)

    rows = []
    for var in VAR_TARGETS:
        codes = PCA(n_components=var, whiten=True, random_state=0).fit_transform(junc)
        _, m = train_inverse_decoder(codes, seqs, epochs=epochs, verbose=False)
        rows.append((f"{var:.1%}", codes.shape[1], m["exact_match"], m["token_acc"],
                     codes.shape[1] * 32, sbits))
    # full-precision ceiling: raw K-dim junction distances (StandardScaler inside the trainer)
    _, m = train_inverse_decoder(junc, seqs, epochs=epochs, verbose=False)
    rows.append((f"full(K={k})", k, m["exact_match"], m["token_acc"], k * 32, sbits))
    return rows


def main(n: int = 6000, k: int = 1000, epochs: int = 40) -> None:
    print(f"lossless-depth sweep  n={n} corpus  k={k} prototypes  epochs={epochs}\n"
          f"(NB: PCA fit on the full corpus — mild optimism, consistent across depths)\n")
    for species, locus in CHAINS:
        rows = sweep_chain(species, locus, n, k, epochs)
        print(f"== {species} {locus} ==")
        print(f"{'depth':<11}{'PCs':>5}{'exact%':>9}{'token%':>9}{'code_bits':>11}{'seq_bits':>10}")
        for depth, pcs, ex, tok, cb, sb in rows:
            print(f"{depth:<11}{pcs:>5}{100 * ex:>9.2f}{100 * tok:>9.2f}{cb:>11}{sb:>10.1f}")
        print()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 6000,
         int(a[1]) if len(a) > 1 else 1000,
         int(a[2]) if len(a) > 2 else 40)
