"""T5 — IGH / somatic hypermutation: embedding drift + reconstruction levers.

1. **SHM drift** (Theory T5): embedding distance vs mutation load is ~linear/sublinear
   (drift *bounded* by mutation load); IGH's longer CDR3 gives the lowest per-mutation drift.
2. **Reconstruction lever**: IGH inverse exact-match vs the variance retained by the compact
   code — isolates whether the compaction (not the frame; only 0.1% of IGH CDR3s exceed 40)
   limits reconstruction of the longer, more diverse IGH junctions.

Run (needs [ml] + [bench]):
    python experiments/benchmark_igh_shm.py [N] [epochs]
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mir.bench.theory import shm_embedding_drift
from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import load_prototypes
from mir.ml.train import train_inverse_decoder


def main(n: int = 8000, epochs: int = 60, k_proto: int = 1000) -> None:
    print("=== SHM embedding drift  D_k = ||emb(k-mutated) - emb(orig)||  (T5) ===")
    for chain in ("IGH", "TRB"):
        protos = load_prototypes("human", chain, n=k_proto)["junction_aa"].to_list()
        seqs = load_prototypes("human", chain, n=2000)["junction_aa"].to_list()[1000:1500]
        d = shm_embedding_drift(seqs, protos, max_mut=8, n_rep=2)
        ks = sorted(d)
        slope = float(np.polyfit(ks, [d[k][0] for k in ks], 1)[0])
        print(f"  {chain}: " + " ".join(f"{k}={d[k][0]:.0f}" for k in ks) + f"  ({slope:.0f}/mut)")

    print("\n=== IGH inverse reconstruction vs variance retained ===")
    protos = load_prototypes("human", "IGH", n=k_proto)["junction_aa"].to_list()
    igh = load_prototypes("human", "IGH", n=10000)["junction_aa"].to_list()[k_proto:k_proto + n]
    junc = junction_distance_matrix(igh, protos)
    print(f"{'variance':>9}{'PCs':>6}{'exact':>8}{'tok_acc':>9}")
    for var in (0.95, 0.99, 0.999):
        codes = PCA(n_components=var, whiten=True, random_state=0).fit_transform(junc)
        _, m = train_inverse_decoder(codes, igh, epochs=epochs, verbose=False)
        print(f"{var:>9}{codes.shape[1]:>6}{m['exact_match']:>8.3f}{m['token_acc']:>9.3f}")
    codes_full = StandardScaler().fit_transform(junc)
    _, mf = train_inverse_decoder(codes_full, igh, epochs=epochs, verbose=False)
    print(f"{'full':>9}{codes_full.shape[1]:>6}{mf['exact_match']:>8.3f}{mf['token_acc']:>9.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 8000, int(a[1]) if len(a) > 1 else 60)
