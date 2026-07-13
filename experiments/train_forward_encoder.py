"""Train the forward codec: CDR3 -> TCREMP junction-distance embedding (Part 2 demo).

The junction (CDR3) component is the expensive part of the embedding (the gapblock
computation); V/J components are cheap germline lookups. This learns seq -> the K-dim
junction-distance-to-prototypes vector with free supervision (targets computed by
seqtree.gapblock), giving a GPU-speed approximation. Reports test mean cosine
(irrm-codec forward metric, ~0.89 target).

Run (needs [ml] + [bench]):
    python experiments/train_forward_encoder.py [N] [K] [epochs]
"""

from __future__ import annotations

import sys
import time

from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import load_prototypes
from mir.ml.train import train_forward_encoder


def main(n: int = 8000, k: int = 1000, epochs: int = 30) -> None:
    from vdjtools.model import generate, load_bundled

    protos = load_prototypes("human", "TRB", n=k)["junction_aa"].to_list()
    gen = generate.generate(load_bundled("TRB", source="olga"), 5 * n, seed=1,
                            productive_only=True)
    seqs = [s for s in gen["junction_aa"].unique(maintain_order=True).to_list()
            if 6 <= len(s) <= 40][:n]

    t0 = time.perf_counter()
    targets = junction_distance_matrix(seqs, protos)  # (N, K) free supervision
    print(f"targets {targets.shape} in {time.perf_counter() - t0:.1f}s (gapblock)")

    enc, metrics = train_forward_encoder(seqs, targets, epochs=epochs)
    print(f"\nRESULT: test mean cosine = {metrics['test_cosine']:.4f}  "
          f"(n={metrics['n']}, K={k} -> {metrics['n_components']} PCs @95% var, epochs={epochs})")

    # timing: DNN vs gapblock for the same batch
    import numpy as np
    probe = seqs[:2000]
    t0 = time.perf_counter(); enc.encode(probe); dt_dnn = time.perf_counter() - t0
    t0 = time.perf_counter(); junction_distance_matrix(probe, protos); dt_gb = time.perf_counter() - t0
    print(f"encode 2000x{k}: DNN {dt_dnn*1e3:.0f}ms  vs  gapblock {dt_gb*1e3:.0f}ms")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 8000,
         int(a[1]) if len(a) > 1 else 1000,
         int(a[2]) if len(a) > 2 else 30)
