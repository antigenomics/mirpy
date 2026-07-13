"""Train the inverse codec: compact junction embedding -> CDR3 sequence (Part 2).

Reconstructs the CDR3 from the 95%-variance PCA-compacted junction embedding —
the hard half of the codec (irrm-codec exact-match ~0.50 on TRB, ~0.16 on IGH).
Reports test exact-match + token accuracy.

Run (needs [ml] + [bench]):
    python experiments/train_inverse_decoder.py [N] [K] [epochs]
"""

from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.decomposition import PCA

from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import load_prototypes
from mir.ml.train import train_inverse_decoder


def main(n: int = 20000, k: int = 1000, epochs: int = 60) -> None:
    from vdjtools.model import generate, load_bundled

    protos = load_prototypes("human", "TRB", n=k)["junction_aa"].to_list()
    gen = generate.generate(load_bundled("TRB", source="learned"), 4 * n, seed=1,
                            productive_only=True)
    seqs = [s for s in gen["junction_aa"].unique(maintain_order=True).to_list()
            if 6 <= len(s) <= 40][:n]

    t0 = time.perf_counter()
    junc = junction_distance_matrix(seqs, protos)             # (N, K)
    codes = PCA(n_components=0.95, whiten=True, random_state=0).fit_transform(junc)
    print(f"codes {codes.shape} (95% var of {k}-D junction) in {time.perf_counter() - t0:.1f}s")

    dec, m = train_inverse_decoder(codes, seqs, epochs=epochs)
    print(f"\nRESULT: exact-match {m['exact_match']:.3f}  token-acc {m['token_acc']:.3f}  "
          f"(n={m['n']}, code_dim={codes.shape[1]})")

    # show a few reconstructions
    recon = dec.decode(codes[:6])
    for orig, r in zip(seqs[:6], recon):
        print(f"  {orig:<22} -> {r:<22} {'OK' if orig == r else ''}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 20000,
         int(a[1]) if len(a) > 1 else 1000,
         int(a[2]) if len(a) > 2 else 60)
