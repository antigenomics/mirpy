"""Per-chain / per-species codec performance breakdown (Part 2).

For every bundled (species, locus) prototype set, trains the forward codec
(seq -> compact junction embedding), the inverse codec (code -> seq), and — where a
rearrangement model exists (human loci) — the Pgen regressor, on the shipped prototype
sequences (first K as prototypes, the rest as the corpus). Prints a per-chain table.

Run (needs [ml] + [bench] + rearrangement):
    python experiments/benchmark_codec_chains.py [N] [K]
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.decomposition import PCA

from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import list_available_prototypes, load_prototypes
from mir.ml.train import train_forward_encoder, train_inverse_decoder, train_pgen_regressor


def _pgen_r(species, locus, seqs, v, j, epochs):
    if species != "human":
        return None
    try:
        from vdjtools.model import load_bundled, native
        model = load_bundled(locus, source="olga")
        pgen = np.asarray(native.pgen_aa_batch(model, seqs, v=v, j=j, mismatches=1, threads=0))
        ok = pgen > 0
        if ok.sum() < 500:
            return None
        _, m = train_pgen_regressor([s for s, k in zip(seqs, ok) if k],
                                    np.log10(pgen[ok]), epochs=epochs, verbose=False)
        return m["pearson"]
    except Exception:
        return None


def main(n: int = 5000, k: int = 1000) -> None:
    rows = []
    for species, locus in list_available_prototypes():
        pool = load_prototypes(species, locus, n=min(10000, k + n + 500))
        protos = pool["junction_aa"].to_list()[:k]
        q = pool[k:k + n]
        seqs = q["junction_aa"].to_list()
        junc = junction_distance_matrix(seqs, protos)
        codes = PCA(n_components=0.95, whiten=True, random_state=0).fit_transform(junc)

        _, fm = train_forward_encoder(seqs, junc, epochs=20, verbose=False)
        _, im = train_inverse_decoder(codes, seqs, epochs=35, verbose=False)
        r = _pgen_r(species, locus, seqs, q["v_call"].to_list(), q["j_call"].to_list(), 20)
        rows.append((species, locus, len(seqs), codes.shape[1],
                     fm["test_cosine"], im["exact_match"], im["token_acc"], r))
        print(f"  done {species}_{locus}: fwd_cos={fm['test_cosine']:.3f} "
              f"inv_exact={im['exact_match']:.3f} pgen_r={r}")

    print(f"\n{'species':<8}{'chain':<6}{'n':>6}{'PCs':>5}{'fwd_cos':>9}"
          f"{'inv_exact':>10}{'tok_acc':>9}{'pgen_r':>8}")
    for sp, lo, nn, pcs, fc, ie, ta, r in rows:
        rt = f"{r:.3f}" if r is not None else "  -"
        print(f"{sp:<8}{lo:<6}{nn:>6}{pcs:>5}{fc:>9.3f}{ie:>10.3f}{ta:>9.3f}{rt:>8}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 5000, int(a[1]) if len(a) > 1 else 1000)
