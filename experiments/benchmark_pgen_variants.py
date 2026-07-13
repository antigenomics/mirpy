"""Pgen regressor breakdown: exact vs 1-mismatch, by V/J conditioning (Part 2).

For a CDR3-only regressor, the target quantity matters: marginalized Pgen (v=None,
j=None) is a pure function of the CDR3, while V/J-conditional Pgen depends on genes the
regressor never sees (a noise ceiling). Reports test Pearson r + RMSE for the 8 variants
{mismatches 0,1} x {V&J, V, J, neither}.

Run (needs [ml] + rearrangement):
    python experiments/benchmark_pgen_variants.py [LOCUS] [N] [epochs]
"""

from __future__ import annotations

import sys

import numpy as np

from mir.ml.train import train_pgen_regressor


def main(locus: str = "TRB", n: int = 10000, epochs: int = 25) -> None:
    from vdjtools.model import load_bundled, native
    from vdjtools.model.generate import generate

    model = load_bundled(locus, source="olga")
    gen = generate(model, 3 * n, seed=1, productive_only=True).unique(
        subset=["junction_aa", "v_call", "j_call"], maintain_order=True)
    gen = gen.filter((gen["junction_aa"].str.len_chars() >= 6)
                     & (gen["junction_aa"].str.len_chars() <= 40)).head(n)
    seqs = gen["junction_aa"].to_list()
    v_all = gen["v_call"].to_list()
    j_all = gen["j_call"].to_list()

    print(f"{locus}: {len(seqs)} seqs\n")
    print(f"{'match':<8}{'mm':>3}{'n>0':>7}{'pearson':>9}{'rmse':>7}")
    for match, (v_arg, j_arg) in [("V&J", (v_all, j_all)), ("V", (v_all, None)),
                                  ("J", (None, j_all)), ("none", (None, None))]:
        for mm in (0, 1):
            pgen = np.asarray(native.pgen_aa_batch(model, seqs, v=v_arg, j=j_arg,
                                                   mismatches=mm, threads=0))
            ok = pgen > 0
            _, m = train_pgen_regressor([s for s, k in zip(seqs, ok) if k],
                                        np.log10(pgen[ok]), epochs=epochs, verbose=False)
            print(f"{match:<8}{mm:>3}{int(ok.sum()):>7}{m['pearson']:>9.4f}{m['rmse']:>7.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "TRB",
         int(a[1]) if len(a) > 1 else 10000,
         int(a[2]) if len(a) > 2 else 25)
