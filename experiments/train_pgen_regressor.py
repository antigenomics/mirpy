"""Train the Pgen regressor: CDR3 -> log10 Pgen(1-mismatch) (Part 2).

Free supervision — targets are 1-mismatch Pgen from vdjtools' native model. The
shared sequence encoder predicts a scalar; far faster than the DP once trained
(irrm-codec r~0.99 on TRB). Reports test Pearson r and RMSE (log10 units).

Run (needs [ml] + rearrangement):
    python experiments/train_pgen_regressor.py [N] [epochs]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from mir.ml.train import train_pgen_regressor


def main(n: int = 20000, epochs: int = 40) -> None:
    from vdjtools.model import load_bundled, native
    from vdjtools.model.generate import generate

    model = load_bundled("TRB", source="olga")
    gen = generate(model, 3 * n, seed=1, productive_only=True).unique(
        subset=["junction_aa", "v_call", "j_call"], maintain_order=True)
    gen = gen.filter(
        (gen["junction_aa"].str.len_chars() >= 6) & (gen["junction_aa"].str.len_chars() <= 40)
    ).head(n)
    seqs = gen["junction_aa"].to_list()
    v = gen["v_call"].to_list()
    j = gen["j_call"].to_list()

    t0 = time.perf_counter()
    pgen = np.asarray(native.pgen_aa_batch(model, seqs, v=v, j=j, mismatches=1, threads=0))
    dt = time.perf_counter() - t0
    ok = pgen > 0
    seqs = [s for s, k in zip(seqs, ok) if k]
    log_pgen = np.log10(pgen[ok])
    print(f"pgen_1mm for {ok.sum()}/{len(ok)} seqs in {dt:.1f}s "
          f"({ok.sum()/dt:.0f} seq/s); log10 range [{log_pgen.min():.1f}, {log_pgen.max():.1f}]")

    reg, m = train_pgen_regressor(seqs, log_pgen, epochs=epochs)
    print(f"\nRESULT: Pearson r {m['pearson']:.4f}  RMSE {m['rmse']:.3f} log10  (n={m['n']})")

    # speed: DNN vs native DP on 5000 seqs
    probe = seqs[:5000]
    t0 = time.perf_counter(); reg.predict(probe); dt_dnn = time.perf_counter() - t0
    t0 = time.perf_counter()
    native.pgen_aa_batch(model, probe, v=v[:5000], j=j[:5000], mismatches=1, threads=0)
    dt_dp = time.perf_counter() - t0
    print(f"pgen 5000 seqs: DNN {dt_dnn*1e3:.0f}ms  vs  native DP {dt_dp*1e3:.0f}ms")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 20000, int(a[1]) if len(a) > 1 else 40)
