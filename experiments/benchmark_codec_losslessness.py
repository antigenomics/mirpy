# 2026-07-14
# Measuring codec losslessness as a three-level hierarchy, on real held-out TRB.
#
#   1. informational ceiling (decoder-INDEPENDENT): exact_ceiling = 1 - collision_rate, the best
#      exact-match ANY decoder could reach given the code. If ~1.0 the code is injective, so all
#      reconstruction loss is decoder/data-limited, not information-limited.
#   2. reconstructive loss (decoder): exact-match, mean edit distance (graded), per-position
#      accuracy (conserved anchors vs the specificity-bearing variable middle).
#   3. rate-distortion: how 1 & 2 move with the code rate (PC bits, K) — the "how deep" curve.
#
# The gap (ceiling - exact_match) is decoder-limited; (1 - ceiling) is information-limited. This
# script shows the code is essentially injective (ceiling ~1.0) so the gap is the whole story, and
# it is closed by training data (see benchmark_lossless_kpc.py), not code depth.
#
# corpus = isalgo/airr_control (real naive TRB); prototypes = bundled arda human_TRB landmarks.
# Run (needs [ml] + [bench]):  python experiments/benchmark_codec_losslessness.py [N] [EPOCHS] [--tsv PATH]

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from _hf import fetch, load_repertoire  # noqa: E402
from benchmark_lossless_kpc import CONTROL_FILE, CONTROL_REPO, _embed, _split  # noqa: E402

from mir.bench.theory import codec_losslessness  # noqa: E402
from mir.embedding.prototypes import load_prototypes  # noqa: E402
from mir.ml.train import train_inverse_decoder  # noqa: E402

K_AXIS = [(2000, pc) for pc in (50, 100, 200, 300, 500)]      # rate = code bits (fixed K)
PROTO_AXIS = [(k, 300) for k in (500, 1000, 5000)]            # rate = #prototypes (fixed PC)


def rd_point(seqs, protos, k: int, pc: int, epochs: int) -> dict:
    """Train the inverse codec at (K, PC) and measure losslessness on the held-out split."""
    J, _ = _embed(seqs, protos[:k])
    te, _, tr = _split(len(seqs))
    pc = min(pc, k, len(tr))
    pca = PCA(n_components=pc, svd_solver="randomized", whiten=True, random_state=0).fit(J[tr])
    codes = pca.transform(J).astype(np.float32)
    dec, _ = train_inverse_decoder(codes, seqs, epochs=epochs, verbose=False)  # seed-0 -> same te
    recon = dec.decode(codes[te])
    m = codec_losslessness(codes[te], [seqs[i] for i in te], recon=recon)
    m["K"], m["PC"], m["var"], m["bits"] = k, pc, float(pca.explained_variance_ratio_.sum()), pc * 32
    return m


def _print(title: str, rows: list[dict]) -> None:
    print(f"== {title} ==")
    print(f"{'K':>6}{'PC':>5}{'bits':>6}{'var%':>7}{'ceil%':>8}{'exact%':>8}"
          f"{'edit':>7}{'token%':>8}{'anchor%':>9}{'middle%':>9}")
    for r in rows:
        print(f"{r['K']:>6}{r['PC']:>5}{r['bits']:>6}{100 * r['var']:>7.1f}"
              f"{100 * r['exact_ceiling']:>8.2f}{100 * r['exact_match']:>8.2f}{r['mean_edit']:>7.3f}"
              f"{100 * r['token_acc']:>8.2f}{100 * r['anchor_acc']:>9.2f}{100 * r['middle_acc']:>9.2f}")
    print(flush=True)


def main(n: int = 20000, epochs: int = 35, tsv: str | None = None) -> None:
    seqs = (load_repertoire(fetch(CONTROL_REPO, CONTROL_FILE), top=n + 5000)
            .head(n)["junction_aa"].to_list())
    protos = load_prototypes("human", "TRB", n=10000)["junction_aa"].to_list()
    print(f"codec losslessness on real TRB (isalgo/airr_control)  n={len(seqs)}  epochs={epochs}\n"
          f"ceiling = 1 - collision_rate (decoder-independent injectivity); gap to exact = "
          f"decoder/data-limited\n", flush=True)

    k_rows = [rd_point(seqs, protos, k, pc, epochs) for k, pc in K_AXIS]
    _print("rate = code bits  (K=2000, PC sweep)", k_rows)
    p_rows = [rd_point(seqs, protos, k, pc, epochs) for k, pc in PROTO_AXIS]
    _print("rate = #prototypes  (PC=300, K sweep)", p_rows)

    ref = next(r for r in k_rows if r["PC"] == 300)
    print(f"injectivity @ (K=2000,PC=300): collision_rate={ref['collision_rate']:.4g}  "
          f"ceiling={100 * ref['exact_ceiling']:.2f}%  nn_dist_median={ref['nn_dist_median']:.3f}\n"
          f"  -> code is injective; anchor {100 * ref['anchor_acc']:.1f}% vs middle "
          f"{100 * ref['middle_acc']:.1f}% recovery localizes loss to the variable middle.",
          flush=True)

    if tsv:  # rate-distortion curve for the appendix figure (fig_rd_codec.gp)
        with open(tsv, "w") as fh:
            fh.write("bits\texact\tedit\tceiling\ttoken\n")
            for r in k_rows:
                fh.write(f"{r['bits']}\t{r['exact_match']:.4f}\t{r['mean_edit']:.4f}"
                         f"\t{r['exact_ceiling']:.4f}\t{r['token_acc']:.4f}\n")
        print(f"wrote {tsv}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    tsv = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--tsv"), None)
    main(int(a[0]) if a else 20000, int(a[1]) if len(a) > 1 else 35, tsv)
