"""Does clonotype identity predict HLA across a panel of A/B alleles? (DeWitt et al. 2018, eLife 7:e38358)

DeWitt et al. showed TCR *occurrence patterns* encode HLA genotype on this very cohort (Emerson HIP). Here we
ask whether the sample-level embedding's **second-moment (co-occurrence) block** predicts carriage of each
common HLA-A/B allele better than a diversity summary — which should be at chance, since an HLA allele shifts
*which* public clones you carry, not your overall diversity (Theory §T.7, Prop. ``prop:interact``).

Efficient design: embed one fixed set of HLA-typed donors **once**, then relabel per allele and CV — so the
cost is one embedding, many labels. Per allele: second-moment vs diversity AUC (repeated 50-fold CV, mean±std).

Data: HF isalgo/airr_hip (Emerson 2017). Cached (needs [bench]).
Run:  python experiments/benchmark_repertoire_hlapanel.py [n_donors] [downsample_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from _cohort import cv_auc, load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

REPO, META = "isalgo/airr_hip", "metadata.txt"
N_PROTO, N_COMPONENTS, N_RFF, N_RFF_SECOND = 1000, 20, 2048, 256
PANEL = ("A*01", "A*02", "A*03", "A*11", "A*24", "B*07", "B*08", "B*44")


def main(n_donors: int = 300, downsample_to: int = 20_000) -> None:
    t0 = time.perf_counter()
    from vdjtools.io.batch import read_metadata
    from _hf import fetch

    typed = set(read_metadata(fetch(REPO, META)).filter(pl.col("hla") != "NA")["file_name"].to_list())
    _, samples = load_cohort(REPO, META, downsample_to=downsample_to,
                             only=typed, cap_samples=n_donors)
    frames = [df for _, df in samples]
    hla = [r["hla"] or "" for r, _ in samples]

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples), n_rff=N_RFF,
                                 n_rff_second=N_RFF_SECOND, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("diversity", "second")) for df in frames]
    div = np.stack([e.diversity for e in embs])
    second = np.stack([e.second for e in embs])

    print(f"{len(samples)} HLA-typed donors, ≤{downsample_to} reads; one embedding, per-allele relabel\n")
    print(f"{'allele':<8}{'n+':>5}{'diversity AUC':>18}{'second-moment AUC':>22}{'Δ':>8}")
    rows = []
    for al in PANEL:
        y = np.array([al in h for h in hla], dtype=int)
        if y.sum() < 15 or (1 - y).sum() < 15:
            print(f"{al:<8}{y.sum():>5}   -- too few carriers --"); continue
        vm, vs = cv_auc(div, y)
        dm, ds = cv_auc(second, y, pca_cols=second.shape[1])
        rows.append((al, dm, ds, vm, vs))
        print(f"{al:<8}{y.sum():>5}{vm:>13.3f}±{vs:.3f}{dm:>16.3f}±{ds:.3f}{dm - vm:>+8.3f}")

    n_win = sum(dm - ds > vm + vs for _, dm, ds, vm, vs in rows)
    n_dir = sum(dm > vm for _, dm, ds, vm, vs in rows)
    verdict = "PASS" if n_win >= len(rows) // 2 else "PARTIAL" if n_dir > len(rows) // 2 else "FAIL"
    print(f"\n[{verdict}] second-moment > diversity in direction for {n_dir}/{len(rows)} alleles; "
          f"intervals separate for {n_win}/{len(rows)}. Clonotype identity encodes HLA "
          f"(DeWitt 2018) — {'broadly' if n_win >= len(rows) // 2 else 'weakly, needs depth/n'}.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 300, int(args[2]) if len(args) > 2 else 20_000)
