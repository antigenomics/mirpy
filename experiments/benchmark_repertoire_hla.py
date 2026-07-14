"""HLA-A*02 inference from the repertoire embedding — the clonotype-identity task where Φ beats diversity.

Unlike age/CMV (clone-size/diversity phenomena — see the aging & cmvhla benchmarks), an HLA allele is a
pure **clonotype-identity** signal: A*02 carriers share *public, A*02-restricted* CDR3 clones, but their
overall repertoire diversity is unchanged. So a diversity summary should be near chance while the
clonotype-resolved embedding Φ (and the k-mer composition) should predict A*02 presence — the clean
demonstration of what the sample-level embedding adds over diversity (Theory §T.7, Prop. ``prop:interact``).

We also **find the motifs**: the supervised MMD witness (:func:`mir.repertoire.class_witness`,
Prop. ``prop:witness``) ranks clonotypes by their contribution to the A*02⁺−A*02⁻ mean-embedding
difference; the top motifs should be *public* — present in far more A*02⁺ than A*02⁻ donors.

Data: HF isalgo/airr_hip (Emerson 2017; HLA-A/B typing). Cached on first run (needs [bench]).
Run:  python experiments/benchmark_repertoire_hla.py [n_per_class] [downsample_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from _cohort import held_out_auc, kmer_matrix, load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import class_witness, fit_repertoire_space, sample_embedding

REPO, META = "isalgo/airr_hip", "metadata.txt"
ALLELE = "HLA-A*02"
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048


def _balanced(n_per_class: int) -> set:
    """Allowlist balanced on ALLELE presence vs a typed (non-NA) absence."""
    from vdjtools.io.batch import read_metadata
    from _hf import fetch

    m = read_metadata(fetch(REPO, META)).filter(pl.col("hla") != "NA")
    pos = m.filter(pl.col("hla").str.contains(r"\*02"))["file_name"].to_list()[:n_per_class]
    neg = m.filter(~pl.col("hla").str.contains(r"\*02"))["file_name"].to_list()[:n_per_class]
    return set(pos + neg)


def _publicness(frames, has, motifs) -> float:
    """Mean (frac of ALLELE⁺ donors carrying a motif − frac of ALLELE⁻), over the top motifs."""
    keys = set(zip(motifs["v_call"].to_list(), motifs["junction_aa"].to_list()))
    pos_sets = [set(zip(df["v_call"].to_list(), df["junction_aa"].to_list()))
                for df, h in zip(frames, has) if h]
    neg_sets = [set(zip(df["v_call"].to_list(), df["junction_aa"].to_list()))
                for df, h in zip(frames, has) if not h]
    enr = []
    for k in keys:
        fp = np.mean([k in s for s in pos_sets])
        fn = np.mean([k in s for s in neg_sets])
        enr.append(fp - fn)
    return float(np.mean(enr))


def main(n_per_class: int = 50, downsample_to: int = 10_000) -> None:
    t0 = time.perf_counter()
    _, samples = load_cohort(REPO, META, downsample_to=downsample_to, only=_balanced(n_per_class))
    frames = [df for _, df in samples]
    has = np.array([bool("*02" in (r["hla"] or "")) for r, _ in samples])
    print(f"{len(samples)} donors: {has.sum()} {ALLELE}+, {(~has).sum()} {ALLELE}-; "
          f"≤{downsample_to} reads/donor")

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("mean", "diversity", "second")) for df in frames]
    mean = np.stack([e.mean for e in embs])
    div = np.stack([e.diversity for e in embs])
    second = np.stack([e.second for e in embs])
    km = kmer_matrix(samples)

    idx = np.arange(len(samples))
    tr, te = train_test_split(idx, test_size=0.3, stratify=has, random_state=0)
    y = has.astype(int)
    auc = {
        "diversity (4)": held_out_auc(div[tr], y[tr], div[te], y[te]),
        "kmer_profile": held_out_auc(km[tr], y[tr], km[te], y[te], pca_cols=10**9),
        "kernel-mean Φ₁": held_out_auc(mean[tr], y[tr], mean[te], y[te], pca_cols=mean.shape[1]),
        "second-moment": held_out_auc(second[tr], y[tr], second[te], y[te], pca_cols=second.shape[1]),
    }

    # find motifs: witness on A*02+ vs A*02-, then check publicness
    cand = pooled_clonotypes([(None, frames[i]) for i in np.where(has)[0]], per_sample=3000)
    motifs = class_witness(space, [frames[i] for i in np.where(has)[0]],
                           [frames[i] for i in np.where(~has)[0]], cand, top=30)
    pub = _publicness(frames, has, motifs)

    print(f"\n{'method':<18}{'A*02 AUC (held-out)':>20}")
    for k, v in auc.items():
        print(f"{k:<18}{v:>20.3f}")
    print(f"\nTop witness motifs (A*02-associated public clones):")
    for r in motifs.head(6).iter_rows(named=True):
        print(f"  {r['v_call']:<12} {r['junction_aa']:<20} score={r['witness_score']:.3f}")
    print(f"mean motif publicness (Δ frac donors, A*02⁺ − A*02⁻) = {pub:+.3f}")

    best_clono = max(auc["kernel-mean Φ₁"], auc["second-moment"], auc["kmer_profile"])
    verdict = "PASS" if best_clono > auc["diversity (4)"] and pub > 0 else "PARTIAL"
    print(f"\n[{verdict}] clonotype-identity AUC {best_clono:.3f} > diversity {auc['diversity (4)']:.3f}; "
          f"witness motifs enriched in A*02⁺ (Δ={pub:+.3f})")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 50, int(args[2]) if len(args) > 2 else 10_000)
