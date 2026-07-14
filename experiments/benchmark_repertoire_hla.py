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

from _cohort import cv_auc, kmer_matrix, load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import class_witness, fit_repertoire_space, sample_embedding

REPO, META = "isalgo/airr_hip", "metadata.txt"
ALLELE = "HLA-A*02"
N_PROTO, N_COMPONENTS, N_RFF, N_RFF_SECOND = 1000, 20, 2048, 256   # deeper 2nd moment for public clusters


def _balanced(n_per_class: int) -> set:
    """Allowlist balanced on ALLELE presence vs a typed (non-NA) absence."""
    from vdjtools.io.batch import read_metadata
    from _hf import fetch

    m = read_metadata(fetch(REPO, META)).filter(pl.col("hla") != "NA")
    pos = m.filter(pl.col("hla").str.contains(r"\*02"))["file_name"].to_list()[:n_per_class]
    neg = m.filter(~pl.col("hla").str.contains(r"\*02"))["file_name"].to_list()[:n_per_class]
    return set(pos + neg)


def _public_candidates(frames, idx_pos, *, min_donors: int = 2) -> pl.DataFrame:
    """Clones present in ≥ ``min_donors`` of the given (train) A*02⁺ donors — the public pool."""
    from collections import Counter

    c = Counter()
    for i in idx_pos:
        for key in zip(frames[i]["v_call"].to_list(), frames[i]["j_call"].to_list(),
                       frames[i]["junction_aa"].to_list()):
            c[key] += 1
    keep = [k for k, v in c.items() if v >= min_donors]
    return pl.DataFrame(keep, schema=["v_call", "j_call", "junction_aa"], orient="row")


def witness_motif_auc(space, frames, y, *, top: int = 300, n_splits: int = 5, n_repeats: int = 3,
                      seed: int = 0):
    """Leakage-free supervised motif classifier: per fold, learn A*02-public motifs on TRAIN only,
    score each donor by its burden of clones near those motifs, CV-AUC. The direct 'find motifs → classify'.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.neighbors import BallTree

    clouds = [space.sample_cloud(f) for f in frames]          # (Z, w) per donor, once
    r = space.rff.length_scale
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                          random_state=seed).split(np.arange(len(frames)), y):
        pos = [i for i in tr if y[i] == 1]
        cand = _public_candidates(frames, pos)                # public in TRAIN A*02+ only
        if cand.height < 20:
            continue
        motifs = class_witness(space, [frames[i] for i in pos],
                               [frames[i] for i in tr if y[i] == 0], cand, top=top)
        tree = BallTree(space.transform_clonotypes(motifs))   # A*02-motif anchors (train-derived)
        feat = np.array([float((w * (tree.query_radius(Z, r=r, count_only=True) > 0)).sum())
                         for Z, w in clouds]).reshape(-1, 1)   # per-donor near-motif burden
        clf = LogisticRegression(max_iter=1000).fit(feat[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(feat[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


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
    space = fit_repertoire_space(model, pooled_clonotypes(samples), n_rff=N_RFF,
                                 n_rff_second=N_RFF_SECOND, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("mean", "diversity", "second")) for df in frames]
    mean = np.stack([e.mean for e in embs])
    div = np.stack([e.diversity for e in embs])
    second = np.stack([e.second for e in embs])
    km = kmer_matrix(samples)

    y = has.astype(int)
    auc = {                                       # repeated 5-fold × 10 CV -> (mean, std), not one split
        "diversity (4)": cv_auc(div, y),
        "kmer_profile": cv_auc(km, y, pca_cols=10**9),
        "kernel-mean Φ₁": cv_auc(mean, y, pca_cols=mean.shape[1]),
        "second-moment": cv_auc(second, y, pca_cols=second.shape[1]),
        "witness-motif (sup.)": witness_motif_auc(space, frames, y),   # leakage-free, train-derived motifs
    }

    # find motifs: witness on A*02+ vs A*02- (publicness cross-checked on the SAME donors — see caveat)
    cand = pooled_clonotypes([(None, frames[i]) for i in np.where(has)[0]], per_sample=3000)
    motifs = class_witness(space, [frames[i] for i in np.where(has)[0]],
                           [frames[i] for i in np.where(~has)[0]], cand, top=30)
    pub = _publicness(frames, has, motifs)

    print(f"\n{'method':<18}{'A*02 AUC (mean ± std, 50-fold CV)':>34}")
    for k, (m, s) in auc.items():
        print(f"{k:<18}{m:>22.3f} ± {s:.3f}")
    print(f"\nTop witness motifs (A*02-associated; publicness is in-sample, indicative only):")
    for r in motifs.head(6).iter_rows(named=True):
        print(f"  {r['v_call']:<12} {r['junction_aa']:<20} score={r['witness_score']:.3f}")
    print(f"mean motif publicness (Δ frac donors, A*02⁺ − A*02⁻) = {pub:+.3f}")

    vm, vs = auc["diversity (4)"]
    best_key = max(("second-moment", "witness-motif (sup.)", "kernel-mean Φ₁"), key=lambda k: auc[k][0])
    dm, ds = auc[best_key]
    separated = dm - ds > vm + vs                  # intervals must actually separate
    verdict = "PASS" if separated and dm > 0.55 else "PARTIAL" if dm > vm else "FAIL"
    print(f"\n[{verdict}] best clonotype-identity ({best_key}) {dm:.3f}±{ds:.3f} vs diversity "
          f"{vm:.3f}±{vs:.3f}; intervals separate = {separated} "
          f"(clonotype identity > diversity for HLA, established only if True)")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 50, int(args[2]) if len(args) > 2 else 10_000)
