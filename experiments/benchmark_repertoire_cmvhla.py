"""Repertoire embedding predicts CMV serostatus, HLA-restricted — where clonotype identity beats diversity.

Unlike age (a diversity phenomenon — see benchmark_repertoire_aging), the CMV response is a set of large,
**public, HLA-restricted** clonal expansions (e.g. the HLA-A*02 NLVPMVATV response). Two CMV⁺ donors are
"close" only if they share the presenting HLA type — so a diversity summary can't tell CMV apart, but the
clonotype-resolved kernel mean / learned set encoder can (Theory §T.7, Prop. ``prop:hla``, ``prop:interact``).

On the Emerson 2017 HIP cohort (HF ``isalgo/airr_hip``, 786 donors, CMV serostatus + HLA-A/B typing) we test:
  (i)   CMV⁺/⁻ classification AUC — Φ (kernel mean + second moment) vs diversity-only and k-mer baselines;
  (ii)  **head-to-head** backbone Φ vs the learned Set-Transformer encoder (mir.ml.set_encoder);
  (iii) **HLA-stratified separation** (Prop. ``prop:hla``): CMV⁺ donors cluster (small within-group MMD
        relative to CMV⁺-vs-CMV⁻) *more* among HLA-A*02 carriers than among A*02⁻ donors — the A*02-restricted
        signal. Each donor is downsampled to the shallow RNA-seq regime; the A*02 CMV expansions are
        high-frequency and survive downsampling (why the antigen signal persists — the depth argument).

Data: HF isalgo/airr_hip. Cached on first run (needs [bench] + [ml]).
Run:  python experiments/benchmark_repertoire_cmvhla.py [n_per_class] [downsample_reads] [epochs]
Full cohort: python experiments/benchmark_repertoire_cmvhla.py 380 20000 80
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _cohort import held_out_auc as _auc
from _cohort import kmer_matrix, load_cohort, pooled_clonotypes
from _hf import fetch

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, mmd_matrix, sample_embedding

REPO, META = "isalgo/airr_hip", "metadata.txt"
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048
STRATIFIER = "HLA-A*02"


def _age_matched_cmv(n_per_class: int) -> set:
    """Allowlist of file_names, CMV +/- **age-matched** within age-decade bins.

    CMV prevalence rises with age and diversity falls with age, so an unmatched split lets a
    diversity scalar predict CMV *via age* — a confound. Pairing CMV⁺/CMV⁻ donors within each
    age decade removes it, so any residual CMV signal is genuinely CMV-specific.
    """
    from vdjtools.io.batch import read_metadata

    m = (read_metadata(fetch(REPO, META))
         .filter(pl.col("cmv").is_in(["+", "-"]) & (pl.col("age") != "NA"))
         .with_columns((pl.col("age").cast(pl.Int64) // 10).alias("decade")))
    keep = []
    for (dec,), grp in m.group_by(["decade"], maintain_order=True):
        pos = grp.filter(pl.col("cmv") == "+")["file_name"].to_list()
        neg = grp.filter(pl.col("cmv") == "-")["file_name"].to_list()
        for i in range(min(len(pos), len(neg))):
            keep.append(pos[i]); keep.append(neg[i])
    return set(keep[:2 * n_per_class])


def _hla_stratified(embs, cmv, a02) -> tuple[float, float]:
    """CMV⁺ clustering strength within the A*02⁺ vs A*02⁻ stratum (Prop. prop:hla).

    Returns ``sep`` per stratum: ``mean MMD(CMV⁺, CMV⁻) − mean MMD(CMV⁺, CMV⁺)`` — positive when CMV⁺
    donors cluster tighter than they sit from CMV⁻. A*02⁺ should show the larger separation.
    """
    D = mmd_matrix(embs)
    out = []
    for stratum in (a02, ~a02):
        pp = np.where(stratum & (cmv == 1))[0]
        pn = np.where(stratum & (cmv == 0))[0]
        if len(pp) < 2 or len(pn) < 1:
            out.append(np.nan); continue
        within = np.mean([D[i, j] for a, i in enumerate(pp) for j in pp[a + 1:]])
        between = np.mean([D[i, j] for i in pp for j in pn])
        out.append(between - within)
    return out[0], out[1]


def main(n_per_class: int = 60, downsample_to: int = 10_000, epochs: int = 50) -> None:
    from mir.ml.set_encoder import train_set_encoder

    t0 = time.perf_counter()
    _, samples = load_cohort(REPO, META, downsample_to=downsample_to,
                             only=_age_matched_cmv(n_per_class))
    cmv = np.array([1 if r["cmv"] == "+" else 0 for r, _ in samples])
    a02 = np.array([STRATIFIER in (r["hla"] or "") for r, _ in samples])
    age = np.array([float(r["age"]) for r, _ in samples])
    print(f"{len(samples)} donors: {cmv.sum()} CMV+, {(cmv == 0).sum()} CMV-, "
          f"{a02.sum()} carry {STRATIFIER}; age-matched "
          f"(CMV+ {age[cmv == 1].mean():.0f}±{age[cmv == 1].std():.0f}y, "
          f"CMV- {age[cmv == 0].mean():.0f}±{age[cmv == 0].std():.0f}y); ≤{downsample_to} reads/donor")

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("mean", "diversity", "second")) for _, df in samples]
    mean = np.stack([e.mean for e in embs])
    div = np.stack([e.diversity for e in embs])
    second = np.stack([e.second for e in embs])
    Phi = np.hstack([mean, second, div])               # mean+second PCA'd, diversity raw
    n_pca = mean.shape[1] + second.shape[1]
    clouds = [space.sample_cloud(df) for _, df in samples]

    km = kmer_matrix(samples)
    idx = np.arange(len(samples))
    tr, te = train_test_split(idx, test_size=0.3, stratify=cmv, random_state=0)
    auc = {
        "age only (confound)": _auc(age[tr], cmv[tr], age[te], cmv[te]),
        "diversity (4)": _auc(div[tr], cmv[tr], div[te], cmv[te]),
        "kmer_profile": _auc(km[tr], cmv[tr], km[te], cmv[te], pca_cols=10**9),
        "kernel-mean Φ₁ only": _auc(mean[tr], cmv[tr], mean[te], cmv[te], pca_cols=mean.shape[1]),
        "second-moment only": _auc(second[tr], cmv[tr], second[te], cmv[te], pca_cols=second.shape[1]),
        "Phi (mean+2nd+div)": _auc(Phi[tr], cmv[tr], Phi[te], cmv[te], pca_cols=n_pca),
    }
    sem, _ = train_set_encoder([clouds[i] for i in tr], cmv[tr].astype(float),
                               task="classification", epochs=epochs, seed=0, verbose=False)
    auc["learned set-encoder"] = roc_auc_score(cmv[te], sem.predict([clouds[i] for i in te]))

    sep_a02, sep_ctrl = _hla_stratified(embs, cmv, a02)

    print(f"\n{'method':<22}{'CMV AUC (held-out)':>20}")
    for k, v in auc.items():
        print(f"{k:<22}{v:>20.3f}")
    print(f"\nHLA-stratified CMV⁺ clustering (Prop. prop:hla): "
          f"A*02⁺ sep={sep_a02:.3f}  vs  A*02⁻ sep={sep_ctrl:.3f}")

    payoff = auc["Phi (mean+2nd+div)"] > auc["diversity (4)"]
    hla_ok = np.isfinite(sep_a02) and np.isfinite(sep_ctrl) and sep_a02 > sep_ctrl
    verdict = "PASS" if payoff and hla_ok else "PARTIAL" if payoff or hla_ok else "FAIL"
    print(f"\n[{verdict}] Φ CMV AUC {auc['Phi (mean+2nd+div)']:.3f} vs diversity "
          f"{auc['diversity (4)']:.3f} (payoff={payoff}); "
          f"A*02-restricted clustering stronger = {hla_ok}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 60,
         int(args[2]) if len(args) > 2 else 10_000,
         int(args[3]) if len(args) > 3 else 50)
