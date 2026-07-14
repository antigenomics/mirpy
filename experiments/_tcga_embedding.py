# 2026-07-14
# Unified TME-aware, multi-chain REPERTOIRE EMBEDDING for TCGA — the reframing of the WS4/WS4b results.
# The biology axes that carry prognosis are not competitors to the repertoire embedding; they ARE its channels.
# Per sample, over all 7 chains, Φ(S) = concat of:
#   identity    : per-chain kernel-mean (RFF) PCA-reduced          (what the clones are)
#   diversity   : per-chain Hill numbers                            (clonality)
#   coverage    : per-chain log receptor-read load                 (infiltration / hot-cold — the magnitude the
#                                                                    frequency-normalized kernel mean discards)
#   isotype     : IGH class-switch composition (IgG/IgA/IgM/switch) (mucosal / plasma TME state)
#   composition : T-vs-B balance + per-chain read fractions         (TME cell-type mix)
#   atypicality : identity-block distance to the tumour-type centroid (selection / divergence — a Φ-geometry op)
# Each sample's TSV is read exactly once. `channels` maps each group -> column indices for ablation.

from __future__ import annotations

import os

import numpy as np
import polars as pl

from _cohort import pooled_clonotypes
from _tcga import CHAINS, _samples_dir, load_metadata
from _tcga_features import B_LOCI, T_LOCI, _isotype

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

_CANON = r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$"
MIN_CLONES = 5                     # per-chain minimum to embed that chain for a sample


def _read(path):
    return pl.read_csv(path, separator="\t", infer_schema_length=0).with_columns(
        pl.col("duplicate_count").cast(pl.Float64, strict=False).fill_null(1.0))


def _chain_frame(df, chain):
    return (df.filter(pl.col("locus") == chain)
            .select("junction_aa", "v_call", "j_call", "duplicate_count")
            .filter(pl.col("junction_aa").str.contains(_CANON))
            .group_by(["junction_aa", "v_call", "j_call"]).agg(pl.col("duplicate_count").sum()))


def _fit_spaces(fit_rows, chains, *, n_rff, n_comp, per_sample, seed):
    """One RepertoireSpace per chain, fit on pooled clonotypes from a subsample of the cohort."""
    sd = _samples_dir()
    pools = {c: [] for c in chains}
    for r in fit_rows:
        p = f"{sd}/{r['sample_id']}.tsv"
        if not os.path.exists(p):
            continue
        df = _read(p)
        for c in chains:
            sub = _chain_frame(df, c)
            if sub.height >= MIN_CLONES:
                pools[c].append(sub.head(per_sample))
    spaces = {}
    for c in chains:
        if len(pools[c]) < 10:
            continue
        model = TCREmp.from_defaults("human", c, n_prototypes=1000)
        spaces[c] = fit_repertoire_space(model, pooled_clonotypes([(None, f) for f in pools[c]]),
                                         n_rff=n_rff, n_rff_second=0, n_components=n_comp, seed=seed)
    return spaces


def _isotype_switch(df):
    """IGH read-weighted (IgG, IgA, IgM, switch) fractions of typed IGH."""
    igh = df.filter(pl.col("locus") == "IGH")
    r = {"IgG": 0.0, "IgA": 0.0, "IgM": 0.0, "IgD": 0.0, "IgE": 0.0}
    for c, n in zip(igh["c_call"].to_list(), igh["duplicate_count"].to_list()):
        k = _isotype(c)
        if k:
            r[k] += float(n)
    t = sum(r.values())
    if not t:
        return np.zeros(4)
    return np.array([r["IgG"] / t, r["IgA"] / t, r["IgM"] / t,
                     (r["IgG"] + r["IgA"] + r["IgE"]) / t])


def build_embedding(cancers, chains=CHAINS, *, n_rff=512, n_comp=20, id_pca=8,
                    n_fit=150, per_sample=2000, seed=0, cache_dir=None):
    """Return (rows, X, channels). X = the TME-aware multi-chain repertoire embedding; channels maps
    group name -> list of X column indices (for ablation). Set ``cache_dir`` to memoize the (expensive,
    ~9.5k-TSV) build keyed by the cancer set + params."""
    import hashlib
    import pickle
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cache = None
    if cache_dir:
        key = hashlib.md5(repr((sorted(cancers), list(chains), n_rff, n_comp, id_pca,
                                n_fit, per_sample, seed)).encode()).hexdigest()[:16]
        cache = os.path.join(cache_dir, f"tme_emb_{key}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as fh:
                return pickle.load(fh)
        os.makedirs(cache_dir, exist_ok=True)

    rows_all = load_metadata(cancers, require_os=True).to_dicts()
    rng = np.random.default_rng(seed)
    fit_rows = [rows_all[i] for i in rng.choice(len(rows_all), min(n_fit, len(rows_all)), replace=False)]
    spaces = _fit_spaces(fit_rows, chains, n_rff=n_rff, n_comp=n_comp, per_sample=per_sample, seed=seed)
    chains = [c for c in chains if c in spaces]           # keep only chains we could fit
    sd = _samples_dir()

    # Pass over samples: read each TSV once, embed every chain + composition/coverage/isotype.
    keep, vus = [], []
    means = {c: [] for c in chains}                        # (row_idx, mean vec)
    per = {c: {} for c in chains}                          # row_idx -> [div(4), log_infil(1), present]
    iso, comp = {}, {}                                     # row_idx -> vec
    for r in rows_all:
        p = f"{sd}/{r['sample_id']}.tsv"
        if not os.path.exists(p):
            continue
        df = _read(p)
        ridx = len(keep)
        tot = df["duplicate_count"].sum()
        reads = {c: float(df.filter(pl.col("locus") == c)["duplicate_count"].sum()) for c in CHAINS}
        rt = sum(reads[c] for c in T_LOCI); rb = sum(reads[c] for c in B_LOCI)
        for c in chains:
            sub = _chain_frame(df, c)
            if sub.height >= MIN_CLONES:
                e = sample_embedding(spaces[c], sub, blocks=("mean", "diversity"))
                means[c].append((ridx, e.mean))
                per[c][ridx] = np.concatenate([e.diversity, [np.log1p(reads[c])], [1.0]])
        iso[ridx] = _isotype_switch(df)
        comp[ridx] = np.array([np.log1p(rt) - np.log1p(rb),                       # T-vs-B balance
                               *[reads[c] / (float(tot) + 1e-9) for c in CHAINS]])  # per-chain read fraction
        vus.append({f"{c}:{v.split('*')[0]}": float(n) for c in ("IGH", "TRB")
                    for v, n in zip(df.filter(pl.col("locus") == c)["v_call"].to_list(),
                                    df.filter(pl.col("locus") == c)["duplicate_count"].to_list()) if v})
        keep.append(r)

    n = len(keep)
    # per-chain identity: PCA-reduce the kernel-mean matrix
    ident = {c: np.zeros((n, id_pca)) for c in chains}
    for c in chains:
        if len(means[c]) < id_pca + 1:
            continue
        idxs, M = zip(*means[c]); M = np.stack(M)
        Mr = PCA(id_pca, random_state=seed).fit_transform(StandardScaler().fit_transform(M))
        for j, ri in enumerate(idxs):
            ident[c][ri] = Mr[j]

    # atypicality on the identity block: distance of concatenated identity to the tumour-type centroid
    ID = np.hstack([ident[c] for c in chains]) if chains else np.zeros((n, 0))
    tt = np.array([r["study_id"] for r in keep])
    aty = np.zeros(n)
    for t in set(tt):
        m = tt == t
        cen = ID[m].mean(0); cn = np.linalg.norm(cen) + 1e-9
        for i in np.where(m)[0]:
            aty[i] = 1.0 - float(ID[i] @ cen) / ((np.linalg.norm(ID[i]) + 1e-9) * cn)

    # assemble X + channel index
    blocks, channels, col = [], {}, 0
    def add(name, mat):
        nonlocal col
        blocks.append(mat); channels.setdefault(name, []).extend(range(col, col + mat.shape[1])); col += mat.shape[1]

    for c in chains:
        add("identity", ident[c])
        pc = np.array([per[c].get(i, np.zeros(6)) for i in range(n)])   # div(4)+log_infil(1)+present(1)
        add("diversity", pc[:, :4]); add("coverage", pc[:, 4:5])
    add("isotype", np.array([iso[i] for i in range(n)]))
    add("composition", np.array([comp[i] for i in range(n)]))
    add("atypicality", aty[:, None])
    X = np.hstack(blocks)
    # median-impute + z-score
    for j in range(X.shape[1]):
        cj = X[:, j]; cj[~np.isfinite(cj)] = np.nanmedian(cj[np.isfinite(cj)]) if np.isfinite(cj).any() else 0.0
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = (keep, X, channels)
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out
