"""The digital donor: one fixed, explainable descriptor per subject, fused across chains (§T.7).

Where :mod:`mir.repertoire` embeds one repertoire (one chain of one sample) and :mod:`mir.explain`
names the pieces of a feature matrix, this module assembles the **cohort** object the clinical
pipelines actually consume — a *digital donor* whose vector concatenates, per locus, the identity
(kernel mean), diversity (Hill) and coverage (log receptor load) channels into one row, ready for
survival / classification via :func:`mir.explain.channel_report`.

The library owns the **geometry, fusion and serialization**; the study owns **which extra channels**
(isotype, composition, HLA, …) and **which scorer**. So :func:`fit_donor_embeddings` builds the
per-chain geometry and fuses it through :class:`mir.explain.ChannelBuilder`, and takes an
``extra_channels`` hook for the analysis to inject its own tissue/clinical blocks into the *same*
matrix — the channels then flow through ``channel_report`` indistinguishably.

Comparability bites twice here (two bases, not one): each locus carries its own prototype hash **and**
a cross-sample identity-PCA rotation. :class:`DonorCohort` stores both; :meth:`DonorCohort.load`
verifies *every* locus's prototype hash, and :meth:`DonorCohort.transform` is the only way to project
held-out donors into the fitted basis. A batch-``residualize``d ``X`` is likewise incomparable to the
raw one — record the correction in provenance.

Torch-free (numpy / sklearn / polars); vdjtools is lazy (only :func:`incidence_biomarkers`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from mir.explain import ChannelBuilder, ChannelSpec
from mir.repertoire import RepertoireSpace, mmd_matrix, sample_embedding

_COUNT = "duplicate_count"


# ------------------------------------------------------------- per-donor block assembly


def _locus_blocks(spaces, donor_frames, *, min_clones):
    """Per (donor, locus): kernel mean, Hill diversity (4), log receptor load. ``nan`` where absent.

    Returns ``(means, div, cov)`` where ``means[locus] = [(row_idx, mean_vec), …]`` (only present
    donors), and ``div``/``cov`` are ``(n, 4)`` / ``(n, 1)`` with ``nan`` rows for donors lacking the
    chain (later imputed by :class:`~mir.explain.ChannelBuilder`).
    """
    n = len(donor_frames)
    loci = list(spaces)
    means = {c: [] for c in loci}
    div = {c: np.full((n, 4), np.nan) for c in loci}
    cov = {c: np.full((n, 1), np.nan) for c in loci}
    for i, frames in enumerate(donor_frames):
        for c in loci:
            f = frames.get(c)
            if f is not None and f.height >= min_clones:
                e = sample_embedding(spaces[c], f, blocks=("mean", "diversity"))
                means[c].append((i, e.mean))
                div[c][i] = e.diversity
                cov[c][i, 0] = np.log1p(float(f[_COUNT].sum()))
    return means, div, cov


def _identity_matrix(means_c, n, id_pca, *, pca=None, seed=0):
    """``(n, id_pca)`` identity block: PCA-reduce the per-donor kernel means (``nan`` where absent).

    Fits a ``StandardScaler`` → ``PCA`` when ``pca is None`` (and enough donors carry the chain);
    otherwise reuses the supplied fitted reducer (held-out :meth:`DonorCohort.transform`).
    Returns ``(matrix, fitted_pca_or_None)``.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.full((n, id_pca), np.nan)
    if not means_c:
        return out, pca
    idxs, M = zip(*means_c)
    M = np.stack(M)
    if pca is None:
        if M.shape[0] <= id_pca:
            return out, None                       # too few to reduce; leave holes (imputed)
        pca = make_pipeline(StandardScaler(), PCA(id_pca, random_state=seed)).fit(M)
    Mr = pca.transform(M)
    for j, i in enumerate(idxs):
        out[i] = Mr[j]
    return out, pca


# --------------------------------------------------------------------- DonorCohort


@dataclass
class DonorCohort:
    """A fitted cohort of digital donors: the fused matrix plus everything needed to reproduce it.

    Attributes:
        X: ``(n_donors, width)`` fused, imputed, z-scored feature matrix.
        spec: :class:`~mir.explain.ChannelSpec` — the channel → column map for ``X``.
        spaces: ``{locus: RepertoireSpace}`` — the per-locus clonotype bases (hash-verified on load).
        identity_pca: ``{locus: fitted StandardScaler→PCA}`` — the cross-sample identity reducer,
            stored (not thrown away) so held-out donors project comparably.
        rows: Per-donor metadata, row-aligned to ``X`` (``None`` if not supplied).
        meta: Provenance — per-locus prototype hashes, ``id_pca``, block order, standardization stats.
    """

    X: np.ndarray
    spec: ChannelSpec
    spaces: dict
    identity_pca: dict
    rows: list | None
    meta: dict = field(default_factory=dict)

    def transform(self, donor_frames: list[dict], *, extra: dict | None = None) -> np.ndarray:
        """Project held-out donors into this fitted basis (the only comparable path for new donors).

        Rebuilds the per-locus identity/diversity/coverage blocks through the *stored* spaces and
        identity PCAs, re-adds any ``extra`` channels the caller supplies (same names/widths as at
        fit — the study owns those), then applies the *fit-cohort* impute medians and z-scores.

        Args:
            donor_frames: Per held-out donor, ``{locus: chain clonotype frame}``.
            extra: ``{name: (m, k) array}`` for the non-core channels used at fit (isotype/…). Required
                iff the cohort was fit with ``extra_channels``; the widths must match.

        Returns:
            ``(len(donor_frames), width)`` matrix in the fitted basis.
        """
        n = len(donor_frames)
        means, div, cov = _locus_blocks(self.spaces, donor_frames, min_clones=self.meta["min_clones"])
        blocks: dict[str, np.ndarray] = {}
        for c in self.spaces:
            ident, _ = _identity_matrix(means[c], n, self.meta["id_pca"], pca=self.identity_pca.get(c))
            blocks[f"identity:{c}"] = ident
            blocks[f"diversity:{c}"] = div[c]
            if self.meta["coverage"]:
                blocks[f"coverage:{c}"] = cov[c]
        extra = extra or {}
        need = set(self.meta["extra_names"])
        if set(extra) != need:
            raise ValueError(f"transform needs extra channels {sorted(need)}, got {sorted(extra)}")
        cols = []
        for name, width in self.meta["order"]:
            mat = blocks.get(name, extra.get(name))
            mat = np.asarray(mat, dtype=np.float64)
            mat = mat.reshape(-1, 1) if mat.ndim == 1 else mat
            if mat.shape != (n, width):
                raise ValueError(f"block {name!r} has shape {mat.shape}, expected {(n, width)}")
            cols.append(mat)
        raw = np.hstack(cols)
        med, mu, sd = self.meta["median"], self.meta["mu"], self.meta["sd"]
        for j in range(raw.shape[1]):
            bad = ~np.isfinite(raw[:, j])
            if bad.any():
                raw[bad, j] = med[j]
        return (raw - mu) / sd

    def save(self, path) -> None:
        """Pickle the cohort: ``X``/``spec``/``rows``/``meta``/identity PCAs + each locus's basis."""
        import pickle

        spaces_ser = {c: {"meta": sp.meta, "space": sp.clono.space, "scaler": sp.clono.scaler,
                          "pca": sp.clono.pca, "rff": sp.rff, "rff2": sp.rff2}
                      for c, sp in self.spaces.items()}
        with open(path, "wb") as fh:
            pickle.dump({"X": self.X, "spec": self.spec, "rows": self.rows, "meta": self.meta,
                         "identity_pca": self.identity_pca, "spaces": spaces_ser}, fh)

    @classmethod
    def load(cls, path, *, verify: bool = True) -> "DonorCohort":
        """Load a cohort, rebuilding every locus basis and verifying **all** prototype hashes."""
        import pickle

        from mir.density import DensitySpace
        from mir.embedding.tcremp import TCREmp
        from mir.ml.bundle import prototype_hash

        with open(path, "rb") as fh:
            d = pickle.load(fh)
        spaces = {}
        for c, s in d["spaces"].items():
            m = s["meta"]
            if verify:
                cur = prototype_hash(m["species"], m["locus"], m["n_prototypes"])
                if cur != m["prototype_hash"]:
                    raise ValueError(
                        f"prototype hash mismatch for locus {c} ({m['species']}_{m['locus']}): this "
                        "cohort was fit on a different prototype set — its embeddings are NOT "
                        "comparable to the current prototypes. Pass verify=False to override."
                    )
            model = TCREmp.from_defaults(m["species"], m["locus"], m["n_prototypes"], mode=m["mode"],
                                         metric=m["metric"], gap_positions=tuple(m["gap_positions"]))
            clono = DensitySpace(model=model, space=s["space"], scaler=s["scaler"], pca=s["pca"])
            spaces[c] = RepertoireSpace(clono, s["rff"], s["rff2"], m)
        return cls(d["X"], d["spec"], spaces, d["identity_pca"], d["rows"], d["meta"])


def fit_donor_embeddings(
    spaces: dict,
    donor_frames: list[dict],
    *,
    rows: list | None = None,
    id_pca: int = 8,
    min_clones: int = 5,
    coverage: bool = True,
    extra_channels=None,
    standardize: bool = True,
    impute: bool = True,
    seed: int = 0,
) -> DonorCohort:
    """Fuse per-chain repertoire embeddings into one digital-donor matrix (§T.7).

    For each donor and each fitted locus, computes the kernel-mean identity (cross-sample PCA-reduced),
    Hill diversity and log receptor-load coverage; per-chain blocks of the same kind merge under one
    channel name. Missing chains leave holes that :class:`~mir.explain.ChannelBuilder` imputes.

    Args:
        spaces: ``{locus: RepertoireSpace}`` from :func:`mir.repertoire.fit_repertoire_spaces`.
        donor_frames: Per donor, ``{locus: chain clonotype frame}`` (already canonical-filtered /
            grouped by the study; the library only embeds). Row-aligned to ``rows``.
        rows: Optional per-donor metadata, carried onto the cohort.
        id_pca: Cross-sample identity-PCA dimensionality per locus.
        min_clones: Skip a chain for a donor with fewer clonotypes (its blocks become holes).
        coverage: Include the per-chain log-receptor-load channel.
        extra_channels: Optional ``(rows, identity_concat) -> {name: (n, k) array}`` — the study's own
            blocks (isotype / composition / atypicality …), fused into the *same* matrix. All are
            treated as non-attributable. ``identity_concat`` is the ``(n, Σ id_pca)`` per-chain
            identity, so e.g. :func:`mir.repertoire.centroid_atypicality` can be computed against it.
        standardize / impute: Passed through to the final assembly (z-score / median-impute).
        seed: RNG seed for the identity PCA.

    Returns:
        A :class:`DonorCohort`; ``.X`` / ``.spec`` feed :func:`mir.explain.channel_report`.
    """
    if not spaces:
        raise ValueError("no fitted spaces; fit_repertoire_spaces returned nothing to embed")
    n = len(donor_frames)
    means, div, cov = _locus_blocks(spaces, donor_frames, min_clones=min_clones)

    identity_pca: dict = {}
    ident: dict = {}
    for c in spaces:
        ident[c], pca = _identity_matrix(means[c], n, id_pca, pca=None, seed=seed)
        if pca is not None:
            identity_pca[c] = pca
    identity_concat = np.hstack([ident[c] for c in spaces]) if spaces else np.zeros((n, 0))

    # ordered blocks: per-locus identity(attributable) / diversity / coverage, then study extras
    ordered = []
    for c in spaces:
        ordered.append((f"identity:{c}", ident[c], True))
        ordered.append((f"diversity:{c}", div[c], False))
        if coverage:
            ordered.append((f"coverage:{c}", cov[c], False))
    extra_names = []
    if extra_channels is not None:
        for name, mat in extra_channels(rows, identity_concat).items():
            mat = np.asarray(mat, dtype=np.float64)
            mat = mat.reshape(-1, 1) if mat.ndim == 1 else mat
            ordered.append((name, mat, False))
            extra_names.append(name)

    # merge per-chain blocks under their bare channel name (identity:TRB -> "identity"), as build_embedding does
    b = ChannelBuilder()
    for name, mat, attr in ordered:
        b.add(name.split(":", 1)[0], mat, attributable=attr)
    raw, spec = b.build(standardize=False, impute=False)

    median = np.zeros(raw.shape[1])
    for j in range(raw.shape[1]):
        good = raw[np.isfinite(raw[:, j]), j]
        median[j] = float(np.median(good)) if good.size else 0.0
        if impute:
            bad = ~np.isfinite(raw[:, j])
            raw[bad, j] = median[j]
    mu = raw.mean(axis=0) if standardize else np.zeros(raw.shape[1])
    sd = raw.std(axis=0) if standardize else np.ones(raw.shape[1])
    sd[sd == 0] = 1.0
    X = (raw - mu) / sd

    meta = {
        "prototype_hashes": {c: sp.meta["prototype_hash"] for c, sp in spaces.items()},
        "id_pca": id_pca, "min_clones": min_clones, "coverage": coverage,
        "order": [(name, mat.shape[1]) for name, mat, _ in ordered],
        "extra_names": extra_names,
        "median": median, "mu": mu, "sd": sd, "standardize": standardize, "impute": impute,
    }
    return DonorCohort(X=X, spec=spec, spaces=spaces, identity_pca=identity_pca, rows=rows, meta=meta)


# --------------------------------------------------------------- cohort operations


def residualize(X: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Subtract each group's mean vector — remove the first-order batch offset (Prop. ``prop:batch``).

    The detect→correct→verify batch cookbook's correction step, applied to a stacked donor matrix.
    NB the corrected matrix is a *different* coordinate system: never compare a residualized ``X`` to
    an uncorrected one.
    """
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for g in np.unique(group):
        m = group == g
        out[m] -= X[m].mean(axis=0)
    return out


def cluster_samples(embs, *, unbiased: bool = True, method: str = "dbscan",
                    eps: float | None = None, min_samples: int = 3, k: int = 4, **kwargs):
    """Cluster repertoires by MMD — the sample-level analog of ``cluster_samples`` (TME states etc.).

    Builds the MMD distance matrix (unbiased by default, since cohorts differ in depth) and clusters
    it with a precomputed-metric density estimator (:func:`mir.bench.metrics.cluster`).

    Args:
        embs: Per-sample :class:`~mir.repertoire.SampleEmbedding` (mean block present).
        unbiased: Use the diagonal-removed MMD² (recommended for unequal-depth cohorts).
        method: ``"dbscan"`` / ``"hdbscan"`` / ``"optics"``.
        eps: DBSCAN radius; ``None`` → median of each sample's ``k``-th nearest MMD (self excluded).
        min_samples / k: density / eps-estimation neighbours.
        **kwargs: forwarded to the estimator.

    Returns:
        Cluster labels (``-1`` = noise), length ``len(embs)``.
    """
    from mir.bench.metrics import cluster

    D = mmd_matrix(embs, unbiased=unbiased)
    if method == "dbscan" and eps is None:
        kth = np.sort(D, axis=1)[:, min(k, D.shape[0] - 1)]     # k-th nearest (row 0 is self)
        eps = float(np.median(kth)) or 1.0
    return cluster(D, eps=eps, min_samples=min_samples, method=method, metric="precomputed", **kwargs)


def incidence_biomarkers(cohort, phenotype, *, pheno_col: str, match: str = "1mm",
                         min_incidence: int = 3, **kwargs):
    """Cohort biomarker detection: per-clonotype **subject-incidence Fisher** test (Emerson 2017).

    A thin delegate to :func:`vdjtools.biomarker.fisher.fisher_association` — the presence/absence
    biomarker that complements the geometry witness (:func:`mir.repertoire.class_witness`) and, at
    realistic donor counts, recovers public motifs the witness misses. ``match="1mm"`` groups
    single-mismatch metaclonotypes (the paper's method; exact-match typically returns ~0 hits).

    Args:
        cohort: Pooled clonotype frame with a sample-id column (vdjtools cohort form).
        phenotype: Per-sample frame carrying the binary ``pheno_col`` label.
        pheno_col: Phenotype column name in ``phenotype``.
        match: ``"1mm"`` (metaclonotypes) or ``"exact"``.
        min_incidence: Minimum subject incidence to test a feature.
        **kwargs: forwarded to ``fisher_association`` (``key``, ``scope``, ``alternative``, …).

    Returns:
        One row per feature: incidence, odds-ratio, ``p_value``, BH ``q_value``, direction.
    """
    from vdjtools.biomarker.fisher import fisher_association

    return fisher_association(cohort, phenotype, pheno_col=pheno_col, match=match,
                              min_incidence=min_incidence, **kwargs)


# --------------------------------------------------------------------- self-check


def _demo() -> None:
    """Self-check on bundled prototypes: a multi-chain cohort fuses, serializes, and a planted
    prognostic channel wins the report."""
    from mir.embedding.prototypes import list_available_prototypes
    from mir.embedding.tcremp import TCREmp
    from mir.repertoire import centroid_atypicality, fit_repertoire_spaces

    avail = [loc for sp, loc in list_available_prototypes() if sp == "human"]
    loci = [c for c in ("TRB", "TRA", "IGH") if c in avail][:2] or [avail[0]]
    models = {c: TCREmp.from_defaults("human", c, n_prototypes=300) for c in loci}
    protos = {c: pl.DataFrame({"v_call": m._proto_v, "j_call": m._proto_j,
                               "junction_aa": m._proto_junction}).unique() for c, m in models.items()}
    spaces = fit_repertoire_spaces(models, protos, n_rff=512, n_rff_second=0, n_components=15, seed=0)

    # 40 donors in two "risk" groups: group 1 gets a public expansion in the identity space
    rng = np.random.default_rng(0)
    donor_frames, rows = [], []
    for i in range(40):
        grp = i % 2
        frames = {}
        for c in spaces:
            base = protos[c].sample(120, seed=i, with_replacement=True).with_columns(
                pl.lit(1.0).alias(_COUNT))
            if grp == 1:
                spike = protos[c].slice(0, 5).with_columns(pl.lit(400.0).alias(_COUNT))
                base = pl.concat([base, spike])
            frames[c] = base
        donor_frames.append(frames)
        rows.append({"group": grp})

    def extra(rows, identity):
        aty = centroid_atypicality(identity, np.array([r["group"] for r in rows]))
        return {"atypicality": aty}

    coh = fit_donor_embeddings(spaces, donor_frames, rows=rows, id_pca=6, extra_channels=extra, seed=0)
    assert coh.X.shape[0] == 40
    assert "identity" in coh.spec and "atypicality" in coh.spec
    assert "identity" in coh.spec.attributable

    # the identity channel separates the two groups; a channel_report with a CV-AUC scorer ranks it top
    from mir.bench.eval import cv_auc
    from mir.explain import channel_report

    y = np.array([r["group"] for r in coh.rows], dtype=float)
    rep = channel_report(coh.X, coh.spec, lambda B: cv_auc(B, y, n_repeats=3, pca_cols=B.shape[1])[0],
                         base=0.5, mode="in")
    assert rep.best == "identity", rep.frame()

    # transform held-out donors + serialize with hash verification
    import tempfile, os
    held = donor_frames[:4]
    Xt = coh.transform(held, extra={"atypicality": np.zeros(4)})
    assert Xt.shape == (4, coh.X.shape[1])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cohort.pkl")
        coh.save(p)
        back = DonorCohort.load(p)
        assert np.allclose(back.X, coh.X)
        assert back.spec.names == coh.spec.names

    # cluster_samples on the two-group embeddings
    embs = [sample_embedding(spaces[loci[0]], f[loci[0]], blocks=("mean",)) for f in donor_frames]
    labels = cluster_samples(embs)
    print(f"mir.cohort OK; donors {coh.X.shape[0]} x {coh.X.shape[1]}, channels {coh.spec.names}; "
          f"best={rep.best}; clusters {len(set(labels)) - (1 if -1 in labels else 0)}")


if __name__ == "__main__":
    _demo()
