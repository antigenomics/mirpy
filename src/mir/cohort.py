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


# ------------------------------------------------------------- multi-locus alignment


@dataclass
class LocusAlignment:
    """Per-locus matrices aligned onto one sample axis, with holes where a locus is missing.

    Attributes:
        ids: The sample ids, sorted — the row order of every block and of ``mask``.
        blocks: ``{locus: (n_samples, k) array}``; rows for samples lacking that locus are ``nan``.
        mask: ``(n_samples, n_loci)`` boolean presence, column order ``loci``.
        loci: Locus names, in the order given.
    """

    ids: list
    blocks: dict
    mask: np.ndarray
    loci: list

    @property
    def n_loci(self) -> np.ndarray:
        """``(n_samples,)`` count of loci present per sample."""
        return self.mask.sum(axis=1)

    @property
    def pattern(self) -> np.ndarray:
        """``(n_samples,)`` presence pattern as a string, e.g. ``"1101100"`` — the missingness stratum."""
        return np.array(["".join("1" if v else "0" for v in r) for r in self.mask])

    def build(self, *, standardize: bool = True, attributable: bool = True):
        """Fuse to ``(X, spec)`` through :class:`~mir.explain.ChannelBuilder`, one channel per locus.

        The holes are imputed and standardized by the builder, which takes both statistics from
        **observed** entries — so a locus present in a third of samples is not upweighted by its own
        sparsity, and a hole lands at exactly 0.

        Args:
            standardize: Z-score each column on its observed entries.
            attributable: Mark the per-locus blocks clonotype-attributable (they are kernel means).

        Returns:
            ``(X, spec)``.
        """
        b = ChannelBuilder()
        for c in self.loci:
            b.add(c, self.blocks[c], attributable=attributable)
        return b.build(standardize=standardize, impute=True)

    def missingness_report(self, labels) -> dict:
        """Whether a grouping tracks *which loci a sample has* rather than what is in them.

        See :func:`missingness_report`.
        """
        return missingness_report(labels, self.mask)


def align_loci(blocks: dict, *, how: str = "union", require: list | None = None,
               min_loci: int = 1) -> LocusAlignment:
    """Align per-locus embedding matrices, keyed by sample id, onto one sample axis.

    The step between "one matrix per locus, each over its own samples" and "one matrix over one
    sample set". Loci are sequenced to different depths, so their sample sets differ — and an
    **inner join across seven loci is bound by the thinnest two**. Measured in a 62,293-sample
    tissue cohort: intersecting all seven left 19,346 samples (31%), while the union keeps every
    one. That is complete-case deletion wearing a join's clothes, so ``how="union"`` is the default.

    A missing locus becomes ``nan`` rows, which is the hole convention
    :class:`~mir.explain.ChannelBuilder` already understands: it imputes them to the observed centre
    and standardizes on observed entries, so an absent locus contributes no information and does not
    set any channel's scale.

    **Do not zero-fill before calling this.** A literal 0 is a value — after a naive global z-score
    it becomes ``-mean/std``, a large shared constant that every sample missing that locus carries,
    and the cohort then clusters by coverage. (Measured: cluster-vs-locus-count AMI 0.741 and
    read-depth eta-squared 0.42 cohort-wide against 0.01 inside the fully-observed subset.) ``nan``
    says "not observed"; ``0`` says "observed to be zero".

    Args:
        blocks: ``{locus: (ids, matrix)}`` — ``ids`` a sequence of sample ids, ``matrix`` an
            ``(len(ids), k)`` array. Per-locus ``k`` may differ.
        how: ``"union"`` keeps every sample seen in any locus (holes where absent); ``"inner"``
            keeps only samples present in all of them. ``"inner"`` exists to be compared against,
            not as a default.
        require: Loci a sample must have to be kept at all — for when one chain genuinely carries
            the question and the others are context.
        min_loci: Drop samples carrying fewer than this many of the loci.

    Returns:
        A :class:`LocusAlignment`.

    Raises:
        ValueError: If ``blocks`` is empty, a matrix's row count disagrees with its ids, ``how`` is
            not recognised, ``require`` names an absent locus, or nothing survives the filters.

    Example:
        >>> al = align_loci({"TRB": (trb_ids, TRB), "IGH": (igh_ids, IGH)})   # doctest: +SKIP
        >>> X, spec = al.build()                                             # doctest: +SKIP
        >>> al.missingness_report(clusters)["ami_vs_pattern"] < 0.1          # doctest: +SKIP
        True
    """
    if not blocks:
        raise ValueError("no blocks to align")
    if how not in ("union", "inner"):
        raise ValueError(f"how must be 'union' or 'inner', got {how!r}")
    loci = list(blocks)
    if require:
        missing = [c for c in require if c not in blocks]
        if missing:
            raise ValueError(f"require names loci not in blocks: {missing}")

    pos = {}
    for c in loci:
        ids_c, mat = blocks[c]
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
        if mat.shape[0] != len(ids_c):
            raise ValueError(f"locus {c!r}: {mat.shape[0]} rows for {len(ids_c)} ids")
        pos[c] = ({sid: i for i, sid in enumerate(ids_c)}, mat)

    sets = [set(pos[c][0]) for c in loci]
    keep = set.union(*sets) if how == "union" else set.intersection(*sets)
    for c in require or []:
        keep &= set(pos[c][0])
    ids = sorted(keep)
    if not ids:
        raise ValueError(f"no samples survive how={how!r} require={require!r}")

    mask = np.zeros((len(ids), len(loci)), dtype=bool)
    out = {}
    for j, c in enumerate(loci):
        p, mat = pos[c]
        rows = np.fromiter((p.get(sid, -1) for sid in ids), dtype=np.int64, count=len(ids))
        m = rows >= 0
        blk = np.full((len(ids), mat.shape[1]), np.nan)
        blk[m] = mat[rows[m]]
        out[c] = blk
        mask[:, j] = m

    if min_loci > 1:
        sel = mask.sum(axis=1) >= min_loci
        if not sel.any():
            raise ValueError(f"no samples carry min_loci={min_loci} of {len(loci)} loci")
        ids = [s for s, k in zip(ids, sel) if k]
        out = {c: v[sel] for c, v in out.items()}
        mask = mask[sel]
    return LocusAlignment(ids=ids, blocks=out, mask=mask, loci=loci)


def missingness_report(labels, mask) -> dict:
    """How much of a grouping is explained by *which* blocks each sample has.

    A cohort assembled from partly-observed blocks can cluster by coverage rather than by content,
    and nothing in ``X`` says which happened. This is the check to run before believing a
    clustering, an enrichment or a trajectory built on holed data.

    Args:
        labels: ``(n_samples,)`` cluster or group assignment to test.
        mask: ``(n_samples, n_blocks)`` boolean presence, e.g. :attr:`LocusAlignment.mask`.

    Returns:
        ``ami_vs_pattern`` — adjusted mutual information between ``labels`` and the presence
        pattern; ``eta2_n_present`` — correlation ratio of the per-sample present-block count;
        ``n_patterns``; ``mean_present``. Near zero on the first two is what you want. High means
        the grouping is a missingness stratification: re-read it inside a fully-observed subset
        before reporting it as biology.

    Example:
        >>> missingness_report(clusters, al.mask)["ami_vs_pattern"]   # doctest: +SKIP
        0.041
    """
    from sklearn.metrics import adjusted_mutual_info_score

    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels)
    if len(labels) != mask.shape[0]:
        raise ValueError(f"{len(labels)} labels for {mask.shape[0]} rows")
    pat = np.array(["".join("1" if v else "0" for v in r) for r in mask])
    n_pres = mask.sum(axis=1).astype(float)

    gi, g = np.unique(labels, return_inverse=True)
    cnt = np.bincount(g, minlength=len(gi))
    tot = np.zeros(len(gi))
    np.add.at(tot, g, n_pres)
    gm = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    ss_tot = float(((n_pres - n_pres.mean()) ** 2).sum())
    return {
        "ami_vs_pattern": (float(adjusted_mutual_info_score(pat, labels))
                           if len(set(pat.tolist())) > 1 else 0.0),
        "eta2_n_present": (float((cnt * (gm - n_pres.mean()) ** 2).sum() / ss_tot)
                           if ss_tot > 0 else 0.0),
        "n_patterns": int(len(set(pat.tolist()))),
        "mean_present": float(mask.mean()),
    }


def depth_report(X, stats: dict, *, n_pc: int = 5, subset=None) -> dict:
    """How much of an embedding's **leading geometry** is explained by sampling depth.

    The companion to :func:`missingness_report`: that one asks whether a grouping tracks *which*
    blocks a sample has, this one whether the embedding's dominant axes track *how deeply* it was
    sequenced. Each of the top ``n_pc`` principal components is regressed on the sampling fingerprint
    (:func:`mir.repertoire.cohort_statistics`) and reported as R².

    Report the leading block, not only PC1 — a correction can move depth off PC1 and leave it in PC2.
    Measured with the deficient measure (:func:`mir.repertoire.contrast_embedding`), R²(PC1, depth)
    fell 0.259 → 0.001 and best-of-PC1–5 0.253 → 0.047, while PC1's *explained variance* was
    unchanged, which is what distinguishes "a different direction" from "a collapsed one" — so read
    ``explained_variance`` alongside the R².

    Args:
        X: ``(n_samples, d)`` embedding matrix (uncentred is fine; PCA centres internally).
        stats: ``{name: (n_samples,) values}`` sampling fingerprint. Non-finite entries are
            mean-filled so one missing library size cannot drop a sample.
        n_pc: Number of leading components to test.
        subset: Optional boolean mask of a *trusted* subset (e.g. samples above
            :func:`mir.repertoire.depth_threshold`'s ``κ``). When given, the same statistics are
            recomputed inside it and returned under ``*_subset``: high overall and low within the
            subset means the structure was between coverage strata.

    Returns:
        ``r2_pc1``, ``r2_best``, ``pc_best`` (1-indexed), ``r2_by_pc``, ``explained_variance``
        (per-PC ratio), ``n_pc``, ``stats`` (names, in fit order), ``n_samples``,
        ``residual_dof`` — plus ``r2_pc1_subset`` / ``r2_best_subset`` / ``n_samples_subset``
        when ``subset`` is given.

    Note:
        Needs meaningfully more samples than statistics. The regression uses an intercept plus one
        column per statistic, so it saturates (R² ≡ 1 on any input, including pure noise) at
        ``len(stats) + 1`` samples; with the 12 :func:`cohort_statistics` that is 13 samples. Below
        one residual degree of freedom the R²s come back ``nan`` with a warning, and below five
        they are warned as upper bounds. ``subset`` is smaller still by construction, so it
        saturates first — check ``residual_dof`` before reading ``r2_*_subset``.

    Raises:
        ValueError: If ``stats`` is empty or a statistic's length disagrees with ``X``.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if not stats:
        raise ValueError("stats is empty: nothing to test the geometry against")
    names = list(stats)
    S = np.empty((n, len(names)))
    for j, k in enumerate(names):
        v = np.asarray(stats[k], dtype=np.float64).ravel()
        if v.size != n:
            raise ValueError(f"stat {k!r} has {v.size} values for {n} samples")
        bad = ~np.isfinite(v)
        if bad.any():
            v = v.copy()
            v[bad] = v[~bad].mean() if (~bad).any() else 0.0
        S[:, j] = v

    # An intercept plus one column per statistic. With len(names) statistics the design saturates
    # at len(names)+1 rows, and a saturated least-squares fit reproduces ANY y exactly — so a
    # 12-sample cohort against the 12 cohort_statistics reports r2 = 1.0 on pure noise, i.e. a
    # confident, entirely false "your embedding is depth-driven". Refuse below one residual dof.
    n_params = len(names) + 1
    _MIN_RESIDUAL_DOF = 5   # below this R² is badly upward-biased even though it is computable

    def r2s(rows):
        n_rows = int(rows.sum())
        dof = n_rows - n_params
        if dof < 1:
            import warnings

            warnings.warn(
                f"depth_report: {n_rows} samples against {len(names)} statistics leaves no residual "
                "degrees of freedom, so every R² would be exactly 1 regardless of the data. "
                "Reporting nan — use more samples or fewer statistics.", stacklevel=3)
            return np.array([]), np.array([])
        if dof < _MIN_RESIDUAL_DOF:
            import warnings

            warnings.warn(
                f"depth_report: only {dof} residual degrees of freedom ({n_rows} samples, "
                f"{len(names)} statistics); R² is strongly upward-biased here — read it as an "
                "upper bound, not an estimate.", stacklevel=3)
        k = min(n_pc, n_rows - 1, X.shape[1])
        if k < 1:
            return np.array([]), np.array([])
        pca = PCA(n_components=k, random_state=0).fit(X[rows])
        P = pca.transform(X[rows])
        A = np.column_stack([np.ones(int(rows.sum())), S[rows]])
        out = []
        for j in range(k):
            y = P[:, j]
            res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            ss = float(((y - y.mean()) ** 2).sum())
            out.append(1.0 - float(res @ res) / ss if ss > 0 else 0.0)
        return np.array(out), pca.explained_variance_ratio_

    r2, ev = r2s(np.ones(n, dtype=bool))
    rep = {
        "r2_pc1": float(r2[0]) if r2.size else float("nan"),
        "r2_best": float(r2.max()) if r2.size else float("nan"),
        "pc_best": int(np.argmax(r2) + 1) if r2.size else 0,
        "r2_by_pc": [float(v) for v in r2],
        "explained_variance": [float(v) for v in ev],
        "n_pc": int(r2.size),
        "stats": names,
        "n_samples": n,
        # how much to trust the R² above: n - (1 intercept + one column per statistic)
        "residual_dof": int(n - n_params),
    }
    if subset is not None:
        sub = np.asarray(subset, dtype=bool)
        if sub.shape != (n,):
            raise ValueError(f"subset has shape {sub.shape}, expected ({n},)")
        r2b, _ = r2s(sub)
        rep["r2_pc1_subset"] = float(r2b[0]) if r2b.size else float("nan")
        rep["r2_best_subset"] = float(r2b.max()) if r2b.size else float("nan")
        rep["n_samples_subset"] = int(sub.sum())
    return rep


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
            import warnings

            # all-nan → imputed to the column median → an all-zero, information-free channel.
            # Loud, because the identity block is the point of the digital donor.
            warnings.warn(
                f"only {M.shape[0]} donors carry this chain but id_pca={id_pca}: the identity "
                "channel is left empty (imputed to a constant). Lower id_pca or drop the chain.",
                stacklevel=3)
            return out, None                       # too few to reduce; leave holes (imputed)
        pca = make_pipeline(StandardScaler(), PCA(id_pca, random_state=seed)).fit(M)
    Mr = pca.transform(M)
    for j, i in enumerate(idxs):
        out[i] = Mr[j]
    return out, pca


def _block_starts(ordered):
    """``[(name, first_column_index)]`` for the assembled block order.

    The observed pattern is per BLOCK, not per column: a donor either has a chain or does not, so
    768 identical flags per locus would be noise in the string and in the AMI.
    """
    out, c = [], 0
    for name, mat, _ in ordered:
        out.append((name, c))
        c += mat.shape[1]
    return out


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

    def missingness_report(self, labels) -> dict:
        """Whether a grouping tracks which chains each donor has, rather than what is in them.

        See :func:`missingness_report`; the presence matrix is the one recorded at fit, one column
        per assembled block.

        Args:
            labels: ``(n_donors,)`` cluster or group assignment to test.

        Returns:
            ``{ami_vs_pattern, eta2_n_present, n_patterns, mean_present}``.

        Raises:
            ValueError: If the cohort predates block-presence recording.

        Example:
            >>> cohort.missingness_report(clusters)["ami_vs_pattern"] < 0.1   # doctest: +SKIP
            True
        """
        mask = self.meta.get("observed_blocks")
        if mask is None:
            raise ValueError("this cohort was built before block-presence was recorded; "
                             "call mir.cohort.missingness_report(labels, mask) directly")
        return missingness_report(labels, mask)

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
            pca = self.identity_pca.get(c)
            if pca is None:
                # The fit cohort had too few donors on this chain to reduce, so its identity block
                # is holes (imputed to a constant). Fitting a reducer *here* would put held-out
                # donors in a basis the fit cohort never saw — unscaled scores in a matrix whose
                # every other column is unit-variance — so keep the holes instead.
                ident = np.full((n, self.meta["id_pca"]), np.nan)
            else:
                ident, _ = _identity_matrix(means[c], n, self.meta["id_pca"], pca=pca)
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
        # the FIT cohort's fill/mu/sd verbatim -- a held-out donor must land in the same basis,
        # and its own holes must land at that basis's centre, not at this batch's median
        fill = self.meta.get("fill", self.meta["median"])
        mu, sd = self.meta["mu"], self.meta["sd"]
        for j in range(raw.shape[1]):
            bad = ~np.isfinite(raw[:, j])
            if bad.any():
                raw[bad, j] = fill[j]
        return (raw - mu) / sd

    def save(self, path) -> None:
        """Pickle the cohort: ``X``/``spec``/``rows``/``meta``/identity PCAs + each locus's basis.

        Raises:
            ValueError: If any locus basis uses a non-default ``matrix`` / ``alignment`` — ``load``
                could not rebuild that coordinate system (see
                :func:`mir.repertoire._check_rebuildable`).
        """
        import pickle

        from mir.repertoire import _check_rebuildable

        for sp in self.spaces.values():
            _check_rebuildable(sp.clono.model)
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
            replicate = m.get("replicate", 0)   # .get: cohorts predating replicates are draw 0
            if verify:
                cur = prototype_hash(m["species"], m["locus"], m["n_prototypes"], replicate)
                if cur != m["prototype_hash"]:
                    raise ValueError(
                        f"prototype hash mismatch for locus {c} ({m['species']}_{m['locus']}): this "
                        "cohort was fit on a different prototype set — its embeddings are NOT "
                        "comparable to the current prototypes. Pass verify=False to override."
                    )
            model = TCREmp.from_defaults(m["species"], m["locus"], m["n_prototypes"], mode=m["mode"],
                                         replicate=replicate,
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

    # Statistics from OBSERVED entries only, computed before imputation. Imputing first and then
    # taking mean/std over the filled matrix deflates sd in proportion to how sparse the column is,
    # so a chain present in 30% of donors has its real values scaled up ~1.8x against a
    # fully-covered one -- the least-observed locus ends up dominating every distance and PCA. The
    # same stats are stored in meta and reused verbatim by transform(), so held-out donors project
    # into the same basis.
    observed = np.isfinite(raw)
    median = np.zeros(raw.shape[1])
    mu = np.zeros(raw.shape[1])
    sd = np.ones(raw.shape[1])
    for j in range(raw.shape[1]):
        good = raw[observed[:, j], j]
        median[j] = float(np.median(good)) if good.size else 0.0
        if standardize and good.size:
            mu[j] = float(good.mean())
            s = float(good.std())
            sd[j] = s if s > 0 else 1.0
    # fill at whatever the column is centred on, so a donor's missing chain lands at exactly 0 --
    # no information, rather than a shared offset every donor lacking that chain carries
    fill = mu if standardize else median
    if impute:
        for j in range(raw.shape[1]):
            raw[~observed[:, j], j] = fill[j]
    X = (raw - mu) / sd

    # Per-donor observed fraction, so a caller can ask whether a result tracks coverage rather than
    # repertoire -- see DonorCohort.missingness_report().
    observed_frac = observed.mean(axis=1)

    meta = {
        "prototype_hashes": {c: sp.meta["prototype_hash"] for c, sp in spaces.items()},
        "id_pca": id_pca, "min_clones": min_clones, "coverage": coverage,
        "order": [(name, mat.shape[1]) for name, mat, _ in ordered],
        "extra_names": extra_names,
        "median": median, "fill": fill, "mu": mu, "sd": sd,
        "standardize": standardize, "impute": impute,
        # one presence flag per BLOCK, not per column: a donor either has a chain or does not, so
        # 768 identical flags per locus would be noise in the AMI
        "observed_blocks": observed[:, [c0 for _, c0 in _block_starts(ordered)]],
        "observed_frac": observed_frac,
    }
    return DonorCohort(X=X, spec=spec, spaces=spaces, identity_pca=identity_pca, rows=rows, meta=meta)


# --------------------------------------------------------------- cohort operations


def residualize(X: np.ndarray, group: np.ndarray, *, shrink: bool = False) -> np.ndarray:
    """Subtract each group's mean vector — remove the first-order batch offset (Prop. ``prop:batch``).

    The detect→correct→verify batch cookbook's correction step, applied to a stacked donor matrix.
    NB the corrected matrix is a *different* coordinate system: never compare a residualized ``X`` to
    an uncorrected one.

    ⚠ **The plain correction can make the batch easier to read, not harder**, and this is measured
    rather than hypothetical: evaluated out-of-sample (leave-one-batch-out for any shared basis, plus
    a donor-level split *inside* each batch), batch-identity AUC went 0.863 raw → **0.985** after
    per-group centring, and 0.978 after ComBat. The mechanism is estimation error, not a bug — with a
    mean fitted from as few as 8–15 samples in ~1,280 dimensions, ``‖µ̂ − µ‖ ≈ √(σ²d/n)`` was ≈16
    against a true offset of norm 7–24, so subtracting it **injects a batch-constant vector as large
    as the one it removes**. In-sample this vanishes by construction, which is why only an
    out-of-sample evaluation sees it. ``shrink=True`` (positive-part James–Stein) recovered most of
    the damage in the same experiment: 0.985 → **0.889**.

    Args:
        X: ``(n_samples, n_features)`` stacked matrix.
        group: Length-``n_samples`` batch label per row.
        shrink: Scale each group's offset by the positive-part James–Stein factor
            ``max(0, 1 − (d−2)·(σ̂²/n_g)/‖µ̂_g‖²)`` — where ``σ̂²`` is the pooled within-group
            per-coordinate variance — so a group whose apparent offset is no bigger than its own
            estimation error is left alone. Recommended whenever a group has few samples relative to
            the feature count; ``False`` keeps the exact previous behaviour.

    Returns:
        The corrected ``(n_samples, n_features)`` matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    grand = X.mean(axis=0)
    d = X.shape[1]
    if shrink:
        # pooled within-group variance per coordinate, averaged over coordinates
        ss = dof = 0.0
        for g in np.unique(group):
            m = group == g
            if m.sum() > 1:
                ss += float(np.sum((X[m] - X[m].mean(axis=0)) ** 2))
                dof += (m.sum() - 1) * d
        sigma2 = ss / dof if dof > 0 else 0.0
    for g in np.unique(group):
        m = group == g
        offset = X[m].mean(axis=0) - (grand if shrink else 0.0)
        if shrink:
            sq = float(offset @ offset)
            c = max(0.0, 1.0 - (d - 2) * (sigma2 / max(int(m.sum()), 1)) / sq) if sq > 0 else 0.0
            offset = c * offset + grand      # shrink the group effect, keep the grand mean
        out[m] -= offset
    return out


def cluster_samples(embs, *, unbiased: bool = True, method: str = "dbscan",
                    eps: float | None = None, min_samples: int = 3, k: int = 4, **kwargs):
    """Cluster repertoires by MMD — the repertoire-level analog of :func:`mir.bench.metrics.cluster` (TME states etc.).

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
    import os
    import tempfile
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
