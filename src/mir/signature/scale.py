"""The corpus-fitted half of the reference: per-column location and scale.

The geometry is fit-free — a rotation and a naive reference derived from bundled resources, with
no cohort involved. This module owns the one part that *must* come from data, because there is no
other way to know it: where each column typically sits and how far it typically moves.

That is deliberately the cheap part to estimate. A rotation over ``p=256`` coordinates is not
identified at a thousand samples, which is why none is fitted here; a per-column median and MAD
are identified at any ``n``, converge as ``1/sqrt(n)``, and are what the whole re-fit story rests
on. Fitting a scale is a different statistical problem from fitting a basis, and only one of them
is safe at the sample sizes anyone actually has.

Three rules the estimator follows, each because the alternative silently corrupts something:

* **Robust, not moment-based.** Median and ``1.4826·MAD``. A handful of pathological repertoires
  in a reference corpus would otherwise set the scale for everyone.
* **Observed entries only, before any imputation.** Filling holes first and measuring afterwards
  deflates the scale in proportion to how sparse a column is, so the least-observed locus ends up
  with the largest apparent values and dominates every distance and principal component.
* **Refuse a column the corpus barely saw.** A location fitted on nine samples is not a reference.
  Below ``min_n_obs`` the column ships as "unscaled" and passes through, rather than carrying a
  confident-looking number derived from almost nothing.

It also measures the two constants the statistics half cannot pick for itself: the per-locus
coverage level ``cstar`` at which Hill numbers are compared, and the Pgen quantile below which a
clonotype counts as atypical. Both are quantiles of what the corpus actually attains, not values
chosen by taste — a textbook ``C* = 0.95`` puts every real repertoire into extrapolation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Bundled alongside the geometry artifact.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "resources" / "signature" / "rsig_scale_v2.npz"

#: A column observed fewer times than this ships unscaled. A reference is a claim about a
#: population; a hundred samples cannot support one for 716 columns.
MIN_N_OBS = 1000

#: Coverage level, as a quantile of what the reference corpus attains. Low on purpose: real
#: repertoires reach Good-Turing coverage 0.24-0.58, so anything near the textbook 0.95 forces
#: every sample into extrapolation, where diversity inflates roughly tenfold.
CSTAR_QUANTILE = 0.10

#: Attained coverage at or above this is treated as "no singleton tail", not as deep sequencing.
#: Real repertoires reach 0.24-0.58; a value here means the input was truncated or pre-collapsed.
COVERAGE_CEILING = 0.99


@dataclass(frozen=True)
class ScaleReference:
    """Per-column location and scale, plus the constants the blocks need."""

    columns: list[str]
    loc: np.ndarray             # (n_cols,)
    scale: np.ndarray           # (n_cols,) — 0 means "never established", pass through
    n_obs: np.ndarray           # (n_cols,) observed samples behind each estimate
    cstar: dict[str, float]
    pgen_q05: dict[str, float]
    meta: dict
    #: Between-corpus spread of the column's median over within-corpus spread, measured on the
    #: reference draw. Above 1 the column separates *cohorts* better than it separates donors
    #: within one. Diagnostic only — nothing here divides by it — but it is the number that says
    #: which columns a cross-cohort model should treat as nuisance. ``nan`` where unmeasured.
    batch_ratio: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Built once here rather than cached on the method: the dataclass holds a list, so it is
        # not hashable and lru_cache cannot key on self.
        object.__setattr__(self, "_idx", {c: i for i, c in enumerate(self.columns)})

    @property
    def scaled(self) -> np.ndarray:
        """Mask of columns with a usable scale."""
        return self.scale > 0

    def apply(self, values: dict[str, float], *, clip: float = 8.0) -> dict[str, float]:
        """Rescale one sample's columns against the reference.

        Unknown columns and columns without an established scale pass through untouched, so a
        caller always gets back exactly the keys it handed in. A hole stays a hole: ``nan`` is
        not something to centre.

        Args:
            values: ``{column: value}``, e.g. from :func:`mir.signature.signature`.
            clip: Bound in robust standard deviations. Wide enough that a genuine outlier stays
                one, narrow enough that a single pathological sample cannot set a model's scale.
        """
        idx = self._idx
        out = {}
        for k, v in values.items():
            i = idx.get(k)
            if i is None or self.scale[i] <= 0 or not np.isfinite(v):
                out[k] = v
                continue
            out[k] = float(np.clip((v - self.loc[i]) / self.scale[i], -clip, clip))
        return out

    @property
    def batch_dominated(self) -> np.ndarray:
        """Mask of columns that separate reference corpora better than donors within one.

        Not a defect of the scaling — a property of the feature. Standardising cannot remove a
        batch effect, and a cross-cohort model should treat these as nuisance or residualise them.
        All ``False`` when no grouping was supplied at fit time.
        """
        if self.batch_ratio is None:
            return np.zeros(len(self.columns), dtype=bool)
        return np.nan_to_num(self.batch_ratio, nan=0.0) > 1.0

    def report(self) -> dict:
        """What this reference can and cannot standardise."""
        out = {"columns": len(self.columns), "scaled": int(self.scaled.sum()),
               "unscaled": int((~self.scaled).sum()),
               "median_n_obs": int(np.median(self.n_obs)),
               "loci_with_cstar": sorted(self.cstar), "min_n_obs": self.meta.get("min_n_obs")}
        if self.batch_ratio is not None:
            # Both counts, always. `batch_dominated` is all-False when the ratio could not be
            # measured at all — every group below the floor, which is what a reference drawn a
            # handful of samples per study looks like — and 0-of-0 reads exactly like 0-of-1403,
            # i.e. "we checked and this reference is clean". Reporting the denominator is the
            # difference between a measurement and a silence.
            out["batch_measured"] = int(np.isfinite(self.batch_ratio).sum())
            out["batch_dominated"] = int(self.batch_dominated.sum())
        return out


def fit_scale(frame, *, min_n_obs: int = MIN_N_OBS, cstar: dict | None = None,
              pgen_q05: dict | None = None, group=None,
              meta: dict | None = None) -> ScaleReference:
    """Fit location and scale from an assembled cohort matrix.

    The estimator is the median and ``1.4826·MAD``, and that is a measured choice rather than a
    stylistic one. Against the full-corpus fit on 4,080 real samples, the fraction of columns whose
    location lands within 0.10 scales *and* whose scale lands within ±10%:

    ===========  =======  =======  =======  =======
    estimator      N=250    N=500   N=1000   N=2000
    ===========  =======  =======  =======  =======
    mean / sd      0.362    0.386    0.576    0.841
    median / MAD   0.669    0.908    0.992    0.999
    ===========  =======  =======  =======  =======

    The moment estimator never gets there, and the half that fails is the *scale*: the columns are
    heavy-tailed after their block transform (excess kurtosis 0.6-207, and 0.4-3% of samples beyond
    five robust deviations against the 6e-7 a normal would give), so a standard deviation is set by
    a handful of samples and moves when they do. Median/MAD converges at N=1000, which is what puts
    the "re-fittable from 1,000-5,000 samples" claim on a footing.

    Args:
        frame: A ``pl.DataFrame`` from :func:`mir.signature.signature_cohort` — one row per
            sample, ``sample_id`` plus signature columns.
        min_n_obs: Columns observed fewer times than this get no scale.
        cstar / pgen_q05: Measured constants to carry alongside (see :func:`measure_constants`).
        group: Optional per-sample corpus/batch label (a column name in ``frame``, or a sequence).
            Supplying it records ``batch_ratio`` per column — diagnostic only, nothing divides by
            it — so a downstream user can see which columns separate cohorts rather than donors.
        meta: Provenance recorded into the artifact.

    Returns:
        A :class:`ScaleReference`.

    Raises:
        ValueError: If the frame has no signature columns.
    """
    labels = None
    if group is not None:
        labels = np.asarray(frame[group].to_list() if isinstance(group, str) else list(group))
        if labels.size != frame.height:
            raise ValueError(f"group has {labels.size} labels for {frame.height} samples")
    # Select by the column contract, not by excluding the names we happen to know about. An
    # emitted frame carries `dataset`, `loci` and whatever else the caller joined on; a
    # deny-list keeps every one of them. A *string* column then dies in `.astype(float)`, which
    # is the lucky outcome -- a numeric one (age, n_reads, year) is silently fitted a loc/scale,
    # frozen into the artifact under a name the layout cannot parse, and applied forever after.
    #
    # `L.parse` is the definition of a signature column, so ask it.
    from vdjtools.signature import layout as L

    drop = {group} if isinstance(group, str) else set()

    def is_signature(c: str) -> bool:
        if c in drop:
            return False
        try:
            L.parse(c)
        except ValueError:
            return False
        return True

    cols = [c for c in frame.columns if is_signature(c)]
    if not cols:
        raise ValueError("frame carries no signature columns")
    X = frame.select(cols).to_numpy().astype(float)

    observed = np.isfinite(X)
    n_obs = observed.sum(0)
    loc = np.zeros(len(cols))
    scale = np.zeros(len(cols))
    for j in range(len(cols)):
        good = X[observed[:, j], j]
        if good.size == 0:
            continue
        loc[j] = float(np.median(good))
        if good.size >= min_n_obs:
            mad = float(np.median(np.abs(good - loc[j])))
            scale[j] = mad * 1.4826
            if scale[j] <= 0:                 # observed but constant: nothing to divide by
                scale[j] = 0.0

    _block_policy(cols, X, observed, loc, scale)
    batch = _batch_ratio(X, observed, loc, scale, labels) if labels is not None else None
    return ScaleReference(columns=cols, loc=loc, scale=scale, n_obs=n_obs.astype(np.int64),
                          cstar=dict(cstar or {}), pgen_q05=dict(pgen_q05 or {}),
                          batch_ratio=batch,
                          meta={"min_n_obs": min_n_obs, "n_samples": int(X.shape[0]),
                                "groups": sorted(set(labels.tolist())) if labels is not None
                                else None, **(meta or {})})


def _block_policy(cols, X, observed, loc, scale) -> None:
    """Honour the layout's ``exempt`` and ``magnitude`` flags, which are contract, not decoration.

    Both were declared, documented and unit-tested from the start, and until this ran nothing on
    the emission path consulted either — every column got the same median/MAD treatment.

    * ``exempt`` (masks) is already 0/1 and has nothing to standardise.
    * ``magnitude`` (the contrast) carries its meaning in its size. Median-centring it sends
      ``Ψ = 0`` — an immune desert, the one state the block exists to express — to
      minus-the-median, which on the fitted reference is ``-7.04/0.081 ≈ -87`` robust deviations
      and clips to the far tail, landing on top of the most violently deviant samples in the
      corpus. So the block is divided by **one uncentred number per locus**: the origin stays the
      origin, and the coordinates keep their sizes relative to each other.

    A column too thinly observed to have earned a scale above keeps none; it is excluded from the
    shared estimate rather than diluting it.
    """
    from vdjtools.signature import layout as L

    blk = {(b.sig, b.name): b for b in L.registry() if b.exempt or b.magnitude}
    groups: dict[tuple, list[int]] = {}
    for j, c in enumerate(cols):
        sig, block, locus, feature = L.parse(c)
        b = blk.get((sig, block))
        if b is None:
            continue
        if b.exempt:
            loc[j] = scale[j] = 0.0
        # Only the *coordinates* of a magnitude block are magnitudes. A summary the layout already
        # declares a transform for — `contrast:norm` is log1p of a length — has been stabilised
        # into a different kind of number, and sharing the coordinates' raw RMS would divide a
        # value near 7 by the same constant as one near 0. It keeps ordinary reference-z.
        elif b.transform(feature) == "none":
            loc[j] = 0.0
            if scale[j] > 0:
                groups.setdefault((sig, block, locus), []).append(j)
    for js in groups.values():
        good = X[:, js][observed[:, js]]
        rms = float(np.sqrt(np.median(good ** 2)))
        scale[js] = rms if rms > 0 else 0.0


def _batch_ratio(X, observed, loc, scale, labels, min_per_group: int = 100) -> np.ndarray:
    """Between-group spread of a column's centre over its typical within-group spread.

    Both measured on the standardised column, so the ratio is dimensionless and comparable across
    blocks. Above 1 the column tells you more about which cohort a sample came from than about the
    donor — which is a fact about the feature, not about the scaling, and cannot be fixed by
    rescaling. Groups smaller than ``min_per_group`` are skipped: a median over twenty samples is
    not a group centre.
    """
    groups = [g for g in sorted(set(labels.tolist())) if (labels == g).sum() >= min_per_group]
    out = np.full(X.shape[1], np.nan)
    if len(groups) < 2:
        return out
    for j in range(X.shape[1]):
        if scale[j] <= 0:
            continue
        centres, spreads = [], []
        for g in groups:
            x = X[(labels == g) & observed[:, j], j]
            if x.size < min_per_group:
                continue
            c = float(np.median(x))
            centres.append((c - loc[j]) / scale[j])
            spreads.append(float(np.median(np.abs(x - c))) * 1.4826 / scale[j])
        if len(centres) >= 2 and np.median(spreads) > 0:
            out[j] = float(np.std(centres) / np.median(spreads))
    return out


def measure_constants(samples, *, loci=None, cstar_quantile: float = CSTAR_QUANTILE,
                      n_pgen: int = 2000, threads: int = 0) -> tuple[dict, dict]:
    """Measure ``cstar`` and ``pgen_q05`` per locus from a reference draw.

    ``cstar`` is a **low quantile of attained** Good–Turing coverage, so most samples interpolate
    rather than extrapolate; ``pgen_q05`` is the 5th percentile of ``log10 Pgen`` pooled over the
    draw, which is what "atypically improbable" is measured against.

    Args:
        samples: Iterable of ``(sample_id, {locus: frame})``.
        loci: Restrict to these loci; ``None`` measures whatever appears.
        cstar_quantile: Quantile of attained coverage to freeze.
        n_pgen: Junctions sampled per repertoire for the Pgen pool, via the same
            :func:`~vdjtools.signature.blocks.pgen_junctions` draw the per-sample block uses. This
            can be far smaller than the per-sample ``n_max``: the pool is one percentile over the
            whole corpus, so 400 samples at 2,000 each is 800,000 junctions to place a single q05
            that ~20,000 already pins. Lower it to make a reference fit cheap -- Pgen is ~all of
            the cost, and IGH alone is ~80% of it at 1.5 ms/junction against 0.0-0.2 elsewhere.
        threads: Worker threads for the Pgen batch; 0 = auto. Pgen is essentially the whole cost
            of this function — coverage is a one-line reduction, and the models load once — so
            leaving this unplumbed pins a reference fit to whatever the library defaults to
            regardless of the machine it was given. ``vsig``'s ``pgen_block`` has always taken it.

    Returns:
        ``(cstar, pgen_q05)``, each ``{locus: value}``.
    """
    from vdjtools.model.bundled import load_bundled
    from vdjtools.model.native import pgen_aa_batch
    from vdjtools.signature.blocks import pgen_junctions
    from vdjtools.stats.inext import sample_coverage

    cov: dict[str, list[float]] = {}
    pg: dict[str, list[float]] = {}
    # Load and collapse each recombination model ONCE. It is the expensive step here by a wide
    # margin — pgen over 2,000 junctions takes ~0.15 s, while building the model takes seconds —
    # so calling it per sample turns a two-minute measurement into an unbounded one.
    models: dict[str, object] = {}

    def model_for(locus: str):
        if locus not in models:
            try:
                models[locus] = load_bundled(locus)
            except Exception:
                models[locus] = None
        return models[locus]

    for _sid, sample in samples:
        for locus, df in sample.items():
            if loci and locus not in loci:
                continue
            if df is None or df.height < 2:
                continue
            counts = df["duplicate_count"].to_numpy().astype(np.int64)
            try:
                cov.setdefault(locus, []).append(float(sample_coverage(counts)))
            except Exception:
                pass
            model = model_for(locus)
            if model is None:
                continue
            juncs = pgen_junctions(df, locus, n_pgen)
            try:
                p = np.asarray(pgen_aa_batch(model, juncs, v=None, j=None, threads=threads),
                               dtype=float)
                p = p[np.isfinite(p) & (p > 0)]
                if p.size:
                    pg.setdefault(locus, []).extend(np.log10(p).tolist())
            except Exception:
                pass

    cstar = {k: float(np.quantile(v, cstar_quantile)) for k, v in cov.items() if v}
    q05 = {k: float(np.quantile(v, 0.05)) for k, v in pg.items() if v}

    # Coverage near 1.0 means essentially no clonotype was seen exactly once. That is not deep
    # sequencing — it is the signature of input whose singleton tail is already gone, either from
    # a top-N cut or from upstream collapsing. Good-Turing coverage is 1 - f1/n, so f1 ~ 0 reads
    # as perfect coverage, and freezing cstar from it would put every honest sample into
    # extrapolation, the regime measured to inflate diversity roughly tenfold.
    #
    # Dropped **per locus**, not raised globally: one pre-collapsed arm of a cohort should not
    # cost the other six their reference. A locus with no cstar simply gets no coverage-
    # standardised diversity, which the `estimable` mask already reports honestly.
    collapsed = sorted(k for k, v in cstar.items() if v >= COVERAGE_CEILING)
    for k in collapsed:
        del cstar[k]
    if collapsed:
        import warnings

        warnings.warn(
            f"no singletons observed for {collapsed} (attained coverage >= {COVERAGE_CEILING}), "
            "so no coverage level could be established there — these loci will have no "
            "standardised diversity. Usually means the input was top-N truncated or collapsed "
            "upstream; reload without a `top=` cut if that is the cause.",
            RuntimeWarning, stacklevel=2)
    return cstar, q05


def save_scale(ref: ScaleReference, path: "str | Path" = DEFAULT_PATH) -> Path:
    """Write the scale artifact (and its json sidecar)."""
    p = Path(path)
    extra = {} if ref.batch_ratio is None else {"batch_ratio": ref.batch_ratio}
    np.savez_compressed(p, columns=np.array(ref.columns), loc=ref.loc, scale=ref.scale,
                        n_obs=ref.n_obs, **extra,
                        cstar_loci=np.array(sorted(ref.cstar)),
                        cstar_vals=np.array([ref.cstar[k] for k in sorted(ref.cstar)]),
                        pgen_loci=np.array(sorted(ref.pgen_q05)),
                        pgen_vals=np.array([ref.pgen_q05[k] for k in sorted(ref.pgen_q05)]))
    p.with_suffix(".json").write_text(json.dumps(
        {**ref.meta, "cstar": ref.cstar, "pgen_q05": ref.pgen_q05, **ref.report()}, indent=2))
    return p


@lru_cache(maxsize=4)
def load_scale(path: "str | Path | None" = None) -> "ScaleReference | None":
    """Load the scale artifact, or ``None`` if none is installed.

    ``None`` rather than an exception **only for the default path**: a signature without a scale
    reference is still a perfectly usable raw feature vector, and the caller is told which it got
    via ``standardize=``.

    An explicitly supplied path that does not exist RAISES. Returning ``None`` there conflates
    "you did not ask for a reference" with "the reference you named is missing" -- so a typo in
    ``--scale`` would silently produce an unstandardised matrix that looks exactly like a
    standardised one, and the caller has already said they want a specific artifact.

    Raises:
        FileNotFoundError: If ``path`` was given and does not exist.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no scale reference at {p}")
    else:
        p = DEFAULT_PATH
        if not p.exists():
            return None
    d = np.load(p, allow_pickle=False)
    meta_path = p.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ScaleReference(
        columns=[str(c) for c in d["columns"]], loc=d["loc"], scale=d["scale"],
        n_obs=d["n_obs"],
        batch_ratio=d["batch_ratio"] if "batch_ratio" in d.files else None,
        cstar={str(k): float(v) for k, v in zip(d["cstar_loci"], d["cstar_vals"])},
        pgen_q05={str(k): float(v) for k, v in zip(d["pgen_loci"], d["pgen_vals"])},
        meta=meta)


def _demo() -> None:
    """Self-check on a synthetic cohort: scaling centres, holes survive, thin columns pass through."""
    import polars as pl

    rng = np.random.default_rng(0)
    n, cols = 60, ["vsig:depth:TRB:reads", "vsig:div:TRB:1D_c", "rsig:phic:TRB:PC01"]
    X = rng.normal(loc=[5.0, 1.5, 0.0], scale=[0.3, 0.2, 2.0], size=(n, 3))
    X[0, 1] = np.nan                                   # a hole must survive untouched
    frame = pl.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                          **{c: X[:, i] for i, c in enumerate(cols)}})

    ref = fit_scale(frame, min_n_obs=10)
    assert ref.scaled.all(), "every column had enough observations"
    assert ref.n_obs[1] == n - 1, "the hole was counted as an observation"

    row = {c: float(X[5, i]) for i, c in enumerate(cols)}
    out = ref.apply(row)
    assert set(out) == set(row), "apply changed the key set"
    assert all(abs(v) <= 8.0 for v in out.values())

    # a hole stays a hole rather than being centred
    assert np.isnan(ref.apply({cols[1]: float("nan")})[cols[1]])
    # an unknown column passes through
    assert ref.apply({"vsig:made:UP:column": 3.0})["vsig:made:UP:column"] == 3.0

    # a column the corpus barely saw gets no scale, and passes through unchanged
    thin = fit_scale(frame, min_n_obs=10_000)
    assert not thin.scaled.any()
    assert thin.apply(row) == row

    # scaling really does centre: the median sample lands near zero
    med = {c: float(np.nanmedian(X[:, i])) for i, c in enumerate(cols)}
    assert all(abs(v) < 1e-9 for v in ref.apply(med).values())

    assert ref.report()["scaled"] == 3
    print(f"scale OK — {ref.report()}")


if __name__ == "__main__":
    _demo()
