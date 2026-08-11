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
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "resources" / "signature" / "rsig_scale_v1.npz"

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

    def report(self) -> dict:
        """What this reference can and cannot standardise."""
        return {"columns": len(self.columns), "scaled": int(self.scaled.sum()),
                "unscaled": int((~self.scaled).sum()),
                "median_n_obs": int(np.median(self.n_obs)),
                "loci_with_cstar": sorted(self.cstar), "min_n_obs": self.meta.get("min_n_obs")}


def fit_scale(frame, *, min_n_obs: int = MIN_N_OBS, cstar: dict | None = None,
              pgen_q05: dict | None = None, meta: dict | None = None) -> ScaleReference:
    """Fit location and scale from an assembled cohort matrix.

    Args:
        frame: A ``pl.DataFrame`` from :func:`mir.signature.signature_cohort` — one row per
            sample, ``sample_id`` plus signature columns.
        min_n_obs: Columns observed fewer times than this get no scale.
        cstar / pgen_q05: Measured constants to carry alongside (see :func:`measure_constants`).
        meta: Provenance recorded into the artifact.

    Returns:
        A :class:`ScaleReference`.

    Raises:
        ValueError: If the frame has no signature columns.
    """
    cols = [c for c in frame.columns if c != "sample_id"]
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
    return ScaleReference(columns=cols, loc=loc, scale=scale, n_obs=n_obs.astype(np.int64),
                          cstar=dict(cstar or {}), pgen_q05=dict(pgen_q05 or {}),
                          meta={"min_n_obs": min_n_obs, "n_samples": int(X.shape[0]),
                                **(meta or {})})


def measure_constants(samples, *, loci=None, cstar_quantile: float = CSTAR_QUANTILE,
                      n_pgen: int = 2000) -> tuple[dict, dict]:
    """Measure ``cstar`` and ``pgen_q05`` per locus from a reference draw.

    ``cstar`` is a **low quantile of attained** Good–Turing coverage, so most samples interpolate
    rather than extrapolate; ``pgen_q05`` is the 5th percentile of ``log10 Pgen`` pooled over the
    draw, which is what "atypically improbable" is measured against.

    Args:
        samples: Iterable of ``(sample_id, {locus: frame})``.
        loci: Restrict to these loci; ``None`` measures whatever appears.
        cstar_quantile: Quantile of attained coverage to freeze.
        n_pgen: Junctions sampled per repertoire for the Pgen pool.

    Returns:
        ``(cstar, pgen_q05)``, each ``{locus: value}``.
    """
    from vdjtools.model.bundled import load_bundled
    from vdjtools.model.native import pgen_aa_batch
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
            juncs = df["junction_aa"].to_list()[:n_pgen]
            try:
                p = np.asarray(pgen_aa_batch(model, juncs, v=None, j=None), dtype=float)
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
    np.savez_compressed(p, columns=np.array(ref.columns), loc=ref.loc, scale=ref.scale,
                        n_obs=ref.n_obs,
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

    ``None`` rather than an exception: a signature without a scale reference is still a perfectly
    usable raw feature vector, and the caller is told which it got via ``standardize=``.
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    meta_path = p.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ScaleReference(
        columns=[str(c) for c in d["columns"]], loc=d["loc"], scale=d["scale"],
        n_obs=d["n_obs"],
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
