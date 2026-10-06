"""``rsig`` and ``rsig_cohort``: the geometry half, through a corpus rotation.

The three stages are the same as the statistics half's, and the machinery is literally the same --
:mod:`vdjtools.signature.corpus` fits and applies both, so the two halves cannot drift into
different notions of "standardised"::

    features.raw_and_channels   ->  corpus.apply  ->  rsig:pc:<locus>:PCnn  +  channels

This module also **declares** the ``rsig`` half of the column contract, by registering its raw groups
and channels into :mod:`vdjtools.signature.layout` at import. The contract lives in vdjtools because
mirpy depends on vdjtools and not the reverse; nothing in vdjtools imports ``mir``.
"""
from __future__ import annotations

import functools

import numpy as np
import polars as pl
from vdjtools.signature import corpus as C
from vdjtools.signature import layout as L
from vdjtools.signature import transform as T
from vdjtools.signature.cohort import parallel_rows, resolve_sample

from . import features as F

_SLOTS = ("phiv", "phij", "phic")

#: Registered once, at import. Each ``Phi`` slot is ``K`` weighted mean distances, so every
#: coordinate is non-negative and its tail runs upward only -- which is what fixes the
#: winsorization side. ``n_eff`` is a count, ``mass`` and the band shares are log-ratio coordinates
#: of a composition and so are two-sided.
_REGISTERED = False


def _register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    names = tuple(f"P{i:03d}" for i in range(1, F.K + 1))
    L.register_raw(
        *(L.RawGroup("rsig", slot, L.feats("none", "nonneg", *names)) for slot in _SLOTS),
        L.RawGroup("rsig", "depth", {**L.feats("log10", "nonneg", "n_eff"),
                                     **L.feats("logit", "real", "mass")}, named=True),
        L.RawGroup("rsig", "band", L.feats("clr", "real", "singleton", "top"), named=True),
        L.RawGroup("rsig", "band_igh", L.feats("clr", "real", "IgM", "IgG", "IgA"),
                   loci=("IGH",), named=True),
    )
    L.register_channel(
        # Carried, not rotated: these are diversity read-outs in their own units, and the point of
        # emitting them beside the Hill numbers of the statistics half is the head-to-head. A
        # channel is pass-through, so ADDING ONE DOES NOT INVALIDATE A FITTED CORPUS -- the
        # rotation is indexed by `raw_columns` and nothing else, and `corpus.apply` fills every
        # registered channel from the sample. Every artifact already published gains these columns
        # with no refit.
        L.Channel("rsig", "div", {
            "rao": "nonneg", "q_v": "nonneg", "q_j": "nonneg", "q_c": "nonneg",
            "q_frac_v": "real", "q_frac_j": "real", "q_frac_c": "real",
            "evenness": "real", "eff_dim": "nonneg", "eff_dim_pr": "nonneg",
            "q_top": "nonneg", "q_singleton": "nonneg", "q_ratio_top": "nonneg"}),
        # Displacements between compartment centroids, within one locus only -- see
        # `features.displacement_channels` for why never between two.
        L.Channel("rsig", "disp", {"top_singleton": "nonneg", "cos_top_singleton": "real",
                                   "norm": "nonneg"}),
        L.Channel("rsig", "disp", {"IgG_IgM": "nonneg", "cos_IgG_IgM": "real",
                                   "IgA_IgM": "nonneg", "cos_IgA_IgM": "real"}, loci=("IGH",)),
        # The same channel under the same name as the statistics half: two products, one
        # convention. It is also what makes holing the div family safe -- the moment a shallow
        # locus stops being a plausible 0.0, something has to say it was absent.
        L.Channel("rsig", "mask", {"present": "unit", "estimable": "unit"}),
        L.Channel("rsig", "qc", {"winsor_frac": "unit"}, loci=()),
    )
    _REGISTERED = True


_register()


#: Where shipped rsig artifacts live. A corpus is a rotation, a set of medians and MADs and a set of
#: percentile bounds -- no sample is recoverable from any of it, which is why these ship publicly
#: while the per-sample matrices they were fitted on do not.
_RES = __import__("pathlib").Path(__file__).resolve().parent.parent / "resources" / "signature"


#: The repo whose release assets carry the rsig artifacts. The vsig half lives in vdjtools' releases;
#: each half is published by the library that owns it, so a corpus name resolves to two downloads from
#: two repos rather than one repo having to know about the other's layout.
CORPUS_REPO = "antigenomics/mirpy"


def bundled_names() -> list[str]:
    """Every corpus name this version knows -- installed, cached, or downloadable."""
    from vdjtools.signature.corpus import corpora_index, corpus_cache_dir

    got = {p.stem.removeprefix("rsig_") for p in _RES.glob("rsig_*.npz")} if _RES.is_dir() else set()
    cache = corpus_cache_dir()
    if cache.is_dir():
        got |= {p.stem.removeprefix("rsig_") for p in cache.glob("rsig_*.npz")}
    return sorted(got | set(corpora_index(_RES)))


def bundled_path(name):
    """Resolve a corpus name or a filesystem path to an ``rsig`` artifact, else ``None``.

    Downloads on first use if the name is in the shipped index and not yet cached. One resolver,
    shared with the vsig half, so the two cannot disagree about precedence -- a local path always
    wins over a download.
    """
    from vdjtools.signature.corpus import resolve_artifact

    return resolve_artifact(name, sig="rsig", res_dir=_RES, repo=CORPUS_REPO)


@functools.lru_cache(maxsize=16)
def _model(species: str, locus: str):
    """One immutable prototype panel per locus; parallelism is across samples."""
    from mir.embedding.tcremp import TCREmp

    return TCREmp.from_defaults(species, locus, n_prototypes=F.K, threads=1)


def _clr(shares: dict[str, float], keep: tuple[str, ...], m: int) -> dict[str, float]:
    """Close a compartment composition and select coordinates from the **whole** closure.

    A clr of a sub-composition is a different number from the corresponding coordinate of the full
    one, so the residual has to be inside the geometric mean even though it is never emitted.
    """
    # Fewer than two parts is not a thin composition, it is no composition: every band fell below
    # the presence floor and only the closing residual is left, so there are no ratios to take. A
    # hole is the honest answer -- clr would raise, and forcing a second part would invent one.
    if len(shares) < 2:
        return dict.fromkeys(keep, np.nan)
    coords = T.clr(shares, m=m)
    return {k: float(coords.get(k, np.nan)) for k in keep}


def raw_and_channels(frames: dict[str, pl.DataFrame], vocab: dict[str, dict], *,
                     species: str = "human", weight: str = "log2p1",
                     min_clonotypes: int = 5, chunk: int = F.CHUNK,
                     ) -> tuple[dict[str, float], dict[str, float]]:
    """Every raw rsig feature and every rsig channel for one sample.

    Args:
        frames: ``{locus: frame}``, already sanitised.
        vocab: ``{locus: {}}`` from the corpus artifact. ``rsig`` has no germline-width group, so the
            per-locus dicts are empty; the keys are what say which loci the corpus models.
        species: Prototype panel species.
        weight: Clone-size weight, a key of ``vdjtools.signature.features.WEIGHTS``.
        min_clonotypes: Floor for a compartment to count as present. Below it the compartment is
            dropped from the composition, not zeroed. Never drops a whole sample.
        chunk: Rows embedded per pass; bounds memory, not time.

    Returns:
        ``(raw, channels)`` -- both ``{column: value}``, holes as ``nan``.
    """
    from mir.repertoire import missing_mass

    raw: dict[str, float] = {}
    chan: dict[str, float] = {}

    def hole(locus: str, *, present: float) -> None:
        """Every div and disp channel of this locus is a hole, and the mask says why.

        A hole, never a zero. Rao of a single-clonotype locus is arithmetically 0.0 and is not a
        diversity measurement -- it is `mask:present` in different units, sitting ~29 robust
        deviations below the 1st percentile of the real values. Emitted as a number it was read as
        one: a Cox screen returned a hazard ratio of 739 per SD at p = 7e-155 off four such
        samples, against a Spearman correlation with the outcome of -0.023.
        """
        for c in L.channel_columns("rsig"):
            _sig, block, loc, _feat = L.parse(c)
            if loc == locus and block in ("div", "disp"):
                chan[c] = np.nan
        chan[f"rsig:mask:{locus}:present"] = present
        chan[f"rsig:mask:{locus}:estimable"] = 0.0

    for locus in L.LOCI:
        # Rao is a CHANNEL: it needs the embedder and the sample, not the rotation. So it is
        # computed for every locus the sample has, whether or not this corpus models that locus --
        # gating it on the vocabulary would report a hole for a number that was perfectly
        # computable, which is the mirror image of the mistake the masks exist to prevent. Only raw
        # features are gated on the vocabulary, because only they are indexed by the rotation.
        modelled = locus in vocab
        cols = L.raw_columns("rsig", locus, vocab.get(locus)) if modelled else []
        df = frames.get(locus)
        if df is None or df.height == 0:
            raw |= dict.fromkeys(cols, np.nan)
            hole(locus, present=0.0)
            continue
        counts = df["duplicate_count"].to_numpy()
        try:
            w = F.weights(counts, weight)
        except ValueError:                       # every clone weight zero: nothing to embed
            raw |= dict.fromkeys(cols, np.nan)
            hole(locus, present=1.0)
            continue

        # The SHM columns must not reach the embedder: their presence silently switches it to
        # SHM-aware V distances, a different coordinate system under the same column names.
        clean = df.drop(*F._SHM_COLUMNS, strict=False)
        masks = {k: np.asarray(pred(counts), dtype=bool) for k, pred in F.BANDS.items()}
        if locus == "IGH":
            masks |= F.isotype_masks(df)
        acc = F.dispersion_pass(clean, _model(species, locus), w, band_masks=masks, chunk=chunk)
        phi, n_eff = acc.phi, acc.n_eff
        mass = 1.0 - float(missing_mass(counts))

        # The DISPERSION is what a handful of clonotypes cannot support; `Phi` itself is fine from
        # three of them. So the div/disp family is holed below the floor and the geometry below is
        # computed regardless -- holing the raw features here instead would throw away a measurable
        # embedding to fix an unmeasurable diversity, which is the opposite trade.
        if df.height < min_clonotypes:
            hole(locus, present=1.0)
        else:
            chan |= F.diversity_channels(acc, locus)
            chan |= F.displacement_channels(acc, locus)
            chan[f"rsig:mask:{locus}:present"] = 1.0
            chan[f"rsig:mask:{locus}:estimable"] = 1.0
        if not modelled:
            continue
        for slot, vec in F.slots(phi).items():
            raw |= {f"rsig:{slot}:{locus}:P{i + 1:03d}": float(v) for i, v in enumerate(vec)}
        raw |= {f"rsig:depth:{locus}:n_eff": T.log10(n_eff),
                f"rsig:depth:{locus}:mass": T.logit(np.clip(mass, 0.0, 1.0), counts.size)}
        bands = F.band_shares(df, w, min_clonotypes=min_clonotypes)
        raw |= {f"rsig:band:{locus}:{k}": v for k, v in
                _clr(bands, ("singleton", "top"), df.height).items()}
        if locus == "IGH":
            iso = F.isotype_shares(df, w, min_clonotypes=min_clonotypes)
            raw |= {f"rsig:band_igh:IGH:{k}": v for k, v in
                    _clr(iso, ("IgM", "IgG", "IgA"), df.height).items()}

    chan[f"rsig:qc:{L.NO_LOCUS}:winsor_frac"] = 0.0
    return raw, chan


#: How `vdjtools.signature.corpus.fit_cohort` turns this half's frames into feature rows. A
#: registry rather than an import, because nothing in vdjtools may import mir -- mirpy depends on
#: vdjtools and not the reverse. Registered here rather than inside `_register`, which runs before
#: this function is defined.
C.register_featuriser("rsig", raw_and_channels)


def rsig(sample, corpus: C.Corpus, *, mode: "str | None" = None,
         winsor_p: "float | None" = None, n_components: "int | float | None" = None,
         species: str = "human", weight: str = "log2p1", sanitise: bool = True,
         on_duplicate: str = "error", min_clonotypes: int = 5, chunk: int = F.CHUNK,
         named: "bool | tuple[str, ...]" = (),
         columns: "list[str] | None" = None) -> dict[str, float]:
    """The geometry half of the signature for one sample.

    Args:
        sample: ``{locus: frame}``, one frame with a ``locus`` column, or a zero-argument callable.
        corpus: A fitted corpus for ``sig="rsig"``.
        mode: Winsorization mode override -- ``"features"``, ``"pcs"`` or ``"none"``.
        winsor_p: Which stored percentile to clamp at.
        n_components: Truncate to this many components, or this variance fraction.
        species: Prototype panel species.
        weight: Clone-size weight.
        sanitise: Drop non-productive clonotypes first.
        on_duplicate: ``"error"`` or ``"sum"`` for a repeated amino-acid clonotype key.
        min_clonotypes: Compartment presence floor, and the floor below which the whole ``div``
            and ``disp`` family of a locus is a hole rather than a number. A one-clonotype locus
            has a Rao of exactly ``0.0`` -- arithmetically right, and not a diversity measurement.
        chunk: Rows embedded per pass.
        named: Also return the reportable raw blocks (``band``, ``band_igh``, ``depth``) --
            ``True`` for all of them, or an explicit sequence. ``()`` emits the rotated columns
            and channels only. Values carry their declared transform;
            :func:`~vdjtools.signature.layout.channel_table` reports which.
        columns: Restrict the output to these columns, in layout order.
    """
    from vdjtools.signature.features import sanitise as vsanitise

    if corpus.sig != "rsig":
        raise ValueError(f"corpus {corpus.name!r} is for {corpus.sig!r}, not 'rsig'")
    frames = _locus_frames(sample)
    if sanitise:
        frames = {k: v for k, v in
                  ((k, vsanitise(v, on_duplicate=on_duplicate)[0]) for k, v in frames.items())
                  if v.height}
    raw, chan = raw_and_channels(frames, corpus.vocab, species=species, weight=weight,
                                 min_clonotypes=min_clonotypes, chunk=chunk)
    out = C.apply(raw, chan, corpus, mode=mode, winsor_p=winsor_p, n_components=n_components,
                  named=named)
    if columns is None:
        return out
    want = [c for c in corpus.columns(n_components, named=named) if c in set(columns)]
    return {c: out[c] for c in want}


def _locus_frames(sample) -> dict[str, pl.DataFrame]:
    from vdjtools.io.schema import LOCUS

    sample = resolve_sample(sample)
    if isinstance(sample, dict):
        return {k: v for k, v in sample.items() if k in L.LOCI}
    if LOCUS not in sample.columns:
        raise ValueError(f"a single frame needs a {LOCUS!r} column to be split by locus; pass "
                         "{locus: frame} instead")
    return {loc: part.drop(LOCUS) for (loc,), part in sample.group_by([LOCUS]) if loc in L.LOCI}


def _one(item, corpus, kw):
    """One sample's row. Module-level so a spawned worker can unpickle it."""
    sid, sample = item
    return {"sample_id": sid, **rsig(resolve_sample(sample), corpus, **kw)}


def rsig_cohort(samples, corpus: C.Corpus, *, n_jobs: int = 1,
                columns: "list[str] | None" = None, **kw) -> pl.DataFrame:
    """One row per sample, ``sample_id`` first.

    Args:
        samples: ``{sample_id: sample}`` or an iterable of pairs. A sample may be a zero-argument
            **picklable** callable, which defers the read into the worker and holds peak memory at
            ``O(n_jobs)`` samples rather than the whole cohort.
        corpus: A fitted corpus.
        n_jobs: Worker processes; ``1`` in-process, ``0`` every available core.
        columns: Restrict the output columns.
        **kw: Forwarded to :func:`rsig`.
    """
    items = list(samples.items() if isinstance(samples, dict) else samples)
    rows = parallel_rows(items, functools.partial(_one, corpus=corpus,
                                                 kw={**kw, "columns": columns}), n_jobs)
    want = ["sample_id", *(columns if columns is not None
                           else corpus.columns(kw.get("n_components"),
                                               named=kw.get("named", ())))]
    return pl.DataFrame(rows).select([c for c in want if any(c in r for r in rows)])


def synthesize(corpus_name: str, *, loci: "tuple[str, ...]" = L.LOCI, n_samples: int = 10_000,
               size: "int | str | None" = None, seed: int = C.SEED,
               n_components: "int | float" = C.DEFAULT_COMPONENTS, mode: str = "features",
               winsor_p: float = 0.01, source: str = "olga", species: str = "human",
               progress=None, n_jobs: int = 1, depth_spread: "float | str | None" = None):
    """Build and fit an ``rsig`` corpus on the same synthetic repertoires ``vsig`` uses.

    The receptors, the seeds and the drawn depths are all
    :func:`vdjtools.signature.corpus.synthesize`'s, so the two halves of a corpus name describe the
    *same* repertoires -- which is what makes joining ``vsig_<name>`` and ``rsig_<name>`` on
    ``sample_id`` meaningful rather than merely type-correct.

    Args:
        corpus_name: One of :data:`vdjtools.signature.corpus.SYNTHETIC` -- ``"naive"`` / ``"memory"``
            for the pure regimes, ``"synthetic-blood"`` / ``"synthetic-tissue"`` for the naive/memory
            mixture drawn across that cohort's measured per-locus richness, read-depth and
            singleton-fraction bands.
        n_jobs: Worker **processes** (not kernel threads). ``0`` means every available core; ``1``,
            the default, runs in-process. The artifact is bit-identical at every value.
        size: ``None`` is the corpus's own -- 10,000 for a pure regime, the geometric centre of the
            cohort's measured richness band for a ``synthetic-*`` one.
        depth_spread: Multiplicative depth range each repertoire's size is drawn log-uniformly
            across, around ``size``; ``None`` is the corpus's own. Must match the ``vsig`` half's
            value for the two to describe the same repertoires.

    Returns:
        ``(corpus, mats)`` -- ``mats`` being ``{locus: (matrix, columns)}``, the corpus matrix itself.
    """
    from functools import partial

    import mir

    regime, cohort, size, depth_spread = C.corpus_plan(corpus_name, size=size,
                                                      depth_spread=depth_spread)
    vocab = {loc: {} for loc in loci}
    # Parallel across samples, in the one shared builder: an rsig row is a pure function of its
    # sample, so a worker draws its own repertoires from the memory-mapped pools and only the row of
    # numbers comes back. ``n_jobs`` does not change a single value.
    mats = C.build_matrices(
        regime, sig="rsig", vocab=vocab, loci=loci, n_samples=n_samples, size=size, seed=seed,
        source=source, n_jobs=n_jobs, progress=progress, depth_spread=depth_spread, cohort=cohort,
        featurise=partial(raw_and_channels, vocab=vocab, species=species))
    return C.fit_matrices(
        mats, vocab, sig="rsig", name=corpus_name, mode=mode, n_components=n_components,
        winsor_p=winsor_p,
        # One manifest builder for both halves: a `vsig_<name>`/`rsig_<name>` pair that disagreed
        # about the depths or the singleton fractions would be a joinability claim nobody can check.
        meta=C.corpus_meta(regime, cohort, loci, n_samples=n_samples, size=size, seed=seed,
                           source=source, depth_spread=depth_spread,
                           mir_version=mir.__version__, species=species, K=F.K)), mats


def _demo() -> None:
    """Self-check: fit a tiny corpus and score held-out repertoires through it."""
    corpus, _ = synthesize("memory", loci=("TRG",), n_samples=24, size=80, seed=5,
                           n_components=4)
    held = C.synthesize("memory", loci=("TRG",), n_samples=2, size=80, seed=707,
                        fit_corpus=False)
    d = rsig_cohort({f"S{i}": s for i, s in enumerate(held)}, corpus)
    assert d.height == 2 and d.columns[0] == "sample_id"
    assert set(d.columns[1:]) == set(corpus.columns())
    assert np.isfinite(d["rsig:div:TRG:rao"].to_numpy()).all()
    # a corpus for the other half must be refused, not silently applied
    try:
        rsig(held[0], C.Corpus(sig="vsig", name="x", vocab={}, fits={}))
    except ValueError as e:
        assert "not 'rsig'" in str(e)
    else:
        raise AssertionError("an rsig call accepted a vsig corpus")
    print(f"signature OK  {d.width - 1} columns, k={corpus.k}")


if __name__ == "__main__":
    _demo()
