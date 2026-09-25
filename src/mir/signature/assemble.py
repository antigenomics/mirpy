"""Assemble the signature: the ``rsig`` geometry columns, and both halves joined.

:mod:`mir.signature.blocks` computes geometric quantities; this puts them in the order and under
the names :mod:`vdjtools.signature.layout` promises, projected through the frozen reference so
that two people who never share data land in the same coordinate system.

:func:`signature` returns both halves — ``vdjtools.signature.vsig`` for the statistics, ``rsig``
for the geometry — as one flat, positional vector. That is the object meant to be handed to
somebody else.

Holes, here as in the statistics half, are never zeros. A locus that was not sequenced, a
compartment with too few clonotypes to be a compartment, a chain whose panel does not match the
frozen reference — each yields ``nan`` and a mask column, because a model that reads "absent" as
"zero" will read an unsequenced chain as a biological finding.
"""
from __future__ import annotations

import numpy as np

import polars as pl

from . import blocks as B

#: Which slot of ``Φ`` each identity block reads, and how many components each tier keeps.
#: Widths come from the layout, so the contract has exactly one home.
_SLOT_OF = {"phiv": "V", "phij": "J", "phic": "C"}

#: Stand-in coverage level for a locus where none could be established. Deliberately
#: unreachable, so the diversity block fails its own estimability check and masks out instead of
#: reporting an extrapolation as a measurement.
_UNREACHABLE_COVERAGE = 1.0

#: Columns that switch :meth:`mir.embedding.tcremp.TCREmp.embed` onto SHM-aware V distances, by
#: being present. That is the right embedding for a B-cell study and the wrong one here: the
#: frozen ``rsig`` reference was fitted on germline V coordinates, so a frame that happens to
#: carry ``v_identity`` would otherwise land in a different coordinate system than the same frame
#: without it — silently, with no mask to say so. The SHM load is still reported, by the
#: statistics half, as ``vsig:shm:IGH:mean_v_identity``.
_SHM_COLUMNS = ("v_identity", "v_mutations")


def _locus_frames(sample) -> dict[str, pl.DataFrame]:
    """Accept ``{locus: frame}``, a single frame with a ``locus`` column, or a callable.

    A **zero-argument callable** returning either of the first two is the deferred form, and it is
    what keeps a large cohort inside a small machine. ``signature_cohort`` pickles each worker its
    slice of the cohort; if the slice holds frames, the parent has already read and materialised
    every sample and then copies them all down a pipe. If it holds callables, the parent holds
    nothing, each worker reads its own samples one at a time, and peak memory stops scaling with
    the number of samples and starts scaling with the number of workers.

    Measured, 1,000 samples x 10,000 clonotypes at ``n_jobs=8``: 6.6 GB peak resident with frames,
    against 1.1 GB with callables. Use ``functools.partial`` over a module-level function, not a
    lambda -- a lambda cannot be pickled and the pool will refuse it.
    """
    from vdjtools.io.schema import LOCUS, add_locus, column_names

    if callable(sample) and not isinstance(sample, (dict, pl.DataFrame)):
        sample = sample()
    if isinstance(sample, dict):
        return {k: v for k, v in sample.items() if v is not None and v.height}
    df = sample if LOCUS in column_names(sample) else add_locus(sample)
    return {k[0] if isinstance(k, tuple) else k: v
            for k, v in df.partition_by(LOCUS, as_dict=True).items()}


def rsig(sample, *, tier: str = "standard", species: str = "human", weight: str = "log2p1",
         reference=None, chunk: int = B.CHUNK, min_clonotypes: int = 5) -> dict[str, float]:
    """The ``rsig`` (geometry) half of one sample's signature, as ``{column_name: value}``.

    Args:
        sample: ``{locus: clonotype frame}``, or one frame with a ``locus`` column. Frames should
            already be sanitised (``vdjtools.signature.blocks.sanitise``); malformed junctions are
            silently accepted by the embedder and would quietly contaminate every coordinate here.
        tier: ``"core"``, ``"standard"`` or ``"full"``.
        species: Prototype panel species.
        weight: Clone-size weight ``g``.
        reference: A :class:`~mir.signature.reference.SignatureReference`, or ``None`` for the
            bundled one.
        chunk: Rows embedded at a time; bounds memory, not time.
        min_clonotypes: Floor for a *compartment* to count as present. Not a floor on the sample —
            a whole repertoire is never dropped for being small.

    Returns:
        Every column the layout lists for ``tier`` and ``"rsig"``, in order, ``nan`` where the
        sample could not support one.

    Raises:
        ValueError: If ``tier`` is unknown.
    """
    from vdjtools.signature import layout as L
    from vdjtools.signature import transform as T

    from .reference import load_reference

    ref = reference if reference is not None else load_reference()
    want = L.columns(tier, "rsig")
    out = dict.fromkeys(want, np.nan)
    frames = _locus_frames(sample)

    for locus in L.LOCI:
        df = frames.get(locus)
        if df is None or df.height == 0 or locus not in ref:
            continue
        lref = ref[locus]

        model = _model(species, locus, lref.n_prototypes)
        counts = df["duplicate_count"].to_numpy()
        try:
            w = B.weights(counts, weight)
        except ValueError:                      # every clone weight zero — nothing to embed
            continue
        phi, mean_sq = B.prototype_sum(df.drop(*_SHM_COLUMNS, strict=False), model, w, chunk=chunk)

        # depth and diversity, read off the geometry rather than off the counts
        n_eff = 1.0 / float(w @ w)
        mass = 1.0 - _missing_mass(counts)
        _put(out, f"rsig:depth:{locus}", B.depth_block(counts, w, mass))
        rao = 2.0 * (mean_sq - float(phi @ phi))
        if n_eff > 1.0:
            rao *= n_eff / (n_eff - 1.0)        # self-pair bias is O(1/n_eff) and tracks depth
        _put(out, f"rsig:div:{locus}", {"rao": T.log1p(max(rao, 0.0))})

        # the contrast: magnitude-scaled, never centred — an immune desert belongs at the origin
        psi = lref.contrast(phi, mass)
        _put(out, f"rsig:contrast:{locus}", _contrast_features(psi, lref, tier, L))

        for block, slot in _SLOT_OF.items():
            k = L.PC_DIMS[block].get(tier)
            coords = lref.project(phi, slot, k=k) if k else None
            feats = {"norm": T.log1p(float(np.linalg.norm(lref.standardize(phi)[
                {"V": 0, "J": 1, "C": 2}[slot]::3])))}
            if coords is not None:
                feats |= {f"PC{i + 1:02d}": float(c) for i, c in enumerate(coords)}
            _put(out, f"rsig:{block}:{locus}", feats)

        if tier in ("standard", "full"):
            _put(out, f"rsig:band:{locus}",
                 _clr_shares(B.band_shares(df, w, min_clonotypes=min_clonotypes),
                             ("singleton", "top"), df.height, T))
            if locus == "IGH":
                _put(out, "rsig:band:IGH",
                     _clr_shares(B.isotype_shares(df, w, min_clonotypes=min_clonotypes),
                                 ("IgM", "IgG", "IgA"), df.height, T))

    return {k: out[k] for k in want}


def _contrast_features(psi: np.ndarray, lref, tier: str, L) -> dict[str, float]:
    """``norm`` plus the rotated contrast coordinates the tier asks for.

    The width comes from the layout, like every other block's. A second copy of ``{standard: 12,
    full: 32}`` here would not raise if the contract widened — ``_put`` drops what it does not
    recognise and leaves what it never got as ``nan`` — so the block would quietly ship a
    short vector padded with holes.
    """
    blk = next(b for b in L.registry("rsig") if b.name == "contrast")
    n_pc = len([c for c in blk.columns(tier) if ":PC" in c]) // len(blk.emitted_loci)
    feats = {"norm": float(np.log1p(np.linalg.norm(psi)))}
    if n_pc:
        # Rotate through the junction basis: the contrast lives in the same space as Phi, and
        # reusing one rotation keeps the two blocks' coordinates comparable rather than each
        # having its own arbitrary axes.
        z = (psi / lref.sd_phi)[2::3]
        coords = (z @ lref.rotations["C"])[:n_pc]
        feats |= {f"PC{i + 1:02d}": float(c) for i, c in enumerate(coords)}
    return feats


def _clr_shares(shares: dict, keys, m: int, T) -> dict[str, float]:
    """Close a share dict into log-ratio coordinates, then select the shipped ones.

    The CLR is taken over the *whole* composition — including the residual or uncalled part —
    before any coordinate is selected, so a tier shipping two of three parts still divides by the
    three-part geometric mean and stays a slice of the wider tier.

    Fewer than two parts is a hole, not an error. A shallow repertoire can put every compartment
    below its clonotype floor, leaving only the residual; a log-ratio needs something to take a
    ratio *against*, so there is nothing to report — and reporting nothing is the contract, while
    raising would take the whole sample down with it.
    """
    if len(shares) < 2 or all(v <= 0 for v in shares.values()):
        return dict.fromkeys(keys, np.nan)
    coords = T.clr(shares, m=max(m, 1))
    return {k: float(coords[k]) if k in coords else np.nan for k in keys}


def _put(out: dict, prefix: str, values: dict) -> None:
    for k, v in values.items():
        key = f"{prefix}:{k}"
        if key in out:
            out[key] = float(v)


def _missing_mass(counts: np.ndarray) -> float:
    """Chao's estimate of the never-drawn mass.

    Deliberately unguarded. ``missing_mass`` is arithmetically total — its Chao form adds 1 to the
    denominator so it cannot divide by zero, and a zero total is handled — so the only things it
    raises are an unknown method and *being handed frequencies where counts belong*. That second
    ValueError exists, in its own words, to stop a shallow sample being "silently declar[ed] a
    complete probability measure".

    This used to sit behind ``except Exception: return 0.0``, i.e. mass = 1.0 — which is exactly
    the outcome the guard was written to prevent, reached by catching the guard. A caller passing
    a frequency column got a confident "we observed this entire repertoire" for every sample, and
    the contrast block scaled at full magnitude on it. Better to raise where the mistake is.
    """
    from mir.repertoire import missing_mass

    return float(missing_mass(counts, "chao"))


_MODELS: dict[tuple, object] = {}


def _model(species: str, locus: str, n_prototypes: int):
    """Cache the embedder per (species, locus, K) — construction reads the bundled panel."""
    key = (species, locus, n_prototypes)
    if key not in _MODELS:
        from mir.embedding.tcremp import TCREmp

        _MODELS[key] = TCREmp.from_defaults(species, locus, n_prototypes=n_prototypes)
    return _MODELS[key]


def signature(sample, *, tier: str = "standard", species: str = "human", weight: str = "log2p1",
              reference=None, scale=None, standardize: str = "reference", clip: float = 8.0,
              sanitise: bool = True, prefiltered: bool = False,
              on_duplicate: str = "error", **vsig_kw) -> dict[str, float]:
    """Both halves of one sample's signature, concatenated — the hand-off object.

    With ``standardize="reference"`` (the default) every column is rescaled against the frozen
    reference, so the result is dimensionless and on a common scale and a downstream model needs
    no scaler of its own. That is the entire point: a collaborator's matrix and ours are directly
    comparable, rather than each being internally consistent and mutually meaningless.

    Args:
        sample: ``{locus: clonotype frame}`` or one frame with a ``locus`` column.
        tier: ``"core"``, ``"standard"`` or ``"full"``.
        species: Prototype panel species.
        weight: Clone-size weight ``g``, shared by both halves so they describe one measure.
        reference: Frozen geometry reference, or ``None`` for the bundled one.
        scale: A :class:`~mir.signature.scale.ScaleReference`, or ``None`` for the bundled one.
        standardize: ``"reference"`` to rescale against it, ``"none"`` for raw values. Asking for
            ``"reference"`` when none is installed raises rather than silently handing back raw
            numbers that look standardised.
        clip: Bound in robust standard deviations when standardising.
        sanitise: Drop unusable clonotypes first. Leave this on. Turning it off no longer
            contaminates the geometry silently — ``TCREmp.embed`` now refuses a junction outside
            the 20 standard amino acids — but it does mean the call raises rather than returning
            a vector, which is the intended outcome and not a reason to reach for
            ``allow_nonstandard``.
        prefiltered: Say ``True`` when non-functional rearrangements were already removed
            upstream. ``vsig:qc:*:nonstd_aa_frac`` then reports ``nan`` rather than a confident
            floor value, because on a pre-filtered input there is nothing left to measure. This
            is the ONLY part of the vector that pre-filtering affects: every other column is
            computed on the rows that survive ``sanitise``, and pre-filtering does not change
            which rows those are. If you do not want the column at all, drop it with a preset
            (``classify``, ``transfer`` and ``compact`` all do) — that is exact, whereas
            filtering the repertoire is not.
        on_duplicate: What to do with a locus frame that has no ``junction_nt`` and repeats an
            amino-acid clonotype key -- ``"error"`` (default) or ``"sum"``. Both halves see the
            same policy, which matters more here than in either library alone: ``vsig`` and
            ``rsig`` sanitise separately, so a policy applied to one and not the other would
            weight the two halves of one vector on different row sets. See
            :func:`vdjtools.io.schema.assert_resolvable`.
        **vsig_kw: Forwarded to ``vdjtools.signature.vsig`` (``cstar``, ``pgen_q05``, ``strict``…).
            ``cstar`` and ``pgen_q05`` default to the measured values in the scale reference.

    Returns:
        ``{column: value}`` for ``vsig`` then ``rsig``, in layout order.

    Raises:
        ValueError: If ``standardize`` is unknown, or is ``"reference"`` with none available.
    """
    from vdjtools.signature import blocks as VB
    from vdjtools.signature import vsig

    from .scale import load_scale

    if standardize not in ("reference", "none"):
        raise ValueError(f"standardize must be 'reference' or 'none'; got {standardize!r}")
    sref = scale if scale is not None else load_scale()
    if standardize == "reference" and sref is None:
        raise ValueError(
            "standardize='reference' but no scale reference is installed. Fit one with "
            "mir.signature.scale.fit_scale over a reference cohort, or pass standardize='none' "
            "to get raw values — which are usable, but not comparable to anyone else's.")

    # The measured constants live with the scale reference, so a caller gets them by default
    # rather than having to know that a hand-picked coverage level would put every sample into
    # extrapolation.
    #
    # The dict must cover EVERY locus. measure_constants deliberately omits a locus whose
    # reference draw had no singleton tail, and a partial dict is indexed positionally
    # downstream — so a missing key is a KeyError rather than a graceful skip. Filling the gaps
    # with an unreachable level is the honest completion: that locus's diversity then fails its
    # own estimability check and masks out, which is exactly what "we could not establish a
    # coverage level here" should look like.
    if sref is not None:
        vsig_kw.setdefault("pgen_q05", sref.pgen_q05 or None)
        if sref.cstar:
            vsig_kw.setdefault("cstar", sref.cstar)

    # Complete the dict AFTER the defaults, and unconditionally — a caller may pass a partial one
    # straight from measure_constants, which is the common case when fitting a reference.
    if isinstance(vsig_kw.get("cstar"), dict):
        from vdjtools.signature.layout import LOCI

        given = vsig_kw["cstar"]
        vsig_kw["cstar"] = {loc: given.get(loc, _UNREACHABLE_COVERAGE) for loc in LOCI}

    # Resolve a deferred sample ONCE: vsig is handed the raw `sample` and rsig the sanitised
    # frames, so leaving it deferred here would read the same files twice per sample.
    if callable(sample) and not isinstance(sample, (dict, pl.DataFrame)):
        sample = sample()
    frames = _locus_frames(sample)
    if sanitise:
        frames = {k: VB.sanitise(v, on_duplicate=on_duplicate)[0] for k, v in frames.items()}
        frames = {k: v for k, v in frames.items() if v.height}
    # vsig sanitises internally and needs the raw frames to report the dropped fraction honestly
    out = {**vsig(sample, tier=tier, weight=weight, prefiltered=prefiltered,
                  on_duplicate=on_duplicate, **vsig_kw),
           **rsig(frames, tier=tier, species=species, weight=weight, reference=reference)}
    return sref.apply(out, clip=clip) if standardize == "reference" else out


def _one(item, tier, kw):
    """One sample's joined row. Module-level so a worker process can unpickle it."""
    sid, sample = item
    return {"sample_id": sid, **signature(sample, tier=tier, **kw)}


def _one_rsig(item, tier, kw):
    """One sample's rsig row -- geometry only, no vsig computed at all."""
    sid, sample = item
    if callable(sample) and not isinstance(sample, (dict, pl.DataFrame)):
        sample = sample()
    from vdjtools.signature import blocks as VB

    frames = _locus_frames(sample)
    if kw.pop("sanitise", True):
        frames = {k: VB.sanitise(v, on_duplicate=kw.pop("on_duplicate", "error"))[0]
                  for k, v in frames.items()}
        frames = {k: v for k, v in frames.items() if v.height}
    scale = kw.pop("scale", None)
    standardize = kw.pop("standardize", "reference")
    clip = kw.pop("clip", 8.0)
    out = rsig(frames, tier=tier, **kw)
    if standardize == "reference":
        from .scale import load_scale

        sref = scale if scale is not None else load_scale()
        if sref is not None:
            out = sref.apply(out, clip=clip)
    return {"sample_id": sid, **out}


def _cohort(samples, one, tier, kw, n_jobs, columns, sig):
    """Shared body: build rows through vdjtools' pool, then select this half's columns."""
    import functools

    from vdjtools.signature import layout as L
    from vdjtools.signature.cohort import parallel_rows

    items = list(samples.items() if isinstance(samples, dict) else samples)
    rows = parallel_rows(items, functools.partial(one, tier=tier, kw=kw), n_jobs)
    if not rows:
        return pl.DataFrame(schema={"sample_id": pl.Utf8})
    want = columns if columns is not None else L.columns(tier, sig)
    if columns is not None and sig is not None:
        want = [c for c in want if c.startswith(f"{sig}:")]
    return pl.DataFrame(rows).select(["sample_id", *want])


def rsig_cohort(samples, *, tier: str = "standard", n_jobs: int = 1,
                columns: list[str] | None = None, **kw) -> pl.DataFrame:
    """A cohort of **this tool's half**: ``sample_id`` plus the ``rsig`` geometry columns.

    The counterpart to ``vdjtools.signature.vsig_cohort``. One tool per half -- join the two
    frames on ``sample_id`` for the full portable signature. Cheap relative to the pair: at
    ``tier="standard"`` this is 528 of the 688 columns for roughly a twelfth of the cost, because
    vdjtools' Pgen block dominates the other half.

    Args:
        samples: ``{sample_id: {locus: frame}}``, an iterable of pairs, or a zero-argument
            callable per sample (deferring the read into the worker keeps peak memory at
            ``O(n_jobs)`` samples rather than the whole cohort).
        tier: Column tier.
        n_jobs: Worker processes; ``1`` in-process, ``0`` every core this process may use.
        columns: Explicit subset, intersected with the ``rsig`` half.
        **kw: Passed through to :func:`rsig` (plus ``sanitise``, ``scale``, ``standardize``,
            ``clip``, ``on_duplicate``).
    """
    return _cohort(samples, _one_rsig, tier, kw, n_jobs, columns, "rsig")


def signature_cohort(samples, *, tier: str = "standard", n_jobs: int = 1,
                     columns: list[str] | None = None, **kw) -> pl.DataFrame:
    """A cohort of the **joined** signature -- both halves, one row per sample.

    This is the library constructor for the portable signature and what the scale-reference
    fitting machinery measures over. The CLI does not use it: ``mir signature`` emits
    :func:`rsig_cohort` and ``vdjtools signature`` emits ``vsig_cohort``, and a caller who wants
    the join runs both and joins on ``sample_id``.
    """
    return _cohort(samples, _one, tier, kw, n_jobs, columns, None)


def _demo() -> None:
    """Self-check: both halves land, in layout order, with holes where they belong."""
    from vdjtools.signature import layout as L

    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")

    def mk(n, v, j, c=None, seed=0):
        r = np.random.default_rng(seed)
        return pl.DataFrame(
            {"v_call": [v] * n, "j_call": [j] * n, "c_call": [c] * n,
             "junction_aa": ["C" + "".join(r.choice(aa, 12)) + "F" for _ in range(n)],
             "duplicate_count": np.ceil(r.zipf(1.5, n).clip(1, 500)).astype(int).tolist()},
            schema_overrides={"c_call": pl.Utf8})

    del rng
    sample = {"TRB": mk(1500, "TRBV20-1", "TRBJ2-2", seed=1),
              "IGH": mk(600, "IGHV1-2", "IGHJ4", "IGHM", seed=2)}

    for tier in L.TIERS:
        r = rsig(sample, tier=tier)
        assert list(r) == L.columns(tier, "rsig"), f"rsig {tier} out of layout order"

    r = rsig(sample)
    assert np.isfinite(r["rsig:depth:TRB:n_eff"]) and np.isfinite(r["rsig:div:TRB:rao"])
    assert np.isfinite(r["rsig:phic:TRB:PC01"]) and np.isfinite(r["rsig:contrast:TRB:norm"])
    assert np.isnan(r["rsig:phic:TRA:PC01"]), "an absent locus must be a hole"
    assert np.isfinite(r["rsig:band:IGH:IgM"]), "IGH isotype share should resolve"

    full = signature(sample, tier="standard")
    assert list(full) == L.columns("standard"), "combined signature out of layout order"
    n_finite = int(np.isfinite(np.array(list(full.values()))).sum())

    F = signature_cohort({"a": sample, "b": {"TRB": mk(900, "TRBV20-1", "TRBJ2-2", seed=3)}})
    assert F.columns == ["sample_id", *L.columns("standard")] and F.height == 2
    print(f"assemble OK — {len(full)} columns (vsig {len(L.columns('standard', 'vsig'))} + "
          f"rsig {len(L.columns('standard', 'rsig'))}), {n_finite} finite on a 2-locus sample")


if __name__ == "__main__":
    _demo()
