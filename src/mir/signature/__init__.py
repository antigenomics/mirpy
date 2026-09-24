"""mir.signature — the geometry half of the portable repertoire signature.

The column contract lives in :mod:`vdjtools.signature`, which mirpy already depends on, so
there is one implementation of the layout, the transforms and the frozen reference rescaling
rather than two that can drift apart. This package supplies the ``rsig`` blocks: features of
the prototype-sum measure ``Φ(S) = Σ_σ w_σ z_σ``, each a linear functional, a norm, or a
mixture coefficient of it.
"""
# Re-exported so a caller of ``mir.signature`` never has to know the contract is implemented in
# vdjtools. One layout, two import paths — not two layouts.
from vdjtools.signature.layout import (
    CHANNELS,
    LOCI,
    TIERS,
    channel,
    channel_table,
    channels,
    columns,
    describe,
    index,
    parse,
)

from .assemble import rsig, signature, signature_cohort
from .blocks import (
    BANDS,
    CHUNK,
    ISOTYPE_BANDS,
    WEIGHTS,
    band_shares,
    depth_block,
    isotype_shares,
    prototype_sum,
    slots,
    weights,
)
from .scale import (
    MIN_N_OBS,
    MODELS,
    ScaleReference,
    fit_scale,
    load_scale,
    measure_constants,
    save_scale,
    KMER_PATH,
    load_kmer_spaces,
)
from .reference import (
    DEFAULT_PATH,
    LocusReference,
    SignatureReference,
    load_reference,
    self_test,
)

def channel_spec(tier: str = "standard", *, columns: list[str] | None = None,
                 per_locus: bool = False):
    """The signature's channel map, as a :class:`mir.explain.ChannelSpec`.

    The bridge between the two halves of "which channel carries this signal": the layout knows
    which columns form a channel and which channels have a clonotype pre-image, and
    :mod:`mir.explain` knows how to ablate them against a scorer. Neither needs to learn the
    other's job::

        from mir.signature import channel_spec, signature_cohort
        from mir.explain import channel_report

        F = signature_cohort(samples, tier="standard")
        X = F.drop("sample_id").to_numpy()
        rep = channel_report(X, channel_spec("standard", columns=F.columns[1:]),
                             lambda B: cv_auc(B, y))
        rep.best                              # -> "vsig:div"

    Args:
        tier: Tier to index, when ``columns`` is not given.
        columns: Index *these* columns instead — typically ``frame.columns[1:]``, the emitted
            frame minus ``sample_id``. Indices are positions in this list, so they line up with
            the matrix you pass to :func:`mir.explain.channel_report`.
        per_locus: Key by ``"<sig>:<channel>:<locus>"``, so an ablation names the locus.

    Returns:
        A :class:`mir.explain.ChannelSpec` over those columns.
    """
    from vdjtools.signature.layout import registry

    from mir.explain import ChannelSpec

    idx = channels(tier, columns=columns, per_locus=per_locus)
    attr = {f"{b.sig}:{b.name}" for b in registry() if b.attributable}
    keys = frozenset(k for k in idx if (k.rsplit(":", 1)[0] if per_locus else k) in attr)
    return ChannelSpec(columns_by_name=idx, attributable=keys)


__all__ = [
    "BANDS",
    "CHANNELS",
    "channel",
    "channel_spec",
    "channel_table",
    "channels",
    "DEFAULT_PATH",
    "CHUNK",
    "LOCI",
    "TIERS",
    "columns",
    "describe",
    "index",
    "parse",
    "ISOTYPE_BANDS",
    "WEIGHTS",
    "band_shares",
    "depth_block",
    "isotype_shares",
    "prototype_sum",
    "slots",
    "KMER_PATH",
    "MIN_N_OBS",
    "MODELS",
    "load_kmer_spaces",
    "LocusReference",
    "ScaleReference",
    "fit_scale",
    "load_scale",
    "measure_constants",
    "save_scale",
    "SignatureReference",
    "load_reference",
    "rsig",
    "self_test",
    "signature",
    "signature_cohort",
    "weights",
]
