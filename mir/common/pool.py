"""Pooling utilities for repertoire objects.

Supports pooling across:
- list[LocusRepertoire]
- list[SampleRepertoire]
- RepertoireDataset

Pooling is performed independently per locus.
"""

from __future__ import annotations

from copy import copy
from typing import Iterable, Literal

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset

PoolRule = Literal["ntvj", "nt", "aavj", "aa"]
_VALID_POOL_RULES: set[str] = {"ntvj", "nt", "aavj", "aa"}


def _pool_key(cl: Clonotype, rule: PoolRule) -> tuple[str, ...]:
    if rule == "ntvj":
        return (str(cl.junction or ""), str(cl.v_gene or ""), str(cl.j_gene or ""))
    if rule == "nt":
        return (str(cl.junction or ""),)
    if rule == "aavj":
        return (str(cl.junction_aa or ""), str(cl.v_gene or ""), str(cl.j_gene or ""))
    return (str(cl.junction_aa or ""),)


def _freeze_metadata(meta: dict) -> tuple[tuple[str, str], ...]:
    if not meta:
        return ()
    return tuple(sorted((str(k), repr(v)) for k, v in meta.items()))


def _row_signature(cl: Clonotype) -> tuple:
    return (
        str(cl.sequence_id or ""),
        str(cl.junction or ""),
        str(cl.junction_aa or ""),
        str(cl.v_gene or ""),
        str(cl.d_gene or ""),
        str(cl.j_gene or ""),
        str(cl.c_gene or ""),
        int(cl.v_sequence_end),
        int(cl.d_sequence_start),
        int(cl.d_sequence_end),
        int(cl.j_sequence_start),
        _freeze_metadata(cl.clone_metadata),
    )


def _iter_sample_repertoires(
    repertoires: RepertoireDataset | Iterable[LocusRepertoire | SampleRepertoire] | LocusRepertoire | SampleRepertoire,
) -> list[tuple[str, SampleRepertoire]]:
    out: list[tuple[str, SampleRepertoire]] = []

    if isinstance(repertoires, RepertoireDataset):
        for sid, srep in repertoires.samples.items():
            sid_norm = str(sid) if sid else "sample"
            out.append((sid_norm, srep))
        return out

    if isinstance(repertoires, SampleRepertoire):
        sid = repertoires.sample_id or "sample_0"
        out.append((sid, repertoires))
        return out

    if isinstance(repertoires, LocusRepertoire):
        sid = repertoires.repertoire_id or "sample_0"
        out.append((sid, SampleRepertoire(loci={repertoires.locus: repertoires}, sample_id=sid)))
        return out

    for i, rep in enumerate(repertoires):
        if isinstance(rep, SampleRepertoire):
            sid = rep.sample_id or f"sample_{i}"
            out.append((sid, rep))
        elif isinstance(rep, LocusRepertoire):
            sid = rep.repertoire_id or f"sample_{i}"
            out.append((sid, SampleRepertoire(loci={rep.locus: rep}, sample_id=sid)))
        else:
            raise TypeError(
                "repertoires must contain LocusRepertoire and/or SampleRepertoire objects"
            )

    return out


def _pool_locus_entries(
    entries: list[tuple[Clonotype, str]],
    *,
    locus: str,
    rule: PoolRule,
    weighted: bool,
    include_sample_ids: bool,
    sample_ids_field: str,
) -> list[Clonotype]:
    grouped: dict[tuple[str, ...], dict] = {}
    for cl, sid in entries:
        k = _pool_key(cl, rule)
        if k not in grouped:
            grouped[k] = {
                "dup_sum": 0,
                "occurrences": 0,
                "sample_ids": set(),
                "rep_scores": {},
                "rep_rows": {},
            }
        g = grouped[k]
        dup = int(cl.duplicate_count)
        g["dup_sum"] += dup
        g["occurrences"] += 1
        g["sample_ids"].add(sid)

        sig = _row_signature(cl)
        score_add = dup if weighted else 1
        g["rep_scores"][sig] = int(g["rep_scores"].get(sig, 0)) + score_add
        if sig not in g["rep_rows"]:
            g["rep_rows"][sig] = cl

    pooled_clonotypes: list[Clonotype] = []
    for g in grouped.values():
        best_sig = max(g["rep_scores"], key=lambda sig: (g["rep_scores"][sig], sig))
        best_row: Clonotype = g["rep_rows"][best_sig]

        rep = copy(best_row)
        rep.clone_metadata = dict(best_row.clone_metadata)
        rep.duplicate_count = int(g["dup_sum"])
        rep.locus = locus
        rep.clone_metadata["incidence"] = len(g["sample_ids"])
        rep.clone_metadata["occurrences"] = int(g["occurrences"])
        if include_sample_ids:
            rep.clone_metadata[sample_ids_field] = sorted(g["sample_ids"])
        pooled_clonotypes.append(rep)

    pooled_clonotypes.sort(key=lambda c: c.duplicate_count, reverse=True)
    return pooled_clonotypes


def pool_samples(
    repertoires: RepertoireDataset | Iterable[LocusRepertoire | SampleRepertoire] | LocusRepertoire | SampleRepertoire,
    *,
    rule: PoolRule = "ntvj",
    weighted: bool = True,
    include_sample_ids: bool = False,
    sample_ids_field: str = "sample_ids",
) -> LocusRepertoire | SampleRepertoire:
    """Pool clonotypes across samples with per-locus grouping.

    Parameters
    ----------
    repertoires
        Input repertoire objects. Can be a dataset, a single repertoire,
        or an iterable of repertoire objects.
    rule
        Pooling key definition:
        - ``"ntvj"``: ``(junction, v_gene, j_gene)``
        - ``"nt"``: ``(junction,)``
        - ``"aavj"``: ``(junction_aa, v_gene, j_gene)``
        - ``"aa"``: ``(junction_aa,)``
    weighted
        If True, representative row is selected by summed ``duplicate_count``.
        If False, representative row is selected by raw row occurrences.
    include_sample_ids
        If True, stores sorted list of contributing ``sample_id`` values in
        clonotype metadata under ``sample_ids_field``.
    sample_ids_field
        Metadata key used when ``include_sample_ids=True``.

    Returns
    -------
    LocusRepertoire | SampleRepertoire
        Pooled repertoire. Returns ``LocusRepertoire`` when exactly one locus
        is pooled, else returns ``SampleRepertoire`` with ``sample_id='pool'``.

    Examples
    --------
    Pool two TRB repertoires by nucleotide+V+J identity::

        pooled = pool_samples([rep_a, rep_b], rule="ntvj", weighted=True)

    Pool a full dataset and attach contributing sample ids to each pooled clone::

        pooled = pool_samples(dataset, rule="aavj", include_sample_ids=True)
    """
    if rule not in _VALID_POOL_RULES:
        raise ValueError("rule must be one of: 'ntvj', 'nt', 'aavj', 'aa'")

    samples = _iter_sample_repertoires(repertoires)
    if not samples:
        return LocusRepertoire(clonotypes=[], locus="", repertoire_id="pool")

    by_locus: dict[str, list[tuple[Clonotype, str]]] = {}
    for sid, srep in samples:
        sid_norm = str(sid)
        for locus, lr in srep.loci.items():
            if locus not in by_locus:
                by_locus[locus] = []
            by_locus[locus].extend((cl, sid_norm) for cl in lr.clonotypes)

    pooled_loci: dict[str, LocusRepertoire] = {}
    for locus, entries in by_locus.items():
        pooled_clonotypes = _pool_locus_entries(
            entries,
            locus=locus,
            rule=rule,
            weighted=weighted,
            include_sample_ids=include_sample_ids,
            sample_ids_field=sample_ids_field,
        )
        pooled_loci[locus] = LocusRepertoire(
            clonotypes=pooled_clonotypes,
            locus=locus,
            repertoire_id="pool",
            repertoire_metadata={"pool_rule": rule, "weighted": weighted, "n_input_samples": len(samples)},
        )

    if len(pooled_loci) == 1:
        return next(iter(pooled_loci.values()))

    return SampleRepertoire(
        loci=pooled_loci,
        sample_id="pool",
        sample_metadata={"pool_rule": rule, "weighted": weighted, "n_input_samples": len(samples)},
    )
