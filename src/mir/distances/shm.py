"""Somatic hypermutation as a distance, for B-cell TCREMP embeddings.

:mod:`mir.distances.germline` resolves a V call to a row of a **baked allele-to-allele** distance
matrix. For T cells that is exactly right: the V gene is germline for life, so the allele name is
a complete description of the V region. For B cells it is wrong in a specific and consequential
way — a somatically hypermutated ``IGHV3-23`` and a germline ``IGHV3-23`` resolve to the same row
and therefore embed to the same V coordinates. Every trace of the germinal centre is discarded
before the embedding starts, and two cells from opposite ends of one affinity-maturation lineage
land on top of each other.

This module restores it. The V-slot distance becomes

    d(query, prototype) = d_germline(V_query, V_prototype) + λ · penalty_SHM(query)

where ``penalty_SHM`` is the BLOSUM62 cost of the mutations the query carries relative to *its
own* germline. The additive form is the first-order model: the germline-to-germline difference
and the somatic mutations are different sets of positions, and squared distances add over
independent coordinates. It is an approximation where they overlap, and the alternative —
realigning every observed V sequence against every prototype V sequence — is a full
Smith-Waterman per pair and does not fit in an embedding hot path.

``penalty_SHM`` accepts two grades of evidence, because that is what real data offers:

**Exact** — a mutation list per clonotype (``"A23V,S31N"``, as parsed from a cigar or an
alignment pair). The penalty is the summed BLOSUM62 dissimilarity of the substitutions, on the
same ``s(a,a) + s(b,b) − 2·s(a,b)`` convention seqtree uses everywhere else, so it is a squared
distance and adds to the germline block coherently.

**Scalar** — only ``v_identity``, the fraction of the V region matching germline. Then the
mutation count is ``(1 − identity) · L_V`` and each mutation is charged the *mean* BLOSUM62
penalty over the substitutions actually observed in SHM. This is a calibrated approximation, and
the calibration is the point: charging a flat cost per mutation would make a conservative
S→T change as expensive as a W→G one.

Our current cohorts only carry the scalar. The AIRR files hold no cigar and no alignment strings,
and the SRA store carries ``v_identity`` alone — so the exact path exists for data we can produce
later (re-running the aligner with alignment output retained) rather than data we have. That is
recorded in the B-cell roadmap rather than hidden here.

**Scale, measured.** ``shm_scale=1.0`` is not a guess that happens to be 1: on human IGH the
median germline V-to-V distance is 506 (IQR 231–690), against which the scalar penalty reads

===============  ==============  ============================
V identity       SHM penalty     as a fraction of that median
===============  ==============  ============================
0.98 (light)               28.3                          0.06
0.95                       70.8                          0.14
0.90                      141.6                          0.28
0.80 (heavy)              283.3                          0.56
===============  ==============  ============================

So even a heavily mutated V sits closer to its own germline than to a *different* V gene, which
is the behaviour to want: somatic hypermutation should perturb a clonotype's position, not
relabel its V gene. Raise ``shm_scale`` to make affinity maturation the dominant axis; set it to
0 to recover T-cell behaviour exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import polars as pl

#: Typical IMGT V-region length in amino acids, used to turn a scalar identity into a mutation
#: count. Real V regions run ~95-100 aa; the constant only sets the scale of the scalar path and
#: cancels out of any comparison made at one setting.
V_REGION_AA = 98

@lru_cache(maxsize=1)
def _matrix():
    import seqtree

    return seqtree.SubstitutionMatrix.blosum62()


def substitution_penalty(a: str, b: str) -> float:
    """BLOSUM62 dissimilarity of one substitution, ``s(a,a) + s(b,b) − 2·s(a,b)``.

    This is seqtree's ``penalty``: a **squared** distance (zero on the diagonal, and a genuine
    Gram-derived metric off it), which is why it can be added to the germline block, whose entries
    are squared distances too until :meth:`TCREmp.embed` takes the uniform square root.
    """
    return float(_matrix().penalty(a, b))


@lru_cache(maxsize=4)
def mean_shm_penalty(alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> float:
    """Mean BLOSUM62 penalty over all ordered substitution pairs.

    Cached on ``alphabet``, which is the function's only argument and so covers every input that
    can change the result; bounded at 4 because a caller realistically uses one alphabet. It
    replaces a hand-rolled module-level dict that was correct but unbounded -- the same key, with
    eviction, and five fewer lines.

    The uniform-spectrum default. It is deliberately *not* the SHM-weighted mean: weighting by the
    real AID spectrum needs a mutability model, and asserting one here would bake an assumption
    into a constant where a caller cannot see it. Pass your own ``lambda_scalar`` to
    :func:`shm_penalty` if you have a spectrum.
    """
    tot = n = 0.0
    for a in alphabet:
        for b in alphabet:
            if a != b:
                tot += substitution_penalty(a, b)
                n += 1
    return tot / n


def parse_mutations(spec: str | None) -> list[tuple[str, str]]:
    """Parse ``"A23V,S31N"`` into ``[("A","V"), ("S","N")]``.

    Position is parsed and discarded: the penalty is positional-agnostic, because a per-position
    weight would need a CDR/FR mask and an argument for how much more a CDR mutation counts,
    which is a modelling choice and not a parsing one. Silent-looking entries (``A23A``) are kept
    and cost zero, so a caller's mutation count still matches.
    """
    if not spec:
        return []
    out: list[tuple[str, str]] = []
    for tok in str(spec).replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if len(tok) < 3:
            continue
        a, b = tok[0].upper(), tok[-1].upper()
        if a.isalpha() and b.isalpha():
            out.append((a, b))
    return out


def shm_penalty(mutations: str | None = None, *, identity: float | None = None,
                v_length: int = V_REGION_AA, lambda_scalar: float | None = None) -> float:
    """The SHM cost of one clonotype, as a squared distance.

    Args:
        mutations: Mutation spec, e.g. ``"A23V,S31N"``. Takes precedence when given: it is the
            better evidence.
        identity: V-region identity to germline in ``[0, 1]``, the fallback when no mutation list
            exists.
        v_length: V-region length in amino acids, for the scalar path.
        lambda_scalar: Cost charged per mutation on the scalar path. Defaults to
            :func:`mean_shm_penalty`.

    Returns:
        A non-negative squared distance. ``0.0`` for a germline sequence, and ``nan`` when there
        is **no** evidence either way — an unsequenced identity is not an unmutated V, and
        returning 0 would quietly assert that it is.
    """
    if mutations:
        return float(sum(substitution_penalty(a, b) for a, b in parse_mutations(mutations)))
    if identity is None or not np.isfinite(identity):
        return float("nan")
    lam = mean_shm_penalty() if lambda_scalar is None else lambda_scalar
    return float(max(1.0 - identity, 0.0) * v_length * lam)


@lru_cache(maxsize=1)
def _penalty_table() -> np.ndarray:
    """The whole BLOSUM62 penalty domain as a ``(26, 26)`` table, indexed by ``ord(aa) - 65``.

    :func:`substitution_penalty` is a seqtree call per substitution, and the batch path made one
    per mutation per clonotype: a real BCR repertoire carries ~15 mutations on each of ~10^5
    clonotypes, so ~1.5 million calls to answer 676 distinct questions. ``nan`` marks the
    upper-case codes BLOSUM62 does not define (J, O and U, and no others), so an unsupported
    residue stays *detectable* rather than silently scoring 0 — the scalar path raises on
    those, so this one must too.
    """
    m = _matrix()
    t = np.full((26, 26), np.nan)
    for i in range(26):
        for j in range(26):
            try:
                t[i, j] = m.penalty(chr(65 + i), chr(65 + j))
            except ValueError:
                pass                                # BLOSUM62 does not define this pair
    return t


#: Upper-case residue to its row in :func:`_penalty_table`, built from the table's own index so
#: the two cannot disagree about what is defined.
_AA_INDEX = {chr(65 + i): i for i in range(26)}


def shm_penalty_batch(mutations=None, identity=None, *, v_length: int = V_REGION_AA,
                      lambda_scalar: float | None = None) -> np.ndarray:
    """Vectorised :func:`shm_penalty` over a batch of clonotypes.

    Either argument may be ``None``; per row, a mutation spec wins over an identity. Rows with
    neither are ``nan``, which :class:`MutatedGermlineDistances` turns into "no SHM adjustment"
    rather than into a distance — see there for why those are different.

    This was a ``for`` loop calling :func:`shm_penalty` per row while its own first line said
    "vectorised", which put a string parse and one seqtree call per substitution on the embedding
    path of every frame carrying mutation evidence: **3.10 microseconds per clonotype** at 15
    mutations, 309.9 ms for 100,000. The parse is one polars pass now and the penalties one
    gather from :func:`_penalty_table`. :func:`shm_penalty` stays the scalar reference, and
    ``test_shm.py`` asserts the two agree — exactly, not within a tolerance, because every
    penalty is a small integer and a float64 sum of those is exact however it is ordered.
    """
    n = len(mutations) if mutations is not None else len(identity)
    lam = mean_shm_penalty() if lambda_scalar is None else lambda_scalar
    if mutations is not None and identity is not None and len(identity) != n:
        raise ValueError(f"mutations has {n} entries and identity {len(identity)}")
    out = np.full(n, np.nan, dtype=np.float64)

    if identity is not None:
        # `pl.Series` turns a None into a null and `to_numpy` a null into nan, which is the same
        # answer the scalar path's `identity is None or not np.isfinite(identity)` gives.
        idt = pl.Series(values=list(identity), dtype=pl.Float64, strict=False).to_numpy()
        ok = np.isfinite(idt)
        out[ok] = np.maximum(1.0 - idt[ok], 0.0) * v_length * lam

    if mutations is not None:
        # Truthiness of the RAW value, never of its string form: the scalar path branches on
        # `if mutations:`, so an int 0 takes the identity path while an int 3 does not. Casting
        # to string first would make "0" truthy and score that row from the wrong branch —
        # plausibly, and with nothing about the number to show it had. One `bool()` per row is
        # ~0.03 us against the 3.10 us this function used to spend on each of them.
        has = np.fromiter((bool(m) for m in mutations), dtype=bool, count=n)
        if has.any():
            out[has] = _spec_penalties([m for m, h in zip(mutations, has) if h])
    return out


def _spec_penalties(specs: list) -> np.ndarray:
    """Summed BLOSUM62 penalty per mutation spec, for specs already known to be truthy.

    Reproduces :func:`parse_mutations` as one polars pass: separators normalised, split, trimmed,
    tokens under three characters dropped, then the first and last characters upper-cased and
    required to be letters. A token that survives all that while naming a residue BLOSUM62 does
    not define raises, exactly as the per-row path did when seqtree rejected it.
    """
    n = len(specs)
    tok = (
        pl.DataFrame({"s": pl.Series(values=specs, strict=False).cast(pl.String, strict=False)})
        .with_row_index("row")
        .with_columns(pl.col("s").str.replace_all("[; ]", ",").str.split(",").alias("t"))
        .drop("s")
        # `empty_as_null` is pinned, not inherited: polars 2.0 flips the default to False,
        # and an empty token between two separators ("A23V,,S31N") is exactly what this
        # parser meets. Either value is dropped by the length filter below, so pinning it
        # keeps that true across the version change rather than relying on it.
        .explode("t", empty_as_null=True)
        .with_columns(pl.col("t").str.strip_chars())
        .filter(pl.col("t").str.len_chars() >= 3)
        .with_columns(a=pl.col("t").str.slice(0, 1).str.to_uppercase(),
                      b=pl.col("t").str.slice(-1, 1).str.to_uppercase())
        .with_columns(
            ai=pl.col("a").replace_strict(_AA_INDEX, default=None, return_dtype=pl.Int64),
            bi=pl.col("b").replace_strict(_AA_INDEX, default=None, return_dtype=pl.Int64))
    )
    if tok.height == 0:
        return np.zeros(n, dtype=np.float64)
    # A null index is either a first/last character that is not a letter, which
    # `parse_mutations` skips in silence, or a letter outside A-Z, which seqtree rejected. Only
    # the second is an error, and telling them apart needs `str.isalpha()`'s own domain
    # (`\p{L}`) — so that regex runs on the null rows ALONE. It was 31.3 ms of a
    # 113.5 ms pass when applied to all 1.5M tokens, and a well-formed spec has no null rows at
    # all.
    null = pl.col("ai").is_null() | pl.col("bi").is_null()
    undefined = tok.filter(null).filter(
        pl.col("a").str.contains(r"^\p{L}$") & pl.col("b").str.contains(r"^\p{L}$"))
    if undefined.height == 0:
        tok = tok.filter(~null) if tok.select(null.any()).item() else tok
        pen = _penalty_table()[tok["ai"].to_numpy(), tok["bi"].to_numpy()]
        if np.isfinite(pen).all():
            return np.bincount(tok["row"].to_numpy(), weights=pen, minlength=n)
        undefined = tok.filter(~np.isfinite(pen))
    raise ValueError(
        f"mutation spec names a residue BLOSUM62 does not define, e.g. "
        f"{undefined['t'].unique().head(5).to_list()}. The defined upper-case codes are "
        f"ABCDEFGHIKLMNPQRSTVWXYZ; J, O and U are not among them."
    )


@dataclass
class MutatedGermlineDistances:
    """A :class:`~mir.distances.germline.GermlineDistances` that can see somatic mutation.

    Wraps the baked lookup and adds the per-clonotype SHM penalty to the ``V`` component. Other
    components pass through untouched: J is mutated far less often and its short alignment makes
    an identity estimate noisy, and CDR1/CDR2 are already inside the V region the penalty covers,
    so charging them again would double-count.

    Example:
        >>> from mir.distances.germline import load_germline_distances
        >>> gd = MutatedGermlineDistances(load_germline_distances("human", "IGH"))
        >>> D = gd.matrix("V", ["IGHV3-23*01"], ["IGHV3-23*01"], shm=[0.0])
        >>> float(D[0, 0])            # germline against itself
        0.0
    """

    base: object
    scale: float = 1.0

    def matrix(self, component: str, query, prototypes, *, shm=None,
               proto_idx=None) -> np.ndarray:
        """``(n_query, n_proto)`` distances, with the SHM penalty added along ``V``.

        Args:
            component: ``V`` / ``J`` / ``CDR1`` / ``CDR2``.
            query: Query allele names.
            prototypes: Prototype allele names.
            proto_idx: The prototype panel pre-resolved by
                :meth:`~mir.distances.germline.GermlineDistances.resolve_all`; forwarded
                unchanged. See that method for why it is worth hoisting.
            shm: Per-query SHM penalty from :func:`shm_penalty_batch`, or ``None``. ``nan``
                entries are treated as **no adjustment**, not as an unknown distance: the query
                still has a germline V call, and that call's distance is a real, usable number.
                Propagating the nan would replace a known quantity with an unknown one because a
                *second*, optional quantity was missing.
        """
        D = np.asarray(self.base.matrix(component, query, prototypes,
                                        proto_idx=proto_idx), dtype=np.float32)
        if component != "V" or shm is None:
            return D
        p = np.asarray(shm, dtype=np.float64)
        if p.shape[0] != D.shape[0]:
            raise ValueError(f"shm has {p.shape[0]} entries for {D.shape[0]} queries")
        p = np.where(np.isfinite(p), p, 0.0) * self.scale
        return (D + p[:, None]).astype(np.float32)


def _demo() -> None:
    """SHM must move a mutated cell away from its own germline, and conservatively."""
    assert substitution_penalty("A", "A") == 0.0
    # A conservative substitution costs less than a radical one -- the whole reason for BLOSUM.
    assert substitution_penalty("I", "V") < substitution_penalty("W", "G")

    exact = shm_penalty("A23V,S31N")
    assert exact == substitution_penalty("A", "V") + substitution_penalty("S", "N")
    assert shm_penalty("A23A") == 0.0                    # silent change costs nothing
    assert shm_penalty(None, identity=1.0) == 0.0        # germline
    assert shm_penalty(None, identity=0.90) > 0.0

    # More mutation costs more, monotonically.
    ladder = [shm_penalty(None, identity=x) for x in (1.0, 0.98, 0.90, 0.80)]
    assert ladder == sorted(ladder), ladder

    # No evidence is nan, NOT zero: an unsequenced identity is not an unmutated V.
    assert np.isnan(shm_penalty(None, identity=None))
    assert np.isnan(shm_penalty(None, identity=float("nan")))

    b = shm_penalty_batch(["A23V", None, ""], [None, 0.9, None])
    assert b[0] > 0 and b[1] > 0 and np.isnan(b[2])

    # The wrapper adds along V only, and a nan penalty leaves the germline distance intact.
    class _Fake:
        def matrix(self, comp, q, p, *, proto_idx=None):
            return np.zeros((len(q), len(p)), dtype=np.float32) + (1.0 if comp == "V" else 5.0)

    g = MutatedGermlineDistances(_Fake())
    v = g.matrix("V", ["IGHV3-23*01", "IGHV3-23*01"], ["p1"], shm=[0.0, 12.0])
    assert v[0, 0] == 1.0 and v[1, 0] == 13.0, v
    j = g.matrix("J", ["x", "y"], ["p1"], shm=[0.0, 12.0])
    assert (j == 5.0).all(), j                            # J untouched
    keep = g.matrix("V", ["a"], ["p1"], shm=[float("nan")])
    assert keep[0, 0] == 1.0, keep                        # nan -> no adjustment, not nan distance
    print(f"shm distance: exact={exact:.1f}, mean penalty per mutation="
          f"{mean_shm_penalty():.2f}; V-only, nan-safe")


if __name__ == "__main__":
    _demo()
