"""Explainable readouts over a repertoire feature matrix: which channel carries the signal (§T.7).

vdjtools-era repertoire analysis answers "which summary statistic separates my groups?" by running a
*fixed menu* of named statistics side by side — diversity, clonality, CDR3 length, V-usage, overlap —
one test per statistic, and reading off which one moved. The menu **is** the explanation: every number
has a name, so a result is a sentence ("the groups differ in clonality").

The embedding replaces the menu with one vector ``Φ(S)`` (:mod:`mir.repertoire`), which scores better
and explains worse: ``Φ.vector`` is an anonymous concatenation, so "the classifier found something"
has no noun. This module restores the noun without giving up the vector. A **channel** is a named
group of ``Φ`` columns (the mean/identity block, the Hill block, the second-moment block, and any
extra blocks a study bolts on — per-chain blocks of the same kind merge under one name). A
:class:`ChannelSpec` is the name→column-index map that ``Φ.vector`` alone does not carry;
:class:`ChannelBuilder` assembles one alongside the matrix, so the layout is declared where the
blocks are stacked, not rediscovered later.

Given the map, "which part of ``Φ`` carries this signal" is an ablation against a user-supplied
scorer ``f: X_block → float`` (higher is better)::

    delta_in(g)  = f(X[:, cols(g)]) − base        # does channel g carry signal on its own?
    delta_out(g) = f(X) − f(X[:, ¬cols(g)])       # is channel g's signal anywhere else?

The scorer is a closure over the study's labels, so the same call works for a Cox C-index, a
cross-validated AUC, or anything else — this module never sees ``y`` and ships no scorers of its own
(model choices belong to the analysis, not the library). ``delta_in`` is the default and is
*marginal*: it is inflated by correlation between channels, so two redundant channels both look
important. ``delta_out`` is *conditional* and deflated by the same correlation. Reported together
they separate the two claims — high in / high out = irreplaceable; high in / ~zero out =
**redundant**, its signal is duplicated elsewhere. Significance, when asked for, is a row permutation
of the block: the scorer holds ``y`` in row order, so shuffling the block's rows breaks the
association exactly as shuffling ``y`` would, and it is the only scorer-agnostic null available.

Leave-one-**in** is the default for a concrete reason: the scorers in practice reduce their input
(an in-fold PCA to ``n_pc`` components), so dropping one narrow channel out of a wide block leaves
the reduction to re-mix the rest and reconstruct nearly the same components — leave-one-out is
structurally near-blind to exactly the 1-column channels that often win. Ask for ``mode="both"``
when the question is whether the winner is *necessary* rather than *sufficient*.

The last hop, channel → **clonotypes**, exists only for channels with a clonotype pre-image: a
kernel-mean block, where :func:`mir.repertoire.class_witness` reads the MMD witness
``s(σ)=⟨μ_pos−μ_neg, ψ(φ(σ))⟩``. A Hill number or a read-count fraction has no such pre-image —
asking which clones drive the diversity channel is a category error, and :func:`channel_drivers`
raises rather than answer it. Attributability is therefore *declared* per channel at build time,
never guessed from its name.

Torch-free (numpy / polars); scorer-agnostic (no sklearn / lifelines import here).

Typical usage::

    from mir.explain import ChannelBuilder, channel_drivers, channel_report, stack_embeddings

    embs = [sample_embedding(space, s) for s in samples]
    X, spec = stack_embeddings(embs)                     # channels: mean ‖ diversity ‖ second
    rep = channel_report(X, spec, lambda B: cv_auc(B, y)[0], base=0.5)
    print(rep.frame())                                   # one row per channel: score, delta, rank
    rep.best                                             # -> "second"   (the HLA imprint)

    # multi-source assembly (per-chain blocks merge by name) + the channel -> clonotypes hop
    b = ChannelBuilder()
    for c in chains:
        b.add("identity", ident[c], attributable=True).add("diversity", hill[c])
    X, spec = b.add("coverage", log_reads).build()       # median-impute + z-score
    rep = channel_report(X, spec, lambda B: cv_cindex(rows, B), base=c_base, mode="both")
    channel_drivers(rep, space=space, pos=pos, neg=neg, candidates=cands)   # if rep.best is a KME
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from mir.repertoire import RepertoireSpace, SampleEmbedding, class_witness

_MODES = ("in", "out", "both")


@dataclass
class ChannelSpec:
    """Name → column-index map for a feature matrix — the label ``Φ.vector`` does not carry.

    ``spec[name]`` is an alias for :meth:`columns`, so a spec drops into code written against the
    plain ``dict[str, list[int]]`` it replaces (``X[:, spec["coverage"]]``, or ``spec["a"] +
    spec["b"]`` for a union of columns).

    Attributes:
        columns_by_name: Channel name → its column indices in ``X``, in assembly order.
        attributable: Channels with a clonotype pre-image (kernel-mean blocks) — the only ones
            :func:`channel_drivers` will attribute. Declared at build time, never inferred.
    """

    columns_by_name: dict[str, list[int]]
    attributable: frozenset[str] = frozenset()

    @property
    def names(self) -> list[str]:
        """Channel names in assembly order."""
        return list(self.columns_by_name)

    @property
    def width(self) -> int:
        """Total column count — must equal ``X.shape[1]``."""
        return sum(len(c) for c in self.columns_by_name.values())

    def columns(self, name: str) -> list[int]:
        """Column indices of ``name``.

        Args:
            name: Channel name.

        Returns:
            The channel's column indices in ``X``.

        Raises:
            ValueError: If ``name`` is not a channel.
        """
        try:
            return self.columns_by_name[name]
        except KeyError:
            raise ValueError(
                f"unknown channel {name!r}; known channels: {sorted(self.columns_by_name)}"
            ) from None

    def __getitem__(self, name: str) -> list[int]:
        return self.columns(name)

    def __contains__(self, name: object) -> bool:
        return name in self.columns_by_name


@dataclass
class ChannelBuilder:
    """Accumulate named blocks into one matrix + its :class:`ChannelSpec`.

    Blocks added under the *same* name merge into one channel (per-chain identity blocks become a
    single ``identity`` channel), which is the granularity ablation asks about. For per-chain
    resolution, add under distinct names (``f"identity:{chain}"``).
    """

    _blocks: list[np.ndarray] = field(default_factory=list, repr=False)
    _cols: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _attr: set[str] = field(default_factory=set, repr=False)
    _col: int = 0

    def add(self, name: str, mat: np.ndarray, *, attributable: bool = False) -> "ChannelBuilder":
        """Append a block. Returns ``self`` for chaining.

        Args:
            name: Channel name; repeat a name to merge blocks into one channel.
            mat: ``(n_samples,)`` or ``(n_samples, k)``. 1-D is treated as a single column.
            attributable: Mark the channel clonotype-attributable (a kernel-mean block). Sticky:
                one attributable block makes the whole merged channel attributable.

        Returns:
            ``self``.

        Raises:
            ValueError: On a row-count mismatch with the blocks already added.
        """
        m = np.asarray(mat, dtype=np.float64)
        if m.ndim == 1:
            m = m[:, None]
        if self._blocks and m.shape[0] != self._blocks[0].shape[0]:
            raise ValueError(
                f"channel {name!r} has {m.shape[0]} rows, expected {self._blocks[0].shape[0]}"
            )
        self._blocks.append(m)
        self._cols.setdefault(name, []).extend(range(self._col, self._col + m.shape[1]))
        self._col += m.shape[1]
        if attributable:
            self._attr.add(name)
        return self

    def build(self, *, standardize: bool = True, impute: bool = True) -> tuple[np.ndarray, ChannelSpec]:
        """Assemble ``(X, spec)``.

        Args:
            impute: Replace non-finite entries with the column median (``0.0`` if a column is all
                non-finite) — chains a sample lacks leave holes.
            standardize: Z-score each column. Default ``True`` because the builder exists for
                *heterogeneous* assembly, where a log-read-count column and a read-fraction column
                differ by orders of magnitude and any distance/PCA/penalised fit is otherwise
                dominated by the loudest unit. Pass ``False`` to keep raw values.

        Returns:
            ``(X, spec)``.

        Raises:
            ValueError: If no blocks were added.
        """
        if not self._blocks:
            raise ValueError("no blocks added; call add() before build()")
        X = np.hstack(self._blocks)
        if impute:
            for j in range(X.shape[1]):
                col = X[:, j]
                bad = ~np.isfinite(col)
                if bad.any():
                    good = col[~bad]
                    col[bad] = float(np.median(good)) if good.size else 0.0
        if standardize:
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd == 0] = 1.0
            X = (X - mu) / sd
        return X, ChannelSpec({k: list(v) for k, v in self._cols.items()}, frozenset(self._attr))


def stack_embeddings(embs: Sequence[SampleEmbedding]) -> tuple[np.ndarray, ChannelSpec]:
    """Stack ``Φ(S)`` embeddings into ``(X, spec)`` with channels named after their blocks.

    ``X`` is row-wise **identical to** ``np.stack([e.vector for e in embs])`` — this only attaches
    the names, it does not transform. No z-scoring: the blocks are one homogeneous ``Φ`` and
    rescaling them would silently change every MMD/PCA downstream.

    Args:
        embs: Per-sample embeddings from :func:`mir.repertoire.sample_embedding`, all fitted from
            one :class:`~mir.repertoire.RepertoireSpace`.

    Returns:
        ``(X, spec)`` with ``X`` shape ``(len(embs), Φ_dim)`` and channels ``mean`` / ``diversity``
        / ``second`` (whichever are present). ``mean`` is marked attributable.

    Raises:
        ValueError: If ``embs`` is empty or the embeddings disagree on present blocks / widths.
    """
    if not embs:
        raise ValueError("embs is empty")
    present = [b for b in ("mean", "diversity", "second") if getattr(embs[0], b) is not None]
    for i, e in enumerate(embs):
        p = [b for b in ("mean", "diversity", "second") if getattr(e, b) is not None]
        if p != present:
            raise ValueError(f"embs[{i}] has blocks {p}, embs[0] has {present}")

    cols: dict[str, list[int]] = {}
    start = 0
    for b in present:
        w = int(np.asarray(getattr(embs[0], b)).size)
        cols[b] = list(range(start, start + w))
        start += w
    X = np.stack([e.vector for e in embs])
    if X.shape[1] != start:
        raise ValueError(f"vector width {X.shape[1]} != summed block widths {start}")
    return X, ChannelSpec(cols, frozenset({"mean"}))


@dataclass
class ChannelReport:
    """Per-channel ablation readout — the explainable answer to "which part of ``Φ`` carries this".

    Attributes:
        channels: Channel names in report order.
        score: ``scorer(X[:, cols(g)])`` — the leave-one-in score of each channel alone.
        delta: ``score − base``. ``nan`` if no ``base`` was given.
        rank: 1 = best, by the primary statistic of ``mode``.
        base: Reference score with no channel features (Cox clinical-only, or 0.5 for chance);
            ``nan`` if not supplied.
        full: ``scorer`` over all reported channels — the "C+Φ" headline.
        n_samples: Rows scored.
        spec: The :class:`ChannelSpec` scored (provenance; also carries ``attributable``).
        mode: ``"in"`` | ``"out"`` | ``"both"``.
        delta_out: ``full − scorer(X[:, ¬cols(g)])`` — the drop caused by *removing* g, so larger =
            more uniquely necessary (same sign convention as ``delta``). ``None`` unless requested.
        pvalue: One-sided row-permutation p per channel, add-one smoothed
            (``≥ 1/(1+n_permutations)``, never 0). ``None`` unless requested.
    """

    channels: list[str]
    score: np.ndarray
    delta: np.ndarray
    rank: np.ndarray
    base: float
    full: float
    n_samples: int
    spec: ChannelSpec
    mode: str
    delta_out: np.ndarray | None = None
    pvalue: np.ndarray | None = None

    @property
    def best(self) -> str:
        """The top-ranked channel name."""
        return self.channels[int(np.argmin(self.rank))]

    @property
    def gain(self) -> float:
        """``full − base`` — the whole embedding's gain over the reference (ΔC / ΔAUC)."""
        return float(self.full - self.base)

    def frame(self) -> pl.DataFrame:
        """Long-form readout, one row per channel, sorted by ``rank``.

        Returns:
            Columns ``channel``, ``n_columns``, ``score``, ``delta``, ``rank``, ``attributable``,
            plus ``delta_out`` / ``pvalue`` when present. Pivot for the wide one-value-per-cell
            table; this returns the tidy form.
        """
        d = {
            "channel": self.channels,
            "n_columns": [len(self.spec[c]) for c in self.channels],
            "score": self.score,
            "delta": self.delta,
            "rank": self.rank,
            "attributable": [c in self.spec.attributable for c in self.channels],
        }
        if self.delta_out is not None:
            d["delta_out"] = self.delta_out
        if self.pvalue is not None:
            d["pvalue"] = self.pvalue
        return pl.DataFrame(d).sort("rank")


def _rank(x: np.ndarray) -> np.ndarray:
    """1 = best (largest). Ties get distinct ranks by first-occurrence, so rank is a permutation."""
    order = np.argsort(-x, kind="stable")
    r = np.empty(x.size, dtype=np.int64)
    r[order] = np.arange(1, x.size + 1)
    return r


def _complement(spec: ChannelSpec, names: Sequence[str], drop: str) -> list[int]:
    out: list[int] = []
    for n in names:
        if n != drop:
            out.extend(spec[n])
    return sorted(out)


def _permutation_pvalues(
    X: np.ndarray, spec: ChannelSpec, names: Sequence[str],
    scorer: Callable[[np.ndarray], float], observed: np.ndarray, n: int, seed: int,
) -> np.ndarray:
    """One-sided row-permutation p per channel; one shared permutation per round.

    The scorer holds ``y`` in row order, so shuffling the block's rows breaks the association
    exactly as shuffling ``y`` would — the only null available without seeing the labels. Assumes
    exchangeable rows. Add-one smoothed, so p is never 0.
    """
    rng = np.random.default_rng(seed)
    ge = np.zeros(len(names), dtype=np.int64)
    for _ in range(n):
        perm = rng.permutation(X.shape[0])
        for i, g in enumerate(names):
            if scorer(X[np.ix_(perm, spec[g])]) >= observed[i]:
                ge[i] += 1
    return (ge + 1.0) / (n + 1.0)


def channel_report(
    X: np.ndarray,
    spec: ChannelSpec,
    scorer: Callable[[np.ndarray], float],
    *,
    base: float | None = None,
    mode: str = "in",
    channels: Sequence[str] | None = None,
    n_permutations: int = 0,
    seed: int = 0,
) -> ChannelReport:
    """Score every channel of ``X`` under ``scorer`` and rank them (§T.7).

    Args:
        X: ``(n_samples, spec.width)`` feature matrix.
        spec: Channel map for ``X``'s columns.
        scorer: ``X_block → float``, **higher is better**, closing over the study's labels
            (``lambda B: cv_auc(B, y)[0]``, ``lambda B: cv_cindex(rows, B, n_pc=8)``). Any
            block-width-dependent choice — in-fold PCA, penaliser — belongs inside the closure;
            this function only slices columns.
        base: Reference score with **no** channel features. Cox: ``cv_cindex(rows, None)``.
            Classification: ``0.5``. ``None`` → ``delta`` is ``nan`` and ranking falls back to
            ``score`` (an identical ordering — ``delta`` is a constant shift of ``score``).
        mode: ``"in"`` (default; each channel alone vs ``base`` — the marginal question),
            ``"out"`` (each channel removed vs ``full`` — the uniqueness question), or ``"both"``.
        channels: Restrict / order the report. ``None`` → ``spec`` order. For ``"out"``/``"both"``,
            ``full`` and the complements are taken over the **union of the reported channels**.
        n_permutations: If > 0, a one-sided permutation p per channel. Costs
            ``n_permutations × len(channels)`` extra scorer calls; off by default.
        seed: Permutation RNG seed.

    Returns:
        A :class:`ChannelReport`; ``.best`` is the winning channel, ``.frame()`` the tidy table.

    Raises:
        ValueError: If ``X.shape[1] != spec.width``, ``channels`` names an unknown channel, the
            channel list is empty, or ``mode`` is not one of ``"in"``/``"out"``/``"both"``.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if X.shape[1] != spec.width:
        raise ValueError(f"X has {X.shape[1]} columns, spec describes {spec.width}")
    names = list(channels) if channels is not None else spec.names
    if not names:
        raise ValueError("no channels to report")
    for n in names:
        spec.columns(n)  # raises with a helpful message on an unknown name

    used = sorted({i for n in names for i in spec[n]})
    full = float(scorer(X[:, used]))
    score = np.array([float(scorer(X[:, spec[g]])) for g in names])
    b = float("nan") if base is None else float(base)
    delta = score - b if base is not None else np.full(score.size, np.nan)

    d_out = None
    if mode in ("out", "both"):
        d_out = np.array([full - float(scorer(X[:, _complement(spec, names, g)]))
                          if len(names) > 1 else full - b
                          for g in names])

    primary = d_out if mode == "out" else (score if base is None else delta)
    pv = None
    if n_permutations > 0:
        pv = _permutation_pvalues(X, spec, names, scorer, score, n_permutations, seed)

    return ChannelReport(
        channels=names, score=score, delta=delta, rank=_rank(primary), base=b, full=full,
        n_samples=int(X.shape[0]), spec=spec, mode=mode, delta_out=d_out, pvalue=pv,
    )


def channel_drivers(
    report: ChannelReport,
    *,
    space: RepertoireSpace,
    pos: list[pl.DataFrame],
    neg: list[pl.DataFrame],
    candidates: pl.DataFrame,
    channel: str | None = None,
    weight: str = "log1p",
    top: int = 30,
) -> pl.DataFrame:
    """Clonotypes driving a channel — the channel → clones hop (Prop. ``prop:witness``).

    Delegates to :func:`mir.repertoire.class_witness` (the MMD witness
    ``s(σ)=⟨μ_pos−μ_neg, ψ(φ(σ))⟩``) and guards it: only a channel declared ``attributable`` — a
    kernel-mean block — has a clonotype pre-image. A Hill number, a read-count fraction or a
    centroid distance is a *summary*; its "drivers" are the count distribution, not any clone, and
    asking for them is a category error this function refuses rather than answers.

    NB the witness is recomputed from ``pos``/``neg`` through ``space``'s **raw** kernel mean, so it
    attributes the channel's *source*, not the exact (typically reduced) columns that were scored.
    That is the intended reading — the reduction is a fitting convenience, the mean is the object —
    but a driver list explains ``μ_pos−μ_neg``, not the specific components.

    Args:
        report: The readout naming the channel.
        space: The shared :class:`~mir.repertoire.RepertoireSpace` the channel was built from.
        pos: Per-sample clonotype frames for the positive group (with ``duplicate_count``).
        neg: Per-sample clonotype frames for the negative group.
        candidates: Clonotype frame to score (e.g. all clonotypes seen in ``pos``).
        channel: Which channel to attribute. ``None`` → ``report.best``.
        weight: Clone-size weight for the per-sample kernel means.
        top: Number of driving clonotypes to return.

    Returns:
        ``candidates`` with ``witness_score`` (descending, truncated to ``top``) plus a ``channel``
        column, so a stacked driver frame stays self-describing.

    Raises:
        ValueError: If ``channel`` is unknown, or is not clonotype-attributable.
    """
    g = channel if channel is not None else report.best
    report.spec.columns(g)  # unknown-name guard
    if g not in report.spec.attributable:
        attr = sorted(report.spec.attributable)
        raise ValueError(
            f"channel {g!r} is not clonotype-attributable: it has no clonotype pre-image, so "
            f"'which clones drive it' is a category error. Attributable channels: "
            f"{attr if attr else '(none declared)'}"
        )
    out = class_witness(space, pos, neg, candidates, weight=weight, top=top)
    return out.with_columns(pl.lit(g).alias("channel"))


def _demo() -> None:
    """Assert-based self-check on synthetic data (no TCREmp needed, so it stays instant)."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n).astype(float)

    signal = (y[:, None] + rng.normal(0, 0.35, (n, 3)))     # carries y
    dup = signal + rng.normal(0, 0.02, (n, 3))              # a near-duplicate of it
    noise = rng.normal(0, 1, (n, 5))                        # carries nothing

    X, spec = (ChannelBuilder()
               .add("signal", signal, attributable=True)
               .add("dup", dup)
               .add("noise", noise)
               .build(standardize=False, impute=False))
    assert spec.width == X.shape[1] == 11
    assert sorted(i for g in spec.names for i in spec[g]) == list(range(11))

    def scorer(B: np.ndarray) -> float:
        # a label-correlation scorer; the point is only that the library never sees y
        return float(max(abs(np.corrcoef(B[:, j], y)[0, 1]) for j in range(B.shape[1])))

    rep = channel_report(X, spec, scorer, base=0.0, mode="both", n_permutations=50)
    assert rep.best == "signal", rep.frame()
    assert rep.rank.tolist().count(1) == 1
    assert np.allclose(rep.delta, rep.score - rep.base, equal_nan=True)

    # the redundancy invariant that justifies mode="both": both copies score high leave-one-IN,
    # but neither is necessary leave-one-OUT, because the other one covers for it.
    f = {c: i for i, c in enumerate(rep.channels)}
    assert rep.delta[f["signal"]] > 0.5 and rep.delta[f["dup"]] > 0.5
    assert abs(rep.delta_out[f["signal"]]) < 0.1 and abs(rep.delta_out[f["dup"]]) < 0.1
    assert rep.delta[f["noise"]] < 0.3

    assert rep.pvalue is not None
    assert (rep.pvalue >= 1.0 / 51).all() and (rep.pvalue <= 1.0).all()
    assert rep.pvalue[f["noise"]] == rep.pvalue.max()

    fr = rep.frame()
    assert fr["channel"][0] == "signal" and fr.height == 3

    try:
        channel_drivers(rep, space=None, pos=[], neg=[], candidates=pl.DataFrame(), channel="noise")
    except ValueError as e:
        assert "not clonotype-attributable" in str(e)
    else:  # pragma: no cover
        raise AssertionError("channel_drivers must refuse a non-attributable channel")

    print(fr)
    print("mir.explain self-check OK")


if __name__ == "__main__":
    _demo()
