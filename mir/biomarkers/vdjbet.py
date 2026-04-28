"""VDJBet: Pgen-matched mock repertoire generation and overlap analysis.

Design
------
The key idea is to pre-build a large pool of OLGA-generated sequences organised
by their log2 Pgen bin, then for each mock simply sample from that pool rather
than running OLGA on the fly.  This avoids the fundamental problem of the old
approach — bins for ultra-rare sequences that OLGA can never fill in finite time.

Algorithm
---------
1. Build a :class:`PgenBinPool` from n (default 1 000 000) OLGA sequences
   generated in parallel.  Winsorize the floor and ceiling bins from the pool's
   own distribution (default 0.1% quantiles).
2. Compute a reference bin histogram: for each clonotype in the reference
   repertoire compute its log2 Pgen and map it to a winsorized bin.
3. For each of the n_mocks iterations sample from the pool per bin.  Any
   reference sequence whose Pgen is below the floor (or above the ceiling) is
   automatically clamped — no infinite loops.
4. Compute real vs mock overlap counts and derive z/p-scores.

V/J gene bias correction
------------------------
Pass a :class:`~mir.basic.pgen.PgenGeneUsageAdjustment` to
:class:`VDJBetOverlapAnalysis` to re-weight each generated sequence's Pgen by
its V-J factor (target usage / OLGA usage).  This calibrates the mock null to
match the target repertoire's V/J gene usage without explicit V/J stratification
of the histogram.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
from scipy.stats import norm as _scipy_norm

from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import Repertoire
from mir.comparative.overlap import (
    compute_overlaps,
    count_overlap,
    make_query_index,
    make_reference_keys,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_allele(gene: str) -> str:
    """'TRBV1*01' -> 'TRBV1'."""
    return gene.split("*")[0]


def _resolve_locus(repertoire: Repertoire) -> str:
    """Return the IMGT locus code for *repertoire*.

    Raises ValueError when the locus cannot be determined.
    """
    if repertoire.locus:
        return repertoire.locus
    from mir.common.repertoire import infer_locus
    for clone in repertoire.clonotypes:
        if clone.locus:
            return clone.locus
        if clone.v_gene:
            loc = infer_locus(clone.v_gene)
            if loc:
                return loc
    raise ValueError(
        "Cannot determine locus from repertoire. "
        "Set Repertoire.locus or populate Clonotype.locus / v_gene fields."
    )


def _log2_pgen_bin(log2_pgen: float) -> int:
    """Round log2 Pgen to the nearest integer bin."""
    return round(log2_pgen)


def compute_pgen_histogram(
    clonotypes: Sequence[Clonotype],
    model: OlgaModel,
) -> dict[int, int]:
    """Build a log2-Pgen bin histogram for *clonotypes*.

    Clonotypes with zero or undefined Pgen are silently skipped.

    Parameters
    ----------
    clonotypes:
        Source clonotypes.
    model:
        OLGA model used to compute Pgen.

    Returns
    -------
    dict
        Maps each log2-Pgen bin to the number of clonotypes in that bin.
    """
    hist: dict[int, int] = defaultdict(int)
    for clone in clonotypes:
        pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
        if pgen_val is None or pgen_val <= 0:
            continue
        hist[_log2_pgen_bin(math.log2(pgen_val))] += 1
    return dict(hist)


# ---------------------------------------------------------------------------
# PgenBinPool
# ---------------------------------------------------------------------------

class PgenBinPool:
    """Pre-built pool of OLGA sequences organized by log2 Pgen bin.

    Generates n sequences in parallel (via :meth:`~mir.basic.pgen.OlgaModel.generate_pool`)
    and groups them by ``round(log2_pgen)``.  The floor and ceiling bins are
    inferred from the pool's own distribution so that query sequences outside
    the observable OLGA range are automatically clamped — eliminating the
    infinite-loop problem of the old bin-fill approach.

    Build once, share across analyses of the same locus.  For a 1 M sequence
    pool on 8 cores the build takes ~80 s; subsequent mock generations are
    sub-second for typical reference sizes (< 1 k clonotypes).

    Parameters
    ----------
    locus:
        IMGT locus code (``'TRB'``, ``'IGH'``, etc.).
    n:
        Number of OLGA sequences to generate.  1 000 000 provides good
        coverage of all observable log2 Pgen bins with thousands of sequences
        per bin.
    n_jobs:
        Parallel worker processes.  Recommended: number of available cores.
    seed:
        Base RNG seed (worker i uses ``seed + i``).
    floor_quantile, ceil_quantile:
        Winsorization bounds.  Default 0.1% / 99.9% cuts only extreme tails
        that OLGA can generate but are essentially unobservable in real
        repertoires.
    species:
        Model species (``'human'`` or ``'mouse'``).
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  When
        supplied every record's ``log2_pgen`` is scaled by the V-J factor
        after generation so the pool reflects the target V/J distribution.

    Examples
    --------
    >>> pool = PgenBinPool("TRB", n=100_000, n_jobs=4)
    >>> analysis = VDJBetOverlapAnalysis(llw_ref, pool=pool, n_mocks=200)
    >>> result = analysis.score(query_sample)
    """

    def __init__(
        self,
        locus: str,
        n: int = 1_000_000,
        n_jobs: int = 4,
        seed: int = 42,
        floor_quantile: float = 0.001,
        ceil_quantile: float = 0.999,
        species: str = "human",
        pgen_adjustment=None,
    ) -> None:
        model = OlgaModel(locus=locus, species=species, seed=seed)
        records = model.generate_pool(n, n_jobs=n_jobs, seed=seed)

        # Apply V/J usage adjustment when requested.
        if pgen_adjustment is not None:
            for rec in records:
                if not math.isinf(rec["log2_pgen"]):
                    p_lin = 2.0 ** rec["log2_pgen"]
                    p_adj = pgen_adjustment.adjust_pgen(
                        locus, rec.get("v_gene", ""), rec.get("j_gene", ""), p_lin
                    )
                    rec["log2_pgen"] = (
                        math.log2(p_adj) if p_adj > 0 else float("-inf")
                    )

        valid_l2p = [r["log2_pgen"] for r in records if not math.isinf(r["log2_pgen"])]
        if not valid_l2p:
            raise ValueError(
                f"PgenBinPool: no valid log2 Pgen values from {n:,} OLGA records "
                f"for locus={locus!r}.  Check that the model is correctly installed."
            )

        arr = np.array(valid_l2p)
        # Winsorize: clamp bins to the observable range of the OLGA model.
        self.floor_bin: int = int(math.floor(np.quantile(arr, floor_quantile)))
        self.ceil_bin: int = int(math.ceil(np.quantile(arr, ceil_quantile)))

        self.locus: str = locus
        self.species: str = species
        self.n_generated: int = len(records)

        # Build bin -> list[record] mapping.
        self.bins: dict[int, list[dict]] = defaultdict(list)
        for rec in records:
            if math.isinf(rec["log2_pgen"]):
                continue
            b = round(rec["log2_pgen"])
            # Clamp to winsorized range.
            b = max(self.floor_bin, min(self.ceil_bin, b))
            self.bins[b].append(rec)

        self._available_bins: np.ndarray = np.array(
            sorted(self.bins), dtype=int
        )

    # ------------------------------------------------------------------
    # Bin lookup helpers
    # ------------------------------------------------------------------

    def winsorize_bin(self, b: int) -> int:
        """Clamp b to [floor_bin, ceil_bin]."""
        return max(self.floor_bin, min(self.ceil_bin, b))

    def nearest_bin(self, b: int) -> int:
        """Return the nearest non-empty bin to b (after winsorization)."""
        b = self.winsorize_bin(b)
        if b in self.bins:
            return b
        if self._available_bins.size == 0:
            raise RuntimeError("PgenBinPool has no available bins.")
        pos = int(np.searchsorted(self._available_bins, b))
        if pos <= 0:
            return int(self._available_bins[0])
        if pos >= self._available_bins.size:
            return int(self._available_bins[-1])
        left = int(self._available_bins[pos - 1])
        right = int(self._available_bins[pos])
        return left if abs(b - left) <= abs(b - right) else right

    # ------------------------------------------------------------------
    # Mock sampling
    # ------------------------------------------------------------------

    def sample_mock(
        self,
        ref_bin_counts: dict[int, int],
        rng: "np.random.Generator",
    ) -> list[tuple[str, str, str]]:
        """Sample one mock as a list of (junction_aa, v_base, j_base) tuples.

        Parameters
        ----------
        ref_bin_counts:
            {log2-Pgen bin -> count} for the reference/query sequences.
            Bins outside [floor_bin, ceil_bin] are clamped automatically.
        rng:
            NumPy random generator for reproducibility.
        """
        result: list[tuple[str, str, str]] = []
        for bin_val, count in ref_bin_counts.items():
            src_bin = self.nearest_bin(bin_val)
            pool = self.bins.get(src_bin, [])
            if not pool:
                continue
            replace = len(pool) < count
            chosen = rng.choice(len(pool), count, replace=replace)
            for i in chosen:
                rec = pool[int(i)]
                result.append((
                    rec["junction_aa"],
                    _strip_allele(rec.get("v_gene", "")),
                    _strip_allele(rec.get("j_gene", "")),
                ))
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def bin_distribution(self) -> dict[int, int]:
        """Return {bin -> count} for all non-empty bins in the pool."""
        return {b: len(recs) for b, recs in self.bins.items()}

    def log2_pgen_array(self) -> np.ndarray:
        """Return a flat array of the pool's log2 Pgen bin values (one per sequence)."""
        out: list[int] = []
        for b, recs in self.bins.items():
            out.extend([b] * len(recs))
        return np.array(out, dtype=int)


# ---------------------------------------------------------------------------
# Overlap result
# ---------------------------------------------------------------------------

@dataclass
class OverlapResult:
    """Overlap statistics for one query sample under one set of match options.

    Produced by :meth:`VDJBetOverlapAnalysis.score`.  Per-mock distributions
    are stored so the object is self-contained; z/p-scores are computed lazily.

    Attributes
    ----------
    n_total, dc_total:
        Total unique clonotypes and total duplicate count in the query.
    n, dc:
        Unique clonotypes and cells overlapping the reference.
    mock_n, mock_dc:
        Per-mock overlap counts (length == n_mocks).
    allow_1mm:
        Whether 1-substitution CDR3 matching was used.
    match_v, match_j:
        Whether V-gene / J-gene matching was required.
    """

    n_total: int
    dc_total: int
    n: int
    dc: int
    mock_n: list[int]
    mock_dc: list[int]
    allow_1mm: bool = False
    match_v: bool = True
    match_j: bool = True

    @staticmethod
    def _z_p(real: float, mocks: list) -> tuple[float, float]:
        arr = np.asarray(mocks, dtype=float)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            return (float("inf") if real > mean else 0.0), (
                0.0 if real > mean else 1.0
            )
        z = float((real - mean) / std)
        return z, float(1.0 - _scipy_norm.cdf(z))

    @property
    def frac_n(self) -> float:
        """Fraction of query clonotypes overlapping the reference."""
        return self.n / self.n_total if self.n_total else 0.0

    @property
    def frac_dc(self) -> float:
        """Fraction of query cells overlapping the reference."""
        return self.dc / self.dc_total if self.dc_total else 0.0

    @cached_property
    def _zp_n(self) -> tuple[float, float]:
        return self._z_p(self.n, self.mock_n)

    @property
    def z_n(self) -> float:
        """Z-score for unique-clonotype overlap vs the null distribution."""
        return self._zp_n[0]

    @property
    def p_n(self) -> float:
        """One-sided p-value for unique-clonotype overlap (upper tail)."""
        return self._zp_n[1]

    @cached_property
    def _dc_log2(self) -> float:
        return math.log2(self.dc + 1)

    @cached_property
    def _mock_dc_log2(self) -> list[float]:
        return [math.log2(x + 1) for x in self.mock_dc]

    @cached_property
    def _zp_dc(self) -> tuple[float, float]:
        return self._z_p(self._dc_log2, self._mock_dc_log2)

    @property
    def z_dc(self) -> float:
        """Z-score for duplicate-count overlap (log2-transformed)."""
        return self._zp_dc[0]

    @property
    def p_dc(self) -> float:
        """One-sided p-value for duplicate-count overlap (log2, upper tail)."""
        return self._zp_dc[1]


# ---------------------------------------------------------------------------
# Reference bin histogram computation
# ---------------------------------------------------------------------------

def _compute_ref_bins(
    reference: Repertoire,
    model: OlgaModel,
    pool: PgenBinPool,
    *,
    pgen_adjustment=None,
) -> dict[int, int]:
    """Compute winsorized log2-Pgen bin counts for *reference*.

    Bins are clamped to [pool.floor_bin, pool.ceil_bin] so every reference
    clonotype contributes even if its Pgen is extremely low.
    """
    locus = _resolve_locus(reference)
    hist: dict[int, int] = defaultdict(int)
    for clone in reference.clonotypes:
        pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
        if pgen_val is None or pgen_val <= 0:
            continue
        if pgen_adjustment is not None:
            pgen_val = pgen_adjustment.adjust_pgen(
                locus, clone.v_gene or "", clone.j_gene or "", pgen_val
            )
            if pgen_val <= 0:
                continue
        b = _log2_pgen_bin(math.log2(pgen_val))
        hist[pool.winsorize_bin(b)] += 1
    return dict(hist)


# ---------------------------------------------------------------------------
# VDJBetOverlapAnalysis
# ---------------------------------------------------------------------------

class VDJBetOverlapAnalysis:
    """Epitope-specific reference with a Pgen-matched mock null.

    Builds *n_mocks* mock key sets by sampling from a pre-built
    :class:`PgenBinPool`.  The pool is shared and reused across multiple
    analyses of the same locus, so the amortised cost per mock is very low.

    Parameters
    ----------
    reference:
        Epitope-specific reference repertoire (e.g. VDJdb LLW TRB clonotypes).
    pool:
        Pre-built :class:`PgenBinPool`.  When ``None``, a pool is built
        automatically using *pool_size* sequences.
    n_mocks:
        Number of Pgen-matched mock key sets.  200 provides stable z-scores.
    pool_size:
        Number of OLGA sequences to generate when *pool* is ``None``.
    n_jobs:
        Parallel workers for pool construction and overlap scoring.
    seed:
        RNG seed for reproducibility.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  Applied
        during pool construction (when *pool* is ``None``) and when computing
        reference bins.  If you supply a pre-built *pool* built with
        adjustment, pass ``None`` here (adjustment already baked in).

    Examples
    --------
    >>> pool = PgenBinPool("TRB", n=1_000_000, n_jobs=8)
    >>> analysis = VDJBetOverlapAnalysis(llw_ref, pool=pool, n_mocks=200)
    >>> result = analysis.score(query_sample)
    >>> print(f"z={result.z_n:.2f}  p={result.p_n:.4f}")
    """

    def __init__(
        self,
        reference: Repertoire,
        *,
        pool: "PgenBinPool | None" = None,
        n_mocks: int = 200,
        pool_size: int = 100_000,
        n_jobs: int = 1,
        seed: int = 42,
        pgen_adjustment=None,
        # Legacy compat — silently accepted but not used:
        cache_size=None,
        max_cache_size=None,
        infer_log2_floor_from_olga=None,
        olga_floor_sample_size=None,
        olga_floor_quantile=None,
        log2_floor_bin=None,
    ) -> None:
        self._reference = reference
        self._n_mocks = n_mocks
        self._n_jobs = n_jobs
        self._seed = seed
        self._pgen_adjustment = pgen_adjustment

        # Build pool lazily if not supplied.
        if pool is None:
            locus = _resolve_locus(reference)
            pool = PgenBinPool(
                locus,
                n=pool_size,
                n_jobs=n_jobs,
                seed=seed,
                pgen_adjustment=pgen_adjustment,
            )
        self._pool = pool

        # Model for computing reference Pgen values.
        self._model = OlgaModel(
            locus=self._pool.locus,
            species=self._pool.species,
            seed=seed,
        )

        # Lazy state.
        self._ref_bin_counts: "dict[int, int] | None" = None
        self._mock_key_sets: "list[frozenset] | None" = None
        self._mock_bin_samples: "list[list[int]] | None" = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ref_bin_counts(self) -> dict[int, int]:
        if self._ref_bin_counts is None:
            self._ref_bin_counts = _compute_ref_bins(
                self._reference,
                self._model,
                self._pool,
                pgen_adjustment=self._pgen_adjustment,
            )
        return self._ref_bin_counts

    def _build_mock_key_sets(self) -> "tuple[list[frozenset], list[list[int]]]":
        """Build n_mocks mock key sets by sampling from the pool per bin."""
        ref_bins = self._get_ref_bin_counts()

        if not ref_bins:
            warnings.warn(
                "VDJBetOverlapAnalysis: all reference clonotypes have zero or "
                "undefined Pgen; mock key sets will be empty.",
                UserWarning,
                stacklevel=3,
            )
            return (
                [frozenset() for _ in range(self._n_mocks)],
                [[] for _ in range(self._n_mocks)],
            )

        rng = np.random.default_rng(self._seed)
        mock_key_sets: list[frozenset] = []
        mock_bin_samples: list[list[int]] = []

        # Pre-resolve bins for diagnostic tracking (avoids repeated nearest_bin calls).
        resolved = {b: self._pool.nearest_bin(b) for b in ref_bins}

        for _ in range(self._n_mocks):
            tuples = self._pool.sample_mock(ref_bins, rng)
            mock_key_sets.append(frozenset(tuples))
            # Record which pool bins were used (for KS/Chi2 diagnostics).
            drawn: list[int] = []
            for b, cnt in ref_bins.items():
                drawn.extend([resolved[b]] * cnt)
            mock_bin_samples.append(drawn)

        return mock_key_sets, mock_bin_samples

    def _get_mock_key_sets(self) -> list[frozenset]:
        if self._mock_key_sets is None:
            self._mock_key_sets, self._mock_bin_samples = self._build_mock_key_sets()
        return self._mock_key_sets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        sample: Repertoire,
        *,
        allow_1mm: bool = False,
        match_v: bool = True,
        match_j: bool = True,
    ) -> OverlapResult:
        """Compute overlap statistics for *sample* against the reference.

        The mock null is built once and cached; subsequent calls with different
        ``allow_1mm`` / ``match_v`` / ``match_j`` values reuse the same mock
        key sets — only query normalization differs.

        Parameters
        ----------
        sample:
            Query repertoire.
        allow_1mm:
            Count clonotypes within one amino-acid substitution of the
            reference CDR3 in addition to exact matches.
        match_v, match_j:
            Require V-gene / J-gene match for the query<->reference overlap.

        Returns
        -------
        OverlapResult
            Overlap statistics with z/p-scores computed from the mock null.
        """
        qi = make_query_index(sample, match_v=match_v, match_j=match_j)
        n_total  = len(qi)
        dc_total = sum(qi.values())

        ref_keys = make_reference_keys(
            self._reference, allow_1mm=False, match_v=match_v, match_j=match_j,
        )
        real = count_overlap(ref_keys, qi, allow_1mm=allow_1mm)

        raw_mocks = self._get_mock_key_sets()
        # Normalise V/J fields in mock keys to match the requested match flags.
        if match_v and match_j:
            norm_mocks = raw_mocks
        else:
            norm_mocks = [
                frozenset(
                    (jaa, v if match_v else "", j if match_j else "")
                    for jaa, v, j in ks
                )
                for ks in raw_mocks
            ]

        mock_res = compute_overlaps(
            norm_mocks, qi, allow_1mm=allow_1mm, n_jobs=self._n_jobs
        )

        return OverlapResult(
            n_total=n_total,
            dc_total=dc_total,
            n=real.n,
            dc=real.dc,
            mock_n=[r.n for r in mock_res],
            mock_dc=[r.dc for r in mock_res],
            allow_1mm=allow_1mm,
            match_v=match_v,
            match_j=match_j,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_mock_bin_samples(self) -> list[list[int]]:
        """Return log2-Pgen bin draws used to build current mock key sets."""
        self._get_mock_key_sets()
        return self._mock_bin_samples or []

    def get_reference_bin_counts(self) -> dict[int, int]:
        """Return {log2-Pgen bin -> count} for the reference clonotypes."""
        return dict(self._get_ref_bin_counts())

    def get_reference_bin_sample(self) -> list[int]:
        """Return a flat list of log2-Pgen bins (one per reference clonotype)."""
        out: list[int] = []
        for b, cnt in self._get_ref_bin_counts().items():
            out.extend([b] * cnt)
        return out


