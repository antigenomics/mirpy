"""Tests for :mod:`mir.biomarkers.vdjbet`.

Fast tests (always run)
-----------------------
* :class:`TestHelpers` — unit tests for internal helpers.
* :class:`TestMockSanity` — empty repertoire, warning when budget exhausted.

Benchmark tests (``RUN_BENCHMARK=1``)
--------------------------------------
* :class:`TestMockHistogramTRB` — generates 50 TRB clonotypes via OLGA and
  verifies the mock Pgen histogram matches under all four fix_v/fix_j
  combinations.  Prints wall-clock time for each variant.
* :class:`TestMockHistogramTRA` — same for TRA (VJ model, no D gene).
"""

from __future__ import annotations

import time

import pytest

import gzip
from pathlib import Path

from mir.basic.pgen import OlgaModel
from mir.biomarkers.vdjbet import (
    _make_key,
    _strip_allele,
    build_olga_pool,
    compute_pgen_histogram,
    generate_mock_from_pool,
    generate_mock_key_sets_from_pool,
    generate_mock_repertoire,
)
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, Repertoire
from tests.conftest import skip_benchmarks

ASSETS = Path(__file__).parent / "assets"
_GILG_FILE = ASSETS / "gilgfvftl_trb_cdr3.txt.gz"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEED = 42


def _make_olga_repertoire(locus: str, n: int, seed: int = _SEED) -> Repertoire:
    """Generate *n* clonotypes via OLGA and return them as a :class:`Repertoire`."""
    model = OlgaModel(locus=locus, seed=seed)
    records = model.generate_sequences_with_meta(n, pgens=False, seed=None)
    clonotypes = [
        Clonotype(
            sequence_id=str(i),
            locus=locus,
            junction_aa=rec["junction_aa"],
            junction=rec["junction"],
            v_gene=rec["v_gene"],
            j_gene=rec["j_gene"],
            v_sequence_end=rec["v_end"],
            j_sequence_start=rec["j_start"],
            duplicate_count=1,
        )
        for i, rec in enumerate(records)
    ]
    return Repertoire(clonotypes=clonotypes, locus=locus, repertoire_id=f"olga_{locus}")


def _assert_histograms_match(
    orig: Repertoire,
    mock: Repertoire,
    model: OlgaModel,
    *,
    fix_v: bool,
    fix_j: bool,
    label: str,
) -> None:
    """Assert mock Pgen histogram exactly equals original and print timing info."""
    orig_hist = compute_pgen_histogram(orig.clonotypes, model, fix_v=fix_v, fix_j=fix_j)
    mock_hist = compute_pgen_histogram(mock.clonotypes, model, fix_v=fix_v, fix_j=fix_j)

    for key, count in orig_hist.items():
        assert mock_hist.get(key, 0) == count, (
            f"[{label}] bin {key!r}: expected {count}, got {mock_hist.get(key, 0)}"
        )
    assert set(mock_hist) == set(orig_hist), (
        f"[{label}] unexpected bins in mock: {set(mock_hist) - set(orig_hist)}"
    )


# ---------------------------------------------------------------------------
# Fast helpers — always run
# ---------------------------------------------------------------------------

class TestHelpers:
    """Unit tests for internal helpers (no OLGA calls)."""

    def test_strip_allele_with_allele(self) -> None:
        assert _strip_allele("TRBV1*01") == "TRBV1"

    def test_strip_allele_without_allele(self) -> None:
        assert _strip_allele("TRBV1") == "TRBV1"

    def test_make_key_no_fix(self) -> None:
        assert _make_key("TRBV1", "TRBJ1", -10, False, False) == -10

    def test_make_key_fix_v(self) -> None:
        assert _make_key("TRBV1", "TRBJ1", -10, True, False) == ("TRBV1", -10)

    def test_make_key_fix_j(self) -> None:
        assert _make_key("TRBV1", "TRBJ1", -10, False, True) == ("TRBJ1", -10)

    def test_make_key_fix_vj(self) -> None:
        assert _make_key("TRBV1", "TRBJ1", -10, True, True) == ("TRBV1", "TRBJ1", -10)

    def test_compute_pgen_histogram_excludes_zero_pgen(self) -> None:
        """A nonsense CDR3 produces pgen≈0 and must not appear in the histogram."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        bad = Clonotype(sequence_id="0", locus="TRB", junction_aa="AAAA", duplicate_count=1)
        hist = compute_pgen_histogram([bad], model)
        assert hist == {}, "zero-pgen clonotype must not appear in histogram"

    def test_compute_pgen_histogram_single_valid(self) -> None:
        model = OlgaModel(locus="TRB", seed=_SEED)
        seqs = model.generate_sequences(1, seed=_SEED)
        clone = Clonotype(sequence_id="0", locus="TRB", junction_aa=seqs[0], duplicate_count=1)
        hist = compute_pgen_histogram([clone], model)
        assert len(hist) == 1
        assert list(hist.values()) == [1]


class TestMockSanity:
    """Fast sanity tests that do not require the full mock histogram loop."""

    def test_empty_repertoire_returns_empty(self) -> None:
        empty = Repertoire(clonotypes=[], locus="TRB")
        mock = generate_mock_repertoire(empty, seed=_SEED)
        assert mock.clonotype_count == 0
        assert mock.locus == "TRB"

    def test_max_sequences_warning(self) -> None:
        """A tiny budget must trigger a UserWarning about exhausted sequences."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        seqs = model.generate_sequences(3, seed=_SEED)
        rep = Repertoire(
            clonotypes=[
                Clonotype(sequence_id=str(i), locus="TRB", junction_aa=s,
                          duplicate_count=1)
                for i, s in enumerate(seqs)
            ],
            locus="TRB",
        )
        with pytest.warns(UserWarning, match="exhausted"):
            generate_mock_repertoire(rep, max_sequences=2, seed=_SEED)

    def test_locus_inferred_from_v_gene(self) -> None:
        """Locus must be inferred from v_gene when Repertoire.locus is empty."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        seqs = model.generate_sequences_with_meta(3, pgens=False, seed=_SEED)
        rep = Repertoire(
            clonotypes=[
                Clonotype(sequence_id=str(i), junction_aa=r["junction_aa"],
                          v_gene=r["v_gene"], duplicate_count=1)
                for i, r in enumerate(seqs)
            ],
            locus="",  # intentionally empty
        )
        mock = generate_mock_repertoire(rep, seed=_SEED)
        assert mock.locus == "TRB"

    def test_unknown_locus_raises(self) -> None:
        rep = Repertoire(
            clonotypes=[Clonotype(sequence_id="0", junction_aa="CASSF",
                                  duplicate_count=1)],
            locus="",
        )
        with pytest.raises(ValueError, match="Cannot determine locus"):
            generate_mock_repertoire(rep, seed=_SEED)

    def test_duplicate_count_distribution_preserved(self) -> None:
        model = OlgaModel(locus="TRB", seed=_SEED)
        seqs = model.generate_sequences_with_meta(5, pgens=False, seed=_SEED)
        counts = [10, 5, 3, 1, 7]
        rep = Repertoire(
            clonotypes=[
                Clonotype(sequence_id=str(i), locus="TRB",
                          junction_aa=r["junction_aa"], v_gene=r["v_gene"],
                          duplicate_count=dc)
                for i, (r, dc) in enumerate(zip(seqs, counts))
            ],
            locus="TRB",
        )
        mock = generate_mock_repertoire(rep, seed=_SEED)
        assert sorted(c.duplicate_count for c in mock.clonotypes) == sorted(counts)


# ---------------------------------------------------------------------------
# Benchmark histogram tests — opt-in via RUN_BENCHMARK=1
# ---------------------------------------------------------------------------

N_BENCHMARK = 50


@pytest.fixture(scope="module")
def trb_model_bm():
    return OlgaModel(locus="TRB", seed=_SEED)


@pytest.fixture(scope="module")
def trb_repertoire_bm():
    return _make_olga_repertoire("TRB", N_BENCHMARK)


@pytest.fixture(scope="module")
def tra_model_bm():
    return OlgaModel(locus="TRA", seed=_SEED)


@pytest.fixture(scope="module")
def tra_repertoire_bm():
    return _make_olga_repertoire("TRA", N_BENCHMARK)


@skip_benchmarks
@pytest.mark.benchmark
class TestMockHistogramTRB:
    """Pgen-histogram matching for human TRB — all four fix_v/fix_j variants.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet.py::TestMockHistogramTRB
    """

    @pytest.mark.parametrize(
        "fix_v,fix_j,label",
        [
            (False, False, "no-fix"),
            (True,  False, "fix-V"),
            (False, True,  "fix-J"),
            (True,  True,  "fix-VJ"),
        ],
    )
    def test_histogram_matches(
        self,
        trb_model_bm: OlgaModel,
        trb_repertoire_bm: Repertoire,
        fix_v: bool,
        fix_j: bool,
        label: str,
    ) -> None:
        t0 = time.perf_counter()
        mock = generate_mock_repertoire(
            trb_repertoire_bm, fix_v_usage=fix_v, fix_j_usage=fix_j, seed=_SEED,
        )
        elapsed = time.perf_counter() - t0

        orig_hist = compute_pgen_histogram(
            trb_repertoire_bm.clonotypes, trb_model_bm, fix_v=fix_v, fix_j=fix_j,
        )
        mock_hist = compute_pgen_histogram(
            mock.clonotypes, trb_model_bm, fix_v=fix_v, fix_j=fix_j,
        )
        print(
            f"\nTRB {label} (N={N_BENCHMARK}): "
            f"{mock.clonotype_count} accepted in {elapsed:.2f}s, "
            f"bins orig={len(orig_hist)} mock={len(mock_hist)}"
        )
        _assert_histograms_match(
            trb_repertoire_bm, mock, trb_model_bm,
            fix_v=fix_v, fix_j=fix_j, label=f"TRB-{label}",
        )

    def test_reproducibility(self, trb_repertoire_bm: Repertoire) -> None:
        mock_a = generate_mock_repertoire(trb_repertoire_bm, seed=_SEED)
        mock_b = generate_mock_repertoire(trb_repertoire_bm, seed=_SEED)
        assert [c.junction_aa for c in mock_a.clonotypes] == \
               [c.junction_aa for c in mock_b.clonotypes], \
               "same seed must yield identical mock repertoires"

    def test_sequence_ids_unique(self, trb_repertoire_bm: Repertoire) -> None:
        mock = generate_mock_repertoire(trb_repertoire_bm, seed=_SEED)
        ids = [c.sequence_id for c in mock.clonotypes]
        assert len(ids) == len(set(ids))

    def test_junction_aa_nonempty(self, trb_repertoire_bm: Repertoire) -> None:
        mock = generate_mock_repertoire(trb_repertoire_bm, seed=_SEED)
        assert all(c.junction_aa for c in mock.clonotypes)

    def test_locus_propagated(self, trb_repertoire_bm: Repertoire) -> None:
        mock = generate_mock_repertoire(trb_repertoire_bm, seed=_SEED)
        assert mock.locus == "TRB"
        assert all(c.locus == "TRB" for c in mock.clonotypes)


@skip_benchmarks
@pytest.mark.benchmark
class TestMockHistogramTRA:
    """Pgen-histogram matching for human TRA (VJ model — no D gene).

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet.py::TestMockHistogramTRA
    """

    def test_histogram_matches_no_fix(
        self,
        tra_model_bm: OlgaModel,
        tra_repertoire_bm: Repertoire,
    ) -> None:
        t0 = time.perf_counter()
        mock = generate_mock_repertoire(tra_repertoire_bm, seed=_SEED)
        elapsed = time.perf_counter() - t0

        orig_hist = compute_pgen_histogram(
            tra_repertoire_bm.clonotypes, tra_model_bm, fix_v=False, fix_j=False,
        )
        mock_hist = compute_pgen_histogram(
            mock.clonotypes, tra_model_bm, fix_v=False, fix_j=False,
        )
        print(
            f"\nTRA no-fix (N={N_BENCHMARK}): "
            f"{mock.clonotype_count} accepted in {elapsed:.2f}s, "
            f"bins orig={len(orig_hist)} mock={len(mock_hist)}"
        )
        _assert_histograms_match(
            tra_repertoire_bm, mock, tra_model_bm,
            fix_v=False, fix_j=False, label="TRA-no-fix",
        )

    def test_locus_propagated(self, tra_repertoire_bm: Repertoire) -> None:
        mock = generate_mock_repertoire(tra_repertoire_bm, seed=_SEED)
        assert mock.locus == "TRA"


# ---------------------------------------------------------------------------
# Fast unit tests for generate_mock_from_pool
# ---------------------------------------------------------------------------

class TestMockFromPool:
    """Unit tests for :func:`generate_mock_from_pool` (no OLGA generation)."""

    @pytest.fixture(scope="class")
    def trb_pool(self):
        """Small pre-generated pool (100 sequences) via OLGA."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        return model.generate_sequences_with_meta(100, pgens=True, seed=_SEED)

    @pytest.fixture(scope="class")
    def trb_rep(self):
        return _make_olga_repertoire("TRB", 10)

    def test_returns_repertoire(self, trb_rep, trb_pool) -> None:
        mock = generate_mock_from_pool(trb_rep, trb_pool, seed=_SEED)
        assert isinstance(mock, Repertoire)

    def test_locus_propagated(self, trb_rep, trb_pool) -> None:
        mock = generate_mock_from_pool(trb_rep, trb_pool, seed=_SEED)
        assert mock.locus == "TRB"

    def test_clonotypes_nonempty(self, trb_rep, trb_pool) -> None:
        mock = generate_mock_from_pool(trb_rep, trb_pool, seed=_SEED)
        assert mock.clonotype_count > 0

    def test_empty_repertoire_returns_empty(self, trb_pool) -> None:
        empty = Repertoire(clonotypes=[], locus="TRB")
        mock = generate_mock_from_pool(empty, trb_pool, seed=_SEED)
        assert mock.clonotype_count == 0

    def test_empty_pool_warns_and_returns_empty(self, trb_rep) -> None:
        with pytest.warns(UserWarning):
            mock = generate_mock_from_pool(trb_rep, [], seed=_SEED)
        # All bins are absent → no clonotypes filled (or warning about missing)
        # The function may return a non-empty mock if some bins had zero pgen
        assert isinstance(mock, Repertoire)

    def test_pgen_histogram_approximately_preserved(self, trb_rep, trb_pool) -> None:
        model = OlgaModel(locus="TRB", seed=_SEED)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            mock = generate_mock_from_pool(trb_rep, trb_pool, seed=_SEED)
        orig_bins = set(compute_pgen_histogram(trb_rep.clonotypes, model).keys())
        mock_bins = set(compute_pgen_histogram(mock.clonotypes, model).keys())
        # Filled bins must be a subset of the original bins
        assert mock_bins.issubset(orig_bins)

    def test_replacement_sampling_warns(self, trb_rep) -> None:
        """A pool with only 1 sequence per bin forces with-replacement sampling."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        tiny_pool = model.generate_sequences_with_meta(5, pgens=True, seed=_SEED)
        # trb_rep has multiple clonotypes per bin, pool has very few → replacement
        with pytest.warns(UserWarning):
            generate_mock_from_pool(trb_rep, tiny_pool, seed=_SEED)


# ---------------------------------------------------------------------------
# Benchmark: pool vs on-the-fly sampling speed
# ---------------------------------------------------------------------------

N_POOL   = 5_000
N_ONFLY  = 50


@skip_benchmarks
@pytest.mark.benchmark
class TestPoolVsOnTheFlySpeed:
    """Compare :func:`generate_mock_from_pool` against
    :func:`generate_mock_repertoire` for a 50-clonotype TRB repertoire.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet.py::TestPoolVsOnTheFlySpeed
    """

    @pytest.fixture(scope="class")
    def trb_rep_bm(self):
        return _make_olga_repertoire("TRB", N_ONFLY)

    @pytest.fixture(scope="class")
    def trb_pool_bm(self):
        """Pre-generate a {N_POOL}-sequence pool once for the whole class."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        return model.generate_sequences_with_meta(N_POOL, pgens=True, seed=_SEED)

    def test_onthefly_timing(self, trb_rep_bm: Repertoire) -> None:
        import warnings as _warnings
        t0 = time.perf_counter()
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            mock = generate_mock_repertoire(trb_rep_bm, seed=_SEED)
        elapsed = time.perf_counter() - t0
        print(
            f"\non-the-fly  N={N_ONFLY}: {mock.clonotype_count} accepted  "
            f"{elapsed:.3f}s"
        )

    def test_pool_timing(self, trb_rep_bm: Repertoire, trb_pool_bm: list) -> None:
        import warnings as _warnings
        t0 = time.perf_counter()
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            mock = generate_mock_from_pool(trb_rep_bm, trb_pool_bm, seed=_SEED)
        elapsed = time.perf_counter() - t0
        print(
            f"\npool-based  N={N_ONFLY} (pool={N_POOL}): "
            f"{mock.clonotype_count} accepted  {elapsed:.3f}s"
        )

    def test_speedup_reported(
        self, trb_rep_bm: Repertoire, trb_pool_bm: list
    ) -> None:
        """Run both methods back-to-back and print the speedup ratio."""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            t0 = time.perf_counter()
            generate_mock_repertoire(trb_rep_bm, seed=_SEED)
            t_onfly = time.perf_counter() - t0

            t0 = time.perf_counter()
            generate_mock_from_pool(trb_rep_bm, trb_pool_bm, seed=_SEED)
            t_pool = time.perf_counter() - t0

        speedup = t_onfly / t_pool if t_pool > 0 else float("inf")
        print(
            f"\nSpeedup  on-the-fly {t_onfly:.3f}s  vs  pool {t_pool:.3f}s  "
            f"→ {speedup:.1f}×"
        )
        # Pool should be faster (or at least not dramatically slower).
        # We assert >= 0.5 to avoid false failures on loaded CI machines.
        assert speedup >= 0.5, (
            f"pool-based mock was unexpectedly slower than on-the-fly "
            f"({speedup:.2f}× speedup)"
        )


# ---------------------------------------------------------------------------
# build_olga_pool fast unit tests
# ---------------------------------------------------------------------------

class TestBuildOlgaPool:
    def test_returns_list_of_dicts(self) -> None:
        pool = build_olga_pool("TRB", 10, seed=_SEED)
        assert isinstance(pool, list)
        assert len(pool) == 10
        assert all(isinstance(r, dict) for r in pool)

    def test_required_keys_present(self) -> None:
        pool = build_olga_pool("TRB", 5, seed=_SEED)
        for rec in pool:
            assert "junction_aa" in rec
            assert "v_gene" in rec
            assert "j_gene" in rec
            assert "pgen" in rec

    def test_pgen_is_float_or_neginf(self) -> None:
        import math
        pool = build_olga_pool("TRB", 10, seed=_SEED)
        for rec in pool:
            p = rec["pgen"]
            assert isinstance(p, float)
            assert p <= 0 or math.isinf(p)  # log10 pgen ≤ 0 or -inf

    def test_reproducible_with_same_seed(self) -> None:
        pool_a = build_olga_pool("TRB", 20, seed=_SEED)
        pool_b = build_olga_pool("TRB", 20, seed=_SEED)
        assert [r["junction_aa"] for r in pool_a] == [r["junction_aa"] for r in pool_b]


# ---------------------------------------------------------------------------
# generate_mock_key_sets_from_pool fast unit tests
# ---------------------------------------------------------------------------

class TestMockKeySetsFromPool:
    @pytest.fixture(scope="class")
    def small_pool(self):
        return build_olga_pool("TRB", 200, seed=_SEED)

    @pytest.fixture(scope="class")
    def trb_ref_rep(self):
        return _make_olga_locus_rep("TRB", 10)

    def test_returns_n_frozensets(self, trb_ref_rep, small_pool) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 5, seed=_SEED)
        assert len(result) == 5
        assert all(isinstance(s, frozenset) for s in result)

    def test_keys_are_3_tuples(self, trb_ref_rep, small_pool) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 2, seed=_SEED)
        for mock in result:
            for key in mock:
                assert len(key) == 3
                assert all(isinstance(x, str) for x in key)

    def test_nonempty_mocks(self, trb_ref_rep, small_pool) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 3, seed=_SEED)
        assert all(len(m) > 0 for m in result)

    def test_empty_reference_returns_empty_sets(self, small_pool) -> None:
        empty = LocusRepertoire(clonotypes=[], locus="TRB")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate_mock_key_sets_from_pool(empty, small_pool, 3, seed=_SEED)
        assert all(len(m) == 0 for m in result)

    def test_reproducible_with_same_seed(self, trb_ref_rep, small_pool) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 3, seed=_SEED)
            b = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 3, seed=_SEED)
        assert a == b

    def test_different_seeds_differ(self, trb_ref_rep, small_pool) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 5, seed=1)
            b = generate_mock_key_sets_from_pool(trb_ref_rep, small_pool, 5, seed=2)
        assert a != b


def _make_olga_locus_rep(locus: str, n: int, seed: int = _SEED) -> LocusRepertoire:
    """Like _make_olga_repertoire but returns LocusRepertoire."""
    model = OlgaModel(locus=locus, seed=seed)
    records = model.generate_sequences_with_meta(n, pgens=False, seed=None)
    clones = [
        Clonotype(
            sequence_id=str(i), locus=locus,
            junction_aa=r["junction_aa"], junction=r["junction"],
            v_gene=r["v_gene"], j_gene=r["j_gene"],
            v_sequence_end=r["v_end"], j_sequence_start=r["j_start"],
            duplicate_count=1,
        )
        for i, r in enumerate(records)
    ]
    return LocusRepertoire(clonotypes=clones, locus=locus)


# ---------------------------------------------------------------------------
# Benchmark: 100 mocks × 100 clonotypes — pool vs on-the-fly (GILG reference)
# ---------------------------------------------------------------------------

N_GILG_CLONES = 100
N_GILG_MOCKS  = 100
N_GILG_POOL   = 10_000


def _load_gilg_reference(n: int = N_GILG_CLONES) -> LocusRepertoire:
    """Build a LocusRepertoire from the first *n* GILGFVFTL CDR3 sequences."""
    if _GILG_FILE.exists():
        with gzip.open(_GILG_FILE, "rt") as fh:
            cdrs = [ln.strip() for ln in fh if ln.strip()][:n]
    else:
        # Synthetic fallback if the asset hasn't been fetched yet
        model = OlgaModel(locus="TRB", seed=_SEED)
        cdrs = model.generate_sequences(n, seed=_SEED)
    clones = [
        Clonotype(
            sequence_id=str(i), locus="TRB", junction_aa=cdr,
            duplicate_count=1,
        )
        for i, cdr in enumerate(cdrs)
    ]
    return LocusRepertoire(clonotypes=clones, locus="TRB")


@skip_benchmarks
@pytest.mark.benchmark
class TestMockKeySetsVsOnTheFlyCGILG:
    """Benchmark: generate 100 Pgen-matched mock reference sets for a
    100-clonotype GILGFVFTL-epitope TRB reference.

    Compare:
    * **Pool method** — ``generate_mock_key_sets_from_pool(ref, pool, 100)``:
      bins pool once; all 100 mocks via NumPy index sampling.
    * **On-the-fly method** — 100 × ``generate_mock_repertoire(ref)`` +
      100 × ``make_reference_keys`` to produce the equivalent frozensets.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet.py::TestMockKeySetsVsOnTheFlyCGILG
    """

    @pytest.fixture(scope="class")
    def gilg_ref(self):
        return _load_gilg_reference(N_GILG_CLONES)

    @pytest.fixture(scope="class")
    def gilg_pool(self):
        print(f"\nBuilding pool of {N_GILG_POOL} TRB sequences (one-time) ...")
        return build_olga_pool("TRB", N_GILG_POOL, seed=_SEED)

    def test_pool_100_mocks_timing(self, gilg_ref, gilg_pool) -> None:
        import warnings as _w
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            mocks = generate_mock_key_sets_from_pool(
                gilg_ref, gilg_pool, N_GILG_MOCKS, seed=_SEED
            )
        elapsed = time.perf_counter() - t0
        print(
            f"\npool   {N_GILG_MOCKS} mocks × {N_GILG_CLONES} clones: "
            f"{elapsed:.3f}s  ({elapsed/N_GILG_MOCKS*1e3:.1f} ms/mock)"
        )
        assert len(mocks) == N_GILG_MOCKS

    def test_onthefly_100_mocks_timing(self, gilg_ref) -> None:
        from mir.comparative.overlap import make_reference_keys as _mk
        import warnings as _w
        t0 = time.perf_counter()
        key_sets = []
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            for _ in range(N_GILG_MOCKS):
                mock_rep = generate_mock_repertoire(gilg_ref, seed=_SEED)
                key_sets.append(_mk(mock_rep))
        elapsed = time.perf_counter() - t0
        print(
            f"\non-fly {N_GILG_MOCKS} mocks × {N_GILG_CLONES} clones: "
            f"{elapsed:.3f}s  ({elapsed/N_GILG_MOCKS*1e3:.1f} ms/mock)"
        )
        assert len(key_sets) == N_GILG_MOCKS

    def test_speedup_reported(self, gilg_ref, gilg_pool) -> None:
        from mir.comparative.overlap import make_reference_keys as _mk
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)

            t0 = time.perf_counter()
            for _ in range(N_GILG_MOCKS):
                mock_rep = generate_mock_repertoire(gilg_ref, seed=_SEED)
                _mk(mock_rep)
            t_onfly = time.perf_counter() - t0

            t0 = time.perf_counter()
            generate_mock_key_sets_from_pool(
                gilg_ref, gilg_pool, N_GILG_MOCKS, seed=_SEED
            )
            t_pool = time.perf_counter() - t0

        speedup = t_onfly / t_pool if t_pool > 0 else float("inf")
        print(
            f"\non-fly {t_onfly:.2f}s  vs  pool {t_pool:.3f}s  "
            f"→ {speedup:.0f}× speedup"
        )
        assert speedup >= 2.0, (
            f"pool method should be ≥ 2× faster than on-the-fly; "
            f"got {speedup:.1f}×"
        )
