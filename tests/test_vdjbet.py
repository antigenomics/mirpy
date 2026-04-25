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

from mir.basic.pgen import OlgaModel
from mir.biomarkers.vdjbet import (
    _make_key,
    _strip_allele,
    compute_pgen_histogram,
    generate_mock_repertoire,
)
from mir.common.clonotype import Clonotype
from mir.common.repertoire import Repertoire
from tests.conftest import skip_benchmarks

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
