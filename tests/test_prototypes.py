"""Unit tests for mir.embedding.prototypes."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import mir.embedding.prototypes as proto_mod
from mir.embedding.prototypes import (
    N_PROTOTYPES,
    list_available_prototypes,
    load_prototypes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_fixture_tsv(path: Path, rows: int = 20) -> None:
    """Write a small prototype TSV with *rows* unique entries."""
    lines = ["v_call\tj_call\tjunction_aa"]
    for i in range(rows):
        lines.append(f"TRBV{(i % 10) + 1}-1*01\tTRBJ{(i % 5) + 1}-1*01\tCASS{'A' * (i % 6 + 1)}F")
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture()
def proto_dir(tmp_path: Path, monkeypatch) -> Path:
    """Temp prototype directory with a single human_TRB.tsv fixture."""
    d = tmp_path / "prototypes"
    d.mkdir()
    _write_fixture_tsv(d / "human_TRB.tsv", rows=20)
    monkeypatch.setattr(proto_mod, "_prototypes_dir", lambda: d)
    return d


# ---------------------------------------------------------------------------
# list_available_prototypes
# ---------------------------------------------------------------------------

def test_list_available_empty(tmp_path: Path, monkeypatch) -> None:
    """Empty directory returns an empty list."""
    monkeypatch.setattr(proto_mod, "_prototypes_dir", lambda: tmp_path)
    assert list_available_prototypes() == []


def test_list_available_single(proto_dir: Path) -> None:
    """Single TSV file appears in the listing."""
    result = list_available_prototypes()
    assert ("human", "TRB") in result


def test_list_available_multiple(tmp_path: Path, monkeypatch) -> None:
    """Multiple TSV files all appear in the listing, sorted."""
    d = tmp_path / "prototypes"
    d.mkdir()
    _write_fixture_tsv(d / "human_TRB.tsv")
    _write_fixture_tsv(d / "human_TRA.tsv")
    _write_fixture_tsv(d / "mouse_TRB.tsv")
    monkeypatch.setattr(proto_mod, "_prototypes_dir", lambda: d)
    result = list_available_prototypes()
    assert set(result) == {("human", "TRB"), ("human", "TRA"), ("mouse", "TRB")}
    assert result == sorted(result)


def test_list_available_ignores_non_tsv(tmp_path: Path, monkeypatch) -> None:
    """Non-TSV files are ignored."""
    d = tmp_path / "prototypes"
    d.mkdir()
    _write_fixture_tsv(d / "human_TRB.tsv")
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    (d / "generate_prototypes.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(proto_mod, "_prototypes_dir", lambda: d)
    result = list_available_prototypes()
    assert result == [("human", "TRB")]


# ---------------------------------------------------------------------------
# load_prototypes — column shape
# ---------------------------------------------------------------------------

def test_load_returns_dataframe(proto_dir: Path) -> None:
    df = load_prototypes("human", "TRB")
    assert isinstance(df, pl.DataFrame)


def test_load_columns(proto_dir: Path) -> None:
    df = load_prototypes("human", "TRB")
    assert df.columns == ["v_call", "j_call", "junction_aa"]


def test_load_all_rows_default(proto_dir: Path) -> None:
    """n=None returns all rows in the file."""
    df = load_prototypes("human", "TRB")
    assert len(df) == 20


# ---------------------------------------------------------------------------
# load_prototypes — n parameter
# ---------------------------------------------------------------------------

def test_load_n_subset(proto_dir: Path) -> None:
    df = load_prototypes("human", "TRB", n=5)
    assert len(df) == 5


def test_load_n_equals_file_size(proto_dir: Path) -> None:
    df = load_prototypes("human", "TRB", n=20)
    assert len(df) == 20


def test_load_n_one(proto_dir: Path) -> None:
    df = load_prototypes("human", "TRB", n=1)
    assert len(df) == 1


def test_load_n_zero(proto_dir: Path) -> None:
    df = load_prototypes("human", "TRB", n=0)
    assert len(df) == 0


def test_load_n_too_large_raises(proto_dir: Path) -> None:
    """n > N_PROTOTYPES raises ValueError before any file I/O."""
    with pytest.raises(ValueError, match="exceeds the maximum"):
        load_prototypes("human", "TRB", n=N_PROTOTYPES + 1)


def test_load_n_exactly_n_prototypes_raises(proto_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """n == N_PROTOTYPES + 1 raises; n == N_PROTOTYPES is fine."""
    # n == N_PROTOTYPES must not raise (boundary)
    d = tmp_path / "exact"
    d.mkdir()
    # write a file with N_PROTOTYPES rows
    lines = ["v_call\tj_call\tjunction_aa"]
    for i in range(N_PROTOTYPES):
        lines.append(f"TRBV1-1*01\tTRBJ1-1*01\tCASS{'A' * (i % 10 + 1)}F")
    (d / "human_TRB.tsv").write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(proto_mod, "_prototypes_dir", lambda: d)

    df = load_prototypes("human", "TRB", n=N_PROTOTYPES)
    assert len(df) == N_PROTOTYPES

    with pytest.raises(ValueError):
        load_prototypes("human", "TRB", n=N_PROTOTYPES + 1)


# ---------------------------------------------------------------------------
# load_prototypes — row order is stable
# ---------------------------------------------------------------------------

def test_load_row_order_is_stable(proto_dir: Path) -> None:
    """Two consecutive loads of the same n return identical rows."""
    df1 = load_prototypes("human", "TRB", n=10)
    df2 = load_prototypes("human", "TRB", n=10)
    assert df1.equals(df2)


def test_load_subset_matches_prefix(proto_dir: Path) -> None:
    """Loading n=5 returns the first 5 rows of a full load."""
    full = load_prototypes("human", "TRB")
    subset = load_prototypes("human", "TRB", n=5)
    assert subset.equals(full.head(5))


# ---------------------------------------------------------------------------
# load_prototypes — species/locus aliases
# ---------------------------------------------------------------------------

def test_load_species_alias(proto_dir: Path) -> None:
    """Species aliases are resolved correctly."""
    df = load_prototypes("hsa", "TRB", n=3)
    assert len(df) == 3


def test_load_locus_alias(proto_dir: Path) -> None:
    """Locus aliases are resolved correctly."""
    df = load_prototypes("human", "beta", n=3)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# load_prototypes — missing file
# ---------------------------------------------------------------------------

def test_load_missing_species_locus_raises(proto_dir: Path) -> None:
    """Missing prototype file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No prototype file found"):
        load_prototypes("mouse", "TRB")


def test_load_invalid_species_raises() -> None:
    """Unknown species raises ValueError from alias normalization."""
    with pytest.raises(ValueError):
        load_prototypes("xenopus", "TRB")


def test_load_invalid_locus_raises() -> None:
    """Unknown locus raises ValueError from alias normalization."""
    with pytest.raises(ValueError):
        load_prototypes("human", "UNKNOWN_LOCUS")


# ---------------------------------------------------------------------------
# Public API surface (import-level check)
# ---------------------------------------------------------------------------

def test_public_exports() -> None:
    """Key symbols are accessible from mir.embedding."""
    from mir.embedding import N_PROTOTYPES as N  # noqa: F401
    from mir.embedding import list_available_prototypes as lap  # noqa: F401
    from mir.embedding import load_prototypes as lp  # noqa: F401

    assert N == 10_000


# ---------------------------------------------------------------------------
# Integration tests against bundled resource files
# (require generate_prototypes.py to have been run at least once)
# ---------------------------------------------------------------------------

_EXPECTED_COMBOS = [
    ("human", "IGH"),
    ("human", "IGK"),
    ("human", "IGL"),
    ("human", "TRA"),
    ("human", "TRB"),
    ("human", "TRD"),
    ("human", "TRG"),
    ("mouse", "TRA"),
    ("mouse", "TRB"),
]


def _resource_files_present() -> bool:
    from mir import get_resource_path
    from pathlib import Path as _Path
    try:
        base = _Path(get_resource_path("prototypes"))
    except Exception:
        return False
    return any(base.glob("*.tsv"))


# Skip the integration suite when resource files haven't been generated yet.
skip_no_resources = pytest.mark.skipif(
    not _resource_files_present(),
    reason="prototype resource files not present — run generate_prototypes.py first",
)


@skip_no_resources
def test_resource_all_combos_present() -> None:
    """All expected (species, locus) pairs have bundled prototype files."""
    available = set(list_available_prototypes())
    for combo in _EXPECTED_COMBOS:
        assert combo in available, f"Missing prototype file for {combo}"


@skip_no_resources
@pytest.mark.parametrize("species,locus", _EXPECTED_COMBOS)
def test_resource_load_full(species: str, locus: str) -> None:
    """Each bundled file loads to a DataFrame with exactly N_PROTOTYPES rows."""
    df = load_prototypes(species, locus)
    assert len(df) == N_PROTOTYPES
    assert df.columns == ["v_call", "j_call", "junction_aa"]


@skip_no_resources
@pytest.mark.parametrize("species,locus", _EXPECTED_COMBOS)
def test_resource_all_unique_triples(species: str, locus: str) -> None:
    """Each bundled file has N_PROTOTYPES unique (v_call, j_call, junction_aa) triples."""
    df = load_prototypes(species, locus)
    n_unique = df.unique(subset=["v_call", "j_call", "junction_aa"]).height
    assert n_unique == N_PROTOTYPES


@skip_no_resources
@pytest.mark.parametrize("species,locus", _EXPECTED_COMBOS)
def test_resource_no_null_values(species: str, locus: str) -> None:
    """No column contains null values."""
    df = load_prototypes(species, locus)
    for col in df.columns:
        assert df[col].null_count() == 0, f"Nulls in {col} for {species}/{locus}"


@skip_no_resources
def test_resource_load_n_subset() -> None:
    """n= parameter returns the first n rows of the bundled file."""
    full = load_prototypes("human", "TRB")
    subset = load_prototypes("human", "TRB", n=500)
    assert len(subset) == 500
    assert subset.equals(full.head(500))


@skip_no_resources
def test_resource_manifest_present() -> None:
    """manifest.json exists next to the prototype TSV files."""
    from mir import get_resource_path
    from pathlib import Path as _Path
    base = _Path(get_resource_path("prototypes"))
    assert (base / "manifest.json").exists()
