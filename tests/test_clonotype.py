"""Tests for the Clonotype dataclass: construction, Polars I/O, C helpers."""
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mir.common.clonotype import Clonotype, JunctionMarkup
from mir.common.parser import ClonotypeTableParser

ASSETS_DIR = Path(__file__).parent / "assets"

# VDJtools → AIRR column rename (mirrors _VDJTOOLS_TO_AIRR in parser.py)
_RENAME = {
    "count":   "duplicate_count",
    "#count":  "duplicate_count",
    "cdr3nt":  "junction",
    "cdr3aa":  "junction_aa",
    "v":       "v_gene",
    "d":       "d_gene",
    "j":       "j_gene",
    "VEnd":    "v_sequence_end",
    "DStart":  "d_sequence_start",
    "DEnd":    "d_sequence_end",
    "JStart":  "j_sequence_start",
}


def load_asset_as_polars(filename: str) -> pl.DataFrame:
    """Read a VDJtools-style asset CSV and normalise to AIRR column names."""
    df = pl.read_csv(ASSETS_DIR / filename)
    # drop the unnamed index column (first column has empty name or "")
    if df.columns[0] in ("", "column_1"):
        df = df.drop(df.columns[0])
    # strip leading # from column names, then apply AIRR rename
    df = df.rename({c: c.lstrip("#") for c in df.columns})
    df = df.rename({c: _RENAME[c] for c in df.columns if c in _RENAME})
    # keep only AIRR columns
    keep = [c for c in df.columns if c in Clonotype._POLARS_SCHEMA]
    return df.select(keep)


# ---------------------------------------------------------------------------
# from_polars
# ---------------------------------------------------------------------------

class TestFromPolars:
    def setup_method(self):
        self.df = load_asset_as_polars("repertoire_1.csv")
        self.clonotypes = Clonotype.from_polars(self.df)

    def test_count(self):
        assert len(self.clonotypes) == self.df.height

    def test_incremental_string_ids(self):
        # No sequence_id column → ids must be "0", "1", …
        assert "sequence_id" not in self.df.columns
        ids = [c.sequence_id for c in self.clonotypes]
        assert ids == [str(i) for i in range(len(self.clonotypes))]

    def test_junction_aa_set(self):
        for c in self.clonotypes:
            assert isinstance(c.junction_aa, str) and c.junction_aa

    def test_v_gene_str(self):
        for c in self.clonotypes:
            assert isinstance(c.v_gene, str)

    def test_duplicate_count(self):
        expected = self.df["duplicate_count"].to_list()
        assert [c.duplicate_count for c in self.clonotypes] == expected

    def test_junction_markup_property(self):
        c = self.clonotypes[0]
        jm = c.junction_markup
        assert isinstance(jm, JunctionMarkup)
        assert jm.v_sequence_end == c.v_sequence_end

    def test_is_coding(self):
        for c in self.clonotypes:
            # All synthetic test assets have clean AA sequences
            assert c.is_coding()

    def test_is_canonical(self):
        # At least some clonotypes should be canonical (starts with C, ends with F/W)
        assert any(c.is_canonical() for c in self.clonotypes)


# ---------------------------------------------------------------------------
# to_polars round-trip
# ---------------------------------------------------------------------------

class TestToPolars:
    def setup_method(self):
        df = load_asset_as_polars("repertoire_1.csv")
        self.clonotypes = Clonotype.from_polars(df)
        self.out = Clonotype.to_polars(self.clonotypes)

    def test_row_count(self):
        assert self.out.height == len(self.clonotypes)

    def test_columns_present(self):
        for col in Clonotype._POLARS_SCHEMA:
            assert col in self.out.columns

    def test_junction_aa_roundtrip(self):
        orig = [c.junction_aa for c in self.clonotypes]
        assert self.out["junction_aa"].to_list() == orig

    def test_sequence_id_roundtrip(self):
        orig = [c.sequence_id for c in self.clonotypes]
        assert self.out["sequence_id"].to_list() == orig

    def test_empty_list(self):
        df = Clonotype.to_polars([])
        assert df.height == 0
        assert set(Clonotype._POLARS_SCHEMA).issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

class TestClonotypeConstruction:
    def test_auto_translate_junction_aa(self):
        c = Clonotype(junction="TGTGCCAGCAGTGCCGAGCAG")
        assert c.junction_aa  # should be auto-translated
        assert c.is_coding()

    def test_gene_entry_normalised_to_str(self):
        from mir.common.gene_library import GeneEntry
        entry = GeneEntry("TRBV3-1*01")
        c = Clonotype(junction_aa="CASSEGF", v_gene=entry)
        assert isinstance(c.v_gene, str)
        assert c.v_gene == "TRBV3-1*01"

    def test_none_gene_becomes_empty_string(self):
        c = Clonotype(junction_aa="CASSEGF", v_gene=None, j_gene=None)
        assert c.v_gene == ""
        assert c.j_gene == ""

    def test_id_property_alias(self):
        c = Clonotype(sequence_id="abc123", junction_aa="CASSEGF")
        assert c.id == "abc123"

    def test_is_not_canonical(self):
        # Valid coding sequence that doesn't start with C → not canonical
        c = Clonotype(junction_aa="ASSEGF")
        assert not c.is_canonical()
        assert c.is_coding()

    def test_is_not_coding(self):
        # Non-standard characters; bypass validation for internal use
        c = Clonotype(junction_aa="XASSEGX", _validate=False)
        assert not c.is_coding()
        assert not c.is_canonical()

    def test_serialize_keys(self):
        c = Clonotype(sequence_id="1", junction_aa="CASSEGF", v_gene="TRBV3-1*01")
        d = c.serialize()
        assert set(d.keys()) == set(Clonotype._POLARS_SCHEMA.keys())


# ---------------------------------------------------------------------------
# Parser integration: ClonotypeTableParser → Clonotype
# ---------------------------------------------------------------------------

class TestParserIntegration:
    def test_parse_inner_returns_clonotypes(self):
        df = pd.read_csv(ASSETS_DIR / "repertoire_1.csv", index_col=0)
        df = ClonotypeTableParser.normalize_df(df)
        parser = ClonotypeTableParser()
        clonotypes = parser.parse_inner(df)
        assert all(isinstance(c, Clonotype) for c in clonotypes)
        assert len(clonotypes) > 0

    def test_sequence_ids_are_strings(self):
        df = pd.read_csv(ASSETS_DIR / "repertoire_1.csv", index_col=0)
        df = ClonotypeTableParser.normalize_df(df)
        clonotypes = ClonotypeTableParser().parse_inner(df)
        for c in clonotypes:
            assert isinstance(c.sequence_id, str)
