"""Unit tests for the mir.utils.parser_cli AIRR-conversion CLI."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from mir.utils.parser_cli import _resolve_output, main


class TestResolveOutput:
    def test_explicit_file_returned_as_is(self):
        out = _resolve_output(Path("sample.txt.gz"), Path("out.airr.tsv"), compress=False)
        assert out == Path("out.airr.tsv")

    def test_directory_target_builds_name_into_dir(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        dst = _resolve_output(Path("sample.txt"), d, compress=False)
        assert dst == d / "sample.airr.tsv"

    def test_directory_compress_extension(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        dst = _resolve_output(Path("donorA.txt.gz"), d, compress=True)
        assert dst == d / "donorA.airr.tsv.gz"

    def test_directory_plain_extension_strips_txt(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        dst = _resolve_output(Path("donorB.tsv"), d, compress=False)
        assert dst == d / "donorB.airr.tsv"


def _write_olga(path: Path) -> None:
    # OLGA format: no header; columns junction, junction_aa, v_call, j_call
    rows = [
        "TGTGCCAGCAGC\tCASS\tTRBV5-1*01\tTRBJ2-7*01",
        "TGTGCCAGCTTT\tCASF\tTRBV20-1*01\tTRBJ2-1*01",
    ]
    path.write_text("\n".join(rows) + "\n")


class TestMain:
    def test_no_files_matched_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / "does_not_exist_*.txt"), "-o", str(tmp_path / "out.airr.tsv")])
        assert exc.value.code == 1
        assert "No files matched" in capsys.readouterr().err

    def test_single_file_olga_conversion(self, tmp_path):
        src = tmp_path / "donor.txt"
        _write_olga(src)
        out = tmp_path / "donor.airr.tsv"
        main([str(src), "-o", str(out), "-f", "olga_gen"])
        assert out.exists()
        text = out.read_text()
        assert "CASS" in text and "CASF" in text

    def test_compressed_output_inferred_from_extension(self, tmp_path):
        src = tmp_path / "donor.txt"
        _write_olga(src)
        out = tmp_path / "donor.airr.tsv.gz"
        main([str(src), "-o", str(out), "-f", "olga_gen"])
        assert out.exists()
        with gzip.open(out, "rt") as fh:
            assert "CASS" in fh.read()
