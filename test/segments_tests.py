import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mir.common.segments import SegmentLibrary


class _FakeResponse:
    def __init__(self, text: str):
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")


class TestSegmentLibraryBootstrap(unittest.TestCase):
    def test_load_default_downloads_missing_segments_and_updates_default_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            segments_path = Path(tmpdir) / "segments.txt"
            segments_path.write_text("organism\tgene\tstype\tid\tseqnt\n", encoding="utf-8")

            def fake_get_resource_path(name=None):
                if name == "segments.txt":
                    return str(segments_path)
                raise AssertionError(f"Unexpected resource request: {name}")

            def fake_urlopen(url):
                if url.endswith("/TR/TRGV.fasta"):
                    return _FakeResponse(">x|TRGV1*01||||||||||||||ATGC\n")
                if url.endswith("/TR/TRGJ.fasta"):
                    return _FakeResponse(">x|TRGJ1*01||||||||||||||TTTGGG\n")
                raise AssertionError(f"Unexpected IMGT URL: {url}")

            with patch("mir.common.segments.get_resource_path", side_effect=fake_get_resource_path):
                with patch("mir.common.segments.urllib.request.urlopen", side_effect=fake_urlopen):
                    lib = SegmentLibrary.load_default(genes={"TRG"}, organisms={"HomoSapiens"})

            self.assertIn("TRGV1*01", lib.segments)
            self.assertIn("TRGJ1*01", lib.segments)

            contents = segments_path.read_text(encoding="utf-8")
            self.assertIn("HomoSapiens\tTRG\tVariable\tTRGV1*01\tATGC", contents)
            self.assertIn("HomoSapiens\tTRG\tJoining\tTRGJ1*01\tTTTGGG", contents)


if __name__ == "__main__":
    unittest.main()
