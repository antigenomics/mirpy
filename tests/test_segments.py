import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mir.common.segments import Segment, SegmentLibrary


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
            requested_urls = []

            def fake_get_resource_path(name=None):
                if name == "segments/segments.txt":
                    return str(segments_path)
                raise AssertionError(f"Unexpected resource request: {name}")

            def fake_urlopen(url):
                requested_urls.append(url)
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
            self.assertTrue(all("/Homo_sapiens/" in url for url in requested_urls))

            contents = segments_path.read_text(encoding="utf-8")
            self.assertIn("HomoSapiens\tTRG\tVariable\tTRGV1*01\tATGC", contents)
            self.assertIn("HomoSapiens\tTRG\tJoining\tTRGJ1*01\tTTTGGG", contents)

    def test_load_default_accepts_string_organism_argument(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            segments_path = Path(tmpdir) / "segments.txt"
            segments_path.write_text("organism\tgene\tstype\tid\tseqnt\n", encoding="utf-8")
            requested_urls = []

            def fake_get_resource_path(name=None):
                if name == "segments/segments.txt":
                    return str(segments_path)
                raise AssertionError(f"Unexpected resource request: {name}")

            def fake_urlopen(url):
                requested_urls.append(url)
                if url.endswith("/IG/IGHV.fasta"):
                    return _FakeResponse(">x|IGHV1*01||||||||||||||ATGC\n")
                if url.endswith("/IG/IGHJ.fasta"):
                    return _FakeResponse(">x|IGHJ1*01||||||||||||||TTTGGG\n")
                raise AssertionError(f"Unexpected IMGT URL: {url}")

            with patch("mir.common.segments.get_resource_path", side_effect=fake_get_resource_path):
                with patch("mir.common.segments.urllib.request.urlopen", side_effect=fake_urlopen):
                    lib = SegmentLibrary.load_default(genes={"IGH"}, organisms="HomoSapiens")

            self.assertIn("IGHV1*01", lib.segments)
            self.assertIn("IGHJ1*01", lib.segments)
            self.assertEqual(len(requested_urls), 2)
            self.assertTrue(all("/Homo_sapiens/" in url for url in requested_urls))

    def test_get_or_create_noallele_uses_min_available_allele(self):
        lib = SegmentLibrary({
            "IGHV3-43D*03": Segment(
                id="IGHV3-43D*03",
                organism="HomoSapiens",
                gene="IGH",
                stype="V",
                seqnt="ATGC",
            ),
            "IGHV3-43D*06": Segment(
                id="IGHV3-43D*06",
                organism="HomoSapiens",
                gene="IGH",
                stype="V",
                seqnt="ATGC",
            ),
        }, complete=True)

        segment = lib.get_or_create_noallele("IGHV3-43D")
        self.assertEqual(segment.id, "IGHV3-43D*03")

    def test_get_or_create_preserves_slash_in_segment_ids(self):
        lib = SegmentLibrary({
            "TRAV29/DV5*01": Segment(
                id="TRAV29/DV5*01",
                organism="HomoSapiens",
                gene="TRA",
                stype="V",
                seqnt="ATGC",
            ),
        }, complete=True)

        self.assertEqual(lib.get_or_create("TRAV29/DV5*01").id, "TRAV29/DV5*01")
        self.assertEqual(lib.get_or_create_noallele("TRAV29/DV5").id, "TRAV29/DV5*01")


if __name__ == "__main__":
    unittest.main()
