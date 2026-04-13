"""Unit tests for the ``mirseq`` C extension.

Covers:
  - Codon translation: linear and bidirectional (comprehensive)
  - BioPython cross-checks for translation
  - AA → reduced alphabet (C and Python paths)
  - Tokenization: plain bytes/str, gapped bytes/str
  - Cross-checking against tokens.py wrappers

Run with ``python -m pytest tests/test_mirseq.py -v``.
"""

import unittest

from Bio.Seq import Seq as BioSeq

from mir.basic import mirseq
from mir.basic.alphabets import (
    AA_MASK,
    AA_TO_REDUCED,
    aa_to_reduced as py_aa_to_reduced,
    matches,
)
from mir.basic.tokens import (
    tokenize as py_tokenize,
    tokenize_gapped as py_tokenize_gapped,
    tokenize_str as py_tokenize_str,
    tokenize_gapped_str as py_tokenize_gapped_str,
)


# ── helpers ────────────────────────────────────────────────────────

_CODON_MAP = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
}


def _py_translate_linear(nt: str) -> str:
    out = []
    for i in range(0, len(nt) - 2, 3):
        codon = nt[i:i + 3]
        out.append("X" if "N" in codon else _CODON_MAP[codon])
    if len(nt) % 3 != 0:
        out.append("_")
    return "".join(out)


def _py_translate_bidi(nt: str) -> str:
    """Reference: insert gap nucleotides then translate linearly.

    Gap length = 3 - (len % 3) nucleotides (N's), inserted after
    fwd_codons * 3 position. The gap codon translates to '_'.
    """
    n = len(nt)
    if n == 0:
        return ""
    if n % 3 == 0:
        return _py_translate_linear(nt)
    remainder = n % 3
    gap_len = 3 - remainder
    n_codons = n // 3
    fwd_codons = 4 if n >= 27 else n_codons // 2
    insert_pos = fwd_codons * 3
    # Build padded sequence
    padded = nt[:insert_pos] + "N" * gap_len + nt[insert_pos:]
    # Translate linearly — all codons are now complete
    result = []
    for i in range(0, len(padded), 3):
        codon = padded[i:i + 3]
        result.append("X" if "N" in codon else _CODON_MAP[codon])
    # Replace the gap codon (at fwd_codons) with '_'
    result[fwd_codons] = "_"
    return "".join(result)


# ── Translation: linear ───────────────────────────────────────────

class TestTranslateLinear(unittest.TestCase):

    def test_single_codon_atg(self) -> None:
        self.assertEqual(mirseq.translate_linear("ATG"), "M")

    def test_multiple_codons(self) -> None:
        self.assertEqual(mirseq.translate_linear("ATGGCTTGA"), "MA*")

    def test_incomplete_trailing_codon(self) -> None:
        self.assertEqual(mirseq.translate_linear("ATGGCTTGAA"), "MA*_")

    def test_single_nt(self) -> None:
        self.assertEqual(mirseq.translate_linear("A"), "_")

    def test_two_nt(self) -> None:
        self.assertEqual(mirseq.translate_linear("AT"), "_")

    def test_n_codon(self) -> None:
        self.assertEqual(mirseq.translate_linear("ATGNCTTGA"), "MX*")

    def test_all_n(self) -> None:
        self.assertEqual(mirseq.translate_linear("NNN"), "X")

    def test_empty(self) -> None:
        self.assertEqual(mirseq.translate_linear(""), "")

    def test_bytes_input(self) -> None:
        self.assertEqual(mirseq.translate_linear(b"ATGGCTTGA"), "MA*")

    def test_stop_codons(self) -> None:
        self.assertEqual(mirseq.translate_linear("TAATGATAG"), "***")

    def test_all_codons(self) -> None:
        for codon, aa in _CODON_MAP.items():
            with self.subTest(codon=codon):
                self.assertEqual(mirseq.translate_linear(codon), aa)

    def test_cross_check_reference(self) -> None:
        seqs = ["ATGGCTTGA", "ATGGCTTGAA", "ATGNCTTGA", "NNN",
                "TTTTTCTTATTG", "GCAGCCGCGGCG"]
        for seq in seqs:
            with self.subTest(seq=seq):
                self.assertEqual(
                    mirseq.translate_linear(seq), _py_translate_linear(seq))

    def test_cross_check_biopython(self) -> None:
        """Verify translate_linear matches BioPython Seq.translate() for
        complete-codon sequences (BioPython doesn't produce '_' for
        incomplete trailing codons)."""
        seqs = ["ATGGCTTGA", "TTTTTCTTATTG", "GCAGCCGCGGCG",
                "TAATGATAG", "ATGATGATGATGATGATGATG",
                "TGTTGCTGATGG"]
        for seq in seqs:
            with self.subTest(seq=seq):
                # Only compare the full-codon portion
                n_full = (len(seq) // 3) * 3
                full_seq = seq[:n_full]
                bio_result = str(BioSeq(full_seq).translate())
                mirseq_result = mirseq.translate_linear(full_seq)
                self.assertEqual(mirseq_result, bio_result)

    def test_cross_check_biopython_all_codons(self) -> None:
        """Verify every codon matches BioPython."""
        for codon in _CODON_MAP:
            with self.subTest(codon=codon):
                self.assertEqual(
                    mirseq.translate_linear(codon),
                    str(BioSeq(codon).translate()))


# ── Translation: bidirectional (comprehensive) ────────────────────

class TestTranslateBidi(unittest.TestCase):

    def test_divisible_by_3(self) -> None:
        self.assertEqual(mirseq.translate_bidi("ATGGCTTGA"), "MA*")

    def test_empty(self) -> None:
        self.assertEqual(mirseq.translate_bidi(""), "")

    def test_bytes_input(self) -> None:
        self.assertEqual(mirseq.translate_bidi(b"ATGGCTTGA"), "MA*")

    # -- short sequences (< 27 nt): gap in middle ----------------------

    def test_4nt(self) -> None:
        # 1 codon, fwd=0, rev=1
        self.assertEqual(mirseq.translate_bidi("TGAA"), "_E")

    def test_5nt(self) -> None:
        self.assertEqual(mirseq.translate_bidi("ATGAA"), "_E")

    def test_7nt(self) -> None:
        # 2 codons, fwd=1, rev=1
        self.assertEqual(mirseq.translate_bidi("ATGGCTA"), "M_L")

    def test_8nt(self) -> None:
        self.assertEqual(mirseq.translate_bidi("ATGGCTAA"), "M_*")

    def test_10nt(self) -> None:
        # 3 codons, fwd=1, rev=2
        self.assertEqual(mirseq.translate_bidi("ATGGCTTGAA"), "M_LE")

    def test_11nt(self) -> None:
        # ATGGCTTGAAC: fwd=ATG→M, gap, rev=TTG→L, AAC→N
        self.assertEqual(mirseq.translate_bidi("ATGGCTTGAAC"), "M_LN")

    def test_13nt(self) -> None:
        # 4 codons, fwd=2, rev=2
        self.assertEqual(mirseq.translate_bidi("ATGGCTTGAAACT"), "MA_ET")

    def test_16nt(self) -> None:
        # 5 codons, fwd=2, rev=3
        result = mirseq.translate_bidi("ATGGCTTGAAACTAAG")
        self.assertEqual(result, "MA_ETK")

    def test_25nt(self) -> None:
        # 8 codons, fwd=4, rev=4
        # ATG*8+A → fwd reads ATG ATG ATG ATG, rev reads TGA TGA TGA TGA
        nt = "ATG" * 8 + "A"
        result = mirseq.translate_bidi(nt)
        self.assertEqual(len(result), 9)
        self.assertEqual(result[:4], "MMMM")
        self.assertEqual(result[4], "_")
        self.assertEqual(result[5:], "****")

    def test_26nt(self) -> None:
        nt = "ATG" * 8 + "AT"
        result = mirseq.translate_bidi(nt)
        self.assertEqual(len(result), 9)
        self.assertEqual(result[4], "_")

    # -- boundary: 27 nt (9*3) — no gap --------------------------------

    def test_27nt_exact(self) -> None:
        nt = "ATG" * 9
        result = mirseq.translate_bidi(nt)
        self.assertEqual(result, "M" * 9)
        self.assertNotIn("_", result)

    # -- long sequences (>= 27 nt): gap after 4th codon ----------------

    def test_28nt(self) -> None:
        # fwd reads ATG*4, rev reads TGA*5 → *****
        nt = "ATG" * 9 + "A"
        result = mirseq.translate_bidi(nt)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[:4], "MMMM")
        self.assertEqual(result[4], "_")
        self.assertEqual(result[5:], "*****")

    def test_29nt(self) -> None:
        nt = "ATG" * 9 + "AT"
        result = mirseq.translate_bidi(nt)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[4], "_")

    def test_31nt(self) -> None:
        nt = "ATG" * 10 + "A"
        result = mirseq.translate_bidi(nt)
        self.assertEqual(len(result), 11)
        self.assertEqual(result[:4], "MMMM")
        self.assertEqual(result[4], "_")

    def test_46nt(self) -> None:
        nt = "ATG" * 15 + "A"
        result = mirseq.translate_bidi(nt)
        self.assertEqual(len(result), 16)
        self.assertEqual(result[4], "_")

    # -- N nucleotides --------------------------------------------------

    def test_n_in_forward(self) -> None:
        self.assertEqual(mirseq.translate_bidi("NNGNCTTGA"), "XX*")

    def test_n_in_reverse_short(self) -> None:
        self.assertEqual(mirseq.translate_bidi("ATGNCTNG"), "M_X")

    # -- cross-check against Python reference ---------------------------

    def test_cross_check_short(self) -> None:
        nt_base = "ATGGCTTGAAACTAAGTTTTTCATA"
        for length in range(1, 27):
            nt = (nt_base * 2)[:length]
            with self.subTest(length=length):
                self.assertEqual(
                    mirseq.translate_bidi(nt), _py_translate_bidi(nt))

    def test_cross_check_long(self) -> None:
        nt_base = "ATGGCTTGAAACTAAGTTTTTCATA" * 3
        for length in range(27, 51):
            nt = nt_base[:length]
            with self.subTest(length=length):
                self.assertEqual(
                    mirseq.translate_bidi(nt), _py_translate_bidi(nt))

    # -- cross-check bidi against BioPython -----------------------------

    def test_bidi_vs_biopython(self) -> None:
        """For divisible-by-3 sequences, bidi == linear == BioPython."""
        seqs = ["ATG" * 3, "ATG" * 9, "ATGGCTTGA", "TTTTTCTTATTG"]
        for seq in seqs:
            with self.subTest(seq=seq):
                self.assertEqual(
                    mirseq.translate_bidi(seq),
                    str(BioSeq(seq).translate()))

    def test_bidi_flanks_vs_biopython(self) -> None:
        """Non-gap codons in bidi output match BioPython translation of
        those same nucleotide regions."""
        nt_base = "ATGGCTTGAAACTAAGTTTTTCATA" * 3
        for length in range(4, 50):
            nt = nt_base[:length]
            if length % 3 == 0:
                continue
            remainder = length % 3
            gap_len = 3 - remainder
            n_codons = length // 3
            fwd_codons = 4 if length >= 27 else n_codons // 2
            with self.subTest(length=length):
                result = mirseq.translate_bidi(nt)
                # Check forward flanking codons against BioPython
                fwd_nt = nt[:fwd_codons * 3]
                if fwd_nt:
                    self.assertEqual(
                        result[:fwd_codons],
                        str(BioSeq(fwd_nt).translate()))
                # Check reverse flanking codons against BioPython
                rev_start = fwd_codons * 3
                rev_nt = nt[rev_start:]
                # After gap insertion, the reverse portion starts at a new
                # codon boundary; extract the codons from end
                rev_codons = n_codons - fwd_codons
                rev_nt_from_end = nt[length - rev_codons * 3:]
                if rev_nt_from_end:
                    self.assertEqual(
                        result[fwd_codons + 1:],
                        str(BioSeq(rev_nt_from_end).translate()))

    # -- structural properties ------------------------------------------

    def test_gap_count(self) -> None:
        for length in range(1, 50):
            nt = ("ATG" * 20)[:length]
            result = mirseq.translate_bidi(nt)
            expected_gaps = 0 if length % 3 == 0 else 1
            self.assertEqual(result.count("_"), expected_gaps, f"length={length}")

    def test_output_length(self) -> None:
        for length in range(1, 50):
            nt = ("ATG" * 20)[:length]
            result = mirseq.translate_bidi(nt)
            n_codons = length // 3
            expected_len = n_codons if length % 3 == 0 else n_codons + 1
            self.assertEqual(len(result), expected_len, f"length={length}")


# ── AA → reduced ─────────────────────────────────────────────────

class TestAaToReduced(unittest.TestCase):

    def test_basic(self) -> None:
        self.assertEqual(mirseq.aa_to_reduced("CASTIVGGLSQDKIVW"),
                         "slhhllGGlhmcbllW")

    def test_matches_python(self) -> None:
        seq = "CASTIVGGLSQDKIVW"
        self.assertEqual(mirseq.aa_to_reduced(seq),
                         py_aa_to_reduced(seq).decode("ascii"))

    def test_special_chars(self) -> None:
        self.assertEqual(mirseq.aa_to_reduced("*_X"), "*_X")

    def test_empty(self) -> None:
        self.assertEqual(mirseq.aa_to_reduced(""), "")

    def test_bytes_input(self) -> None:
        self.assertEqual(mirseq.aa_to_reduced(b"CAST"), "slhh")

    def test_all_aa_mapped(self) -> None:
        for aa, expected in AA_TO_REDUCED.items():
            with self.subTest(aa=aa):
                self.assertEqual(mirseq.aa_to_reduced(aa), expected)


# ── Tokenize bytes ────────────────────────────────────────────────

class TestTokenizeBytes(unittest.TestCase):

    def test_aa_k3(self) -> None:
        self.assertEqual(mirseq.tokenize_bytes("CASSL", 3),
                         [b"CAS", b"ASS", b"SSL"])

    def test_nt_k4(self) -> None:
        self.assertEqual(mirseq.tokenize_bytes("ATCGAT", 4),
                         [b"ATCG", b"TCGA", b"CGAT"])

    def test_k_equals_len(self) -> None:
        self.assertEqual(mirseq.tokenize_bytes("CAST", 4), [b"CAST"])

    def test_k1(self) -> None:
        self.assertEqual(mirseq.tokenize_bytes("ATG", 1),
                         [b"A", b"T", b"G"])

    def test_bytes_input(self) -> None:
        self.assertEqual(mirseq.tokenize_bytes(b"CASSL", 3),
                         [b"CAS", b"ASS", b"SSL"])

    def test_invalid_k(self) -> None:
        with self.assertRaises(Exception):
            mirseq.tokenize_bytes("CAST", 0)
        with self.assertRaises(Exception):
            mirseq.tokenize_bytes("CAST", 5)

    def test_cross_check_wrapper(self) -> None:
        for seq in ["CASSL", "ATCGATCGATCG", "slhhllGG"]:
            for k in [1, 2, 3, 4]:
                if k > len(seq):
                    continue
                with self.subTest(seq=seq, k=k):
                    self.assertEqual(
                        mirseq.tokenize_bytes(seq, k),
                        py_tokenize(seq, k))


# ── Tokenize str ──────────────────────────────────────────────────

class TestTokenizeStr(unittest.TestCase):

    def test_basic(self) -> None:
        self.assertEqual(mirseq.tokenize_str("CASSL", 3),
                         ["CAS", "ASS", "SSL"])

    def test_cross_check_wrapper(self) -> None:
        for seq in ["CASSL", "ATCGATCGATCG"]:
            for k in [1, 2, 3]:
                with self.subTest(seq=seq, k=k):
                    self.assertEqual(
                        mirseq.tokenize_str(seq, k),
                        py_tokenize_str(seq, k))


# ── Tokenize gapped bytes ────────────────────────────────────────

class TestTokenizeGappedBytes(unittest.TestCase):

    def test_aa_gapped_k3(self) -> None:
        expected = [
            b"XAS", b"CXS", b"CAX",
            b"XSS", b"AXS", b"ASX",
            b"XSL", b"SXL", b"SSX",
        ]
        self.assertEqual(
            mirseq.tokenize_gapped_bytes("CASSL", 3, AA_MASK), expected)

    def test_nt_gapped_k2(self) -> None:
        self.assertEqual(
            mirseq.tokenize_gapped_bytes("ATG", 2, ord("N")),
            [b"NT", b"AN", b"NG", b"TN"])

    def test_gapped_k1(self) -> None:
        self.assertEqual(
            mirseq.tokenize_gapped_bytes("CA", 1, AA_MASK), [b"X", b"X"])

    def test_invalid_k(self) -> None:
        with self.assertRaises(Exception):
            mirseq.tokenize_gapped_bytes("CAST", 0, AA_MASK)

    def test_cross_check_wrapper(self) -> None:
        for seq in ["CASSL", "ATCGAT", "slhh"]:
            for k in [1, 2, 3]:
                if k > len(seq):
                    continue
                with self.subTest(seq=seq, k=k):
                    self.assertEqual(
                        mirseq.tokenize_gapped_bytes(seq, k, AA_MASK),
                        py_tokenize_gapped(seq, k, AA_MASK))

    def test_gapped_match_plain(self) -> None:
        plain = mirseq.tokenize_bytes("CASSL", 3)
        gapped = mirseq.tokenize_gapped_bytes("CASSL", 3, AA_MASK)
        for i, kmer in enumerate(plain):
            for var in gapped[i * 3 : (i + 1) * 3]:
                self.assertTrue(matches(kmer, var, AA_MASK))


# ── Tokenize gapped str ──────────────────────────────────────────

class TestTokenizeGappedStr(unittest.TestCase):

    def test_basic(self) -> None:
        gapped = mirseq.tokenize_gapped_str("CASSL", 3, AA_MASK)
        self.assertEqual(len(gapped), 9)
        self.assertEqual(gapped[0], "XAS")
        self.assertIsInstance(gapped[0], str)

    def test_cross_check_wrapper(self) -> None:
        self.assertEqual(
            mirseq.tokenize_gapped_str("CASSL", 3, AA_MASK),
            py_tokenize_gapped_str("CASSL", 3, "X"))


if __name__ == "__main__":
    unittest.main()
