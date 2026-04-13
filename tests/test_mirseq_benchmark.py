"""Benchmarks: C extensions vs pure-Python for key operations.

Run with ``python -m pytest tests/test_mirseq_benchmark.py -v -s``.
"""

import time
import unittest

from mir.basic import mirseq
from mir.distances import seqdist_c
from mir.basic.alphabets import (
    AA_MASK,
    AA_TO_REDUCED_TABLE,
    _to_bytes,
    aa_to_reduced as py_aa_to_reduced,
)


def _time_fn(fn, *args, n: int = 5000) -> float:
    start = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return time.perf_counter() - start


class TestBenchmarks(unittest.TestCase):

    def _report(self, name: str, py_t: float, c_t: float) -> None:
        ratio = py_t / c_t if c_t > 0 else float("inf")
        print(f"  {name:30s}  Python={py_t:.4f}s  C={c_t:.4f}s  speedup={ratio:.1f}x")

    # ── translate_linear ──────────────────────────────────────────

    def test_translate_linear_speed(self) -> None:
        nt = "ATG" * 40
        n = 10_000

        def py_translate(s: str) -> str:
            codon_map = {
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
            out = []
            for i in range(0, len(s) - 2, 3):
                codon = s[i:i + 3]
                out.append(codon_map.get(codon, "X"))
            return "".join(out)

        py_t = _time_fn(py_translate, nt, n=n)
        c_t = _time_fn(mirseq.translate_linear, nt, n=n)
        self._report("translate_linear", py_t, c_t)

    # ── aa_to_reduced ─────────────────────────────────────────────

    def test_aa_to_reduced_speed(self) -> None:
        aa = "CASTIVGGLSQDKIVW" * 5
        n = 20_000
        py_t = _time_fn(py_aa_to_reduced, aa, n=n)
        c_t = _time_fn(mirseq.aa_to_reduced, aa, n=n)
        self._report("aa_to_reduced (Python bytes.translate vs C)", py_t, c_t)

    # ── tokenize_bytes ────────────────────────────────────────────

    def test_tokenize_bytes_speed(self) -> None:
        seq = "CASTIVGGLSQDKIVW" * 5
        n = 10_000

        def py_tokenize(s: str, k: int) -> list[bytes]:
            b = s.encode()
            return [b[i:i + k] for i in range(len(b) - k + 1)]

        py_t = _time_fn(py_tokenize, seq, 3, n=n)
        c_t = _time_fn(mirseq.tokenize_bytes, seq, 3, n=n)
        self._report("tokenize_bytes", py_t, c_t)

    # ── tokenize_gapped_bytes ─────────────────────────────────────

    def test_tokenize_gapped_bytes_speed(self) -> None:
        seq = "CASTIVGGLSQDKIVW" * 3
        n = 5_000

        def py_tokenize_gapped(s: str, k: int, m: int) -> list[bytes]:
            b = s.encode()
            out = []
            for i in range(len(b) - k + 1):
                kmer = bytearray(b[i:i + k])
                for j in range(k):
                    v = bytearray(kmer)
                    v[j] = m
                    out.append(bytes(v))
            return out

        py_t = _time_fn(py_tokenize_gapped, seq, 3, AA_MASK, n=n)
        c_t = _time_fn(mirseq.tokenize_gapped_bytes, seq, 3, AA_MASK, n=n)
        self._report("tokenize_gapped_bytes", py_t, c_t)

    # ── hamming ───────────────────────────────────────────────────

    def test_hamming_speed(self) -> None:
        a = "CASTIVGGLSQDKIVW" * 5
        b = "CASTIVGGLSQEKIVW" * 5
        n = 20_000

        def py_hamming(s1: str, s2: str) -> int:
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))

        py_t = _time_fn(py_hamming, a, b, n=n)
        c_t = _time_fn(seqdist_c.hamming, a, b, n=n)
        self._report("hamming", py_t, c_t)

    # ── levenshtein ───────────────────────────────────────────────

    def test_levenshtein_speed(self) -> None:
        a = "CASTIVGGLSQDKIVW" * 3
        b = "CASTGGLSQEKIVW" * 3
        n = 5_000

        def py_levenshtein(s1: str, s2: str) -> int:
            m, n_ = len(s1), len(s2)
            prev = list(range(n_ + 1))
            for i in range(1, m + 1):
                curr = [i] + [0] * n_
                for j in range(1, n_ + 1):
                    cost = 0 if s1[i - 1] == s2[j - 1] else 1
                    curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
                prev = curr
            return prev[n_]

        py_t = _time_fn(py_levenshtein, a, b, n=n)
        c_t = _time_fn(seqdist_c.levenshtein, a, b, n=n)
        self._report("levenshtein", py_t, c_t)


if __name__ == "__main__":
    unittest.main()
