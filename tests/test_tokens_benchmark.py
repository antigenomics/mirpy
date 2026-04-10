"""Benchmark: tokenize() vs naive Python string slicing for 3-mer extraction.

Generates N=10 000 random amino-acid sequences of length 15 and compares
wall-clock time for splitting each into overlapping 3-mers using:

1. ``tokenize()`` from :mod:`mir.basic.tokens` (sequence + memoryview path).
2. Naive Python: plain string slicing producing ``list[str]``.

Run with ``python -m unittest -v tests/test_tokens_benchmark.py``.
"""

import random
import string
import time
import unittest

from mir.basic.sequence import AminoAcidSequence
from mir.basic.tokens import tokenize

N = 10_000
SEQ_LEN = 15
K = 3

# 20 canonical amino acids
_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def _random_aa_strings(n: int, length: int) -> list[str]:
    rng = random.Random(42)
    return ["".join(rng.choices(_AA_LETTERS, k=length)) for _ in range(n)]


class TestTokenizeBenchmark(unittest.TestCase):
    """Wall-clock comparison of tokenize() vs naive string slicing."""

    def test_benchmark_3mer_tokenization(self) -> None:
        strings = _random_aa_strings(N, SEQ_LEN)

        # -- naive Python string slicing ------------------------------------
        t0 = time.perf_counter()
        naive_total = 0
        for s in strings:
            kmers = [s[i : i + K] for i in range(len(s) - K + 1)]
            naive_total += len(kmers)
        t_naive = time.perf_counter() - t0

        # -- tokenize (sequence objects) ------------------------------------
        sequences = [AminoAcidSequence.from_string(s) for s in strings]
        t0 = time.perf_counter()
        tok_total = 0
        for seq in sequences:
            kmers = tokenize(seq, k=K)
            tok_total += len(kmers)
        t_tokenize = time.perf_counter() - t0

        # Both must produce the same number of k-mers
        self.assertEqual(naive_total, tok_total)

        expected_per_seq = SEQ_LEN - K + 1
        self.assertEqual(naive_total, N * expected_per_seq)

        print(
            f"\n{'Method':<22} {'Time (s)':>10} {'k-mers/s':>14}\n"
            f"{'-' * 48}"
        )
        for label, elapsed in [
            ("naive str slicing", t_naive),
            ("tokenize()", t_tokenize),
        ]:
            rate = tok_total / elapsed if elapsed > 0 else float("inf")
            print(f"{label:<22} {elapsed:>10.4f} {rate:>14,.0f}")

        ratio = t_tokenize / t_naive if t_naive > 0 else float("inf")
        print(f"\ntokenize / naive ratio: {ratio:.2f}x")


if __name__ == "__main__":
    unittest.main()
