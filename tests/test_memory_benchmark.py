"""Memory benchmark for k-mer tokenisation.

Uses ``tracemalloc`` to measure memory for:

1. Plain k-mers: tokenize() vs naive str slicing vs naive bytes slicing.
2. Gapped k-mers: tokenize_gapped() vs naive approaches.

Run with ``python -m pytest tests/test_memory_benchmark.py -s``.
"""

import random
import tracemalloc
import unittest

from mir.basic.sequence import AA_MASK
from mir.basic.tokens import tokenize, tokenize_gapped

N = 100_000
SEQ_LEN = 15
K = 3
MASK_STR = "X"

_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def _random_strings(n: int, length: int) -> list[str]:
    rng = random.Random(42)
    return ["".join(rng.choices(_AA_LETTERS, k=length)) for _ in range(n)]


def _fmt(nbytes: int) -> str:
    return f"{nbytes / 1024:.1f} KiB"


class TestMemoryBenchmark(unittest.TestCase):

    def test_plain_kmer_memory(self) -> None:
        """Compare memory: tokenize() vs naive str/bytes slicing."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]
        expected = N * (SEQ_LEN - K + 1)

        # naive str slices
        tracemalloc.start()
        str_kmers = []
        for s in strings:
            str_kmers.extend(s[i : i + K] for i in range(len(s) - K + 1))
        cur_str, peak_str = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # naive bytes slices
        tracemalloc.start()
        bytes_kmers = []
        for b in byte_strings:
            bytes_kmers.extend(b[i : i + K] for i in range(len(b) - K + 1))
        cur_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # tokenize(bytes)
        tracemalloc.start()
        tok_kmers = []
        for b in byte_strings:
            tok_kmers.extend(tokenize(b, K))
        cur_tok, peak_tok = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(len(str_kmers), expected)
        self.assertEqual(len(bytes_kmers), expected)
        self.assertEqual(len(tok_kmers), expected)

        print(
            f"\n{'Approach':<32} {'Count':>8} {'Current':>12} {'Peak':>12} "
            f"{'Per-item':>10}\n"
            f"{'-' * 76}"
        )
        for lbl, count, cur, peak in [
            ("naive str slices", len(str_kmers), cur_str, peak_str),
            ("naive bytes slices", len(bytes_kmers), cur_bytes, peak_bytes),
            ("tokenize(bytes)", len(tok_kmers), cur_tok, peak_tok),
        ]:
            per = cur / count if count else 0
            print(
                f"{lbl:<32} {count:>8} {_fmt(cur):>12} {_fmt(peak):>12} "
                f"{per:>8.0f} B"
            )

    def test_gapped_kmer_memory(self) -> None:
        """Compare memory: tokenize_gapped() vs naive gapped str slicing."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]
        n_windows = SEQ_LEN - K + 1
        expected = N * n_windows * K

        # naive str gapped
        tracemalloc.start()
        str_gapped = []
        for s in strings:
            for i in range(len(s) - K + 1):
                w = s[i : i + K]
                for j in range(K):
                    str_gapped.append(w[:j] + MASK_STR + w[j + 1 :])
        cur_str, peak_str = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # tokenize_gapped(bytes)
        tracemalloc.start()
        tok_gapped = []
        for b in byte_strings:
            tok_gapped.extend(tokenize_gapped(b, K, AA_MASK))
        cur_tok, peak_tok = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(len(str_gapped), expected)
        self.assertEqual(len(tok_gapped), expected)

        print(
            f"\n{'Approach':<32} {'Count':>8} {'Current':>12} {'Peak':>12} "
            f"{'Per-item':>10}\n"
            f"{'-' * 76}"
        )
        for lbl, count, cur, peak in [
            ("naive str gapped", len(str_gapped), cur_str, peak_str),
            ("tokenize_gapped(bytes)", len(tok_gapped), cur_tok, peak_tok),
        ]:
            per = cur / count if count else 0
            print(
                f"{lbl:<32} {count:>8} {_fmt(cur):>12} {_fmt(peak):>12} "
                f"{per:>8.0f} B"
            )


if __name__ == "__main__":
    unittest.main()
