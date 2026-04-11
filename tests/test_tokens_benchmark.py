"""Speed benchmark: tokenize / tokenize_gapped vs naive Python.

Compares bytes-based tokenisation functions against naive ``str`` slicing
for both plain and gapped k-mers.  Also benchmarks ``str`` vs ``bytes``
input to verify conversion overhead is negligible.

Run with ``python -m pytest tests/test_tokens_benchmark.py -s``.
"""

import random
import time
import unittest

from mir.basic.sequence import AA_MASK
from mir.basic.tokens import tokenize, tokenize_gapped

N = 10_000
SEQ_LEN = 15
K = 3
MASK_STR = "X"

_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


def _random_strings(n: int, length: int) -> list[str]:
    rng = random.Random(42)
    return ["".join(rng.choices(_AA_LETTERS, k=length)) for _ in range(n)]


def _print_table(title: str, rows: list[tuple[str, float, int]]) -> None:
    print(
        f"\n{title}\n"
        f"{'Method':<36} {'Time (s)':>10} {'items/s':>14}\n"
        f"{'-' * 62}"
    )
    for label, elapsed, count in rows:
        rate = count / elapsed if elapsed > 0 else float("inf")
        print(f"{label:<36} {elapsed:>10.4f} {rate:>14,.0f}")


class TestTokenizeBenchmark(unittest.TestCase):

    def test_plain_kmers(self) -> None:
        """Plain k-mers: tokenize(bytes) vs naive str slicing."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]
        expected = N * (SEQ_LEN - K + 1)

        # naive str slicing
        t0 = time.perf_counter()
        cnt = 0
        for s in strings:
            for i in range(len(s) - K + 1):
                _ = s[i : i + K]
                cnt += 1
        t_naive_str = time.perf_counter() - t0

        # naive bytes slicing
        t0 = time.perf_counter()
        cnt2 = 0
        for b in byte_strings:
            for i in range(len(b) - K + 1):
                _ = b[i : i + K]
                cnt2 += 1
        t_naive_bytes = time.perf_counter() - t0

        # tokenize(str input)
        t0 = time.perf_counter()
        cnt3 = 0
        for s in strings:
            cnt3 += len(tokenize(s, K))
        t_tok_str = time.perf_counter() - t0

        # tokenize(bytes input)
        t0 = time.perf_counter()
        cnt4 = 0
        for b in byte_strings:
            cnt4 += len(tokenize(b, K))
        t_tok_bytes = time.perf_counter() - t0

        self.assertEqual(cnt, expected)
        self.assertEqual(cnt2, expected)
        self.assertEqual(cnt3, expected)
        self.assertEqual(cnt4, expected)

        _print_table(
            f"Plain {K}-mers  (N={N:,}, len={SEQ_LEN})",
            [
                ("naive str slicing", t_naive_str, expected),
                ("naive bytes slicing", t_naive_bytes, expected),
                ("tokenize(str input)", t_tok_str, expected),
                ("tokenize(bytes input)", t_tok_bytes, expected),
            ],
        )
        ratio = t_tok_bytes / t_naive_str if t_naive_str > 0 else float("inf")
        print(f"\ntokenize(bytes) / naive str: {ratio:.2f}x")

    def test_gapped_kmers(self) -> None:
        """Gapped k-mers: tokenize_gapped vs naive str concatenation."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]
        n_windows = SEQ_LEN - K + 1
        expected = N * n_windows * K

        # naive str: slice + replace
        t0 = time.perf_counter()
        cnt = 0
        for s in strings:
            for i in range(len(s) - K + 1):
                w = s[i : i + K]
                for j in range(K):
                    _ = w[:j] + MASK_STR + w[j + 1 :]
                    cnt += 1
        t_naive_str = time.perf_counter() - t0

        # naive bytes: slice + replace
        mask_b = bytes([AA_MASK])
        t0 = time.perf_counter()
        cnt2 = 0
        for b in byte_strings:
            for i in range(len(b) - K + 1):
                w = b[i : i + K]
                for j in range(K):
                    _ = w[:j] + mask_b + w[j + 1 :]
                    cnt2 += 1
        t_naive_bytes = time.perf_counter() - t0

        # tokenize_gapped(str input)
        t0 = time.perf_counter()
        cnt3 = 0
        for s in strings:
            cnt3 += len(tokenize_gapped(s, K, AA_MASK))
        t_tok_str = time.perf_counter() - t0

        # tokenize_gapped(bytes input)
        t0 = time.perf_counter()
        cnt4 = 0
        for b in byte_strings:
            cnt4 += len(tokenize_gapped(b, K, AA_MASK))
        t_tok_bytes = time.perf_counter() - t0

        self.assertEqual(cnt, expected)
        self.assertEqual(cnt2, expected)
        self.assertEqual(cnt3, expected)
        self.assertEqual(cnt4, expected)

        _print_table(
            f"Gapped {K}-mers  (N={N:,}, len={SEQ_LEN})",
            [
                ("naive str slice+replace", t_naive_str, expected),
                ("naive bytes slice+replace", t_naive_bytes, expected),
                ("tokenize_gapped(str input)", t_tok_str, expected),
                ("tokenize_gapped(bytes input)", t_tok_bytes, expected),
            ],
        )
        ratio = t_tok_bytes / t_naive_str if t_naive_str > 0 else float("inf")
        print(f"\ntokenize_gapped(bytes) / naive str: {ratio:.2f}x")


if __name__ == "__main__":
    unittest.main()
