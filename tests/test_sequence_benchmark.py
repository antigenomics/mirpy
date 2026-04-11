"""Speed benchmarks for sequence operations: validation, translation,
slicing, matching, and cross-alphabet matching.

Each benchmark compares the ``mir.basic.sequence`` function against one
or more naive Python implementations.

Run with ``python -m pytest tests/test_sequence_benchmark.py -s``.
"""

import random
import time
import unittest

from mir.basic.sequence import (
    AA_ALPHABET,
    AA_MASK,
    AA_TO_REDUCED,
    AA_TO_REDUCED_TABLE,
    NT_MASK,
    _AA_TO_REDUCED_LUT,
    _to_bytes,
    aa_to_reduced,
    matches,
    matches_aa_reduced,
    validate,
)

N = 10_000
SEQ_LEN = 15
K = 3

_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(_AA_LETTERS + "*_X")


def _random_strings(n: int, length: int) -> list[str]:
    rng = random.Random(42)
    return ["".join(rng.choices(_AA_LETTERS, k=length)) for _ in range(n)]


def _print_table(title: str, rows: list[tuple[str, float, int]]) -> None:
    print(
        f"\n{title}\n"
        f"{'Method':<40} {'Time (s)':>10} {'ops/s':>14}\n"
        f"{'-' * 66}"
    )
    for label, elapsed, count in rows:
        rate = count / elapsed if elapsed > 0 else float("inf")
        print(f"{label:<40} {elapsed:>10.4f} {rate:>14,.0f}")


class TestValidationBenchmark(unittest.TestCase):

    def test_validate_lut_vs_set(self) -> None:
        """Alphabet validation: LUT (bytes[256]) vs frozenset membership."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]

        # LUT validation (validate function)
        t0 = time.perf_counter()
        for b in byte_strings:
            validate(b, AA_ALPHABET)
        t_lut = time.perf_counter() - t0

        # frozenset[int] validation
        aa_ords = frozenset(ord(c) for c in _AA_SET)
        t0 = time.perf_counter()
        for b in byte_strings:
            for ch in b:
                if ch not in aa_ords:
                    raise ValueError
        t_fset = time.perf_counter() - t0

        # naive str 'in' check
        t0 = time.perf_counter()
        for s in strings:
            for ch in s:
                if ch not in _AA_SET:
                    raise ValueError
        t_str_in = time.perf_counter() - t0

        _print_table(
            f"Validation  (N={N:,}, len={SEQ_LEN})",
            [
                ("validate() [bytes LUT]", t_lut, N),
                ("frozenset[int] loop", t_fset, N),
                ("str 'in' frozenset[str]", t_str_in, N),
            ],
        )


class TestTranslationBenchmark(unittest.TestCase):

    def test_translate_lut_vs_dict(self) -> None:
        """Translation: bytes.translate vs dict lookup vs manual byte loop."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]

        # bytes.translate (aa_to_reduced)
        t0 = time.perf_counter()
        for b in byte_strings:
            _ = b.translate(AA_TO_REDUCED_TABLE)
        t_translate = time.perf_counter() - t0

        # aa_to_reduced with str input (includes encode)
        t0 = time.perf_counter()
        for s in strings:
            _ = aa_to_reduced(s)
        t_aa_str = time.perf_counter() - t0

        # naive dict[str,str] lookup + join
        t0 = time.perf_counter()
        for s in strings:
            _ = "".join(AA_TO_REDUCED.get(ch, ch) for ch in s)
        t_dict_join = time.perf_counter() - t0

        # manual byte LUT loop
        lut = _AA_TO_REDUCED_LUT
        t0 = time.perf_counter()
        for b in byte_strings:
            _ = bytes(lut[ch] for ch in b)
        t_manual = time.perf_counter() - t0

        _print_table(
            f"Translation AA→reduced  (N={N:,}, len={SEQ_LEN})",
            [
                ("bytes.translate (bytes in)", t_translate, N),
                ("aa_to_reduced (str in)", t_aa_str, N),
                ("dict[str,str] + join", t_dict_join, N),
                ("manual byte LUT loop", t_manual, N),
            ],
        )
        ratio = t_dict_join / t_translate if t_translate > 0 else float("inf")
        print(f"\ndict+join / bytes.translate: {ratio:.1f}x slower")


class TestSlicingBenchmark(unittest.TestCase):

    def test_bytes_vs_str_slicing(self) -> None:
        """Substring slicing: bytes[i:j] vs str[i:j] at various k."""
        strings = _random_strings(N, SEQ_LEN)
        byte_strings = [s.encode() for s in strings]

        for k in (3, 5, 10):
            n_slices = SEQ_LEN - k + 1
            expected = N * n_slices

            # str slicing
            t0 = time.perf_counter()
            cnt = 0
            for s in strings:
                for i in range(len(s) - k + 1):
                    _ = s[i : i + k]
                    cnt += 1
            t_str = time.perf_counter() - t0

            # bytes slicing
            t0 = time.perf_counter()
            cnt2 = 0
            for b in byte_strings:
                for i in range(len(b) - k + 1):
                    _ = b[i : i + k]
                    cnt2 += 1
            t_bytes = time.perf_counter() - t0

            # str slicing via encode→slice→decode
            t0 = time.perf_counter()
            cnt3 = 0
            for s in strings:
                b = s.encode()
                for i in range(len(b) - k + 1):
                    _ = b[i : i + k]
                    cnt3 += 1
            t_enc_slice = time.perf_counter() - t0

            self.assertEqual(cnt, expected)
            self.assertEqual(cnt2, expected)
            self.assertEqual(cnt3, expected)

            _print_table(
                f"Slicing k={k}  (N={N:,}, len={SEQ_LEN}, {n_slices} slices/seq)",
                [
                    ("str[i:i+k]", t_str, expected),
                    ("bytes[i:i+k]", t_bytes, expected),
                    ("str.encode + bytes[i:i+k]", t_enc_slice, expected),
                ],
            )
            ratio = t_str / t_bytes if t_bytes > 0 else float("inf")
            print(f"  str/bytes ratio: {ratio:.2f}x")


class TestMatchingBenchmark(unittest.TestCase):

    def test_matches_vs_naive(self) -> None:
        """Wildcard matching: matches() vs naive Python loop."""
        rng = random.Random(42)
        strings_a = _random_strings(N, SEQ_LEN)
        # create pairs: 50% identical, 50% with 1 mask position
        strings_b = []
        for s in strings_a:
            if rng.random() < 0.5:
                strings_b.append(s)
            else:
                pos = rng.randint(0, SEQ_LEN - 1)
                strings_b.append(s[:pos] + "X" + s[pos + 1 :])

        bytes_a = [s.encode() for s in strings_a]
        bytes_b = [s.encode() for s in strings_b]

        # matches() function
        t0 = time.perf_counter()
        res1 = 0
        for a, b in zip(bytes_a, bytes_b):
            if matches(a, b, AA_MASK):
                res1 += 1
        t_func = time.perf_counter() - t0

        # naive Python: zip + compare
        mask_val = AA_MASK
        t0 = time.perf_counter()
        res2 = 0
        for a, b in zip(bytes_a, bytes_b):
            if len(a) == len(b) and all(
                x == y or x == mask_val or y == mask_val
                for x, y in zip(a, b)
            ):
                res2 += 1
        t_naive = time.perf_counter() - t0

        # naive str comparison
        t0 = time.perf_counter()
        res3 = 0
        for a, b in zip(strings_a, strings_b):
            if len(a) == len(b) and all(
                x == y or x == "X" or y == "X"
                for x, y in zip(a, b)
            ):
                res3 += 1
        t_str = time.perf_counter() - t0

        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

        _print_table(
            f"Wildcard matching  (N={N:,}, len={SEQ_LEN})",
            [
                ("matches() [bytes]", t_func, N),
                ("naive bytes zip+all", t_naive, N),
                ("naive str zip+all", t_str, N),
            ],
        )

    def test_matches_aa_reduced_vs_naive(self) -> None:
        """Cross-alphabet matching: matches_aa_reduced() vs naive."""
        strings = _random_strings(N, SEQ_LEN)
        reduced = [aa_to_reduced(s) for s in strings]

        bytes_aa = [s.encode() for s in strings]

        # matches_aa_reduced()
        t0 = time.perf_counter()
        cnt = 0
        for a, r in zip(bytes_aa, reduced):
            if matches_aa_reduced(a, r):
                cnt += 1
        t_func = time.perf_counter() - t0

        # naive: translate then compare
        t0 = time.perf_counter()
        cnt2 = 0
        for a, r in zip(bytes_aa, reduced):
            if a.translate(AA_TO_REDUCED_TABLE) == r:
                cnt2 += 1
        t_naive = time.perf_counter() - t0

        self.assertEqual(cnt, N)
        self.assertEqual(cnt2, N)

        _print_table(
            f"Cross-alphabet matching  (N={N:,}, len={SEQ_LEN})",
            [
                ("matches_aa_reduced()", t_func, N),
                ("translate + bytes ==", t_naive, N),
            ],
        )


if __name__ == "__main__":
    unittest.main()
