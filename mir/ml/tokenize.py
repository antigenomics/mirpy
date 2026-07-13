"""Fixed-length-40 one-hot tokenization of CDR3/junction sequences.

Mirrors the irrm-codec representation: every sequence is placed into a length-40
frame by anchoring the conserved ends and inserting a contiguous **gap block** in
the variable middle — the N-terminal 4 residues left-align at the start, the
C-terminal 3 residues right-align at the end, the middle left-aligns after
position 4, and gaps fill the remainder (mirrors the TCREMP gap placement). Long
IGH middles that overflow are truncated (the hard case). The alphabet is the 20
amino acids plus a gap symbol (21 tokens); unknown characters map to gap.
"""

from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"
ALPHABET = AA + GAP
N_TOKENS = len(ALPHABET)  # 21
FIXED_LEN = 40
_GAP_IDX = N_TOKENS - 1
_LEFT_ANCHOR = 4   # residues held at the N-terminus
_RIGHT_ANCHOR = 3  # residues held at the C-terminus
_MID_CAP = FIXED_LEN - _LEFT_ANCHOR - _RIGHT_ANCHOR  # 33 middle slots

_IDX = np.full(256, _GAP_IDX, dtype=np.int64)
for _i, _c in enumerate(AA):
    _IDX[ord(_c)] = _i


def encode_indices(cdr3s) -> np.ndarray:
    """Return ``(N, 40)`` int64 token indices (gap-padded, ends anchored)."""
    seqs = list(cdr3s)
    out = np.full((len(seqs), FIXED_LEN), _GAP_IDX, dtype=np.int64)
    for r, s in enumerate(seqs):
        s = s or ""
        b = s.encode("ascii", "ignore")
        codes = np.frombuffer(b, dtype=np.uint8)
        idx = _IDX[codes]
        L = idx.size
        if L <= _LEFT_ANCHOR + _RIGHT_ANCHOR:
            out[r, :L] = idx  # too short to split: left-align verbatim
            continue
        left = idx[:_LEFT_ANCHOR]
        right = idx[-_RIGHT_ANCHOR:]
        mid = idx[_LEFT_ANCHOR:L - _RIGHT_ANCHOR][:_MID_CAP]  # truncate overflow (IGH)
        out[r, :_LEFT_ANCHOR] = left
        out[r, _LEFT_ANCHOR:_LEFT_ANCHOR + mid.size] = mid
        out[r, FIXED_LEN - _RIGHT_ANCHOR:] = right
    return out


def encode_onehot(cdr3s) -> np.ndarray:
    """Return ``(N, 40, 21)`` float32 one-hot encoding."""
    idx = encode_indices(cdr3s)
    oh = np.zeros((*idx.shape, N_TOKENS), dtype=np.float32)
    np.put_along_axis(oh, idx[..., None], 1.0, axis=2)
    return oh


if __name__ == "__main__":
    x = encode_onehot(["CASSIRSSYEQYF", "CAS", "C" * 60])
    assert x.shape == (3, FIXED_LEN, N_TOKENS)
    assert np.allclose(x.sum(axis=2), 1.0)          # exactly one token per position
    idx = encode_indices(["CASSIRSSYEQYF"])[0]
    assert idx[0] == AA.index("C") and idx[-1] == AA.index("F")  # ends anchored
    assert (idx == _GAP_IDX).any()                  # gap block present
    print("mir.ml.tokenize self-check OK; fixed_len", FIXED_LEN, "tokens", N_TOKENS)
