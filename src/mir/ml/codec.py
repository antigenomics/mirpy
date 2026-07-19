"""Unified codec: jointly train encoder + decoder, anchored to the embedding geometry.

The encoder (seq → code) and decoder (code → seq) are trained together, but a
geometry-preservation term keeps the encoder's code close to the true (compact PCA)
embedding — so the codec round-trips sequences *without* the code drifting away from
the embedding space that makes distances meaningful (Theory T1). ``lambda_embed``
controls the anchor strength: large → geometry preserved, decoder adapts to it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mir.ml.decoder import SequenceDecoder, tokens_to_seq
from mir.ml.encoder import SequenceEncoder
from mir.ml.tokenize import N_TOKENS, encode_indices, encode_onehot
from mir.ml.train import _mean_cosine, pick_device, seed_everything


class UnifiedCodec:
    """Trained encoder+decoder: ``encode`` (seq→code), ``decode`` (code→seq), ``roundtrip``."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, cdr3s, batch: int = 1024) -> np.ndarray:
        X = encode_onehot(cdr3s)
        out = [self.encoder(torch.from_numpy(X[i:i + batch]).to(self.device)).cpu().numpy()
               for i in range(0, len(X), batch)]
        return np.concatenate(out)

    @torch.no_grad()
    def decode(self, codes, batch: int = 1024) -> list[str]:
        Z = np.asarray(codes, dtype=np.float32)
        seqs: list[str] = []
        for i in range(0, len(Z), batch):
            idx = self.decoder(torch.from_numpy(Z[i:i + batch]).to(self.device)).argmax(-1).cpu().numpy()
            seqs += [tokens_to_seq(r) for r in idx]
        return seqs

    def roundtrip(self, cdr3s, batch: int = 1024) -> list[str]:
        return self.decode(self.encode(cdr3s, batch), batch)


def train_unified_codec(
    cdr3s,
    codes: np.ndarray,
    *,
    lambda_embed: float = 1.0,
    epochs: int = 60,
    batch: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[UnifiedCodec, dict]:
    """Jointly train encoder+decoder to reconstruct *cdr3s* while matching *codes*.

    Args:
        codes: The true compact embedding codes (whitened PCA), one per sequence —
            the geometry anchor and the encoder target.
        lambda_embed: Weight of the encoder geometry term vs the reconstruction term.

    Returns the :class:`UnifiedCodec` and metrics: ``encode_cosine`` (encoder code vs
    true code — geometry preservation), ``roundtrip_exact`` (seq→code→seq), and
    ``decode_true_exact`` (decode the *true* code — reference upper bound).
    """
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    codes = np.asarray(codes, dtype=np.float32)
    n, code_dim = len(cdr3s), codes.shape[1]

    X = encode_onehot(cdr3s)
    T = encode_indices(cdr3s)
    perm = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    def _dev(idx):
        return (torch.from_numpy(X[idx]).to(dev),
                torch.from_numpy(codes[idx]).to(dev),
                torch.from_numpy(T[idx]).to(dev))

    Xtr, Ctr, Ttr = _dev(tr)
    Xva, Cva, Tva = _dev(va)
    Xte, Cte, Tte = _dev(te)

    enc = SequenceEncoder(code_dim).to(dev)
    dec = SequenceDecoder(code_dim).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    best, best_state = -1.0, None
    for ep in range(epochs):
        enc.train(); dec.train()
        order = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            j = order[i:i + batch]
            opt.zero_grad()
            code_pred = enc(Xtr[j])
            loss = lambda_embed * mse(code_pred, Ctr[j]) + ce(
                dec(code_pred).reshape(-1, N_TOKENS), Ttr[j].reshape(-1))
            loss.backward()
            opt.step()
        enc.eval(); dec.eval()
        with torch.no_grad():
            rt = (dec(enc(Xva)).argmax(-1) == Tva).float().mean().item()
        if rt > best:
            best = rt
            best_state = ({k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()})
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_roundtrip_token_acc {rt:.3f}")

    if best_state is not None:
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    enc.eval(); dec.eval()
    with torch.no_grad():
        code_pred = enc(Xte).cpu().numpy()
        rt_idx = dec(enc(Xte)).argmax(-1).cpu().numpy()
        true_idx = dec(Cte).argmax(-1).cpu().numpy()
    enc_cos = _mean_cosine(code_pred, codes[te])
    orig = [cdr3s[i] for i in te]
    rt_exact = float(np.mean([tokens_to_seq(r) == o for r, o in zip(rt_idx, orig)]))
    true_exact = float(np.mean([tokens_to_seq(r) == o for r, o in zip(true_idx, orig)]))
    if verbose:
        print(f"test encode_cosine {enc_cos:.4f}  roundtrip_exact {rt_exact:.3f}  "
              f"decode_true_exact {true_exact:.3f}  (n_test={len(te)})")
    return UnifiedCodec(enc, dec, dev), {
        "encode_cosine": enc_cos, "roundtrip_exact": rt_exact,
        "decode_true_exact": true_exact, "n": n,
    }
