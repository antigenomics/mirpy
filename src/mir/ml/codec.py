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

from mir.ml.decoder import SequenceDecoder, tokens_to_seqs
from mir.ml.encoder import SequenceEncoder
from mir.ml.tokenize import N_TOKENS
from mir.ml.train import (_batch_size, _decode_batches, _mean_cosine, _predict_batches,
                          _training_batches, _training_tokens, pick_device, seed_everything,
                          train_val_test_split)


class UnifiedCodec:
    """Trained encoder+decoder: ``encode`` (seq→code), ``decode`` (code→seq), ``roundtrip``."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.device = device

    def encode(self, cdr3s, batch: int = 1024) -> np.ndarray:
        return _predict_batches(self.encoder, cdr3s, self.device, batch)

    def decode(self, codes, batch: int = 1024) -> list[str]:
        return _decode_batches(self.decoder, codes, self.device, batch)

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
    batch = _batch_size(batch)
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    codes = np.asarray(codes, dtype=np.float32)
    n, code_dim = len(cdr3s), codes.shape[1]

    tokens = _training_tokens(cdr3s, batch)
    te, va, tr = train_val_test_split(n, test_frac, val_frac, seed)

    enc = SequenceEncoder(code_dim).to(dev)
    dec = SequenceDecoder(code_dim).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

    best, best_state = -1.0, None
    for ep in range(epochs):
        enc.train(); dec.train()
        order = torch.randperm(len(tr), device=dev).cpu().numpy()
        for _, tb, cb in _training_batches(tokens, codes, tr[order], dev, batch):
            opt.zero_grad()
            code_pred = enc(nn.functional.one_hot(tb, N_TOKENS).float())
            loss = lambda_embed * mse(code_pred, cb) + ce(
                dec(code_pred).reshape(-1, N_TOKENS), tb.reshape(-1))
            loss.backward()
            opt.step()
        enc.eval(); dec.eval()
        with torch.no_grad():
            correct, count = 0, 0
            for _, tb, _ in _training_batches(tokens, codes, va, dev, batch):
                code = enc(nn.functional.one_hot(tb, N_TOKENS).float())
                correct = correct + (dec(code).argmax(-1) == tb).sum()
                count += tb.numel()
            rt = correct.item() / count
        if rt > best:
            best = rt
            best_state = ({k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()})
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_roundtrip_token_acc {rt:.3f}")

    if best_state is not None:
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    enc.eval(); dec.eval()
    cosine, roundtrips, true_decodes = 0.0, 0, 0
    with torch.no_grad():
        for ids, tb, cb in _training_batches(tokens, codes, te, dev, batch):
            code_pred = enc(nn.functional.one_hot(tb, N_TOKENS).float())
            cosine += len(ids) * _mean_cosine(code_pred.cpu().numpy(), codes[ids])
            rt_idx = dec(code_pred).argmax(-1).cpu().numpy()
            true_idx = dec(cb).argmax(-1).cpu().numpy()
            roundtrips += sum(r == cdr3s[i] for r, i in zip(tokens_to_seqs(rt_idx), ids))
            true_decodes += sum(r == cdr3s[i] for r, i in zip(tokens_to_seqs(true_idx), ids))
    enc_cos = cosine / len(te)
    rt_exact, true_exact = roundtrips / len(te), true_decodes / len(te)
    if verbose:
        print(f"test encode_cosine {enc_cos:.4f}  roundtrip_exact {rt_exact:.3f}  "
              f"decode_true_exact {true_exact:.3f}  (n_test={len(te)})")
    return UnifiedCodec(enc, dec, dev), {
        "encode_cosine": enc_cos, "roundtrip_exact": rt_exact,
        "decode_true_exact": true_exact, "n": n,
    }
