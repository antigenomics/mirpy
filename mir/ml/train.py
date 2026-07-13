"""Train the forward codec (CDR3 → TCREMP embedding) with free supervision.

Targets are TCREMP embeddings computed by :class:`mir.embedding.tcremp.TCREmp`, so
training data is unlimited — sample sequences from ``vdjtools.model.generate`` (or a
real repertoire) and embed them. Reports test-set mean cosine similarity (irrm-codec's
forward metric).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from mir.ml.encoder import SequenceEncoder
from mir.ml.tokenize import encode_onehot


def pick_device(device: str | None = None) -> torch.device:
    """``mps`` on Apple silicon, else ``cpu`` (or an explicit override)."""
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a * b).sum(axis=1).mean())


class ForwardEncoder:
    """A trained forward codec: CDR3 strings → embedding vectors."""

    def __init__(self, model: nn.Module, scaler: StandardScaler, device: torch.device):
        self.model = model.eval()
        self.scaler = scaler
        self.device = device

    @torch.no_grad()
    def encode(self, cdr3s, batch: int = 1024) -> np.ndarray:
        """Predict embeddings for *cdr3s* (un-normalized to the target scale)."""
        X = encode_onehot(cdr3s)
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(self.device)
            outs.append(self.model(xb).cpu().numpy())
        return self.scaler.inverse_transform(np.concatenate(outs))


def train_forward_encoder(
    cdr3s,
    targets: np.ndarray,
    *,
    epochs: int = 40,
    batch: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[ForwardEncoder, dict]:
    """Train a :class:`SequenceEncoder` to predict *targets* from *cdr3s*.

    Returns the fitted :class:`ForwardEncoder` and a metrics dict
    (``test_cosine``, ``val_mse``, ``n``). Targets are standardized on the train
    split only (no leakage); the model predicts the normalized embedding.
    """
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    targets = np.asarray(targets, dtype=np.float32)
    n, dim = len(cdr3s), targets.shape[1]
    if verbose:
        print(f"torch {torch.__version__} | device={dev} | n={n} | embed_dim={dim}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    scaler = StandardScaler().fit(targets[tr])
    Y = scaler.transform(targets).astype(np.float32)
    X = encode_onehot(cdr3s)

    def _dev(idx):
        return torch.from_numpy(X[idx]).to(dev), torch.from_numpy(Y[idx]).to(dev)

    Xtr, Ytr = _dev(tr)
    Xva, Yva = _dev(va)
    Xte, Yte = _dev(te)

    model = SequenceEncoder(dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            j = order[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xtr[j]), Ytr[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), Yva).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_mse {vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).cpu().numpy()
    cos = _mean_cosine(pred, Yte.cpu().numpy())
    if verbose:
        print(f"test mean cosine {cos:.4f}  (n_test={len(te)})")
    return ForwardEncoder(model, scaler, dev), {"test_cosine": cos, "val_mse": best_val, "n": n}
