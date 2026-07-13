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
from mir.ml.tokenize import N_TOKENS, encode_indices, encode_onehot


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
    """A trained forward codec: CDR3 strings → embedding vectors.

    ``transform`` maps the model output back to the original embedding space —
    either an inverse ``StandardScaler`` (raw target) or an inverse whitened
    ``PCA`` (compact 95%-variance target). ``code`` returns the compact code.
    """

    def __init__(self, model: nn.Module, transform, device: torch.device, is_pca: bool):
        self.model = model.eval()
        self.transform = transform  # sklearn StandardScaler or PCA (whitened)
        self.device = device
        self.is_pca = is_pca

    @torch.no_grad()
    def _predict(self, cdr3s, batch: int) -> np.ndarray:
        X = encode_onehot(cdr3s)
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(self.device)
            outs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outs)

    def code(self, cdr3s, batch: int = 1024) -> np.ndarray:
        """Return the compact code the model predicts (PCA coords, or normalized target)."""
        return self._predict(cdr3s, batch)

    def encode(self, cdr3s, batch: int = 1024) -> np.ndarray:
        """Predict embeddings in the original space (inverse PCA / scaler)."""
        return self.transform.inverse_transform(self._predict(cdr3s, batch))


def train_forward_encoder(
    cdr3s,
    targets: np.ndarray,
    *,
    target_pca: float | None = 0.95,
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

    Args:
        target_pca: If a float in (0, 1], compact the (redundant) target embedding
            with a whitened PCA keeping that fraction of variance — fit on the train
            split only. ``None`` trains on the raw standardized target.

    Returns the fitted :class:`ForwardEncoder` and a metrics dict (``test_cosine`` —
    reconstruction cosine in the *original* embedding space — ``val_mse``, ``n``,
    ``n_components``). No leakage: PCA / scaler are fit on train only.
    """
    from sklearn.decomposition import PCA

    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    raw = np.asarray(targets, dtype=np.float32)
    n, dim0 = len(cdr3s), raw.shape[1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    # compact target (PCA, whitened) or raw (standardized) — fit on train only
    if target_pca is not None:
        transform = PCA(n_components=target_pca, whiten=True, random_state=seed).fit(raw[tr])
        is_pca = True
        n_comp = transform.n_components_
    else:
        transform = StandardScaler().fit(raw[tr])
        is_pca = False
        n_comp = dim0
    Y = transform.transform(raw).astype(np.float32)
    dim = Y.shape[1]
    if verbose:
        _tag = f"{dim0} -> {n_comp} PCs ({target_pca:.0%} var)" if is_pca else f"{dim0} (raw)"
        print(f"torch {torch.__version__} | device={dev} | n={n} | target {_tag}")

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
        pred_code = model(Xte).cpu().numpy()
    # reconstruction cosine in the ORIGINAL embedding space (what downstream uses)
    pred_orig = transform.inverse_transform(pred_code)
    cos = _mean_cosine(pred_orig, raw[te])
    if verbose:
        print(f"test mean cosine {cos:.4f} (original space, n_test={len(te)})")
    encoder = ForwardEncoder(model, transform, dev, is_pca)
    return encoder, {"test_cosine": cos, "val_mse": best_val, "n": n, "n_components": n_comp}


# ---------------------------------------------------------------------------
# Inverse codec: compact embedding code -> CDR3 sequence
# ---------------------------------------------------------------------------


class InverseDecoder:
    """A trained inverse codec: embedding codes → CDR3 strings."""

    def __init__(self, model: nn.Module, code_scaler: StandardScaler, device: torch.device):
        self.model = model.eval()
        self.code_scaler = code_scaler
        self.device = device

    @torch.no_grad()
    def decode(self, codes, batch: int = 1024) -> list[str]:
        from mir.ml.decoder import tokens_to_seq

        Z = self.code_scaler.transform(np.asarray(codes, dtype=np.float32)).astype(np.float32)
        out: list[str] = []
        for i in range(0, len(Z), batch):
            zb = torch.from_numpy(Z[i:i + batch]).to(self.device)
            idx = self.model(zb).argmax(dim=-1).cpu().numpy()
            out += [tokens_to_seq(r) for r in idx]
        return out


def train_inverse_decoder(
    codes: np.ndarray,
    cdr3s,
    *,
    epochs: int = 50,
    batch: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[InverseDecoder, dict]:
    """Train a :class:`SequenceDecoder` to reconstruct *cdr3s* from *codes*.

    Returns the fitted :class:`InverseDecoder` and metrics (``exact_match`` — full
    sequence reconstructed correctly — ``token_acc``, ``n``). Codes are standardized
    on the train split only.
    """
    from mir.ml.decoder import SequenceDecoder, tokens_to_seq

    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    codes = np.asarray(codes, dtype=np.float32)
    n = len(cdr3s)
    tgt = encode_indices(cdr3s)  # (n, 40) int64

    perm = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    cs = StandardScaler().fit(codes[tr])
    Z = cs.transform(codes).astype(np.float32)

    def _dev(idx):
        return torch.from_numpy(Z[idx]).to(dev), torch.from_numpy(tgt[idx]).to(dev)

    Ztr, Ttr = _dev(tr)
    Zva, Tva = _dev(va)
    Zte, Tte = _dev(te)

    model = SequenceDecoder(codes.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def _tok_acc(logits, T):
        return (logits.argmax(dim=-1) == T).float().mean().item()

    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            j = order[i:i + batch]
            opt.zero_grad()
            logits = model(Ztr[j])
            loss_fn(logits.reshape(-1, N_TOKENS), Ttr[j].reshape(-1)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = _tok_acc(model(Zva), Tva)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_token_acc {va_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Zte).argmax(dim=-1).cpu().numpy()
    token_acc = float((pred == tgt[te]).mean())
    recon = [tokens_to_seq(r) for r in pred]
    exact = float(np.mean([r == cdr3s[i] for r, i in zip(recon, te)]))
    if verbose:
        print(f"test exact-match {exact:.3f}  token-acc {token_acc:.3f}  (n_test={len(te)})")
    return InverseDecoder(model, cs, dev), {"exact_match": exact, "token_acc": token_acc, "n": n}


# ---------------------------------------------------------------------------
# Pgen-from-sequence regressor
# ---------------------------------------------------------------------------


class PgenRegressor:
    """A trained regressor: CDR3 strings → log10 Pgen."""

    def __init__(self, model: nn.Module, scaler: StandardScaler, device: torch.device):
        self.model = model.eval()
        self.scaler = scaler
        self.device = device

    @torch.no_grad()
    def predict(self, cdr3s, batch: int = 1024) -> np.ndarray:
        X = encode_onehot(cdr3s)
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(self.device)
            outs.append(self.model(xb).cpu().numpy())
        return self.scaler.inverse_transform(np.concatenate(outs)).ravel()


def train_pgen_regressor(
    cdr3s,
    log_pgen,
    *,
    epochs: int = 40,
    batch: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[PgenRegressor, dict]:
    """Train the shared sequence encoder to predict ``log10 Pgen`` from CDR3.

    Returns the fitted :class:`PgenRegressor` and metrics (``pearson``, ``rmse`` in
    log10 units, ``n``). Targets standardized on the train split only.
    """
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    y = np.asarray(log_pgen, dtype=np.float32).reshape(-1, 1)
    n = len(cdr3s)

    perm = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    scaler = StandardScaler().fit(y[tr])
    Y = scaler.transform(y).astype(np.float32)
    X = encode_onehot(cdr3s)

    def _dev(idx):
        return torch.from_numpy(X[idx]).to(dev), torch.from_numpy(Y[idx]).to(dev)

    Xtr, Ytr = _dev(tr)
    Xva, Yva = _dev(va)
    Xte, _ = _dev(te)

    model = SequenceEncoder(1).to(dev)
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
        pred = scaler.inverse_transform(model(Xte).cpu().numpy()).ravel()
    true = y[te].ravel()
    pearson = float(np.corrcoef(pred, true)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    if verbose:
        print(f"test Pearson r {pearson:.4f}  RMSE {rmse:.3f} (log10)  (n_test={len(te)})")
    return PgenRegressor(model, scaler, dev), {"pearson": pearson, "rmse": rmse, "n": n}
