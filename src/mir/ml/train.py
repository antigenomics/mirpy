"""Train the forward codec (CDR3 → TCREMP embedding) with free supervision.

Targets are TCREMP embeddings computed by :class:`mir.embedding.tcremp.TCREmp`, so
training data is unlimited — sample sequences from ``vdjtools.model.generate`` (or a
real repertoire) and embed them. Reports test-set mean cosine similarity (irrm-codec's
forward metric).
"""

from __future__ import annotations

import os
import operator
import random
import threading
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from mir.ml.encoder import SequenceEncoder
from mir.ml.tokenize import FIXED_LEN, N_TOKENS, encode_indices, encode_onehot


def pick_device(device: str | None = None, *, threads: int | None = None) -> torch.device:
    """Best available accelerator: explicit ``device`` override, else CUDA, else Apple ``mps``, else ``cpu``.

    Set ``MIR_DEVICE`` (e.g. ``cuda:1``) to override without threading ``device=`` through every call.
    Device selection preserves the intra-op budget unless ``threads`` is explicit.
    High-level training/inference calls own a scoped ``threads=0`` (allocated CPU) budget.
    """
    from vdjtools.cores import kernel_threads

    if threads is not None:
        torch.set_num_threads(kernel_threads(threads))
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("Torch inter-op pool was already configured with multiple threads; "
                               "start a fresh interpreter and import mir before Torch") from exc
    if device is None:
        device = os.environ.get("MIR_DEVICE")
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ponytail: seeded fits share framework RNGs; use separate processes for independent fits.
_TRAINING_LOCK = threading.RLock()


@contextmanager
def _torch_runtime(threads):
    """Borrow vdjtools phase ownership; one Torch intra-op layer, serial BLAS/inter-op."""
    from threadpoolctl import threadpool_limits
    from vdjtools.cores import kernel_threads, numerical_kernels

    budget = kernel_threads(threads)
    with numerical_kernels(1):
        if kernel_threads(0) == 1:
            # Outer workers share Torch's process-wide setting and must never resize it.
            if torch.get_num_threads() != 1:
                raise RuntimeError("Torch was configured with multiple intra-op threads; "
                                   "restore one thread before starting sample workers")
            with _TRAINING_LOCK:
                pick_device('cpu')  # Initialize the one-shot inter-op pool before compute.
            with threadpool_limits(limits=1, user_api='openmp'):
                yield
            return
        previous = torch.get_num_threads()
        with threadpool_limits(limits=budget, user_api='openmp'):
            try:
                with _TRAINING_LOCK:
                    pick_device('cpu', threads=budget)
                yield
            finally:
                torch.set_num_threads(previous)


def _torch_call(func):
    @wraps(func)
    def call(*args, threads=0, **kwargs):
        with _torch_runtime(threads):
            return func(*args, threads=threads, **kwargs)
    return call


def _training_call(func):
    @wraps(func)
    def call(*args, **kwargs):
        with _TRAINING_LOCK:
            return func(*args, **kwargs)
    return _torch_call(call)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train_val_test_split(n: int, test_frac: float, val_frac: float, seed: int):
    """Permuted ``(test, val, train)`` index arrays, refusing a degenerate split.

    ``int(n * frac)`` floors to 0 on a small ``n``, so the validation loop would run over an empty
    set and every metric would come back ``nan`` — best-epoch selection then compares against
    ``nan`` and silently keeps an arbitrary epoch, returning a model with NaN metrics rather than
    an error. That is always a too-small dataset, never a real result, so raise instead.

    Args:
        n: Number of examples.
        test_frac: Held-out test fraction.
        val_frac: Held-out validation fraction.
        seed: RNG seed for the permutation.

    Returns:
        ``(test_idx, val_idx, train_idx)`` index arrays.

    Raises:
        ValueError: If any test, validation or training split is empty.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    n_train = n - n_test - n_val
    if n_test < 1 or n_val < 1 or n_train < 1:
        raise ValueError(
            f"n={n} with test_frac={test_frac}, val_frac={val_frac} leaves {n_test} test, "
            f"{n_val} validation and {n_train} training rows; all must be >= 1. "
            "Use more data, or adjust the fractions."
        )
    return perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]


def _mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a * b).sum(axis=1).mean())


def _batch_size(batch):
    batch = operator.index(batch)
    if batch < 1:
        raise ValueError('batch must be positive')
    return batch


def _training_tokens(seqs, batch):
    """One compact host representation; only a batch uses the int64 tokenizer scratch."""
    tokens = np.empty((len(seqs), FIXED_LEN), dtype=np.uint8)
    for start in range(0, len(seqs), batch):
        tokens[start:start + batch] = encode_indices(seqs[start:start + batch])
    return tokens


def _transform_batches(values, transform, batch):
    """Prepare host targets once, avoiding a full transformed temporary and split copies."""
    output = None
    for start in range(0, len(values), batch):
        block = np.asarray(transform(values[start:start + batch]), dtype=np.float32)
        if output is None:
            output = np.empty((len(values), *block.shape[1:]), dtype=np.float32)
        output[start:start + len(block)] = block
    return output


def _training_batches(tokens, values, rows, device, batch):
    """Stage only selected rows; one-hot expansion belongs to the native device kernel."""
    for start in range(0, len(rows), batch):
        ids = rows[start:start + batch]
        tb = torch.from_numpy(tokens[ids]).to(device=device, dtype=torch.long)
        vb = torch.from_numpy(values[ids]).to(device)
        yield ids, tb, vb


@torch.no_grad()
def _predict_batches(model, cdr3s, device, batch, transform=None):
    """Materialize raw strings and one output; expand only the current token batch."""
    batch = _batch_size(batch)
    seqs = list(cdr3s)
    if not seqs:
        raise ValueError('at least one sequence is required')
    out = None
    for start in range(0, len(seqs), batch):
        block = seqs[start:start + batch]
        xb = torch.from_numpy(encode_onehot(block)).to(device)
        values = model(xb).cpu().numpy()
        if transform is not None:
            values = transform(values.copy())
        if len(values) != len(block):
            raise ValueError('model must return one prediction per sequence')
        if out is None:
            out = np.empty((len(seqs), *values.shape[1:]), dtype=values.dtype)
        out[start:start + len(block)] = values
    return out


@torch.no_grad()
def _decode_batches(model, codes, device, batch, transform=None):
    from mir.ml.decoder import tokens_to_seqs

    batch = _batch_size(batch)
    out = []
    for start in range(0, len(codes), batch):
        values = np.array(codes[start:start + batch], dtype=np.float32, copy=True)
        if transform is not None:
            values = np.asarray(transform(values), dtype=np.float32)
        indices = model(torch.from_numpy(values).to(device)).argmax(-1).cpu().numpy()
        if len(indices) != len(values):
            raise ValueError('model must return one prediction per code')
        out += tokens_to_seqs(indices)
    return out


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

    def _predict(self, cdr3s, batch: int, transform=None) -> np.ndarray:
        return _predict_batches(self.model, cdr3s, self.device, batch, transform)

    @_torch_call
    def code(self, cdr3s, batch: int = 1024, *, threads: int = 0) -> np.ndarray:
        """Return the compact code the model predicts (PCA coords, or normalized target)."""
        return self._predict(cdr3s, batch)

    @_torch_call
    def encode(self, cdr3s, batch: int = 1024, *, threads: int = 0) -> np.ndarray:
        """Predict embeddings in the original space (inverse PCA / scaler)."""
        return self._predict(cdr3s, batch, self.transform.inverse_transform)


@_training_call
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
    threads: int = 0,
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

    batch = _batch_size(batch)
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    raw = np.asarray(targets, dtype=np.float32)
    n, dim0 = len(cdr3s), raw.shape[1]

    te, va, tr = train_val_test_split(n, test_frac, val_frac, seed)

    # compact target (PCA, whitened) or raw (standardized) — fit on train only
    if target_pca is not None:
        transform = PCA(n_components=target_pca, whiten=True, random_state=seed).fit(raw[tr])
        is_pca = True
        n_comp = transform.n_components_
    else:
        transform = StandardScaler().fit(raw[tr])
        is_pca = False
        n_comp = dim0
    Y = _transform_batches(raw, transform.transform, batch)
    dim = Y.shape[1]
    if verbose:
        _tag = f"{dim0} -> {n_comp} PCs ({target_pca:.0%} var)" if is_pca else f"{dim0} (raw)"
        print(f"torch {torch.__version__} | device={dev} | n={n} | target {_tag}")

    tokens = _training_tokens(cdr3s, batch)

    model = SequenceEncoder(dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), device=dev).cpu().numpy()
        for _, tb, yb in _training_batches(tokens, Y, tr[order], dev, batch):
            opt.zero_grad()
            xb = nn.functional.one_hot(tb, N_TOKENS).float()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            total, count = 0, 0
            for _, tb, yb in _training_batches(tokens, Y, va, dev, batch):
                prediction = model(nn.functional.one_hot(tb, N_TOKENS).float())
                total = total + nn.functional.mse_loss(prediction, yb, reduction='sum')
                count += yb.numel()
            vl = (total / count).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_mse {vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # Reduce original-space cosine per held-out batch; do not retain full predictions.
    cosine = 0.0
    with torch.no_grad():
        for ids, tb, _ in _training_batches(tokens, Y, te, dev, batch):
            code = model(nn.functional.one_hot(tb, N_TOKENS).float()).cpu().numpy()
            cosine += len(ids) * _mean_cosine(transform.inverse_transform(code), raw[ids])
    cos = cosine / len(te)
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

    @_torch_call
    def decode(self, codes, batch: int = 1024, *, threads: int = 0) -> list[str]:
        return _decode_batches(self.model, codes, self.device, batch, self.code_scaler.transform)


@_training_call
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
    threads: int = 0,
    verbose: bool = True,
) -> tuple[InverseDecoder, dict]:
    """Train a :class:`SequenceDecoder` to reconstruct *cdr3s* from *codes*.

    Returns the fitted :class:`InverseDecoder` and metrics (``exact_match`` — full
    sequence reconstructed correctly — ``token_acc``, ``n``). Codes are standardized
    on the train split only.
    """
    from mir.ml.decoder import SequenceDecoder, tokens_to_seqs

    batch = _batch_size(batch)
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    codes = np.asarray(codes, dtype=np.float32)
    n = len(cdr3s)
    tokens = _training_tokens(cdr3s, batch)

    te, va, tr = train_val_test_split(n, test_frac, val_frac, seed)

    cs = StandardScaler().fit(codes[tr])
    Z = _transform_batches(codes, cs.transform, batch)

    model = SequenceDecoder(codes.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), device=dev).cpu().numpy()
        for _, tb, zb in _training_batches(tokens, Z, tr[order], dev, batch):
            opt.zero_grad()
            logits = model(zb)
            loss_fn(logits.reshape(-1, N_TOKENS), tb.reshape(-1)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct, count = 0, 0
            for _, tb, zb in _training_batches(tokens, Z, va, dev, batch):
                correct = correct + (model(zb).argmax(-1) == tb).sum()
                count += tb.numel()
            va_acc = correct.item() / count
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_token_acc {va_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    correct, count, exact_count = 0, 0, 0
    with torch.no_grad():
        for ids, tb, zb in _training_batches(tokens, Z, te, dev, batch):
            pred = model(zb).argmax(-1)
            correct = correct + (pred == tb).sum()
            count += tb.numel()
            recon = tokens_to_seqs(pred.cpu().numpy())
            exact_count += sum(r == cdr3s[i] for r, i in zip(recon, ids))
    token_acc = correct.item() / count
    exact = exact_count / len(te)
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

    @_torch_call
    def predict(self, cdr3s, batch: int = 1024, *, threads: int = 0) -> np.ndarray:
        return _predict_batches(self.model, cdr3s, self.device, batch,
                                self.scaler.inverse_transform).ravel()


@_training_call
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
    threads: int = 0,
    verbose: bool = True,
) -> tuple[PgenRegressor, dict]:
    """Train the shared sequence encoder to predict ``log10 Pgen`` from CDR3.

    Returns the fitted :class:`PgenRegressor` and metrics (``pearson``, ``rmse`` in
    log10 units, ``n``). Targets standardized on the train split only.
    """
    batch = _batch_size(batch)
    seed_everything(seed)
    dev = pick_device(device)
    cdr3s = list(cdr3s)
    y = np.asarray(log_pgen, dtype=np.float32).reshape(-1, 1)
    n = len(cdr3s)

    te, va, tr = train_val_test_split(n, test_frac, val_frac, seed)

    scaler = StandardScaler().fit(y[tr])
    Y = _transform_batches(y, scaler.transform, batch)
    tokens = _training_tokens(cdr3s, batch)

    model = SequenceEncoder(1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), device=dev).cpu().numpy()
        for _, tb, yb in _training_batches(tokens, Y, tr[order], dev, batch):
            opt.zero_grad()
            xb = nn.functional.one_hot(tb, N_TOKENS).float()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            total, count = 0, 0
            for _, tb, yb in _training_batches(tokens, Y, va, dev, batch):
                prediction = model(nn.functional.one_hot(tb, N_TOKENS).float())
                total = total + nn.functional.mse_loss(prediction, yb, reduction='sum')
                count += yb.numel()
            vl = (total / count).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val_mse {vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    pred = np.empty(len(te), dtype=np.float32)
    start = 0
    with torch.no_grad():
        for ids, tb, _ in _training_batches(tokens, Y, te, dev, batch):
            values = model(nn.functional.one_hot(tb, N_TOKENS).float()).cpu().numpy()
            pred[start:start + len(ids)] = scaler.inverse_transform(values).ravel()
            start += len(ids)
    true = y[te].ravel()
    pearson = float(np.corrcoef(pred, true)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    if verbose:
        print(f"test Pearson r {pearson:.4f}  RMSE {rmse:.3f} (log10)  (n_test={len(te)})")
    return PgenRegressor(model, scaler, dev), {"pearson": pearson, "rmse": rmse, "n": n}
