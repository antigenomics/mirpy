"""Learned permutation-invariant repertoire encoder — the co-equal learned track (Theory §T.7/§T.8).

The fixed backbone (:mod:`mir.repertoire`) pools clonotypes by a size-weighted sum of *fixed* RFF
features. This module replaces that with a **trained** set-pooling network (Deep Sets form,
Prop. ``prop:sampinv``): a per-clonotype MLP ``ψ_θ`` followed by **pooling-by-multihead-attention**
(PMA, Lee et al. 2019 Set Transformer) whose learned seed vectors act as *inducing points* — i.e.
trained metaclonotype / public-cluster detectors (the interaction signal of Prop. ``prop:interact``,
the alternative to the fixed second-moment ``Σ_S``). It reads the same PCA-coordinate clonotype
cloud as the fixed backbone (so the two are comparable) and is fit end-to-end to a label (age / CMV).

Depth-robustness is **engineered in**, not free (§T.7.5): clonotypes enter the attention weighted by
their frequency ``g(a_σ)`` (a log-weight bias on the attention logits), and each epoch every sample
is randomly *subsampled* so the network learns to read shallow repertoires — the RNA-seq regime.

Torch (``[ml]`` extra). Shipped as a :class:`SetEncoderBundle` (weights + PCA transform + prototype
hash), refusing a prototype mismatch on load, exactly like :class:`mir.ml.bundle.CodecBundle`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mir.ml.train import pick_device, seed_everything


class SetEncoder(nn.Module):
    """Per-clonotype MLP + frequency-weighted PMA pooling → fixed vector / prediction.

    ``forward(Z, w)`` maps one sample's clonotype cloud ``Z`` (``n×p`` PCA coords) and weights
    ``w`` (``n``, summing to 1) to an ``out_dim`` output, permutation-invariantly. The ``n_seeds``
    learned queries are the inducing points (metaclonotype detectors); frequency enters as an
    additive ``log w`` bias on the attention logits so abundant clones weigh more.
    """

    def __init__(self, p: int, d: int = 128, n_seeds: int = 16, heads: int = 4, out_dim: int = 1):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(p, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.seeds = nn.Parameter(torch.randn(n_seeds, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Flatten(0), nn.Linear(n_seeds * d, d), nn.GELU(), nn.Linear(d, out_dim))

    def forward(self, Z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = self.phi(Z).unsqueeze(0)                       # (1, n, d)
        q = self.seeds.unsqueeze(0)                        # (1, k, d)
        bias = torch.log(w + 1e-8).unsqueeze(0).expand(self.seeds.shape[0], -1)  # (k, n)
        pooled, _ = self.attn(q, h, h, attn_mask=bias)     # (1, k, d)
        return self.head(self.norm(pooled.squeeze(0)))     # (out_dim,)


@dataclass
class SetEncoderModel:
    """A trained :class:`SetEncoder` plus its device — turns a sample cloud into a prediction."""

    model: nn.Module
    device: torch.device
    task: str                                              # "regression" | "classification"

    @torch.no_grad()
    def predict(self, clouds) -> np.ndarray:
        """Predict for a list of ``(Z, w)`` clonotype clouds (probabilities if classification)."""
        self.model.eval()
        out = []
        for Z, w in clouds:
            zt = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
            wt = torch.as_tensor(w, dtype=torch.float32, device=self.device)
            y = self.model(zt, wt)
            out.append(torch.sigmoid(y).item() if self.task == "classification" else y.item())
        return np.array(out)


def _subsample(Z, w, rng, min_frac=0.2):
    """Depth-robustness augmentation: keep a random fraction of clonotypes, renormalize weights."""
    n = Z.shape[0]
    k = max(1, int(n * rng.uniform(min_frac, 1.0)))
    idx = rng.choice(n, size=k, replace=False)
    wk = w[idx]
    return Z[idx], wk / wk.sum()


def train_set_encoder(
    clouds,
    y: np.ndarray,
    *,
    task: str = "regression",
    epochs: int = 60,
    lr: float = 1e-3,
    n_seeds: int = 16,
    d: int = 128,
    val_frac: float = 0.2,
    augment: bool = True,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[SetEncoderModel, dict]:
    """Train the set encoder on ``(clouds, y)`` (clouds = list of ``(Z, w)`` per sample).

    Args:
        clouds: per-sample ``(Z: (n_i, p) float, w: (n_i,) float)`` from ``RepertoireSpace.sample_cloud``.
        y: per-sample target (float for regression, {0,1} for classification).
        task: ``"regression"`` (MSE) or ``"classification"`` (BCE-with-logits).
        augment: subsample each sample every epoch to engineer depth-robustness (§T.7.5).

    Returns:
        ``(SetEncoderModel, metrics)`` — metrics carry the held-out score (Spearman / AUC).
    """
    seed_everything(seed)
    dev = pick_device(device)
    rng = np.random.default_rng(seed)
    p = clouds[0][0].shape[1]
    n = len(clouds)

    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_i, tr_i = set(perm[:n_val].tolist()), perm[n_val:]

    model = SetEncoder(p, d=d, n_seeds=n_seeds, out_dim=1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)

    best_score, best_state = -np.inf, None
    for ep in range(epochs):
        model.train()
        for i in rng.permutation(tr_i):
            Z, w = clouds[i]
            if augment:
                Z, w = _subsample(Z, w, rng)
            zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
            wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
            pred = model(zt, wt)
            loss = loss_fn(pred.squeeze(), yt[i])
            opt.zero_grad(); loss.backward(); opt.step()

        score = _val_score([clouds[i] for i in sorted(val_i)],
                           y[sorted(val_i)], model, dev, task)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3}  val_score={score:.3f}  (best {best_score:.3f})")

    model.load_state_dict(best_state)
    return SetEncoderModel(model, dev, task), {"val_score": float(best_score), "n_val": len(val_i)}


@torch.no_grad()
def _val_score(clouds, y, model, dev, task) -> float:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    model.eval()
    pred = np.array([model(torch.as_tensor(Z, dtype=torch.float32, device=dev),
                           torch.as_tensor(w, dtype=torch.float32, device=dev)).item()
                     for Z, w in clouds])
    if task == "classification":
        return roc_auc_score(y, pred) if len(set(y.tolist())) > 1 else 0.5
    r = spearmanr(y, pred).correlation
    return abs(r) if np.isfinite(r) else 0.0


@dataclass
class SetEncoderBundle:
    """A trained set encoder plus its embedding-space identity (PCA transform + prototype hash)."""

    meta: dict
    transform: object          # the RepertoireSpace's clonotype PCA (sklearn) — the coordinate system
    model: object              # SetEncoder (on CPU)

    @classmethod
    def from_model(cls, sem: SetEncoderModel, space, task: str) -> "SetEncoderBundle":
        m = dict(space.meta)
        m["task"] = task
        return cls(m, space.clono, copy.deepcopy(sem.model).to("cpu"))

    def save(self, path: str | Path) -> None:
        torch.save({"meta": self.meta, "transform": self.transform, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "SetEncoderBundle":
        from mir.ml.bundle import prototype_hash

        d = torch.load(path, weights_only=False)
        m = d["meta"]
        if verify:
            cur = prototype_hash(m["species"], m["locus"], m["n_prototypes"])
            if cur != m["prototype_hash"]:
                raise ValueError(
                    f"prototype hash mismatch for {m['species']}_{m['locus']}: this encoder was "
                    "trained on a different prototype set — its inputs are NOT comparable. Pass "
                    "verify=False only if the difference is intentional.")
        return cls(m, d["transform"], d["model"])


def _demo() -> None:
    """Synthetic smoke: two clouds classes separable by a public cluster -> AUC > 0.5, save/load."""
    rng = np.random.default_rng(0)
    clouds, y = [], []
    for c in range(40):
        label = c % 2
        n = rng.integers(80, 200)
        Z = rng.standard_normal((n, 12))
        if label:                                  # class 1 carries a tight "public" cluster
            Z[:10] = np.full(12, 3.0) + 0.05 * rng.standard_normal((10, 12))
        w = rng.random(n); w /= w.sum()
        clouds.append((Z.astype(np.float32), w.astype(np.float32))); y.append(float(label))
    sem, metrics = train_set_encoder(clouds, np.array(y), task="classification",
                                     epochs=40, n_seeds=8, d=64, verbose=False)
    assert metrics["val_score"] > 0.6, f"learned encoder failed to separate: {metrics}"
    print(f"[ok] set encoder val AUC={metrics['val_score']:.3f} on the public-cluster toy")


if __name__ == "__main__":
    _demo()
