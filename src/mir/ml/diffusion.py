"""Diffusion generator over a compact embedding/descriptor space (Part 2, needs ``[ml]``).

The *research* half of the generative loop (ROADMAP Phase 2/3), complementing
:mod:`mir.generate`'s linear-Gaussian ``DescriptorDensity``: a small conditional denoising diffusion
model (DDPM, Ho et al. 2020) trained directly on a compact coordinate system — a
:class:`~mir.repertoire.RepertoireDescriptor` vector, a codec's PCA-compacted code
(:mod:`mir.ml.bundle`), or any other ``(N, dim)`` matrix — that can express a genuinely non-linear
generative manifold where the Gaussian baseline can only fit an ellipsoid. Sampling uses **DDIM**
(Song et al. 2021) rather than the full ancestral chain: deterministic given a seed and needs far
fewer steps (tens, not the training horizon ``T``), the standard modern choice over vanilla DDPM
sampling. **Classifier-free guidance** (Ho & Salimans 2022) — random label dropout during training,
so one model samples both unconditionally and conditionally, with ``guidance_scale`` trading
diversity for how strongly a sample matches its condition — lets one model serve as the digital-twin
generator :mod:`mir.generate`'s ``sample``/``evolve`` API mirrors: draw a synthetic donor state
unconditionally, or conditioned on a covariate (tumor type, HLA, batch), with the same call shape.

The coordinate space here is already compact (tens to a few hundred dimensions after PCA/descriptor
reduction), so the epsilon-predictor is a small MLP (:class:`DiffusionMLP`) — matching mir.ml's
existing compact-network style (:class:`~mir.ml.decoder.SequenceDecoder`), not a U-Net or
transformer built for pixel/token grids that would be over-parameterized for this data.

:class:`DiffusionModel` ships the same way :class:`~mir.ml.bundle.CodecBundle` does — ``meta`` +
weights via :meth:`~DiffusionModel.save`/:meth:`~DiffusionModel.load` — so a trained generator
carries its own provenance; pass a ``prototype_hash`` (:func:`mir.ml.bundle.prototype_hash`) in
``meta`` when the coordinate space came from a specific prototype set, and check it with
:meth:`~DiffusionModel.matches_prototype_hash` before sampling.

Typical usage::

    from mir.ml.diffusion import train_diffusion

    model, metrics = train_diffusion(X, tumor_type, epochs=300)     # X: (n, dim) descriptor/code matrix
    synth = model.sample(50, condition="SKCM", steps=50, guidance_scale=2.0)  # 50 synthetic SKCM-like
    model.save("skcm_diffusion.pt")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mir.ml.train import pick_device, seed_everything


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021) — smoother SNR decay than linear DDPM betas,
    the modern default over the original linear schedule."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * torch.pi / 2) ** 2
    alphabar = f / f[0]
    betas = 1 - alphabar[1:] / alphabar[:-1]
    return torch.clip(betas, 1e-5, 0.999).float()


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionMLP(nn.Module):
    """Epsilon-predictor: noised vector + timestep (+ optional class) -> predicted noise.

    Class index ``0`` is reserved as the "null" conditioning token (classifier-free guidance): real
    classes are indices ``1..n_classes``, so ``y=None`` (or the null token during training-time
    label dropout) samples the unconditional model.

    Args:
        dim: Data dimensionality (the descriptor/code width).
        time_dim: Sinusoidal timestep embedding width.
        hidden: Hidden layer width.
        n_classes: Number of real classes; ``0`` disables conditioning entirely.
        class_dim: Class embedding width (unused if ``n_classes == 0``).
    """

    def __init__(self, dim: int, time_dim: int = 64, hidden: int = 256,
                 n_classes: int = 0, class_dim: int = 32):
        super().__init__()
        self.time_embed = nn.Sequential(
            _SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
        )
        self.n_classes = n_classes
        if n_classes:
            self.class_embed = nn.Embedding(n_classes + 1, class_dim)
        in_dim = dim + time_dim + (class_dim if n_classes else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x_t, self.time_embed(t)]
        if self.n_classes:
            if y is None:
                y = torch.zeros(x_t.shape[0], dtype=torch.long, device=x_t.device)
            parts.append(self.class_embed(y))
        return self.net(torch.cat(parts, dim=-1))


@dataclass
class DiffusionModel:
    """A trained diffusion generator: noised-vector-space -> new samples via DDIM.

    Attributes:
        model: The trained :class:`DiffusionMLP` (eval mode).
        betas: ``(T,)`` noise schedule the model was trained with.
        meta: Provenance dict — always carries ``dim``, ``n_classes``, ``T``, and the per-dimension
            ``mu``/``sd`` the training data was standardized with (:meth:`sample` un-standardizes
            with these before returning); carries ``classes`` (the original label values, index
            ``i`` -> class token ``i+1``) when class-conditional; carries a ``prototype_hash`` iff
            the caller supplied one at train time.
    """

    model: nn.Module
    betas: torch.Tensor
    meta: dict = field(default_factory=dict)

    def matches_prototype_hash(self, current: str) -> bool:
        """``True`` iff ``meta`` has no ``prototype_hash`` recorded, or it equals ``current``."""
        return "prototype_hash" not in self.meta or self.meta["prototype_hash"] == current

    def save(self, path: str | Path) -> None:
        """Serialize model + schedule + meta (mirrors :meth:`mir.ml.bundle.CodecBundle.save`)."""
        torch.save({"state_dict": self.model.state_dict(), "betas": self.betas, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DiffusionModel":
        """Rebuild a saved :class:`DiffusionModel` (weights on CPU unless ``device`` given)."""
        d = torch.load(path, weights_only=False)
        meta = d["meta"]
        # .get() with the constructor defaults so bundles written before the architecture
        # was recorded still load (they were all trained at the defaults).
        model = DiffusionMLP(
            meta["dim"], n_classes=meta["n_classes"], hidden=meta.get("hidden", 256),
            time_dim=meta.get("time_dim", 64), class_dim=meta.get("class_dim", 32),
        )
        model.load_state_dict(d["state_dict"])
        model.eval()
        if device is not None:
            model = model.to(device)
        return cls(model=model, betas=d["betas"], meta=meta)

    @torch.no_grad()
    def sample(
        self, n: int = 1, *, condition=None, steps: int = 50, guidance_scale: float = 1.0,
        seed: int = 0, device: str | None = None,
    ) -> np.ndarray:
        """Draw ``n`` synthetic vectors via deterministic DDIM (Song et al. 2021, ``eta=0``).

        Same call shape as :meth:`mir.generate.DescriptorDensity.sample` (``n``, ``condition``,
        ``seed``) so the two generators are drop-in alternatives.

        Args:
            n: Number of samples to draw.
            condition: Class label to condition on (must be one of the labels ``train_diffusion``
                saw); ``None`` samples unconditionally.
            steps: DDIM steps — far fewer than the training horizon ``T`` (state-of-art vs. the full
                ancestral chain); clipped to ``T`` if larger.
            guidance_scale: Classifier-free guidance weight. ``1.0`` = plain conditional/
                unconditional prediction; ``>1.0`` sharpens toward the condition (requires
                ``condition`` set and the model to be class-conditional).
            seed: RNG seed.
            device: Override device (default: wherever the model currently lives).

        Returns:
            ``(n, dim)`` array.

        Raises:
            ValueError: If ``condition`` is unknown, guidance is requested without ``condition``, or
                the model isn't class-conditional but ``condition`` was given.
        """
        dev = torch.device(device) if device is not None else next(self.model.parameters()).device
        model = self.model.to(dev)
        betas = self.betas.to(dev)
        alphas = 1.0 - betas
        alphabar = torch.cumprod(alphas, dim=0)
        T, dim = self.meta["T"], self.meta["dim"]
        steps = min(steps, T)

        y_idx = None
        if condition is not None:
            classes = self.meta.get("classes")
            if not classes:
                raise ValueError("condition was given but this model was trained unconditionally (n_classes=0)")
            if condition not in classes:
                raise ValueError(f"label {condition!r} not in fitted classes {classes}")
            y_idx = classes.index(condition) + 1
        if guidance_scale != 1.0 and y_idx is None:
            raise ValueError("guidance_scale != 1.0 needs a condition to guide toward")

        gen = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(n, dim, generator=gen, device=dev)
        ts = torch.linspace(T - 1, 0, steps, device=dev).round().long()
        for i in range(steps):
            t = ts[i]
            t_batch = torch.full((n,), int(t), device=dev, dtype=torch.long)
            ab_t = alphabar[t]
            ab_prev = alphabar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=dev)

            if y_idx is not None and guidance_scale != 1.0:
                y_batch = torch.full((n,), y_idx, device=dev, dtype=torch.long)
                y_null = torch.zeros(n, device=dev, dtype=torch.long)
                eps_cond = model(x, t_batch, y_batch)
                eps_uncond = model(x, t_batch, y_null)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                y_batch = None if y_idx is None else torch.full((n,), y_idx, device=dev, dtype=torch.long)
                eps = model(x, t_batch, y_batch)

            x0_hat = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            # clip the x0 estimate: at high-noise timesteps alphabar_t~0, so a none-too-accurate eps
            # prediction makes x0_hat blow up -- unclipped, that error compounds every remaining step
            # (the standard DDPM/DDIM stability fix; data is standardized, so +/-6 sigma is generous).
            x0_hat = torch.clamp(x0_hat, -6.0, 6.0)
            # Re-derive eps from the CLIPPED x0, or the step still uses the raw unbounded network
            # output and the blow-up propagates anyway: at exactly the timesteps the clamp targets
            # (ab_t~0, sqrt(1-ab_prev)~1) the update degenerates to x <- eps. Recomputing also keeps
            # (x0_hat, eps) a consistent pair, so this stays the deterministic DDIM path
            # (cf. diffusers DDIMScheduler(clip_sample=True)).
            eps = (x - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1 - ab_t)
            x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1 - ab_prev) * eps
        mu = torch.as_tensor(self.meta["mu"], device=dev, dtype=x.dtype)
        sd = torch.as_tensor(self.meta["sd"], device=dev, dtype=x.dtype)
        return (x * sd + mu).cpu().numpy()


def train_diffusion(
    X: np.ndarray,
    condition=None,
    *,
    T: int = 200,
    epochs: int = 200,
    batch: int = 64,
    lr: float = 1e-3,
    hidden: int = 256,
    time_dim: int = 64,
    class_dim: int = 32,
    class_drop_prob: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
    meta: dict | None = None,
) -> tuple[DiffusionModel, dict]:
    """Train a conditional (or unconditional) diffusion model over ``X``.

    Args:
        X: ``(n_samples, dim)`` data — a descriptor/code matrix.
        condition: Optional length-``n_samples`` class labels (any hashable value); ``None`` trains
            unconditionally.
        T: Diffusion horizon (training timesteps). Sampling can use far fewer via DDIM.
        epochs, batch, lr: Standard training hyperparameters.
        hidden, time_dim, class_dim: :class:`DiffusionMLP` architecture knobs.
        class_drop_prob: Per-example probability of replacing the true label with the null token
            during training (classifier-free guidance) — ignored if ``condition is None``.
        val_frac: Held-out fraction for the reported validation loss (best-epoch checkpointing).
        seed: RNG seed (data split, weight init, training stochasticity).
        device: Override device (default: :func:`mir.ml.train.pick_device`'s auto-pick).
        verbose: Print progress.
        meta: Extra provenance to carry on the returned model (e.g. ``{"prototype_hash": ...}``).

    Returns:
        ``(model, metrics)`` — the fitted :class:`DiffusionModel` (best-validation-loss weights) and
        a metrics dict (``val_loss``, ``n``, ``n_classes``).

    Raises:
        ValueError: If ``X`` has fewer than 4 rows (too few to hold out a validation split).
    """
    seed_everything(seed)
    dev = pick_device(device)
    X = np.asarray(X, dtype=np.float32)
    n, dim = X.shape
    if n < 4:
        raise ValueError(f"need >= 4 samples to train+validate a diffusion model, got {n}")
    # standardize: the noise schedule assumes roughly unit-scale data, and an arbitrary raw scale
    # (e.g. a log-count descriptor coordinate next to a [0,1] fraction) would train and denoise
    # unevenly across dimensions. sample() un-standardizes with the same mu/sd before returning.
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    X = (X - mu) / sd

    classes = None
    y_idx = None
    n_classes = 0
    if condition is not None:
        classes = sorted(set(condition), key=str)
        class_to_idx = {c: i + 1 for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[c] for c in condition], dtype=np.int64)
        n_classes = len(classes)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]

    betas = cosine_beta_schedule(T).to(dev)
    alphabar = torch.cumprod(1.0 - betas, dim=0)

    model = DiffusionMLP(dim, time_dim=time_dim, hidden=hidden, n_classes=n_classes, class_dim=class_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xtr = torch.from_numpy(X[tr]).to(dev)
    Xva = torch.from_numpy(X[va]).to(dev)
    ytr = torch.from_numpy(y_idx[tr]).to(dev) if y_idx is not None else None
    yva = torch.from_numpy(y_idx[va]).to(dev) if y_idx is not None else None

    if verbose:
        print(f"torch {torch.__version__} | device={dev} | n={n} dim={dim} n_classes={n_classes}")

    # One FIXED validation noising, drawn once. Eps-prediction MSE varies systematically with the
    # timestep, so re-drawing (t, eps) every epoch makes consecutive val_losses incomparable and
    # `best_state` latches onto whichever epoch happened to draw easy timesteps — at these defaults
    # (n_val is 40 at n=400, 4 at n=40) that noise swamps real late-training improvement.
    g = torch.Generator(device="cpu").manual_seed(seed)
    t_va = torch.randint(0, T, (Xva.shape[0],), generator=g).to(dev)
    eps_va = torch.randn(Xva.shape, generator=g).to(dev)
    ab_va = alphabar[t_va][:, None]
    x_t_va = torch.sqrt(ab_va) * Xva + torch.sqrt(1 - ab_va) * eps_va

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), batch):
            j = order[i:i + batch]
            xb = Xtr[j]
            t = torch.randint(0, T, (xb.shape[0],), device=dev)
            eps = torch.randn_like(xb)
            ab = alphabar[t][:, None]
            x_t = torch.sqrt(ab) * xb + torch.sqrt(1 - ab) * eps
            yb = None
            if ytr is not None:
                yb = ytr[j].clone()
                drop = torch.rand(yb.shape[0], device=dev) < class_drop_prob
                yb[drop] = 0
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_t, t, yb), eps)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(x_t_va, t_va, yva), eps_va).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep}: val_loss={val_loss:.4f} (best={best_val:.4f})")

    model.load_state_dict(best_state)
    model.eval().to("cpu")

    full_meta = dict(meta or {})
    # hidden/time_dim/class_dim all change state-dict shapes, so they belong in meta —
    # without them `load` rebuilds at constructor defaults and any non-default architecture
    # fails with a size mismatch.
    full_meta.update({"dim": dim, "n_classes": n_classes, "T": T, "classes": classes, "mu": mu, "sd": sd,
                      "hidden": hidden, "time_dim": time_dim, "class_dim": class_dim})
    return DiffusionModel(model=model, betas=betas.cpu(), meta=full_meta), {
        "val_loss": best_val, "n": n, "n_classes": n_classes,
    }


def _demo() -> None:
    """Self-check on a synthetic two-moons-like mixture: unconditional samples match the data
    moments and class-conditional sampling separates two well-separated synthetic groups."""
    rng = np.random.default_rng(0)
    n, dim = 400, 4
    z = rng.standard_normal((n, dim)).astype(np.float32)
    labels = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    z[labels == "b"] += 4.0   # two well-separated synthetic clusters

    model, metrics = train_diffusion(z, labels, T=100, epochs=80, batch=64, seed=0, verbose=False)
    assert metrics["n_classes"] == 2

    synth_a = model.sample(200, condition="a", steps=30, guidance_scale=2.0, seed=1)
    synth_b = model.sample(200, condition="b", steps=30, guidance_scale=2.0, seed=1)
    sep = float(synth_b.mean() - synth_a.mean())
    assert sep > 2.0, f"conditional samples not separated: {sep:.2f}"

    uncond = model.sample(400, steps=30, seed=2)
    assert abs(float(uncond.mean()) - float(z.mean())) < 1.5

    print(f"[ok] val_loss={metrics['val_loss']:.3f}; conditional a/b mean separation={sep:.2f}; "
          f"unconditional mean {uncond.mean():.2f}~{z.mean():.2f}")


if __name__ == "__main__":
    _demo()
