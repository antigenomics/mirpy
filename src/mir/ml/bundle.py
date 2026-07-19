"""Ship a trained codec together with the things that define its embedding space.

Two embeddings are **only comparable if they share the same prototype set and the same
PCA rotation** — different prototypes or a different PCA give an incomparable coordinate
system. A :class:`CodecBundle` therefore serializes, as one artifact:

* the **PCA transform** (the rotation/whitening — the exact coordinate system),
* a **prototype hash** (identity of the prototype set the embedding was built on),
* the model weights and the metadata needed to reproduce/verify it.

Loading verifies the prototype hash against the currently-bundled prototypes, so a codec
can never be silently used to produce embeddings in a different, incomparable space.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import mir
from mir.distances.junction import DEFAULT_GAP_POSITIONS
from mir.embedding.prototypes import load_prototypes


def prototype_hash(species: str, locus: str, n: int) -> str:
    """Stable 16-hex-char hash of the ordered prototype junction set (its identity)."""
    protos = load_prototypes(species, locus, n=n)["junction_aa"].to_list()
    h = hashlib.sha256()
    for s in protos:
        h.update(s.encode("ascii", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass
class CodecBundle:
    """A trained codec plus its embedding-space identity (PCA + prototype hash)."""

    meta: dict          # species, locus, n_prototypes, prototype_hash, gap_positions, ...
    transform: object   # sklearn PCA (the rotation) or StandardScaler — the coordinate system
    model: object       # torch nn.Module (on CPU)

    @classmethod
    def from_forward(
        cls,
        encoder,
        species: str,
        locus: str,
        n_prototypes: int,
        gap_positions: tuple[int, ...] = DEFAULT_GAP_POSITIONS,
        kind: str = "forward",
    ) -> "CodecBundle":
        meta = {
            "kind": kind,
            "species": species,
            "locus": locus,
            "n_prototypes": n_prototypes,
            "prototype_hash": prototype_hash(species, locus, n_prototypes),
            "gap_positions": list(gap_positions),
            "n_components": int(getattr(encoder.transform, "n_components_", 0)) or None,
            "is_pca": bool(encoder.is_pca),
            "mirpy_version": mir.__version__,
        }
        return cls(meta, encoder.transform, copy.deepcopy(encoder.model).to("cpu"))

    def save(self, path: str | Path) -> None:
        import torch

        torch.save({"meta": self.meta, "transform": self.transform, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "CodecBundle":
        import torch

        d = torch.load(path, weights_only=False)
        return cls(d["meta"], d["transform"], d["model"])

    def matches_current_prototypes(self) -> bool:
        """True iff the current bundled prototypes reproduce this codec's embedding space."""
        return self.meta["prototype_hash"] == prototype_hash(
            self.meta["species"], self.meta["locus"], self.meta["n_prototypes"]
        )

    def forward_encoder(self, device: str | None = None, verify: bool = True):
        """Reconstruct the :class:`ForwardEncoder`, verifying prototype comparability."""
        if verify and not self.matches_current_prototypes():
            raise ValueError(
                f"prototype hash mismatch for {self.meta['species']}_{self.meta['locus']}: "
                "this codec was trained on a different prototype set — its embeddings are "
                "NOT comparable to embeddings from the current prototypes. Pass verify=False "
                "only if you know the prototype set is intentionally different."
            )
        from mir.ml.train import ForwardEncoder, pick_device

        dev = pick_device(device)
        return ForwardEncoder(self.model.to(dev), self.transform, dev, self.meta["is_pca"])
