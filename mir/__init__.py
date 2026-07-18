"""mirpy — ML-oriented embeddings for immune receptor repertoires (TCR/BCR).

Version 3 is a slim, embedding-focused rewrite. Heavy machinery is delegated to
sibling packages rather than duplicated:

* alignment → ``seqtree`` (``seqtree.gapblock``),
* VDJ-rearrangement / Pgen model + sampling → ``vdjtools.model`` (core dependency),
* VDJdb annotation / E-values → ``vdjmatch`` (extra ``[annotate]``),
* build-time germline region annotation → ``arda`` (extra ``[build]``).

The classical repertoire toolkit lives on the ``legacy-v2`` branch (``mirpy-lib`` 2.x).
"""

from __future__ import annotations

import os

__version__ = "3.4.0"

__all__ = ["__version__", "get_resource_path", "TCREmp", "PairedTCREmp"]


def get_resource_path(name: str | None = None):
    """Return the absolute path to a bundled resource under ``mir/resources``.

    Args:
        name: Resource file or directory name. When ``None``, return the sorted
            list of top-level resource names instead of a path.

    Returns:
        The absolute path as a string, or the sorted list of resource names
        when ``name`` is ``None``.

    Raises:
        FileNotFoundError: If ``name`` does not resolve to a bundled resource.
    """
    resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    if name is None:
        return sorted(os.listdir(resources))
    path = os.path.join(resources, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing resource: {name!r}")
    return path


def __getattr__(name: str):
    # Lazy top-level re-exports so ``import mir`` stays light.
    if name in ("TCREmp", "PairedTCREmp"):
        from mir.embedding.tcremp import PairedTCREmp, TCREmp

        return {"TCREmp": TCREmp, "PairedTCREmp": PairedTCREmp}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
