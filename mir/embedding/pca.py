"""PCA denoising of TCREMP embeddings.

The raw prototype-distance coordinates are highly redundant — neighbouring
prototypes and correlated V/J scores span a low-rank subspace (Theory T3). A
``StandardScaler`` → ``PCA`` step smooths the embedding, removes that redundancy,
and yields a compact basis that is well behaved for clustering / visualization
(the paper uses ``n_components=50``).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_denoise(
    X: np.ndarray, n_components: int = 50, scale: bool = True, random_state: int = 0
) -> np.ndarray:
    """Standardize then PCA-reduce a TCREMP embedding.

    Args:
        X: ``(n_clonotypes, n_features)`` embedding.
        n_components: Target dimensionality (clamped to ``min(n_samples, n_features)``).
        scale: Standardize each coordinate to zero mean / unit variance first.
        random_state: Seed for the (randomized) SVD solver.

    Returns:
        ``(n_clonotypes, k)`` float array of principal-component scores,
        ``k = min(n_components, n_samples, n_features)``.
    """
    if scale:
        X = StandardScaler().fit_transform(X)
    k = min(n_components, X.shape[0], X.shape[1])
    return PCA(n_components=k, random_state=random_state).fit_transform(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # low-rank signal + noise -> PCA should recover the rank
    Z = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 300))
    Y = pca_denoise(Z + 0.01 * rng.standard_normal((200, 300)), n_components=50)
    assert Y.shape == (200, 50)
    # variance concentrates in the first few PCs
    var = Y.var(axis=0)
    assert var[:5].sum() > 0.9 * var.sum(), var[:8]
    print("mir.embedding.pca self-check OK; top-5 PC var fraction",
          round(float(var[:5].sum() / var.sum()), 3))
