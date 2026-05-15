"""Statistical utility helpers shared across the project."""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import false_discovery_control as _false_discovery_control
except ImportError:  # pragma: no cover
    _false_discovery_control = None

try:
    from statsmodels.stats.multitest import multipletests as _multipletests
except ImportError:  # pragma: no cover
    _multipletests = None


def bh_fdr(pvals: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment with robust NaN handling.

    NaN entries are preserved as NaN in the output while finite p-values are
    adjusted on their own subset.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return out

    finite_p = p[finite_mask]

    if _false_discovery_control is not None:
        q = _false_discovery_control(finite_p, method="bh")
        out[finite_mask] = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        return out

    if _multipletests is not None:
        q = _multipletests(finite_p, method="fdr_bh")[1]
        out[finite_mask] = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        return out

    # Fallback implementation when neither SciPy nor statsmodels API is available.
    n = len(finite_p)
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    adjusted = np.empty_like(q)
    adjusted[order] = q
    out[finite_mask] = adjusted
    return out
