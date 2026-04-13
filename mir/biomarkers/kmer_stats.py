"""K-mer frequency analysis and differential comparison of immune repertoires.

Extracts overlapping k-mers from CDR3 amino-acid sequences, counts their
occurrences across a :class:`~mir.common.repertoire.Repertoire`, and performs
chi-squared differential analysis between two repertoires.

K-mer tokenisation is delegated to the C-accelerated
:func:`mir.basic.tokens.tokenize_str` for speed.

Classes
-------
* :class:`KmerCounter` — Extract and count k-mers from a single repertoire.

Functions
---------
* :func:`compare_kmer_counts` — Chi-squared comparison of two count tables
  with multiple-testing correction.
* :func:`compare_repertoire_kmers` — End-to-end comparison of two
  :class:`~mir.common.repertoire.Repertoire` objects.
* :func:`plot_comparison` — Scatter / volcano visualisation of comparison
  results.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from mir.basic.tokens import tokenize_str
from mir.common.repertoire import Repertoire


# ---------------------------------------------------------------------------
# K-mer counting
# ---------------------------------------------------------------------------

class KmerCounter:
    """Count overlapping k-mers across all CDR3 sequences in a repertoire.

    Uses the C-accelerated :func:`~mir.basic.tokens.tokenize_str` to extract
    k-mers from each clonotype's ``cdr3aa`` field.

    Parameters
    ----------
    k : int
        K-mer length (must be >= 1).
    repertoire : Repertoire
        Source repertoire whose clonotypes will be scanned.

    Examples
    --------
    >>> counter = KmerCounter(k=3, repertoire=rep)
    >>> counts = counter.counts()          # dict[str, int]
    >>> df = counter.counts_dataframe()    # single-column DataFrame
    """

    def __init__(self, k: int, repertoire: Repertoire) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.repertoire = repertoire
        self._counts: Counter[str] | None = None

    def counts(self) -> dict[str, int]:
        """Return k-mer counts for the repertoire.

        Results are cached after the first call.

        Returns
        -------
        dict[str, int]
            Mapping from k-mer string to occurrence count.
        """
        if self._counts is None:
            self._counts = Counter()
            for cl in self.repertoire.clonotypes:
                self._counts.update(tokenize_str(cl.cdr3aa, self.k))
        return dict(self._counts)

    def counts_dataframe(self, column: str = "count") -> pd.DataFrame:
        """Return counts as a single-column :class:`~pandas.DataFrame`.

        Parameters
        ----------
        column : str
            Name of the count column (default ``"count"``).

        Returns
        -------
        pandas.DataFrame
            Indexed by k-mer string.
        """
        return pd.DataFrame.from_dict(self.counts(), orient="index", columns=[column])


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_kmer_counts(
    counts_1: dict[str, int],
    counts_2: dict[str, int],
    p_adj_method: str = "holm",
    p_adj_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Chi-squared comparison of two k-mer count dictionaries.

    For every k-mer observed in either repertoire a 2 × 2 contingency test
    is performed.  P-values are corrected for multiple testing.

    Parameters
    ----------
    counts_1, counts_2 : dict[str, int]
        K-mer occurrence counts (e.g. from :meth:`KmerCounter.counts`).
    p_adj_method : str
        Method for :func:`statsmodels.stats.multitest.multipletests`
        (default ``"holm"``).  Ignored when *p_adj_func* is given.
    p_adj_func : callable, optional
        Custom function that accepts and returns an array of p-values.
        When provided, *p_adj_method* is ignored.

    Returns
    -------
    pandas.DataFrame
        Columns: ``count_1``, ``count_2``, ``freq_1``, ``freq_2``,
        ``freq_fc``, ``p_val``, ``p_val_adj``.  Indexed by k-mer.
    """
    df1 = pd.DataFrame.from_dict(counts_1, orient="index", columns=["count_1"])
    df2 = pd.DataFrame.from_dict(counts_2, orient="index", columns=["count_2"])
    df = df1.join(df2, how="outer").fillna(0).astype({"count_1": int, "count_2": int})

    n1 = df["count_1"].sum()
    n2 = df["count_2"].sum()
    if n1 == 0 or n2 == 0:
        raise ValueError("Both count tables must be non-empty")

    df["freq_1"] = df["count_1"] / n1
    df["freq_2"] = df["count_2"] / n2

    # Vectorised contingency tables → chi-squared p-values
    pvals = np.empty(len(df))
    c1 = df["count_1"].values
    c2 = df["count_2"].values
    for i in range(len(df)):
        table = [[c1[i], n1 - c1[i]], [c2[i], n2 - c2[i]]]
        pvals[i] = chi2_contingency(table)[1]
    df["p_val"] = pvals

    # Fold change (freq_1 / freq_2); 0-frequency guarded by fillna above
    with np.errstate(divide="ignore", invalid="ignore"):
        df["freq_fc"] = df["freq_1"] / df["freq_2"]

    # Multiple testing correction
    if p_adj_func is not None:
        df["p_val_adj"] = p_adj_func(df["p_val"].values)
    else:
        df["p_val_adj"] = multipletests(df["p_val"].values, method=p_adj_method)[1]

    return df


def compare_repertoire_kmers(
    repertoire_1: Repertoire,
    repertoire_2: Repertoire,
    k: int,
    p_adj_method: str = "holm",
    p_adj_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Compare two repertoires by k-mer frequency using chi-squared tests.

    Convenience wrapper that builds :class:`KmerCounter` instances,
    extracts counts, and delegates to :func:`compare_kmer_counts`.

    Parameters
    ----------
    repertoire_1, repertoire_2 : Repertoire
        Repertoires to compare.
    k : int
        K-mer length.
    p_adj_method : str
        Multiple-testing correction method (default ``"holm"``).
    p_adj_func : callable, optional
        Custom p-value adjustment function.

    Returns
    -------
    pandas.DataFrame
        Same format as :func:`compare_kmer_counts`.
    """
    c1 = KmerCounter(k, repertoire_1).counts()
    c2 = KmerCounter(k, repertoire_2).counts()
    return compare_kmer_counts(c1, c2, p_adj_method=p_adj_method, p_adj_func=p_adj_func)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_comparison(
    df: pd.DataFrame,
    kind: str = "volcano",
    ax=None,
    top_n: int = 10,
):
    """Plot the results of a k-mer comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`compare_kmer_counts` or
        :func:`compare_repertoire_kmers`.
    kind : ``"volcano"`` | ``"scatter"``
        Plot type.  ``"scatter"`` shows log₂ frequencies; ``"volcano"``
        shows log₂ fold-change vs. −log₂ p-value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created automatically if *None*.
    top_n : int
        Number of top hits to label on volcano plots (default 10).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # lazy import to avoid hard dep at module level

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6) if kind == "volcano" else (10, 10))

    if kind == "scatter":
        # Guard against log2(0)
        nonzero = df[(df["freq_1"] > 0) & (df["freq_2"] > 0)]
        ax.scatter(np.log2(nonzero["freq_1"]), np.log2(nonzero["freq_2"]), s=1)
        lo = np.log2(nonzero[["freq_1", "freq_2"]].min().min())
        hi = np.log2(nonzero[["freq_1", "freq_2"]].max().max())
        ax.plot([lo, hi], [lo, hi], "--", c="red")
        ax.set_xlabel("log₂(freq_1)")
        ax.set_ylabel("log₂(freq_2)")

    elif kind == "volcano":
        floor = 2.0 ** -100
        pv = df["p_val"].clip(lower=floor)
        fc = df["freq_fc"].replace([np.inf, -np.inf], np.nan).dropna()
        log_fc = np.log2(fc)
        neg_log_p = -np.log2(pv.loc[log_fc.index])

        ax.scatter(log_fc, neg_log_p, s=1)
        ax.set_xlabel("log₂ FC")
        ax.set_ylabel("−log₂(p)")

        # Label top hits
        top = df.loc[log_fc.index].sort_values("p_val").head(top_n)
        for kmer, row in top.iterrows():
            x = np.log2(row["freq_fc"]) if np.isfinite(np.log2(row["freq_fc"])) else 0
            y = -np.log2(max(row["p_val"], floor))
            ax.annotate(
                kmer, (x, y),
                textcoords="offset points", ha="center",
                xytext=(0, 10), arrowprops=dict(arrowstyle="-", lw=0.5),
            )
    else:
        raise ValueError(f"Unknown plot kind {kind!r}; use 'scatter' or 'volcano'")

    return ax
