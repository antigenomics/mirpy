"""Token frequency analysis and differential comparison of immune repertoires.

Extracts overlapping k-mers from CDR3 amino-acid sequences, counts their
occurrences across a :class:`~mir.common.repertoire.Repertoire`, and performs
chi-squared differential analysis between two repertoires.

K-mer tokenisation is delegated to the C-accelerated
:func:`mir.basic.tokens.tokenize_str` for speed.

Classes
-------
* :class:`KmerCounter` — Extract and count k-mers from a single repertoire.
* :class:`TokenCounter` — Alias of :class:`KmerCounter`.

Functions
---------
* :func:`compare_kmer_counts` — Chi-squared comparison of two count tables
  with multiple-testing correction.
* :func:`compare_repertoire_kmers` — End-to-end comparison of two
  :class:`~mir.common.repertoire.Repertoire` objects.
* :func:`compare_repertoire_tokens` — Alias of
    :func:`compare_repertoire_kmers`.
* :func:`plot_comparison` — Scatter / volcano visualisation of comparison
  results.
"""

from __future__ import annotations

from typing import Literal
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import binom, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from mir.basic.tokens import count_tokens_str
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
        self._counts: dict[str, int] | None = None

    def counts(self) -> dict[str, int]:
        """Return k-mer counts for the repertoire.

        Results are cached after the first call.

        Returns
        -------
        dict[str, int]
            Mapping from k-mer string to occurrence count.
        """
        if self._counts is None:
            self._counts = count_tokens_str(
                (cl.junction_aa for cl in self.repertoire.clonotypes),
                self.k,
            )
        return dict(self._counts)

    def counts_dataframe(self, column: str = "count") -> pl.DataFrame:
        """Return counts as a two-column :class:`~polars.DataFrame`.

        Parameters
        ----------
        column : str
            Name of the count column (default ``"count"``).

        Returns
        -------
        polars.DataFrame
            Columns: ``kmer``, *column*.
        """
        d = self.counts()
        return pl.DataFrame({"kmer": list(d.keys()), column: list(d.values())})


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_kmer_counts(
    counts_1: dict[str, int],
    counts_2: dict[str, int],
    test: Literal["chi2", "fisher", "binom"] = "chi2",
    p_adj_method: str = "holm",
    p_adj_func: Callable[[np.ndarray], np.ndarray] | None = None,
    pseudocount: int = 0,
) -> pl.DataFrame:
    """Statistical comparison of two k-mer count dictionaries.

    For every k-mer observed in either repertoire a per-k-mer test is
    performed. P-values are corrected for multiple testing.

    Parameters
    ----------
    counts_1, counts_2 : dict[str, int]
        K-mer occurrence counts (e.g. from :meth:`KmerCounter.counts`).
    test : {"chi2", "fisher", "binom"}
        Statistical test mode.

        ``"chi2"`` and ``"fisher"`` use a 2 x 2 table per k-mer:
        ``[[count_1, total_1-count_1], [count_2, total_2-count_2]]``.
        ``"binom"`` treats ``count_1`` as successes in ``total_1`` Bernoulli
        trials with background rate ``p_background = count_2 / total_2`` and
        computes one-sided enrichment p-values via
        ``binom.sf(count_1 - 1, total_1, p_background)``.
    p_adj_method : str
        Method for :func:`statsmodels.stats.multitest.multipletests`
        (default ``"holm"``). Ignored when *p_adj_func* is given.
    p_adj_func : callable, optional
        Custom function that accepts and returns an array of p-values.
        When provided, *p_adj_method* is ignored.
    pseudocount : int, optional
        Non-negative integer added to every k-mer count in both tables before
        computing totals and test statistics (default ``0`` = no pseudocount).
        Adding a small pseudocount (e.g. ``1``) prevents zero-cell contingency
        tables, stabilizes frequency estimates for rare k-mers, and makes the
        Fisher test more conservative for very low-count tokens.

    Returns
    -------
    polars.DataFrame
        Columns: ``kmer``, ``count_1``, ``count_2``, ``freq_1``, ``freq_2``,
        ``freq_fc``, ``odds_ratio``, ``p_background``, ``p_val``, ``p_val_adj``.
    """
    if pseudocount < 0:
        raise ValueError(f"pseudocount must be >= 0, got {pseudocount}")

    all_kmers = sorted(set(counts_1) | set(counts_2))
    c1_arr = np.array([counts_1.get(k, 0) + pseudocount for k in all_kmers], dtype=int)
    c2_arr = np.array([counts_2.get(k, 0) + pseudocount for k in all_kmers], dtype=int)

    n1 = int(c1_arr.sum())
    n2 = int(c2_arr.sum())
    if n1 == 0 or n2 == 0:
        raise ValueError("Both count tables must be non-empty")

    freq_1 = c1_arr / n1
    freq_2 = c2_arr / n2

    if test not in {"chi2", "fisher", "binom"}:
        raise ValueError(f"Unknown test {test!r}; use 'chi2', 'fisher', or 'binom'")

    pvals = np.empty(len(all_kmers))
    odds = np.empty(len(all_kmers))
    for i in range(len(all_kmers)):
        table = [[int(c1_arr[i]), int(n1 - c1_arr[i])], [int(c2_arr[i]), int(n2 - c2_arr[i])]]
        if test == "fisher":
            odds_i, p_i = fisher_exact(table, alternative="two-sided")
        elif test == "binom":
            p_background = float(c2_arr[i] / n2)
            odds_i = np.inf if p_background == 0 else float((c1_arr[i] / n1) / p_background)
            p_i = binom.sf(int(c1_arr[i]) - 1, int(n1), p_background)
        else:
            odds_i = np.inf if c2_arr[i] == 0 else float(c1_arr[i] / c2_arr[i])
            p_i = chi2_contingency(table)[1]
        odds[i] = float(odds_i)
        pvals[i] = float(p_i)

    with np.errstate(divide="ignore", invalid="ignore"):
        freq_fc = freq_1 / freq_2

    if p_adj_func is not None:
        p_adj = p_adj_func(pvals)
    else:
        p_adj = multipletests(pvals, method=p_adj_method)[1]

    return pl.DataFrame({
        "kmer": all_kmers,
        "count_1": c1_arr.tolist(),
        "count_2": c2_arr.tolist(),
        "freq_1": freq_1.tolist(),
        "freq_2": freq_2.tolist(),
        "freq_fc": freq_fc.tolist(),
        "odds_ratio": odds.tolist(),
        "p_background": freq_2.tolist(),
        "p_val": pvals.tolist(),
        "p_val_adj": p_adj.tolist(),
    })


def compare_repertoire_kmers(
    repertoire_1: Repertoire,
    repertoire_2: Repertoire,
    k: int,
    test: Literal["chi2", "fisher", "binom"] = "chi2",
    p_adj_method: str = "holm",
    p_adj_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pl.DataFrame:
    """Compare two repertoires by k-mer frequency using 2 x 2 tests.

    Convenience wrapper that builds :class:`KmerCounter` instances,
    extracts counts, and delegates to :func:`compare_kmer_counts`.

    Parameters
    ----------
    repertoire_1, repertoire_2 : Repertoire
        Repertoires to compare.
    k : int
        K-mer length.
    test : {"chi2", "fisher", "binom"}
        Statistical test mode passed to :func:`compare_kmer_counts`.
    p_adj_method : str
        Multiple-testing correction method (default ``"holm"``).
    p_adj_func : callable, optional
        Custom p-value adjustment function.

    Returns
    -------
    polars.DataFrame
        Same format as :func:`compare_kmer_counts`.
    """
    c1 = KmerCounter(k, repertoire_1).counts()
    c2 = KmerCounter(k, repertoire_2).counts()
    return compare_kmer_counts(
        c1,
        c2,
        test=test,
        p_adj_method=p_adj_method,
        p_adj_func=p_adj_func,
    )


class TokenCounter(KmerCounter):
    """Token-oriented alias for :class:`KmerCounter`."""


def compare_repertoire_tokens(
    repertoire_1: Repertoire,
    repertoire_2: Repertoire,
    k: int,
    test: Literal["chi2", "fisher", "binom"] = "chi2",
    p_adj_method: str = "holm",
    p_adj_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pl.DataFrame:
    """Token-oriented alias for :func:`compare_repertoire_kmers`."""
    return compare_repertoire_kmers(
        repertoire_1,
        repertoire_2,
        k,
        test=test,
        p_adj_method=p_adj_method,
        p_adj_func=p_adj_func,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_comparison(
    df: pl.DataFrame,
    kind: str = "volcano",
    ax=None,
    top_n: int = 10,
):
    """Plot the results of a k-mer comparison.

    Parameters
    ----------
    df : polars.DataFrame
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
        nonzero = df.filter((pl.col("freq_1") > 0) & (pl.col("freq_2") > 0))
        f1 = nonzero["freq_1"].to_numpy()
        f2 = nonzero["freq_2"].to_numpy()
        ax.scatter(np.log2(f1), np.log2(f2), s=1)
        lo = min(np.log2(f1).min(), np.log2(f2).min())
        hi = max(np.log2(f1).max(), np.log2(f2).max())
        ax.plot([lo, hi], [lo, hi], "--", c="red")
        ax.set_xlabel("log₂(freq_1)")
        ax.set_ylabel("log₂(freq_2)")

    elif kind == "volcano":
        floor = 2.0 ** -100
        finite_fc = df.filter(
            pl.col("freq_fc").is_finite()
            & pl.col("freq_fc").is_not_null()
            & (pl.col("freq_fc") > 0)
        )
        log_fc = np.log2(finite_fc["freq_fc"].to_numpy())
        pv = np.maximum(finite_fc["p_val"].to_numpy(), floor)
        neg_log_p = -np.log2(pv)

        ax.scatter(log_fc, neg_log_p, s=1)
        ax.set_xlabel("log₂ FC")
        ax.set_ylabel("−log₂(p)")

        top = finite_fc.sort("p_val").head(top_n)
        for row in top.iter_rows(named=True):
            fc = row["freq_fc"]
            x = float(np.log2(fc)) if fc > 0 and np.isfinite(fc) else 0.0
            y = -np.log2(max(row["p_val"], floor))
            ax.annotate(
                row["kmer"], (x, y),
                textcoords="offset points", ha="center",
                xytext=(0, 10), arrowprops=dict(arrowstyle="-", lw=0.5),
            )
    else:
        raise ValueError(f"Unknown plot kind {kind!r}; use 'scatter' or 'volcano'")

    return ax


__all__ = [
    "KmerCounter",
    "TokenCounter",
    "compare_kmer_counts",
    "compare_repertoire_kmers",
    "compare_repertoire_tokens",
    "plot_comparison",
]
