"""Position-weight matrix and sequence logo visualisation for CDR3 motifs.

Implements per-position amino-acid frequency matrices (PWMs) from raw CDR3
sequences and renders standard (Shannon information-content) and
background-normalised sequence logos as publication-quality matplotlib figures.

**Standard IC logo** (Schneider *et al.* 1986; Schneider & Stephens 1990)::

    IC[p]       = log₂(20) + Σₐ f[p,a] · log₂(f[p,a])
    h_IC[p,a]   = f[p,a] · IC[p]

**Background-normalised logo** (log-odds KL divergence per residue)::

    h_norm[p,a] = f[p,a] · log₂(f[p,a] / f_bg[p,a])

Enriched residues (h_norm > 0) are stacked above zero; depleted residues
(h_norm < 0) are stacked below zero with letters drawn upside-down.

Note: the ``motif_pwms.txt.gz`` ``height.I.norm`` column follows the VDJdb-motifs
cross-entropy formula ``I.norm[p] = −Σₐ f·ln(f_bg) / ln(20) / 2``, which is
always ≥ 0.  :func:`compute_logo` uses log-odds (KL per residue) which can be
negative for depleted residues — a more informative visualization for motif work.

Background frequencies are obtained from OLGA-generated synthetic sequences
matched to the same V-gene / J-gene / CDR3-length combination, stored in the
``motif_pwms.txt.gz`` file distributed via the ``isalgo/airr_benchmark``
HuggingFace dataset.

Classes
-------
* None — the module is purely functional.

Functions
---------
* :func:`compute_pwm` — Build a PWM from raw CDR3 sequences.
* :func:`compute_logo` — Add IC and background-normalised height columns.
* :func:`load_motif_pwms` — Load ``motif_pwms.txt.gz`` as a Polars DataFrame.
* :func:`pwm_from_motif_pwms` — Extract a pre-computed logo from ``motif_pwms``.
* :func:`get_vj_background` — Look up a VJ-specific background PWM.
* :func:`plot_logo` — Render a single logo panel on given axes.
* :func:`plot_motif_logos` — Two-panel (IC + bg-normalised) figure.

References
----------
Pogorelyy M.V., Minervina A.A., Shugay M. *et al.* Identifying the target of
an antigen-specific T cell response from combined TCR repertoire and binding
assay data. *eBioMedicine* **43**, 545–553 (2019).
https://doi.org/10.1016/j.ebiom.2019.04.050

Schneider T.D., Stormo G.D., Gold L. and Ehrenfeucht A. Information content
of binding sites on nucleotide sequences. *J. Mol. Biol.* **188**, 415–431
(1986). — *primary source for the IC formula*

Schneider T.D. and Stephens R.M. Sequence logos: a new way to display
consensus sequences. *Nucleic Acids Res.* **18**, 6097–6100 (1990).
— *introduced the logo visualization*

Crooks G.E. *et al.* WebLogo: A sequence logo generator.
*Genome Res.* **14**, 1188–1190 (2004).
"""

from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical order of the 20 standard amino acids.
AA_ORDER: list[str] = list("ACDEFGHIKLMNPQRSTVWY")

_AA_SET: frozenset[str] = frozenset(AA_ORDER)
_LOG2_20: float = np.log2(20.0)

#: Chemistry-based colour scheme (ClustalX-inspired, publication-safe palette).
#: Hydrophobic residues → amber; polar/uncharged → teal; positive → blue;
#: negative → red; glycine → silver; tyrosine → indigo.
CHEMISTRY_COLORS: dict[str, str] = {
    "A": "#e09c3c",  # hydrophobic
    "V": "#e09c3c",
    "I": "#e09c3c",
    "L": "#e09c3c",
    "M": "#e09c3c",
    "F": "#e09c3c",
    "W": "#e09c3c",
    "P": "#e09c3c",
    "S": "#3dae77",  # polar uncharged
    "T": "#3dae77",
    "N": "#3dae77",
    "Q": "#3dae77",
    "C": "#c5a928",  # cysteine (special)
    "G": "#aaaaaa",  # glycine (flexible)
    "H": "#3366cc",  # positive
    "K": "#3366cc",
    "R": "#3366cc",
    "D": "#cc3322",  # negative
    "E": "#cc3322",
    "Y": "#7f3fb0",  # aromatic (tyrosine)
}


# ---------------------------------------------------------------------------
# PWM computation
# ---------------------------------------------------------------------------

def compute_pwm(
    sequences: Sequence[str] | list[str],
    *,
    pseudocount: float = 0.5,
    length: int | None = None,
) -> pl.DataFrame:
    """Build a per-position amino-acid frequency table from CDR3 sequences.

    Sequences shorter or longer than the modal length are silently dropped
    unless *length* is given explicitly.  A Laplace pseudocount is applied to
    every (position, residue) cell before normalising to frequencies.

    Args:
        sequences: Iterable of CDR3 amino-acid strings.
        pseudocount: Laplace smoothing count added to every cell (default 0.5).
            Set to 0 for maximum-likelihood frequencies (may produce log(0)
            in downstream logo computation).
        length: CDR3 length to retain.  If ``None``, the modal length is used.

    Returns:
        Polars DataFrame with columns ``pos``, ``aa``, ``count``,
        ``frequency``.  Every (pos, aa) pair is present for all 20 standard
        amino acids.

    Raises:
        ValueError: If *sequences* is empty or contains no valid CDR3s.

    Example:
        >>> seqs = ["CASSRS", "CASSTS", "CASSRG"]
        >>> pwm = compute_pwm(seqs)
        >>> pwm.filter(pl.col("pos") == 4)["aa"].to_list()  # variable pos
        ['A', 'C', 'D', ...]
    """
    if pseudocount < 0:
        raise ValueError(f"pseudocount must be >= 0, got {pseudocount}")

    seqs = list(sequences)
    if not seqs:
        raise ValueError("sequences must be non-empty")

    if length is None:
        length_counts = Counter(len(s) for s in seqs)
        length = length_counts.most_common(1)[0][0]

    seqs = [s for s in seqs if len(s) == length]
    if not seqs:
        raise ValueError(f"No sequences of length {length} found")

    n_seqs = len(seqs)
    total_with_pc = n_seqs + pseudocount * len(AA_ORDER)

    records: list[dict] = []
    for pos in range(length):
        col_counts = Counter(s[pos] for s in seqs if s[pos] in _AA_SET)
        for aa in AA_ORDER:
            cnt = col_counts.get(aa, 0)
            records.append({
                "pos": pos,
                "aa": aa,
                "count": cnt,
                "frequency": (cnt + pseudocount) / total_with_pc,
            })

    return pl.DataFrame(records).with_columns(
        pl.col("pos").cast(pl.Int32),
        pl.col("count").cast(pl.Int32),
    )


# ---------------------------------------------------------------------------
# Logo height computation
# ---------------------------------------------------------------------------

def compute_logo(
    pwm: pl.DataFrame,
    *,
    background: pl.DataFrame | None = None,
    bg_floor: float = 1e-6,
) -> pl.DataFrame:
    """Add Shannon IC and background-normalised height columns to a PWM.

    Args:
        pwm: Output of :func:`compute_pwm` — must contain ``pos``, ``aa``,
            ``frequency`` columns.
        background: Optional background PWM in the same format (``pos``,
            ``aa``, ``frequency``).  When provided, a ``bg_height`` column is
            added containing per-residue log-odds heights.  Values may be
            negative (depleted residues).
        bg_floor: Minimum background frequency used to prevent ``log2(0)``
            when *background* frequencies are very small (default ``1e-6``).

    Returns:
        Input *pwm* extended with an ``ic_height`` column and, if *background*
        was given, a ``bg_height`` column.  Both are in bits.

    Example:
        >>> logo = compute_logo(pwm)
        >>> logo.filter(pl.col("pos") == 0)["ic_height"].sum()  # ≈ IC at pos 0
    """
    positions = sorted(pwm["pos"].unique().to_list())
    ic_rows: list[dict] = []
    bg_rows: list[dict] = []

    bg_lookup: dict[tuple[int, str], float] = {}
    if background is not None:
        for row in background.iter_rows(named=True):
            bg_lookup[(row["pos"], row["aa"])] = float(row["frequency"])

    for pos in positions:
        pos_data = pwm.filter(pl.col("pos") == pos)
        freqs = pos_data["frequency"].to_numpy().astype(float)
        aas = pos_data["aa"].to_list()

        # Shannon IC: IC = log2(20) + sum(f * log2(f)) for f > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_f = np.where(freqs > 0, np.log2(freqs), 0.0)
        ic = max(0.0, _LOG2_20 + float(np.dot(freqs, log_f)))

        for aa, f in zip(aas, freqs.tolist()):
            ic_rows.append({"pos": pos, "aa": aa, "ic_height": float(f) * ic})

        if background is not None:
            for aa, f in zip(aas, freqs.tolist()):
                fb = max(float(bg_lookup.get((pos, aa), bg_floor)), bg_floor)
                h_bg = float(f) * np.log2(float(f) / fb) if f > 0 else 0.0
                bg_rows.append({"pos": pos, "aa": aa, "bg_height": h_bg})

    ic_df = pl.DataFrame(ic_rows).with_columns(pl.col("pos").cast(pl.Int32))
    result = pwm.join(ic_df, on=["pos", "aa"])

    if bg_rows:
        bg_df = pl.DataFrame(bg_rows).with_columns(pl.col("pos").cast(pl.Int32))
        result = result.join(bg_df, on=["pos", "aa"])

    return result


# ---------------------------------------------------------------------------
# motif_pwms.txt.gz helpers
# ---------------------------------------------------------------------------

def load_motif_pwms(path: str | Path) -> pl.DataFrame:
    """Load ``motif_pwms.txt.gz`` into a Polars DataFrame.

    The file is tab-separated and gzip-compressed.  Expected columns include
    ``cid``, ``csz``, ``species``, ``gene``, ``antigen.epitope``,
    ``v.segm.repr``, ``j.segm.repr``, ``len``, ``pos``, ``aa``,
    ``freq``, ``freq.bg``, ``height.I``, ``height.I.norm``.

    Args:
        path: Path to ``motif_pwms.txt.gz``.

    Returns:
        Full table as a Polars DataFrame.
    """
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        return pl.read_csv(fh, separator="\t")


def pwm_from_motif_pwms(
    motif_pwms: pl.DataFrame,
    cid: str,
) -> pl.DataFrame:
    """Extract a pre-computed logo for a single cluster from ``motif_pwms``.

    The returned DataFrame uses the internal logo column names so it can be
    passed directly to :func:`plot_logo` or :func:`plot_motif_logos`.

    Args:
        motif_pwms: Full ``motif_pwms`` table from :func:`load_motif_pwms`.
        cid: Cluster identifier (e.g. ``"H.B.GILGFVFTL.1"``).

    Returns:
        Polars DataFrame with columns ``pos``, ``aa``, ``frequency``,
        ``bg_frequency``, ``ic_height``, ``bg_height``.

        .. note::
            The ``ic_height`` values are in the motif_pwms normalised scale
            ``[0, 1]`` (IC / log₂20), not absolute bits.  Multiply by
            ``log2(20) ≈ 4.32`` to convert to bits for comparison with
            :func:`compute_logo` output.

    Raises:
        KeyError: If *cid* is not present in *motif_pwms*.
    """
    cluster = motif_pwms.filter(pl.col("cid") == cid)
    if cluster.is_empty():
        raise KeyError(f"Cluster {cid!r} not found in motif_pwms")
    return cluster.select(
        pl.col("pos").cast(pl.Int32),
        pl.col("aa"),
        pl.col("freq").alias("frequency"),
        pl.col("freq.bg").alias("bg_frequency"),
        pl.col("height.I").alias("ic_height"),
        pl.col("height.I.norm").alias("bg_height"),
    )


def get_vj_background(
    motif_pwms: pl.DataFrame,
    *,
    v_gene: str,
    j_gene: str,
    length: int,
    species: str = "HomoSapiens",
    gene: str = "TRB",
    min_bg_seqs: int = 100,
) -> pl.DataFrame | None:
    """Return a background PWM for a given V-gene / J-gene / CDR3 length.

    Searches *motif_pwms* for the cluster with the largest synthetic background
    (``total.bg``) matching the given V, J, and length constraints.  The
    returned DataFrame contains per-position background frequencies suitable
    for passing to :func:`compute_logo` as the *background* argument.

    Args:
        motif_pwms: Full ``motif_pwms`` table from :func:`load_motif_pwms`.
        v_gene: V-gene representative name as it appears in ``v.segm.repr``
            (e.g. ``"TRBV19*01"``).  Prefix matching is used if no exact match.
        j_gene: J-gene representative name as it appears in ``j.segm.repr``
            (e.g. ``"TRBJ2-7*01"``).  Prefix matching is used if no exact match.
        length: CDR3 amino-acid length.
        species: Species filter (default ``"HomoSapiens"``).
        gene: Receptor gene filter (default ``"TRB"``).
        min_bg_seqs: Minimum ``total.bg`` required to use the cluster.
            Clusters with fewer synthetic background sequences are skipped
            (default 100).

    Returns:
        Polars DataFrame with columns ``pos``, ``aa``, ``frequency`` (the
        background frequencies), or ``None`` if no matching cluster was found.

    Example:
        >>> bg = get_vj_background(pwms, v_gene="TRBV19*01", j_gene="TRBJ2-7*01", length=13)
        >>> compute_logo(my_pwm, background=bg)
    """
    # Try exact match first, then prefix match
    for v_match, j_match in [
        (pl.col("v.segm.repr") == v_gene, pl.col("j.segm.repr") == j_gene),
        (pl.col("v.segm.repr").str.starts_with(v_gene.split("*")[0]),
         pl.col("j.segm.repr").str.starts_with(j_gene.split("*")[0])),
    ]:
        candidates = motif_pwms.filter(
            v_match & j_match
            & (pl.col("len") == length)
            & (pl.col("species") == species)
            & (pl.col("gene") == gene)
        )
        if not candidates.is_empty():
            break
    else:
        return None

    # Pick the cluster with the largest background sample
    cluster_stats = (
        candidates
        .select(["cid", "total.bg"])
        .unique()
        .sort("total.bg", descending=True)
    )
    best = cluster_stats.filter(pl.col("total.bg") >= min_bg_seqs)
    if best.is_empty():
        return None

    best_cid = best[0]["cid"][0]
    bg_data = candidates.filter(pl.col("cid") == best_cid)

    return bg_data.select(
        pl.col("pos").cast(pl.Int32),
        pl.col("aa"),
        pl.col("freq.bg").alias("frequency"),
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _get_color_scheme(color_scheme: str | dict) -> dict[str, str]:
    if isinstance(color_scheme, dict):
        return color_scheme
    if color_scheme == "chemistry":
        return CHEMISTRY_COLORS
    raise ValueError(
        f"Unknown color_scheme {color_scheme!r}; use 'chemistry' or a dict."
    )


def _draw_letter(
    ax,
    letter: str,
    x: float,
    y_bottom: float,
    width: float,
    height: float,
    color: str,
    font_prop,
) -> None:
    """Draw a single amino-acid letter scaled to the given dimensions.

    For positive *height* the letter is drawn upright from *y_bottom*.
    For negative *height* the letter is drawn inverted (upside-down),
    filling [*y_bottom*, *y_bottom* + |*height*|] with the top of the glyph
    at the larger y-coordinate — the standard convention for depleted residues
    in differential sequence logos.
    """
    if abs(height) < 1e-10:
        return
    from matplotlib.patches import PathPatch  # lazy import
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    tp = TextPath((0, 0), letter, size=1, prop=font_prop)
    bb = tp.get_extents()
    char_w = bb.x1 - bb.x0
    char_h = bb.y1 - bb.y0
    if char_w <= 0 or char_h <= 0:
        return

    sx = width / char_w
    sy = abs(height) / char_h

    if height < 0:
        # Flip vertically: map (bb.x0, bb.y0) → (x, y_bottom + |height|) = y_top
        y_top = y_bottom + abs(height)
        t = (
            Affine2D()
            .translate(-bb.x0, -bb.y0)
            .scale(sx, -sy)
            .translate(x, y_top)
        )
    else:
        t = (
            Affine2D()
            .translate(-bb.x0, -bb.y0)
            .scale(sx, sy)
            .translate(x, y_bottom)
        )

    tp_t = t.transform_path(tp)
    ax.add_patch(PathPatch(tp_t, facecolor=color, linewidth=0))


# ---------------------------------------------------------------------------
# Logo plotting
# ---------------------------------------------------------------------------

def plot_logo(
    logo_df: pl.DataFrame,
    ax,
    *,
    height_col: str = "ic_height",
    color_scheme: str | dict = "chemistry",
    ylabel: str | None = None,
    letter_width: float = 0.85,
    show_xaxis: bool = True,
) -> None:
    """Render a sequence logo onto *ax*.

    Args:
        logo_df: Logo DataFrame with ``pos``, ``aa``, and *height_col*.
            All positions need not have all 20 residues — only rows present
            are drawn.
        ax: Matplotlib ``Axes`` instance to draw on.
        height_col: Name of the column containing per-residue heights
            (default ``"ic_height"``).  Use ``"bg_height"`` for the
            background-normalised panel.
        color_scheme: ``"chemistry"`` (default) or a custom ``dict[aa, color]``.
        ylabel: Y-axis label.  If ``None``, inferred from *height_col*.
        letter_width: Fraction of one position width used by each letter
            (default 0.85; the remainder is left as whitespace).
        show_xaxis: Whether to show position tick labels (default ``True``).

    Example:
        >>> fig, axes = plt.subplots(2, 1, figsize=(8, 4))
        >>> plot_logo(logo, axes[0], height_col="ic_height")
        >>> plot_logo(logo, axes[1], height_col="bg_height")
    """
    from matplotlib.font_manager import FontProperties  # lazy import

    prop = FontProperties(family="monospace", weight="bold")
    colors = _get_color_scheme(color_scheme)
    positions = sorted(logo_df["pos"].unique().to_list())
    x_offset = (1.0 - letter_width) / 2.0

    all_heights = logo_df[height_col].to_numpy()
    has_negatives = bool(np.any(all_heights < -1e-10))

    y_max = 0.0
    y_min = 0.0

    for pos in positions:
        pos_data = logo_df.filter(pl.col("pos") == pos)

        if not has_negatives:
            # Standard IC logo: sort descending, stack upward
            pos_data = pos_data.sort(height_col, descending=True)
            y_cursor = 0.0
            for row in pos_data.iter_rows(named=True):
                h = float(row[height_col])
                if h <= 0:
                    continue
                aa = row["aa"]
                _draw_letter(
                    ax, aa,
                    pos + x_offset, y_cursor,
                    letter_width, h,
                    colors.get(aa, "#888888"), prop,
                )
                y_cursor += h
            y_max = max(y_max, y_cursor)
        else:
            # Background-normalised: positives up, negatives down (flipped)
            pos_pos = pos_data.filter(pl.col(height_col) > 1e-10).sort(height_col, descending=True)
            pos_neg = pos_data.filter(pl.col(height_col) < -1e-10).sort(height_col)

            y_cursor_pos = 0.0
            for row in pos_pos.iter_rows(named=True):
                h = float(row[height_col])
                aa = row["aa"]
                _draw_letter(
                    ax, aa,
                    pos + x_offset, y_cursor_pos,
                    letter_width, h,
                    colors.get(aa, "#888888"), prop,
                )
                y_cursor_pos += h

            y_cursor_neg = 0.0
            for row in pos_neg.iter_rows(named=True):
                h = float(row[height_col])  # h < 0
                aa = row["aa"]
                # Draw flipped letter below cursor; y_bottom = y_cursor_neg + h
                _draw_letter(
                    ax, aa,
                    pos + x_offset, y_cursor_neg + h,
                    letter_width, h,
                    colors.get(aa, "#888888"), prop,
                )
                y_cursor_neg += h

            y_max = max(y_max, y_cursor_pos)
            y_min = min(y_min, y_cursor_neg)

    # Axes formatting
    n_pos = len(positions)
    ax.set_xlim(-0.1, n_pos + 0.1)
    pad = max(0.05, (y_max - y_min) * 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)

    if has_negatives:
        ax.axhline(0, color="#333333", linewidth=0.6, zorder=0)

    if show_xaxis:
        ax.set_xticks(range(n_pos))
        ax.set_xticklabels([str(p + 1) for p in positions], fontsize=8)
        ax.set_xlabel("CDR3 position", fontsize=9)
    else:
        ax.set_xticks([])

    if ylabel is None:
        ylabel = "IC (bits)" if not has_negatives else "log-odds height (bits)"
    ax.set_ylabel(ylabel, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def plot_motif_logos(
    logo_df: pl.DataFrame,
    *,
    v_gene: str | None = None,
    j_gene: str | None = None,
    n_seqs: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    color_scheme: str | dict = "chemistry",
    show_bg: bool = True,
) -> tuple:
    """Two-panel (IC logo + background-normalised logo) figure.

    Args:
        logo_df: DataFrame with ``pos``, ``aa``, ``ic_height`` columns and
            optionally ``bg_height``.  Produced by :func:`compute_logo` or
            :func:`pwm_from_motif_pwms`.
        v_gene: V-gene name annotated on the left margin.
        j_gene: J-gene name annotated on the right margin.
        n_seqs: Cluster size annotated in the title/subtitle.
        title: Figure suptitle.  Auto-generated from V/J genes if ``None``.
        figsize: Figure size in inches.  Defaults to ``(9, 4)`` for one panel
            or ``(9, 6)`` for two panels.
        color_scheme: Passed to :func:`plot_logo`.
        show_bg: Whether to draw the background-normalised panel when
            ``bg_height`` is present in *logo_df* (default ``True``).

    Returns:
        ``(fig, axes)`` — the Matplotlib Figure and an ndarray of Axes objects
        (shape (1,) or (2,) depending on whether the bg panel is shown).

    Example:
        >>> logo = compute_logo(pwm, background=bg)
        >>> fig, axes = plot_motif_logos(logo, v_gene="TRBV9", j_gene="TRBJ2-3")
    """
    import matplotlib.pyplot as plt  # lazy import

    has_bg = show_bg and "bg_height" in logo_df.columns
    n_panels = 2 if has_bg else 1

    if figsize is None:
        figsize = (9.0, 3.5 * n_panels)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"hspace": 0.45},
    )
    axes = axes.ravel()

    # Top panel: standard IC logo
    plot_logo(
        logo_df, axes[0],
        height_col="ic_height",
        color_scheme=color_scheme,
        ylabel="IC (bits)",
        show_xaxis=not has_bg,
    )
    axes[0].set_title("Standard IC logo", fontsize=9, loc="left", pad=4)

    if has_bg:
        plot_logo(
            logo_df, axes[1],
            height_col="bg_height",
            color_scheme=color_scheme,
            ylabel="log-odds (bits)",
            show_xaxis=True,
        )
        axes[1].set_title("Background-normalised logo", fontsize=9, loc="left", pad=4)

    # V / J gene annotations on leftmost and rightmost axes
    n_pos = logo_df["pos"].n_unique()
    for ax in axes:
        if v_gene:
            ax.text(
                -0.05, 0.5, v_gene,
                transform=ax.transAxes,
                ha="right", va="center", fontsize=8,
                rotation=90, style="italic", color="#333333",
            )
        if j_gene:
            ax.text(
                1.02, 0.5, j_gene,
                transform=ax.transAxes,
                ha="left", va="center", fontsize=8,
                rotation=270, style="italic", color="#333333",
            )

    # Suptitle
    if title is None:
        parts = []
        if v_gene:
            parts.append(v_gene)
        if j_gene:
            parts.append(j_gene)
        title = " · ".join(parts) if parts else "Sequence logo"
    if n_seqs is not None:
        title = f"{title}  (n = {n_seqs:,})"
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.subplots_adjust(hspace=0.5, top=0.90, bottom=0.10, left=0.10, right=0.90)

    return fig, axes


# ---------------------------------------------------------------------------
# Aggregate cluster profile computation
# ---------------------------------------------------------------------------

def compute_cluster_profiles(
    motif_pwms: pl.DataFrame,
    *,
    min_csz: int = 30,
    species: str | None = None,
    gene: str | None = None,
) -> pl.DataFrame:
    """Compute per-position IC, entropy and cross-entropy profiles for clusters.

    For every cluster in *motif_pwms* with ``csz >= min_csz``, recovers the
    per-position Shannon information content (IC), entropy (H), and the
    VDJdb-motifs background-normalised cross-entropy score (I_norm) directly
    from the stored ``height.I`` and ``height.I.norm`` columns.

    Args:
        motif_pwms: Full table from :func:`load_motif_pwms`.
        min_csz: Minimum cluster size to include (default 30).
        species: Optional species filter (e.g. ``"HomoSapiens"``).
        gene: Optional receptor gene filter (e.g. ``"TRB"``).

    Returns:
        Polars DataFrame with columns ``cid``, ``species``, ``gene``,
        ``antigen.epitope``, ``v.segm.repr``, ``j.segm.repr``, ``len``,
        ``csz``, ``pos``, ``IC`` (bits), ``H`` (entropy, bits), ``I_norm``
        (cross-entropy score in VDJdb-motifs scale).

        ``IC = Σₐ height.I · log₂(20)`` (position-level IC recovered from
        pre-normalised heights).  ``H = log₂(20) − IC``.
        ``I_norm = Σₐ height.I.norm`` (position-level cross-entropy).

    Example:
        >>> profiles = compute_cluster_profiles(pwms, gene="TRB")
        >>> profiles.filter(pl.col("cid") == "H.B.GILGFVFTL.1")
    """
    mask = pl.col("csz") >= min_csz
    if species is not None:
        mask = mask & (pl.col("species") == species)
    if gene is not None:
        mask = mask & (pl.col("gene") == gene)

    subset = motif_pwms.filter(mask)
    if subset.is_empty():
        return pl.DataFrame()

    profiles = (
        subset
        .group_by([
            "cid", "species", "gene", "antigen.epitope",
            "v.segm.repr", "j.segm.repr", "len", "csz", "pos",
        ])
        .agg([
            (pl.col("height.I").sum() * _LOG2_20).alias("IC"),
            pl.col("height.I.norm").sum().alias("I_norm"),
        ])
        .with_columns(
            (pl.lit(_LOG2_20) - pl.col("IC")).clip(lower_bound=0.0).alias("H")
        )
        .sort(["species", "gene", "len", "cid", "pos"])
    )
    return profiles


__all__ = [
    "AA_ORDER",
    "CHEMISTRY_COLORS",
    "compute_pwm",
    "compute_logo",
    "load_motif_pwms",
    "pwm_from_motif_pwms",
    "get_vj_background",
    "plot_logo",
    "plot_motif_logos",
    "compute_cluster_profiles",
]
