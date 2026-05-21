"""Position-weight matrix and sequence logo visualisation for CDR3 motifs.

The central goal of CDR3 motif logos is to reveal *antigen-driven selection*
by removing the germline baseline.  CDR3 sequences of antigen-specific T-cell
clones are enriched for particular residues at certain positions — but V-gene
and J-gene templates already encode highly conserved residues at the CDR3 ends
(e.g. the N-terminal Cys and the J-gene STDTQYF stretch in TRBV9/TRBJ2-3
CDR3s).  If we show a plain IC logo, these germline-encoded letters dominate
and obscure the motif.  Subtracting an OLGA-derived background for the *same*
V-gene / J-gene / length combination collapses the germline signal to ≈0 and
reveals only what is enriched (or depleted) *relative to recombination
expectation*.

**Standard IC logo** (Schneider *et al.* 1986; Schneider & Stephens 1990)::

    IC[p]       = log₂(20) + Σₐ f[p,a] · log₂(f[p,a])   (bits, ≥ 0)
    h_IC[p,a]   = f[p,a] · IC[p]

**Selection logo** (log-odds KL divergence per residue, this module)::

    h_sel[p,a]  = f[p,a] · log₂(f[p,a] / f_bg[p,a])

h_sel > 0 means the residue is *more* frequent than the OLGA expectation →
antigen-driven enrichment.  h_sel < 0 means *depletion* (drawn inverted).
At germline-encoded positions f ≈ f_bg so h_sel ≈ 0: the germline signal is
removed.

**Two background regimes**:

* **Per-VJ-len background** (from :func:`get_vj_background`): OLGA sequences
  with the *same* V, J, and CDR3 length.  Removes both V-gene and J-gene
  germline signal; the selection logo shows only the CDR3-centre motif.
* **All-VJ aggregate background** (from :func:`aggregate_vj_background`):
  OLGA sequences of the *same CDR3 length* averaged over all V/J combinations
  weighted by background pool size.  Retains V-gene and J-gene contributions;
  useful as a weaker baseline or when motif sequences span multiple V/J genes.

Note on motif_pwms heights
--------------------------
The ``motif_pwms.txt.gz`` ``height.I.norm`` column uses the VDJdb-motifs
cross-entropy formula ``I.norm[p] = −Σₐ f · ln(f_bg) / ln(20) / 2`` (always
≥ 0) and ``height.I`` is in [0, 1] scale (IC / log₂20), not bits.  These
differ from the log-odds h_sel formula above.  :func:`pwm_from_motif_pwms`
exposes these columns as ``ic_height`` / ``bg_height`` in the motif_pwms scale.

References
----------
Pogorelyy M.V., Minervina A.A., Shugay M. *et al.* Detecting T cell receptors
involved in immune responses from single repertoire snapshots.
*PLoS Biol.* **17**, e3000314 (2019).
https://doi.org/10.1371/journal.pbio.3000314

Schneider T.D., Stormo G.D., Gold L. and Ehrenfeucht A. Information content
of binding sites on nucleotide sequences. *J. Mol. Biol.* **188**, 415–431
(1986).

Schneider T.D. and Stephens R.M. Sequence logos: a new way to display
consensus sequences. *Nucleic Acids Res.* **18**, 6097–6100 (1990).

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

#: Five-category biochemistry colour scheme matching the VDJdb-motifs / Pogorelyy
#: *et al.* 2019 Fig 2e convention.  Letter colours indicate chemical class:
#: aromatic (purple), nonpolar aliphatic (green), polar uncharged (gold),
#: negatively charged (blue), positively charged (red).
BIOCHEMISTRY_COLORS: dict[str, str] = {
    # Aromatic — W F Y H
    "W": "#7b3f9e",
    "F": "#7b3f9e",
    "Y": "#7b3f9e",
    "H": "#7b3f9e",
    # Nonpolar aliphatic — A V I L M G P
    "A": "#3aac5e",
    "V": "#3aac5e",
    "I": "#3aac5e",
    "L": "#3aac5e",
    "M": "#3aac5e",
    "G": "#3aac5e",
    "P": "#3aac5e",
    # Polar uncharged — S T N Q C
    "S": "#d4a520",
    "T": "#d4a520",
    "N": "#d4a520",
    "Q": "#d4a520",
    "C": "#d4a520",
    # Negatively charged — D E
    "D": "#2c5aa0",
    "E": "#2c5aa0",
    # Positively charged — K R
    "K": "#c0392b",
    "R": "#c0392b",
}

#: Alias kept for backward compatibility.
CHEMISTRY_COLORS: dict[str, str] = BIOCHEMISTRY_COLORS


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
    """Add Shannon IC and selection-logo height columns to a PWM.

    Args:
        pwm: Output of :func:`compute_pwm` — must contain ``pos``, ``aa``,
            ``frequency`` columns.
        background: Optional background PWM in the same format (``pos``,
            ``aa``, ``frequency``).  Typically from :func:`get_vj_background`
            or :func:`aggregate_vj_background`.  When provided, a ``bg_height``
            column is added containing per-residue log-odds heights.  Values
            may be negative (depleted residues).
        bg_floor: Minimum background frequency used to prevent ``log2(0)``
            when *background* frequencies are very small (default ``1e-6``).

    Returns:
        Input *pwm* extended with an ``ic_height`` column and, if *background*
        was given, a ``bg_height`` column.  Both are in bits.

    Example:
        >>> bg = get_vj_background(pwms, v_gene="TRBV9*01", j_gene="TRBJ2-3*01", length=15)
        >>> logo = compute_logo(pwm, background=bg)
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

    The resulting background captures both V-gene and J-gene germline residue
    frequencies.  Passing it to :func:`compute_logo` yields a selection logo
    where germline-encoded positions collapse to ≈0 (since f ≈ f_bg), leaving
    only antigen-driven enrichment visible.

    Args:
        motif_pwms: Full ``motif_pwms`` table from :func:`load_motif_pwms`.
        v_gene: V-gene representative name as it appears in ``v.segm.repr``
            (e.g. ``"TRBV9*01"``).  Prefix matching is used if no exact match.
        j_gene: J-gene representative name as it appears in ``j.segm.repr``
            (e.g. ``"TRBJ2-3*01"``).  Prefix matching is used if no exact match.
        length: CDR3 amino-acid length.
        species: Species filter (default ``"HomoSapiens"``).
        gene: Receptor gene filter (default ``"TRB"``).  Pass ``"TRA"`` for
            alpha-chain CDR3s — always set explicitly to avoid mixing TRA/TRB.
        min_bg_seqs: Minimum ``total.bg`` required to use the cluster.
            Clusters with fewer synthetic background sequences are skipped
            (default 100; ≥ 1 000 recommended for production).

    Returns:
        Polars DataFrame with columns ``pos``, ``aa``, ``frequency`` (the
        OLGA background frequencies), or ``None`` if no matching cluster was
        found.

    Example:
        >>> bg = get_vj_background(
        ...     pwms, v_gene="TRBV9*01", j_gene="TRBJ2-3*01",
        ...     length=15, species="HomoSapiens", gene="TRB",
        ... )
        >>> logo = compute_logo(my_pwm, background=bg)
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


def aggregate_vj_background(
    motif_pwms: pl.DataFrame,
    *,
    length: int,
    species: str = "HomoSapiens",
    gene: str = "TRB",
    min_bg_seqs: int = 100,
) -> pl.DataFrame | None:
    """Compute a length-specific background pooled across all V/J combinations.

    Averages OLGA background frequencies over all V/J/length clusters present
    in *motif_pwms*, weighted by ``total.bg`` (background pool size per
    cluster).  This background captures length-specific amino-acid composition
    bias (e.g. the overall CDR3 length-15 baseline) but retains V-gene and
    J-gene contributions.

    Use this background when the motif sequences span multiple V/J genes or
    when you want a conservative baseline.  For full germline removal — the
    standard approach for public TCR motifs — use :func:`get_vj_background`
    with the specific V/J gene of the motif.

    Args:
        motif_pwms: Full ``motif_pwms`` table from :func:`load_motif_pwms`.
        length: CDR3 amino-acid length to aggregate over.
        species: Species filter (default ``"HomoSapiens"``).
        gene: Receptor gene filter (default ``"TRB"``).
        min_bg_seqs: Minimum ``total.bg`` for a VJ cluster to be included in
            the average (default 100).

    Returns:
        Polars DataFrame with columns ``pos``, ``aa``, ``frequency`` (the
        weighted-average OLGA background frequencies), or ``None`` if no
        qualifying clusters were found.

    Example:
        >>> bg_agg = aggregate_vj_background(pwms, length=15, gene="TRB")
        >>> logo = compute_logo(pwm, background=bg_agg)
    """
    subset = motif_pwms.filter(
        (pl.col("len") == length)
        & (pl.col("species") == species)
        & (pl.col("gene") == gene)
    )
    if subset.is_empty():
        return None

    # One representative cid per (V, J) — the one with the largest background pool.
    # Use explicit max aggregation to avoid sort-order issues with group_by.
    cid_stats = (
        subset
        .select(["cid", "v.segm.repr", "j.segm.repr", "total.bg"])
        .unique()
        .filter(pl.col("total.bg") >= min_bg_seqs)
    )
    if cid_stats.is_empty():
        return None

    max_bg_per_vj = (
        cid_stats
        .group_by(["v.segm.repr", "j.segm.repr"])
        .agg(pl.col("total.bg").max().alias("_max_bg"))
    )

    # Join to select the cid with the maximum total.bg per VJ pair.
    # Deduplicate by cid so the same cluster is never double-counted when it
    # happens to be the best representative for two different VJ gene groups.
    best_cids = (
        cid_stats
        .join(max_bg_per_vj, on=["v.segm.repr", "j.segm.repr"])
        .filter(pl.col("total.bg") == pl.col("_max_bg"))
        .unique(["v.segm.repr", "j.segm.repr"])
        .select(
            pl.col("cid"),
            pl.col("total.bg").alias("_weight"),
        )
        .unique("cid")
    )

    if best_cids.is_empty():
        return None

    # Global denominator: total weight across ALL selected clusters.
    # Must be constant across all (pos, aa) groups so that frequencies at each
    # position sum to ≤ 1 (= 1 exactly when all AAs are represented in motif_pwms).
    total_weight = float(best_cids["_weight"].sum())

    # Background frequencies for the selected clusters, using "_weight" alias to
    # avoid a naming conflict with the "total.bg" column already present in subset.
    bg_data = (
        subset
        .join(best_cids, on="cid")
        .select(["pos", "aa", "freq.bg", "_weight"])
        .with_columns(
            (pl.col("freq.bg") * pl.col("_weight")).alias("_wfreq")
        )
    )

    # Weighted average: Σ_vj(f_bg[p,a] * w_vj) / Σ_vj(w_vj)
    # Denominator is the same global total_weight for ALL (pos, aa) pairs so that
    # Σ_a freq[p,a] = 1.0 (since Σ_a f_bg[p,a,c] = 1.0 for every cluster c).
    return (
        bg_data
        .group_by(["pos", "aa"])
        .agg(pl.col("_wfreq").sum().alias("_wsum"))
        .with_columns(
            (pl.col("_wsum") / pl.lit(total_weight)).alias("frequency")
        )
        .select(
            pl.col("pos").cast(pl.Int32),
            pl.col("aa"),
            pl.col("frequency"),
        )
        .sort(["pos", "aa"])
    )


def build_motif_logos_vj(
    sequences_df: pl.DataFrame,
    motif_pwms: pl.DataFrame,
    *,
    species: str = "HomoSapiens",
    gene: str = "TRB",
    pseudocount: float = 0.5,
    min_seqs: int = 3,
    cdr3_col: str = "junction_aa",
    v_col: str = "v_gene",
    j_col: str = "j_gene",
) -> dict[tuple[str | None, str | None, int], pl.DataFrame]:
    """Build per-VJ-len and per-len PWM logos with matched OLGA backgrounds.

    Groups the input sequences by (V-gene, J-gene, CDR3 length) and produces
    a logo DataFrame for each group using the corresponding OLGA per-VJ-len
    background.  Also produces a length-aggregated logo for each distinct CDR3
    length using :func:`aggregate_vj_background`.

    This is the recommended entry point when building selection logos for
    ALICE or TCRNET hits, 1-mismatch connected components, or any set of
    antigen-associated CDR3 sequences that span multiple V/J gene families.

    Args:
        sequences_df: Polars DataFrame containing at least *cdr3_col*,
            *v_col*, and *j_col* columns.  Typically the ``table`` output of
            :func:`~mir.biomarkers.alice.compute_alice` or the hit DataFrame
            produced by :func:`~mir.biomarkers.alice.alice_hit_clusters`.
        motif_pwms: Full ``motif_pwms`` table from :func:`load_motif_pwms`.
        species: Species for OLGA background lookup (default ``"HomoSapiens"``).
        gene: Receptor gene for background lookup (default ``"TRB"``).
        pseudocount: Laplace smoothing for PWM construction (default 0.5).
        min_seqs: Minimum number of sequences required to build a logo for a
            given group (default 3).
        cdr3_col: Column name containing CDR3 sequences (default
            ``"junction_aa"``).
        v_col: Column name containing V-gene (default ``"v_gene"``).
        j_col: Column name containing J-gene (default ``"j_gene"``).

    Returns:
        Dictionary mapping group keys to logo DataFrames:

        * ``(v_gene, j_gene, length)`` — per-VJ-len logo with per-VJ OLGA
          background (``bg_height`` present when a background was found).
        * ``(None, None, length)`` — length-aggregated logo built from *all*
          sequences of that length, using the all-VJ weighted-average
          background from :func:`aggregate_vj_background`.

    Example:
        >>> logos = build_motif_logos_vj(alice_hits, pwms, gene="TRB")
        >>> vj_logo = logos[("TRBV9*01", "TRBJ2-3*01", 15)]
        >>> agg_logo = logos[(None, None, 15)]
        >>> fig, axes = plot_motif_logos(
        ...     vj_logo, v_gene="TRBV9*01", j_gene="TRBJ2-3*01", n_seqs=41
        ... )
    """
    df = sequences_df.with_columns(
        pl.col(cdr3_col).str.len_chars().alias("_cdr3_len")
    )

    results: dict[tuple[str | None, str | None, int], pl.DataFrame] = {}

    # Per-VJ-len logos
    for (v, j, length), group in df.group_by([v_col, j_col, "_cdr3_len"]):
        seqs = group[cdr3_col].to_list()
        if len(seqs) < min_seqs:
            continue
        pwm = compute_pwm(seqs, pseudocount=pseudocount, length=int(length))
        bg = get_vj_background(
            motif_pwms, v_gene=str(v), j_gene=str(j),
            length=int(length), species=species, gene=gene,
        )
        logo = compute_logo(pwm, background=bg)
        results[(str(v), str(j), int(length))] = logo

    # Per-len aggregate logos (all V/J pooled for that length)
    for (length,), group in df.group_by(["_cdr3_len"]):
        seqs = group[cdr3_col].to_list()
        if len(seqs) < min_seqs:
            continue
        pwm = compute_pwm(seqs, pseudocount=pseudocount, length=int(length))
        bg_agg = aggregate_vj_background(
            motif_pwms, length=int(length), species=species, gene=gene,
        )
        logo = compute_logo(pwm, background=bg_agg)
        results[(None, None, int(length))] = logo

    return results


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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _get_color_scheme(color_scheme: str | dict) -> dict[str, str]:
    if isinstance(color_scheme, dict):
        return color_scheme
    if color_scheme in ("biochemistry", "chemistry"):
        return BIOCHEMISTRY_COLORS
    raise ValueError(
        f"Unknown color_scheme {color_scheme!r}; use 'biochemistry' or a dict."
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
    color_scheme: str | dict = "biochemistry",
    ylabel: str | None = None,
    letter_width: float = 0.85,
    show_xaxis: bool = True,
) -> None:
    """Render a sequence logo onto *ax*.

    Letters are stacked from bottom to top with the **most frequent / tallest
    letter at the top** (WebLogo convention).  For background-normalised logos,
    enriched residues (h > 0) stack upward and depleted residues (h < 0) stack
    downward with letters drawn inverted.

    Args:
        logo_df: Logo DataFrame with ``pos``, ``aa``, and *height_col*.
            All positions need not have all 20 residues — only rows present
            are drawn.
        ax: Matplotlib ``Axes`` instance to draw on.
        height_col: Name of the column containing per-residue heights
            (default ``"ic_height"``).  Use ``"bg_height"`` for the
            selection-logo panel.
        color_scheme: ``"biochemistry"`` (default, 5-category biochemical
            classification matching Pogorelyy *et al.* 2019 Fig 2e) or a
            custom ``dict[aa, color]``.
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
            # IC logo: sort ascending so the tallest letter is drawn last (on top).
            # Letters stack bottom → top; smallest frequency letters are at the
            # bottom, the dominant residue is the top-most visible letter.
            pos_data = pos_data.sort(height_col, descending=False)
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
            # Selection logo: sort ascending (smallest enrichment first, highest
            # enrichment letter on top for positive stack).
            pos_pos = pos_data.filter(pl.col(height_col) > 1e-10).sort(
                height_col, descending=False
            )
            # For negative stack: most-depleted letter is outermost (furthest from 0).
            # Sort descending so most-negative is processed last (drawn furthest down).
            pos_neg = pos_data.filter(pl.col(height_col) < -1e-10).sort(
                height_col, descending=True
            )

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
        ylabel = "IC (bits)" if not has_negatives else "Selection (bits)"
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
    color_scheme: str | dict = "biochemistry",
    show_bg: bool = True,
) -> tuple:
    """Two-panel (IC logo + selection logo) figure.

    Reproduces the format of Pogorelyy *et al.* 2019 Fig 2e: the top panel
    shows the standard IC logo (germline-encoded positions are tall), and the
    bottom panel shows the selection logo (germline positions collapse to ≈0,
    only antigen-driven residues are visible).  V and J gene names appear in
    the figure title.

    Args:
        logo_df: DataFrame with ``pos``, ``aa``, ``ic_height`` columns and
            optionally ``bg_height``.  Produced by :func:`compute_logo` or
            :func:`pwm_from_motif_pwms`.
        v_gene: V-gene name shown in the title (e.g. ``"TRBV9*01"``).
        j_gene: J-gene name shown in the title (e.g. ``"TRBJ2-3*01"``).
        n_seqs: Cluster size shown as ``(n = …)`` in the title.
        title: Figure suptitle.  Auto-generated from V/J genes if ``None``.
        figsize: Figure size in inches.  Defaults to ``(9, 3.5)`` for one
            panel or ``(9, 6)`` for two panels.
        color_scheme: Passed to :func:`plot_logo` (default ``"biochemistry"``).
        show_bg: Whether to draw the selection-logo panel when ``bg_height``
            is present in *logo_df* (default ``True``).

    Returns:
        ``(fig, axes)`` — the Matplotlib Figure and an ndarray of Axes objects
        (shape ``(1,)`` or ``(2,)`` depending on whether the bg panel is shown).

    Example:
        >>> bg = get_vj_background(
        ...     pwms, v_gene="TRBV9*01", j_gene="TRBJ2-3*01",
        ...     length=15, species="HomoSapiens", gene="TRB",
        ... )
        >>> logo = compute_logo(pwm, background=bg)
        >>> fig, axes = plot_motif_logos(logo, v_gene="TRBV9*01", j_gene="TRBJ2-3*01")
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
    axes[0].set_title("IC logo", fontsize=9, loc="left", pad=4)

    if has_bg:
        plot_logo(
            logo_df, axes[1],
            height_col="bg_height",
            color_scheme=color_scheme,
            ylabel="Selection (bits)",
            show_xaxis=True,
        )
        axes[1].set_title("Selection logo (VJ-background subtracted)", fontsize=9, loc="left", pad=4)

    # V / J gene names and cluster size in the suptitle — not on the axes margins
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
    fig.subplots_adjust(hspace=0.5, top=0.90, bottom=0.10, left=0.10, right=0.95)

    return fig, axes


__all__ = [
    "AA_ORDER",
    "BIOCHEMISTRY_COLORS",
    "CHEMISTRY_COLORS",
    "compute_pwm",
    "compute_logo",
    "load_motif_pwms",
    "pwm_from_motif_pwms",
    "get_vj_background",
    "aggregate_vj_background",
    "build_motif_logos_vj",
    "plot_logo",
    "plot_motif_logos",
    "compute_cluster_profiles",
]
