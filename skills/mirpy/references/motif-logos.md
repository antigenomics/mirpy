# CDR3 Motif Logo Reference

CDR3 motif logos visualise position-specific amino acid enrichment in antigen-specific
TCR repertoires. The background-subtracted (selection) logo approach removes germline
signal to reveal antigen-driven positions, as described in:

- Pogorelyy et al. (2019) PLoS Biol. PMID:31194732 doi:10.1371/journal.pbio.3000314

## Scientific Background

V-gene and J-gene templates encode conserved residues at CDR3 ends. A plain IC logo shows
these germline residues as the tallest columns, hiding the antigen-specific motif. Subtracting
an OLGA background for the same V/J/length removes the germline signal: h_sel ≈ 0 at germline
positions, h_sel >> 0 at antigen-driven positions (Pogorelyy et al. 2019, PLoS Biol.).

## CDR3 Omega Loop Geometry

V-gene encodes the first ~5 residues; J-gene encodes the last ~4; the centre varies in length.
CDR3s of different lengths are NOT linearly aligned — aggregate profiles must use fractional
position p / (L−1):
- 0 → conserved N-terminal Cys
- 1 → conserved C-terminal Phe/Trp
- 0.5 → approximate hypervariable centre

## Key Formulas

| Logo type | Formula | Notes |
|---|---|---|
| IC logo | `h_IC[p,a] = f[p,a] · (log₂20 + Σₐ f·log₂f)` | Always ≥ 0 (bits) |
| Selection logo | `h_sel[p,a] = f[p,a] · log₂(f[p,a] / f_bg[p,a])` | Negative = depleted |
| motif_pwms height.I | IC / log₂(20) | [0,1] scale — **not bits**; multiply by log₂20 to convert |
| motif_pwms height.I.norm | −Σₐ f·ln(f_bg) / ln(20) / 2 | Cross-entropy, always ≥ 0 |

## Build a PWM From Raw Sequences

```python
from mir.biomarkers.motif_logo import compute_pwm, compute_logo, get_vj_background

pwm = compute_pwm(sequences, pseudocount=0.5)   # → pos, aa, count, frequency
logo = compute_logo(pwm)                         # adds ic_height (bits)
bg   = get_vj_background(pwms, v_call="TRBV19*01", j_call="TRBJ2-7*01",
                          length=13, species="HomoSapiens", gene="TRB")
logo_bg = compute_logo(pwm, background=bg)       # adds ic_height + bg_height
```

`compute_pwm` filters to the modal CDR3 length; set `length=` to override.
Always pass `species=` and `gene=` to avoid mixing TRA/TRB or mouse/human OLGA pools.

## Two Background Regimes

| Regime | Function | Removes |
|---|---|---|
| Per-VJ-len | `get_vj_background(v, j, len, species, gene)` | V-gene AND J-gene germline |
| All-VJ aggregate | `aggregate_vj_background(pwms, length=L, species=S, gene=G)` | Length-composition bias only |

```python
from mir.biomarkers.motif_logo import aggregate_vj_background

agg_bg = aggregate_vj_background(pwms, length=13, species="HomoSapiens", gene="TRB")
# Returns pl.DataFrame[pos, aa, frequency] — weighted average over all VJ clusters
# of the given length. Returns None if no matching clusters.
```

`get_vj_background` picks the cluster with the largest `total.bg` for the matching V/J/length;
prefix matching (strip allele suffix) is tried if exact match fails.

## Automated Per-VJ-len Logos From ALICE / TCRNET Hits

```python
from mir.biomarkers.motif_logo import build_motif_logos_vj
import polars as pl

# hits_df must have columns: junction_aa, v_call, j_call
logos = build_motif_logos_vj(
    hits_df, pwms, species="HomoSapiens", gene="TRB", min_seqs=5, pseudocount=0.5,
)
# Returns {(v, j, len): logo_df, ..., (None, None, len): logo_df, ...}
# Each logo_df has columns: pos, aa, count, frequency, ic_height, bg_height

vj_logo  = logos.get(("TRBV9", "TRBJ2-3", 15))
agg_logo = logos.get((None, None, 15))
```

## Load Pre-Computed Cluster Logos From motif_pwms.txt.gz

```python
from mir.biomarkers.motif_logo import load_motif_pwms, pwm_from_motif_pwms

pwms = load_motif_pwms(path)
logo = pwm_from_motif_pwms(pwms, "H.B.GILGFVFTL.1")
```

`motif_pwms.txt.gz` is in `isalgo/airr_benchmark` on HuggingFace (`vdjdb/**`).
Key columns: `cid`, `csz`, `species`, `gene`, `antigen.epitope`, `v.segm.repr`, `j.segm.repr`,
`len`, `pos`, `aa`, `freq`, `freq.bg`, `height.I`, `height.I.norm`.

**Sparse storage**: `freq.bg` stores only non-zero residues per position. `height.I` is in [0,1] scale, not bits.

## Plotting

```python
from mir.biomarkers.motif_logo import plot_motif_logos, plot_logo, BIOCHEMISTRY_COLORS

fig, axes = plot_motif_logos(
    logo_with_bg, v_call="TRBV19*01", j_call="TRBJ2-7*01",
    n_seqs=896, title="GILGFVFTL (Influenza A, HLA-A*02:01)",
)
# axes[0] = IC logo (always ≥ 0); axes[1] = selection logo (can be negative)
# Letters sorted ascending so the tallest letter is drawn on top (WebLogo convention).
```

`BIOCHEMISTRY_COLORS` maps all 20 amino acids to 5 colour categories:
- Aromatic: W, F, Y, H (purple)
- Nonpolar aliphatic: A, V, I, L, M, G, P (green)
- Polar: S, T, N, Q, C (yellow)
- Negatively charged: D, E (blue)
- Positively charged: K, R (red)

## Background From Real / Synthetic Control (Without motif_pwms.txt.gz)

```python
from mir.biomarkers.motif_logo import get_vj_background_from_control
from mir.common.control import ControlManager

cm = ControlManager()
ctrl_real  = cm.load_control_df("real",      "human", "TRB")
ctrl_synth = cm.load_control_df("synthetic", "human", "TRB", n=100_000)

bg_real  = get_vj_background_from_control(ctrl_real,  v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=100)
bg_synth = get_vj_background_from_control(ctrl_synth, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=20)
```

Background correlations (158 matched VJ/len combos, 43,580 frequency pairs):
- motif_pwms vs real control: R = 0.97
- motif_pwms vs synthetic 100K: R = 0.96
- Real vs synthetic: R = 0.98

Use `min_seqs=20` for synthetic controls; `min_seqs=100` for real repertoires.

## Mixed-Length Terminal-Anchored Logos

```python
from mir.biomarkers.motif_logo import build_terminal_anchored_logo
import polars as pl

seqs_pl = pl.from_pandas(hits_df[["junction_aa", "v_call", "j_call"]])
logo_anchored = build_terminal_anchored_logo(
    seqs_pl, pwms, n_term=8, c_term=8, species="HomoSapiens", gene="TRB",
)
# Returns pl.DataFrame[pos_label, aa, count, frequency, ic_height, bg_height]
# pos_label: "1","2",…,"n_term","gap","-c_term",…,"-1"
```

Background subtraction happens in the original linear CDR3 coordinate space per length, THEN positions are mapped to the terminal display — required for valid h_sel.

## De-Novo Motif Discovery: Per-VJ-len Connected Components

When running `alice_hit_clusters` for motif discovery, **always build CCs per VJ/len group separately**.

```python
from mir.biomarkers.alice import alice_hit_clusters

# CORRECT: build CCs per (v_call, j_call, length) group
for v, j, L in top_vj_len_groups:
    sub = hits_df[(hits_df.v_call.str.startswith(v)) &
                  (hits_df.j_call.str.startswith(j)) &
                  (hits_df.junction_aa.str.len() == L)]
    clustered = alice_hit_clusters(sub)
    cc_sizes  = clustered.groupby("cluster_id").size().sort_values(ascending=False)
    top_seqs  = clustered[clustered.cluster_id == cc_sizes.index[0]]

# WRONG: calling on all sequences creates one giant mixed CC
# alice_hit_clusters(all_hits_df)  ← never do this for motif discovery
```

## Background Pool Size

≥ 1,000 OLGA sequences per VJ/length gives stable background frequencies (MAD < 0.002);
`motif_pwms.txt.gz` uses ~23,000 per combination (well above threshold for all cases).

## Important Cluster IDs For Reference

| Cluster | Epitope | V | J | len | csz |
|---------|---------|---|---|-----|-----|
| H.B.GILGFVFTL.1 | GILGFVFTL (InfluenzaA, HLA-A*02) | TRBV19*01 | TRBJ2-7*01 | 13 | 896 |
| H.B.GILGFVFTL.4 | GILGFVFTL | TRBV19*01 | TRBJ1-5*01 | 13 | 129 |

The B27 AS CASSVGL[YF]STDTQYF motif is NOT in motif_pwms — use VDJdb TRBV9/TRBJ2-3/len=15 sequences.
