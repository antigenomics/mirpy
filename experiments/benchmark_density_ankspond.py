"""Ankylosing-spondylitis benchmark: continuous-density enrichment (mir.density) recovers
the public HLA-B27-linked CD8 motif CASSVGL[YF]STDTQYF (TRBV9 / TRBJ2-3) in B27+ patients
and not in B27- controls (Komech et al. 2018/2022).

Gate: the VGLYST motif appears among the enriched hits pooled across B27+ AS CD8 repertoires
and is absent (or far weaker) in the B27- pool — the antigen-driven convergent group that
background subtraction against a vdjtools P_gen model is meant to surface.

Data: isalgo/airr_ankspond (new/ CD8 repertoires). Downloaded + cached (needs [bench]).
Run:  python experiments/benchmark_density_ankspond.py
"""

from __future__ import annotations

import time

import polars as pl

from _hf import fetch, load_repertoire

from mir.density import enriched_mask, fit_density_space, generate_background, neighbor_enrichment
from mir.embedding.tcremp import TCREmp

REPO = "isalgo/airr_ankspond"
# B27+ AS synovial-fluid CD8 (where the motif concentrates) vs the B27- SF CD8 control (Kal);
# see new/metadata.tsv. The full CD8 pools also work but are large — SF CD8 is motif-rich + fast.
B27_POS = ["new/as_Dv_SFCD8.tsv.gz", "new/as_Mikh_SFCD8.tsv.gz", "new/as_Shep_SFCD8.tsv.gz"]
B27_NEG = ["new/as_Kal_SFCD8.tsv.gz"]

# the public B27 motif family: CASS[x]G[L/V][Y/F]STDTQYF on TRBV9 / TRBJ2-3
MOTIF_RE = r"CASS[A-Z]G[LV][YF]STDTQYF"
N_PROTO = 1000
N_COMPONENTS = 20
BG_CAP = 200000


def _pool(files: list[str]) -> pl.DataFrame:
    frames = [load_repertoire(fetch(REPO, f)) for f in files]
    return (pl.concat(frames)
            .group_by(["junction_aa", "v_call", "j_call"])
            .agg(pl.col("duplicate_count").sum()))


def _motif(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col("junction_aa").str.contains(MOTIF_RE)
        & pl.col("v_call").str.contains("TRBV9")
        & pl.col("j_call").str.contains("TRBJ2-3")
    )


def _enriched(model, obs: pl.DataFrame) -> pl.DataFrame:
    bg = generate_background("TRB", min(max(5 * obs.height, 5000), BG_CAP), seed=0)
    space, obs_emb, bg_emb = fit_density_space(
        model, obs, bg, n_components=N_COMPONENTS, space="full", pca_fit_cap=40000)
    res = neighbor_enrichment(obs_emb, bg_emb, lambda0=5.0)
    return obs.with_columns(
        pl.Series("fold", res.fold), pl.Series("qvalue", res.qvalue)
    ).filter(enriched_mask(res))


def main() -> None:
    t0 = time.perf_counter()
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    result = {}
    print(f"{'group':<8}{'clones':>8}{'motif_in':>10}{'hits':>7}{'motif_hits':>12}{'top_rank':>9}")
    for group, files in (("B27+", B27_POS), ("B27-", B27_NEG)):
        obs = _pool(files)
        motif_in = _motif(obs).height
        hits = _enriched(model, obs)
        mhits = _motif(hits)
        # rank of the best motif hit among all hits by fold enrichment (1 = most enriched)
        ranked = hits.sort("fold", descending=True).with_row_index("rank")
        top_rank = _motif(ranked)["rank"].min()
        top_rank = int(top_rank) + 1 if top_rank is not None else 0
        result[group] = mhits.height
        print(f"{group:<8}{obs.height:>8}{motif_in:>10}{hits.height:>7}{mhits.height:>12}{top_rank:>9}")

    verdict = "PASS" if result["B27+"] > result["B27-"] else "FAIL"
    print(f"\n[{verdict}] VGLYST motif enriched hits: B27+={result['B27+']} vs B27-={result['B27-']}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
