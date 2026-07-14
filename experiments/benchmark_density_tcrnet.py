"""TCRNET-vs-ALICE benchmark: continuous-density enrichment (mir.density) run with the two
classical backgrounds recovers concordant, VDJdb-annotated antigen-specific clusters.

Same enrichment test, two backgrounds (appendix §T.6):
* **TCRNET** — a real control repertoire, binomial test.
* **ALICE**  — a generated vdjtools P_gen background, Poisson test.

Gate: the two enriched-clone sets are strongly concordant (Jaccard overlap far above chance) —
the classical result that ALICE is a special case of TCRNET. VDJdb-annotation enrichment is
reported too but is a weak gate here (VDJdb covers only a few percent of repertoire clones).

Data: isalgo/airr_benchmark tcrnet/{CMV+,control}.txt.gz + bundled vdjdb.slim. Cached (needs [bench]).
Run:  python experiments/benchmark_density_tcrnet.py [CMV+|B35+] [cap] [weight]
      (weight distinct|log1p|anscombe folds in clone sizes; both backgrounds then share the
      compound-Poisson Gamma test — the abundance channel, §T.6 sec:dens-abund)
"""

from __future__ import annotations

import sys
import time

import polars as pl

from _hf import fetch, load_repertoire

from mir.bench.vdjdb import load_vdjdb
from mir.density import enriched_mask, fit_density_space, generate_background, neighbor_enrichment
from mir.embedding.tcremp import TCREmp

REPO = "isalgo/airr_benchmark"
N_PROTO = 1000
N_COMPONENTS = 20
BG_MULT = 5


def _hits(model, obs: pl.DataFrame, bg: pl.DataFrame, test: str, weight: str = "distinct") -> set[str]:
    space, obs_emb, bg_emb = fit_density_space(
        model, obs, bg, n_components=N_COMPONENTS, space="full", pca_fit_cap=40000)
    abund = obs["duplicate_count"].to_numpy() if weight != "distinct" else None
    # NB weight!=distinct uses the compound-Poisson Gamma tail (both backgrounds), so the
    # poisson/binomial distinction folds into the shared abundance test — the flag is opt-in.
    res = neighbor_enrichment(obs_emb, bg_emb, lambda0=3.0, test=test, abundance=abund, weight=weight)
    return set(obs.filter(enriched_mask(res))["junction_aa"].to_list())


def _annotated_fraction(junctions, vdjdb_set) -> float:
    j = list(junctions)
    return sum(x in vdjdb_set for x in j) / len(j) if j else 0.0


def main(sample: str = "CMV+", cap: int = 40000, weight: str = "distinct") -> None:
    t0 = time.perf_counter()
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)

    obs = load_repertoire(fetch(REPO, f"tcrnet/{sample}.txt.gz"))
    if obs.height > cap:
        obs = obs.sample(cap, seed=0)
    control = load_repertoire(fetch(REPO, "tcrnet/control.txt.gz"))
    if control.height > BG_MULT * obs.height:
        control = control.sample(BG_MULT * obs.height, seed=0)
    generated = generate_background("TRB", BG_MULT * obs.height, seed=0)
    vdjdb = set(load_vdjdb("tests/assets/vdjdb.slim.txt.gz")
                .filter(pl.col("locus") == "TRB")["junction_aa"].to_list())

    tcrnet_hits = _hits(model, obs, control, test="binomial", weight=weight)   # TCRNET: control bg
    alice_hits = _hits(model, obs, generated, test="poisson", weight=weight)   # ALICE: P_gen bg

    inter = len(tcrnet_hits & alice_hits)
    union = len(tcrnet_hits | alice_hits)
    jaccard = inter / union if union else 0.0
    base = _annotated_fraction(obs["junction_aa"].to_list(), vdjdb)

    print(f"sample {sample}, {obs.height} clones, control {control.height}, "
          f"generated {generated.height}, {len(vdjdb)} VDJdb-TRB junctions")
    print(f"\n{'method':<10}{'background':<12}{'hits':>7}{'VDJdb%':>9}{'lift_vs_base':>14}")
    lifts = {}
    for name, bgname, hits in (("TCRNET", "control", tcrnet_hits), ("ALICE", "P_gen", alice_hits)):
        frac = _annotated_fraction(hits, vdjdb)
        lift = frac / base if base else float("nan")
        lifts[name] = lift
        print(f"{name:<10}{bgname:<12}{len(hits):>7}{frac:>8.1%}{lift:>13.2f}x")
    print(f"\nbaseline VDJdb-annotated fraction: {base:.2%}  (VDJdb lift: TCRNET {lifts['TCRNET']:.2f}x, "
          f"ALICE {lifts['ALICE']:.2f}x — weak: sparse VDJdb coverage)")
    # chance Jaccard for two independent sets of these sizes, as a reference point
    r_t, r_a = len(tcrnet_hits) / obs.height, len(alice_hits) / obs.height
    chance = (r_t * r_a) / (r_t + r_a - r_t * r_a) if (r_t + r_a) else 0.0
    print(f"TCRNET∩ALICE Jaccard: {jaccard:.2f}  ({inter} shared / {union} union; chance ≈ {chance:.2f})")

    # gate: the real-control (TCRNET) and generated-P_gen (ALICE) backgrounds agree on which
    # clones are enriched, far above chance — ALICE is a special case of TCRNET.
    verdict = "PASS" if (jaccard > 0.5 and jaccard > 2 * chance) else "FAIL"
    print(f"\n[{verdict}] TCRNET≈ALICE concordance: Jaccard {jaccard:.2f} > 0.5 and ≫ chance {chance:.2f}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(args[1] if len(args) > 1 else "CMV+", int(args[2]) if len(args) > 2 else 40000,
         args[3] if len(args) > 3 else "distinct")
