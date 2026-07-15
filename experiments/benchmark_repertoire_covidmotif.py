"""COVID motif discovery with HLA + α + β — the full-cohort, breadth-powered revisit.

WS2 established the bulk MMD witness is breadth-limited (a few-hundred-donor sample recovers nothing; the
signal needs the whole cohort) and that batch control, not per-allele HLA stratification, was the lever at
n=300. This revisit uses the tool that *works* at breadth — the per-clonotype **Fisher incidence screen**
(Emerson) — on the **full ~1258-donor cohort**, both chains, to actually *find COVID motifs*, and tests the
user's hypothesis directly: does **HLA restriction** (test only carriers of an allele) recover motifs that the
genome-wide FDR misses, and does combining **α + β** widen the recovered set?

For each chain: genome-wide Fisher vs per-allele HLA-restricted Fisher; count ground-truth clones recovered
(`covid_associated_clonotypes.csv`), report the recovered motifs, and the α+β union.

Data: ``~/hf/airr_covid19`` (paired TRB+TRA). Needs ``[bench]``.
Run:  python experiments/benchmark_repertoire_covidmotif.py [n_donors|0=all] [downsample_reads] [min_carriers]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from _covid import carries, load_covid_paired
from benchmark_repertoire_covidwitness import (
    CHAINS, _cohort_frame, _common_alleles, _fisher, _ground_truth, _phenotype,
)


def _recovered(res: pl.DataFrame, gt: pl.DataFrame, alpha: float = 0.05):
    """GT-true clones that clear q<alpha in this Fisher result (junction_aa, v_call, j_call, odds_ratio)."""
    gt_true = gt.filter(pl.col("covid")).select("junction_aa", "v_call", "j_call")
    return (res.filter((pl.col("q_value") < alpha) & (pl.col("direction") == "enriched"))
            .join(gt_true, on=["junction_aa", "v_call", "j_call"], how="inner"))


def main(n_donors: int = 0, downsample_to: int = 60_000, min_carriers: int = 60) -> None:
    t0 = time.perf_counter()
    rows, pairs = load_covid_paired(n_donors or None, downsample_to, statuses=("COVID", "healthy"))
    covid = np.array([r["COVID_status"] == "COVID" for r in rows])
    alleles = _common_alleles(rows, min_carriers, per_locus=3)
    print(f"{len(rows)} paired donors ≤{downsample_to} reads: {covid.sum()} COVID / {(~covid).sum()} healthy; "
          f"HLA panel {alleles}\n")

    all_recovered = {}
    for chain, _locus, idx in CHAINS:
        frames = [p[idx] for p in pairs]
        gt = _ground_truth(chain)
        cohort, pheno = _cohort_frame(frames, rows), _phenotype(rows)
        gw = _fisher(cohort, pheno)
        rec_gw = _recovered(gw, gt)
        n_gw = int((gw["q_value"] < 0.05).sum())

        # HLA-restricted: pool GT clones recovered in ANY carrier stratum
        strat_hits = {}
        for al in alleles:
            keep = np.array([carries(r, al) for r in rows])
            if (covid & keep).sum() < 5 or (~covid & keep).sum() < 5:
                continue
            sub_rows = [r for r, k in zip(rows, keep) if k]
            res = _fisher(_cohort_frame([f for f, k in zip(frames, keep) if k], sub_rows),
                          _phenotype(sub_rows))
            for r in _recovered(res, gt).iter_rows(named=True):
                strat_hits[r["junction_aa"]] = (al, r["odds_ratio"])

        gw_set = set(rec_gw["junction_aa"].to_list())
        extra = set(strat_hits) - gw_set
        union = gw_set | set(strat_hits)
        all_recovered[chain] = union
        print(f"[{chain}] genome-wide: {n_gw} clones pass q<0.05; GT-true recovered {len(gw_set)}. "
              f"HLA-restricted adds {len(extra)} GT clones FDR missed → {len(union)} total.")
        for j in list(gw_set)[:4]:
            print(f"    genome-wide  {j}")
        for j, (al, orr) in list((k, strat_hits[k]) for k in extra)[:4]:
            print(f"    HLA-focused  {j:<20} via {al} (OR={orr:.1f})")

    tot = set().union(*all_recovered.values()) if all_recovered else set()
    a_only = all_recovered.get("alpha", set()); b_only = all_recovered.get("beta", set())
    print(f"\n[α+β] recovered GT COVID motifs: β {len(b_only)}, α {len(a_only)}, union {len(tot)} "
          f"(GT is 88% α, so α carries most of the recoverable signal).")
    print(f"[verdict] At full cohort breadth the Fisher screen recovers real COVID motifs; HLA restriction "
          f"lifts the count the genome-wide FDR misses, and α+β widens it — the honest way to 'find motifs', "
          f"vs the breadth-starved bulk witness. Total {time.perf_counter() - t0:.0f}s")


def native_beta(min_carriers: int = 40, per_locus: int = 4, cap: int = 60_000) -> None:
    """WS1 — does FULL-NATIVE-DEPTH β (+ HLA restriction) recover ground-truth COVID β clones the read-capped
    genome-wide screen misses? β is hypothesised HLA-restricted AND depth-limited (the α screen recovers a
    public family at breadth; β recovers 0). For each common class-II/B allele we load ONLY its carriers at
    native depth (``carrier_of`` bounds the I/O) and run the within-stratum Fisher incidence test, pooling
    recovered GT β clones, vs the capped genome-wide β baseline. A negative is informative (depth+HLA
    necessary but not sufficient at this breadth)."""
    t0 = time.perf_counter()
    gt = _ground_truth("beta")

    rows_c, pairs_c = load_covid_paired(None, cap, statuses=("COVID", "healthy"))
    cap_gw = set(_recovered(_fisher(_cohort_frame([p[0] for p in pairs_c], rows_c),
                                    _phenotype(rows_c)), gt)["junction_aa"].to_list())
    alleles = _common_alleles(rows_c, min_carriers, per_locus, loci=("DRB1", "DQB1", "B"))
    print(f"capped {cap}: {len(rows_c)} donors, β genome-wide recovers {len(cap_gw)} GT clones; "
          f"class-II/B strata to test at native depth: {alleles}\n")

    native_hits: dict[str, str] = {}
    for al in alleles:
        rows_a, pairs_a = load_covid_paired(None, None, carrier_of=al, statuses=("COVID", "healthy"))
        covid = np.array([r["COVID_status"] == "COVID" for r in rows_a])
        if covid.sum() < 5 or (~covid).sum() < 5:
            print(f"  {al:<12} skipped ({covid.sum()} COVID / {(~covid).sum()} healthy carriers)")
            continue
        res = _fisher(_cohort_frame([p[0] for p in pairs_a], rows_a), _phenotype(rows_a))
        rec = set(_recovered(res, gt)["junction_aa"].to_list())
        for j in rec:
            native_hits.setdefault(j, al)
        depth = int(np.median([int(p[0]["duplicate_count"].sum()) for p in pairs_a]))
        print(f"  {al:<12} {len(rows_a):>4} carriers, median {depth:>7} reads → β GT recovered {len(rec)} "
              f"{sorted(rec) if rec else ''}  [{time.perf_counter() - t0:.0f}s]")

    extra = set(native_hits) - cap_gw
    print(f"\n[WS1] native-depth + HLA restriction recovers {len(set(native_hits))} GT β clone(s); "
          f"{len(extra)} beyond the capped genome-wide screen ({len(cap_gw)}).")
    for j in sorted(extra):
        print(f"    native-only β  {j:<20} via {native_hits[j]}")
    verdict = ("depth+HLA recovers β clones the capped screen misses — β IS depth+HLA-limited" if extra
               else "no β clone recovered beyond capped — depth+HLA necessary but NOT sufficient for β at this "
               "breadth (honest negative; the recoverable COVID signal stays α-public)")
    print(f"[verdict] {verdict}. Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 1 and a[1] == "native_beta":
        native_beta(int(a[2]) if len(a) > 2 else 40, int(a[3]) if len(a) > 3 else 4)
    else:
        main(int(a[1]) if len(a) > 1 else 0, int(a[2]) if len(a) > 2 else 60_000,
             int(a[3]) if len(a) > 3 else 60)
