"""Fast COVID-19 biomarker association scan.

Reads only the cdr3aa column from each AIRR gzip file (no full repertoire
objects). Builds binary donor×CDR3 presence, finds public clonotypes (>=5%
donors), runs Fisher test per CDR3, saves results as parquet.

Runtime: ~2-3 min for 1137 paired donors.
Outputs:
  tmp/fisher_trb.parquet
  tmp/fisher_tra.parquet
"""
import gzip, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

REPO      = pathlib.Path(__file__).parents[1]
DATA_ROOT = REPO / "notebooks" / "assets" / "large" / "airr_covid19"
MIN_CDR3_LEN  = 7
MIN_FRACTION  = 0.05

# ── Metadata ─────────────────────────────────────────────────────────────────
meta = pd.read_csv(DATA_ROOT / "metadata.tsv", sep="\t", dtype={"donor_id": "string"})
meta_filt = meta[meta["COVID_status"].isin(["COVID", "healthy"])].copy()
meta_filt["locus_upper"] = meta_filt["locus"].str.upper()

trb_meta = meta_filt[meta_filt["locus_upper"] == "TRB"].set_index("donor_id")
tra_meta = meta_filt[meta_filt["locus_upper"] == "TRA"].set_index("donor_id")
paired_donors = sorted(set(trb_meta.index) & set(tra_meta.index))
print(f"paired_donors={len(paired_donors)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
_STOP = frozenset("*_X")

def _read_cdr3_set(path: pathlib.Path, locus: str) -> frozenset[str]:
    """Read cdr3aa column from AIRR gzip TSV; return functional CDR3 set."""
    try:
        df = pd.read_csv(path, sep="\t", usecols=["cdr3aa"], dtype=str)
    except Exception:
        return frozenset()
    col = df["cdr3aa"].dropna()
    return frozenset(
        x for x in col
        if len(x) >= MIN_CDR3_LEN
        and not any(c in x for c in _STOP)
    )

def _scan_locus(locus_meta: pd.DataFrame, locus: str) -> pd.DataFrame:
    """Build per-CDR3 Fisher test for one locus."""
    covid_cnt  : Counter[str] = Counter()
    healthy_cnt: Counter[str] = Counter()
    n_covid_tot   = 0
    n_healthy_tot = 0
    t0 = time.perf_counter()

    for i, donor_id in enumerate(paired_donors):
        if donor_id not in locus_meta.index:
            continue
        row = locus_meta.loc[donor_id]
        status = str(row["COVID_status"])
        path = DATA_ROOT / str(row["file_name"])
        cdr3s = _read_cdr3_set(path, locus)
        if status == "COVID":
            covid_cnt.update(cdr3s)
            n_covid_tot += 1
        elif status == "healthy":
            healthy_cnt.update(cdr3s)
            n_healthy_tot += 1

        if (i + 1) % 200 == 0:
            print(f"  {locus} [{i+1}/{len(paired_donors)}]  {time.perf_counter()-t0:.1f}s", flush=True)

    all_cdr3s = set(covid_cnt) | set(healthy_cnt)
    threshold  = int(len(paired_donors) * MIN_FRACTION + 0.9999)

    # public = present in ≥ threshold donors (across both groups)
    total_cnt = Counter()
    total_cnt.update(covid_cnt)
    total_cnt.update(healthy_cnt)
    public = [c for c, n in total_cnt.items() if n >= threshold]
    print(f"  {locus}: {len(all_cdr3s)} distinct CDR3s → {len(public)} public (≥{threshold} donors)")

    rows = []
    eps = 0.5
    for cdr3 in public:
        nc = covid_cnt[cdr3]
        nh = healthy_cnt[cdr3]
        _, p = fisher_exact([[nc, nh], [n_covid_tot - nc, n_healthy_tot - nh]],
                            alternative="two-sided")
        fc = (nc + eps) / (n_covid_tot + 2 * eps)
        fh = (nh + eps) / (n_healthy_tot + 2 * eps)
        fe = fc / fh
        rows.append({
            "junction_aa"   : cdr3,
            "locus"         : locus,
            "n_covid_hit"   : nc,
            "n_healthy_hit" : nh,
            "n_covid_total" : n_covid_tot,
            "n_healthy_total": n_healthy_tot,
            "freq_covid"    : fc,
            "freq_healthy"  : fh,
            "fold_enrichment": fe,
            "log2_fe"       : float(np.log2(max(fe, 1e-6))),
            "p_value"       : p,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    _, padj, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["p_value_adj"]    = padj
    df["neg_log10_padj"] = -np.log10(np.clip(padj, 1e-300, None))
    df = df.sort_values(["p_value_adj", "p_value"]).reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    sig = int((df["p_value_adj"] < 0.05).sum())
    print(f"  {locus}: done in {elapsed:.1f}s | q<0.05: {sig}")
    return df

# ── Run ───────────────────────────────────────────────────────────────────────
print("\n=== TRB ===")
df_trb = _scan_locus(trb_meta, "TRB")
out_trb = REPO / "tmp" / "fisher_trb.parquet"
df_trb.to_parquet(out_trb, index=False)
print(f"Saved {out_trb} ({len(df_trb)} rows)")

print("\n=== TRA ===")
df_tra = _scan_locus(tra_meta, "TRA")
out_tra = REPO / "tmp" / "fisher_tra.parquet"
df_tra.to_parquet(out_tra, index=False)
print(f"Saved {out_tra} ({len(df_tra)} rows)")
