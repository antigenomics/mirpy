"""Compute VJ-corrected feature matrix X_vj and save to tmp/ for use in notebook."""
import pandas as pd
import numpy as np
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import os, json

DATASET_ROOT = Path("notebooks/assets/large/airr_covid19")
meta = pd.read_csv(DATASET_ROOT / "metadata.tsv", sep="\t", low_memory=False)

# Load reference biomarkers + their VJ genes
ref = pd.read_csv(DATASET_ROOT / "covid_associated_clonotypes.csv")
pos = ref[ref["has_covid_association"] == True]
tra_bms = sorted(pos[pos["chain"] == "alpha"]["cdr3"].astype(str).tolist())
trb_bms = sorted(pos[pos["chain"] == "beta"]["cdr3"].astype(str).tolist())
print(f"Biomarkers: TRA={len(tra_bms)}, TRB={len(trb_bms)}")

# Build paired donor list
MIN_READS = 10_000
tra_meta = meta[(meta["locus"] == "TRA") & (meta["reads"] >= MIN_READS)]
trb_meta = meta[(meta["locus"] == "TRB") & (meta["reads"] >= MIN_READS)]
tra_df = tra_meta.set_index("donor_id")
trb_df = trb_meta.set_index("donor_id")
paired_donors = sorted(set(tra_df.index) & set(trb_df.index))
print(f"Paired donors: {len(paired_donors)}")


def _extract(file_path: str, biomarkers: list):
    """Load biomarker freq vector and VJ usage from a single TSV file."""
    try:
        df = pd.read_csv(
            file_path, sep="\t",
            usecols=["cdr3aa", "freq", "v", "j"],
            compression="gzip", low_memory=False,
        )
        df = df.dropna(subset=["cdr3aa"])
        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        # Biomarker frequencies
        cdr3_freq = df.groupby("cdr3aa")["freq"].sum().to_dict()
        freq_vec = np.array([cdr3_freq.get(b, 0.0) for b in biomarkers], dtype=np.float32)
        # VJ usage
        df_vj = df.dropna(subset=["v", "j"])
        df_vj = df_vj.assign(
            v_b=df_vj["v"].astype(str).str.split("*").str[0],
            j_b=df_vj["j"].astype(str).str.split("*").str[0],
        )
        vj_series = df_vj.groupby(["v_b", "j_b"])["freq"].sum()
        total = vj_series.sum()
        vj_dict = (
            {(v, j): float(f / total) for (v, j), f in vj_series.items()}
            if total > 0 else {}
        )
        return freq_vec, vj_dict
    except Exception as e:
        print(f"  Warning: {Path(file_path).name}: {e}")
        return None, {}


print("Loading all files (freq + VJ in one pass)...")
t0 = time.perf_counter()
n_workers = os.cpu_count() or 4

tra_vecs, trb_vecs, vj_tra, vj_trb = {}, {}, {}, {}
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    ft_tra = {
        executor.submit(_extract, str(DATASET_ROOT / tra_df.loc[d, "file_name"]), tra_bms): d
        for d in paired_donors
    }
    ft_trb = {
        executor.submit(_extract, str(DATASET_ROOT / trb_df.loc[d, "file_name"]), trb_bms): d
        for d in paired_donors
    }
    for f, d in ft_tra.items():
        vec, vj = f.result()
        if vec is not None:
            tra_vecs[d] = vec
            vj_tra[d] = vj
    for f, d in ft_trb.items():
        vec, vj = f.result()
        if vec is not None:
            trb_vecs[d] = vec
            vj_trb[d] = vj

print(f"  done in {time.perf_counter() - t0:.1f}s")

# Assemble X, y
X_rows, y_labels, donor_ids = [], [], []
for d in paired_donors:
    if d not in tra_vecs or d not in trb_vecs:
        continue
    X_rows.append(np.concatenate([tra_vecs[d], trb_vecs[d]]))
    row = tra_df.loc[d]
    y_labels.append(1 if str(row["COVID_status"]) == "COVID" else 0)
    donor_ids.append(d)

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_labels, dtype=np.int32)
print(f"X shape: {X.shape}  (COVID={y.sum()}, healthy={(y==0).sum()})")


def _bm_vj_lookup(bm_df, cdr3_list):
    """Return list of (v_base, j_base) for each CDR3, allele-stripped."""
    lookup = {}
    for _, row in bm_df.iterrows():
        v = str(row.get("v", "")).split("*")[0]
        j = str(row.get("j", "")).split("*")[0]
        lookup[str(row["cdr3"])] = (v, j)
    return [lookup.get(c, ("", "")) for c in cdr3_list]


tra_bm_vj = _bm_vj_lookup(pos[pos["chain"] == "alpha"], tra_bms)
trb_bm_vj = _bm_vj_lookup(pos[pos["chain"] == "beta"], trb_bms)

# Global reference VJ = mean across all donors
agg_tra: dict = defaultdict(list)
agg_trb: dict = defaultdict(list)
for d in donor_ids:
    for vj, p in vj_tra.get(d, {}).items():
        agg_tra[vj].append(p)
    for vj, p in vj_trb.get(d, {}).items():
        agg_trb[vj].append(p)

vj_ref_tra = {vj: float(np.mean(ps)) for vj, ps in agg_tra.items()}
vj_ref_trb = {vj: float(np.mean(ps)) for vj, ps in agg_trb.items()}

# Apply VJ correction
CLAMP = 20.0
X_vj = X.copy()
n_tra = len(tra_bms)

for i, d in enumerate(donor_ids):
    for j, (vj, raw) in enumerate(zip(tra_bm_vj, X[i, :n_tra])):
        if raw == 0.0 or not vj[0]:
            continue
        s_vj = vj_tra.get(d, {}).get(vj, 0.0)
        g_vj = vj_ref_tra.get(vj, s_vj)
        if s_vj > 1e-10:
            X_vj[i, j] = min(raw * g_vj / s_vj, raw * CLAMP)
    for j, (vj, raw) in enumerate(zip(trb_bm_vj, X[i, n_tra:])):
        if raw == 0.0 or not vj[0]:
            continue
        s_vj = vj_trb.get(d, {}).get(vj, 0.0)
        g_vj = vj_ref_trb.get(vj, s_vj)
        if s_vj > 1e-10:
            X_vj[i, n_tra + j] = min(raw * g_vj / s_vj, raw * CLAMP)

n_mod = int((np.abs(X_vj - X) > 1e-9).sum())
print(f"X_vj: {n_mod} values modified ({100 * n_mod / X.size:.1f}% of matrix)")

# Save outputs
os.makedirs("tmp", exist_ok=True)
np.save("tmp/X_vj.npy", X_vj)
np.save("tmp/X_raw.npy", X)
np.save("tmp/y.npy", y)
with open("tmp/donor_ids.json", "w") as fh:
    json.dump(donor_ids, fh)

print("Saved: tmp/X_vj.npy  tmp/X_raw.npy  tmp/y.npy  tmp/donor_ids.json")
print("Done.")
