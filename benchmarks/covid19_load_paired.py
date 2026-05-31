"""Pre-load all 1137 paired COVID/healthy donors (TRA+TRB) into SampleRepertoire
objects and cache as a pkl for use in covid19_biomarkers.ipynb.

Runtime: ~3-4 min. Output: tmp/covid19_paired_samples.pkl
"""
import pickle, pathlib, time, sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

import pandas as pd
from mir.common.parser import ClonotypeTableParser
from mir.common.filter import filter_functional
from mir.common.repertoire import LocusRepertoire, SampleRepertoire

REPO = pathlib.Path(__file__).parents[1]
DATA_ROOT = REPO / "notebooks" / "assets" / "large" / "airr_covid19"
OUT_PKL   = REPO / "tmp" / "covid19_paired_samples.pkl"

meta = pd.read_csv(DATA_ROOT / "metadata.tsv", sep="\t", dtype={"donor_id": "string"})
meta_filt = meta[meta["COVID_status"].isin(["COVID", "healthy"])].copy()
meta_filt["locus_upper"] = meta_filt["locus"].str.upper()

trb_donors = set(meta_filt[meta_filt["locus_upper"] == "TRB"]["donor_id"])
tra_donors = set(meta_filt[meta_filt["locus_upper"] == "TRA"]["donor_id"])
paired_donors = sorted(trb_donors & tra_donors)
print(f"paired_donors={len(paired_donors)}", flush=True)

cohort_trb = meta_filt[meta_filt["locus_upper"] == "TRB"].set_index("donor_id")
cohort_tra = meta_filt[meta_filt["locus_upper"] == "TRA"].set_index("donor_id")

parser = ClonotypeTableParser()
samples: list[SampleRepertoire] = []
t0 = time.perf_counter()

for i, donor_id in enumerate(paired_donors):
    if donor_id not in cohort_trb.index or donor_id not in cohort_tra.index:
        continue

    trb_row = cohort_trb.loc[donor_id]
    tra_row = cohort_tra.loc[donor_id]
    loci = {}

    for locus_key, row in [("TRB", trb_row), ("TRA", tra_row)]:
        path = DATA_ROOT / str(row["file_name"])
        if not path.exists():
            continue
        clones_raw = parser.parse(str(path))
        clones_filt = [c for c in clones_raw if not c.locus or c.locus == locus_key]
        if not clones_filt:
            continue
        rep = filter_functional(
            LocusRepertoire(clonotypes=clones_filt, locus=locus_key,
                            repertoire_id=str(row["sample_id"]))
        )
        if rep.clonotype_count > 0:
            loci[locus_key] = rep

    if len(loci) < 2:
        continue  # skip donors missing one locus after filtering

    samples.append(SampleRepertoire(
        loci=loci,
        sample_id=str(trb_row["sample_id"]),
        sample_metadata={
            "donor_id": donor_id,
            "COVID_status": str(trb_row["COVID_status"]),
            "batch_id": str(trb_row["batch_id"]),
            "reads": int(trb_row["reads"]),
        },
    ))

    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  [{i+1}/{len(paired_donors)}] {elapsed:.1f}s  samples={len(samples)}", flush=True)

elapsed = time.perf_counter() - t0
n_covid   = sum(1 for s in samples if s.sample_metadata["COVID_status"] == "COVID")
n_healthy = sum(1 for s in samples if s.sample_metadata["COVID_status"] == "healthy")
print(f"\nLoaded {len(samples)} paired samples in {elapsed:.1f}s: COVID={n_covid}, healthy={n_healthy}")

print(f"Saving to {OUT_PKL} ...", flush=True)
with open(OUT_PKL, "wb") as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

size_mb = OUT_PKL.stat().st_size / 1024**2
print(f"Done. File size: {size_mb:.1f} MB")
