"""Exercise the installed Windows wheel through batched receptor embedding and PCA."""
import subprocess

import mir
import numpy as np
import polars as pl
from mir.embedding.tcremp import TCREmp
from vdjtools.cores import available_cores, physical_cores
from vdjtools.signature.corpus import TablePCA

subprocess.run(["mir", "--help"], check=True, capture_output=True, text=True)
assert 1 <= physical_cores() <= available_cores()
sample = pl.DataFrame({
    "v_call": ["TRBV10-3*01", "TRBV20-1*01", "TRBV10-3*01"],
    "j_call": ["TRBJ2-7*01", "TRBJ1-2*01", "TRBJ2-7*01"],
    "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF", "CASSIRSSYEQFF"],
})
model = TCREmp.from_defaults("human", "TRB", n_prototypes=8, threads=0)
coordinates = model.embed(sample)
assert coordinates.shape == (3, 24) and np.isfinite(coordinates).all()
table = pl.from_numpy(coordinates, schema=[f"phi:{i}" for i in range(24)]).with_columns(
    pl.lit("windows-smoke").alias("embedding_space"))
rotation = TablePCA.fit(table, n_components=2)
assert rotation.transform(table).height == 3
print("Installed wheel batched embedding and sample PCA passed")
