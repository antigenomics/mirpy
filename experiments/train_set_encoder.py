"""Train the learned repertoire set encoder (mir.ml.set_encoder) on the aging cohort (age regression).

Demonstrates the co-equal learned track end-to-end: fit ONE RepertoireSpace on the pooled cloud,
turn each donor into a ``(Z, w)`` clonotype cloud, train the attention set-pooling network with
subsample augmentation (depth-robustness), and ship a :class:`SetEncoderBundle`. Writes config +
metrics + bundle into a fresh run-id'd directory (never overwriting), per the ml-experiments rules.

Data: HF isalgo/airr_benchmark (aging). Cached on first run (needs [bench] + [ml]).
Run:  python experiments/train_set_encoder.py [n_donors] [downsample_reads] [epochs]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from _cohort import load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.ml.set_encoder import SetEncoderBundle, train_set_encoder
from mir.repertoire import fit_repertoire_space

REPO, META = "isalgo/airr_benchmark", "vdjtools/metadata_aging.txt"
PREFIX, SUFFIX = "vdjtools/", ".gz"
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048
RUNS = Path(__file__).parent / "runs"


def main(n_donors: int = 0, downsample_to: int = 20_000, epochs: int = 60) -> None:
    t0 = time.perf_counter()
    seed = 0
    _, samples = load_cohort(REPO, META, prefix=PREFIX, suffix=SUFFIX,
                             downsample_to=downsample_to, cap_samples=n_donors or None)
    ages = np.array([float(r["age"]) for r, _ in samples])

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=seed)
    clouds = [space.sample_cloud(df) for _, df in samples]

    print(f"training set encoder on {len(samples)} donors (age), ≤{downsample_to} reads each")
    sem, metrics = train_set_encoder(clouds, ages, task="regression", epochs=epochs, seed=seed)

    run = RUNS / f"set_encoder_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    config = {"cohort": "aging", "n_donors": len(samples), "downsample_to": downsample_to,
              "epochs": epochs, "n_prototypes": N_PROTO, "n_components": N_COMPONENTS,
              "n_rff": N_RFF, "seed": seed, "task": "regression"}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    SetEncoderBundle.from_model(sem, space, task="regression").save(run / "bundle.pt")

    print(f"[done] held-out |Spearman(age)|={metrics['val_score']:.3f}  ->  {run}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 0,
         int(args[2]) if len(args) > 2 else 20_000,
         int(args[3]) if len(args) > 3 else 60)
