User guide
==========

Runnable examples for each part of ``mir``. All embeddings operate on ``polars`` frames keyed by the
AIRR column names ``v_call`` / ``j_call`` / ``junction_aa`` (and ``duplicate_count`` for clone sizes).

Install
-------

.. code-block:: bash

   pip install mirpy-lib            # core: import mir
   pip install "mirpy-lib[ml]"      # + neural codecs (torch)
   pip install "mirpy-lib[bench]"   # + benchmark harness (huggingface_hub, pynndescent, ...)

Requires ``vdjtools>=2.3.0`` and ``seqtree>=0.3.0``; ``mir`` itself is a pure-Python
``py3-none-any`` wheel.

Clonotype embedding
-------------------

``TCREmp`` maps each clonotype to a fixed vector — the concatenation of its distances to a set of
prototype clonotypes, per component (V, J, junction). Distance in this space approximates the
pairwise alignment distance (Theory T1).

.. code-block:: python

   from mir.embedding.tcremp import TCREmp

   model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
   X = model.embed(df)                      # (N, 3K) float32, interleaved [v, j, junction]

   # paired chains: a dict of per-locus frames -> concatenated embedding
   from mir.embedding.tcremp import PairedTCREmp
   paired = PairedTCREmp.from_defaults("human", ("TRA", "TRB"))
   Xp = paired.embed({"TRA": tra_df, "TRB": trb_df})

Pick prototype counts / PCA dims from the per-chain presets, and denoise with PCA:

.. code-block:: python

   from mir.embedding.presets import get_preset
   from mir.embedding.pca import pca_denoise

   preset = get_preset("human", "TRB")      # n_prototypes, n_components (95% var), recon dims
   Xd = pca_denoise(X, n_components=preset.n_components)

Density and background subtraction
----------------------------------

``mir.density`` finds antigen-driven convergent clusters by neighbour enrichment in the embedding
space (graph-free TCRNET/ALICE, Theory T6): ``E(z) = f_obs(z) / f_gen(z)`` estimated by an
adaptive-bandwidth balloon estimator with a Poisson/binomial test and BH q-values.

.. code-block:: python

   from mir.density import fit_density_space, neighbor_enrichment, enriched_mask, denoise_and_cluster

   # background = a biological control (TCRNET) or generate_background(...) (ALICE, P_gen)
   space, obs_emb, bg_emb = fit_density_space(model, obs_df, control_df, n_components=20)
   res  = neighbor_enrichment(obs_emb, bg_emb, test="binomial")   # backend="kdtree" for multicore
   hits = obs_df.filter(enriched_mask(res, alpha=0.05))           # background-subtracted clones
   labels, mask = denoise_and_cluster(obs_emb, res)               # noise-filter + cluster the hits

Prefer a biological control (e.g. pre/post-vaccination) over the P_gen background — differential
enrichment cancels generic public convergence and isolates the antigen-specific response.

Sample-level (repertoire) embedding
-----------------------------------

``mir.repertoire`` embeds a whole repertoire — an order-invariant multiset of clonotypes with clone
sizes — into one fixed vector ``Φ(S)`` (kernel mean ‖ Hill diversity ‖ second moment), depth-robust
into the RNA-seq regime (Theory T7). Every sample in a cohort must share one basis.

.. code-block:: python

   from mir.repertoire import (fit_repertoire_space, sample_embedding,
                               mmd_matrix, hla_stratified_mmd, class_witness)
   import polars as pl

   space = fit_repertoire_space(model, pl.concat(samples))    # ONE basis for the cohort
   embs  = [sample_embedding(space, s) for s in samples]      # Φ(S)
   D     = mmd_matrix(embs, unbiased=True)                    # pairwise MMD (unbiased when depth varies)

   # supervised motif finder: public clones separating two groups
   motifs = class_witness(space, pos_samples, neg_samples, candidates)

Use ``unbiased=True`` whenever samples differ in depth/diversity. For a batch-confounded contrast,
compare *within-batch* (residualise ``Φ`` on the batch indicator): a batch offset is first-order and
cancels, while a batch-orthogonal signal (e.g. HLA) survives.

Neural codecs
-------------

The optional ``mir.ml`` tier (``[ml]`` extra, torch) trains fast neural approximations of the
embedding: a forward encoder (sequence → code), an inverse decoder (code → sequence), a Pgen
regressor, and a unified codec; plus a learned repertoire set encoder. Device selection is
automatic (CUDA → MPS → CPU; override with ``device=`` or ``MIR_DEVICE``).

.. code-block:: python

   from mir.ml.bundle import CodecBundle
   bundle = CodecBundle.load("path/to/codec")   # refuses a prototype-hash mismatch
   codes  = bundle.encode(X)

Benchmark harness
-----------------

``mir.bench`` provides the VDJdb clustering benchmark (F1 / retention / purity) with selectable
clustering methods, and the reproduced theory experiments.

.. code-block:: python

   from mir.bench.vdjdb import load_vdjdb
   from mir.bench.metrics import cluster, cluster_metrics

   labels = cluster(X, method="dbscan")         # or "hdbscan" / "optics"
