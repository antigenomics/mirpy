API reference
=============

Every public module is documented below, grouped by subpackage. See the :doc:`user guide <usage>` for runnable examples.

mir
---

``mir``
~~~~~~~

Package root: version, `get_resource_path`, re-exported `TCREmp` / `PairedTCREmp`.

.. automodule:: mir
   :members:
   :undoc-members:
   :show-inheritance:

Clonotype embedding (``mir.embedding``)
---------------------------------------

``mir.embedding.prototypes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bundled prototype loader + manifest.

.. automodule:: mir.embedding.prototypes
   :members:
   :undoc-members:
   :show-inheritance:

``mir.embedding.tcremp``
~~~~~~~~~~~~~~~~~~~~~~~~

`TCREmp` / `PairedTCREmp` — the prototype (TCREMP) embedding.

.. automodule:: mir.embedding.tcremp
   :members:
   :undoc-members:
   :show-inheritance:

``mir.embedding.pca``
~~~~~~~~~~~~~~~~~~~~~

PCA denoising / de-redundancy of embeddings.

.. automodule:: mir.embedding.pca
   :members:
   :undoc-members:
   :show-inheritance:

``mir.embedding.presets``
~~~~~~~~~~~~~~~~~~~~~~~~~

Per-chain recommended prototype counts + PCA dims.

.. automodule:: mir.embedding.presets
   :members:
   :undoc-members:
   :show-inheritance:

Distances (``mir.distances``)
-----------------------------

``mir.distances.junction``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Junction distance via `seqtree.gapblock` (metric / matrix / alignment knobs).

.. automodule:: mir.distances.junction
   :members:
   :undoc-members:
   :show-inheritance:

``mir.distances.germline``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Resource-backed V / J / CDR1 / CDR2 germline distance lookup.

.. automodule:: mir.distances.germline
   :members:
   :undoc-members:
   :show-inheritance:

Density and background subtraction (``mir.density``)
----------------------------------------------------

Graph-free continuous-density TCRNET/ALICE neighbour enrichment (Theory T6).

.. automodule:: mir.density
   :members:
   :undoc-members:
   :show-inheritance:

Sample-level (repertoire) embedding (``mir.repertoire``)
--------------------------------------------------------

One vector per repertoire: RFF kernel mean ‖ Hill diversity ‖ second moment; MMD (Theory T7).
Optionally a **sub-probability** measure: ``missing_mass`` (Good-Turing / bias-corrected Chao1),
``naive_reference`` (the germline location of the unseen) and ``contrast_embedding``
(``Ψ = mass·(Φ − naive)``, signed and magnitude-carrying). Also the measure algebra around it:
``rao_q`` (Rao's quadratic entropy = ``1 − ‖Φ₁‖²``), ``depth_threshold`` (the ``κ`` below which ``Φ``
is mostly sampling noise), ``sample_statistics`` / ``cohort_statistics``, the compartment
decomposition ``band_frames`` / ``band_embeddings`` / ``mixture_weights``, and ``rarefy_embedding``.
Derivations: :doc:`math`.

.. automodule:: mir.repertoire
   :members:
   :undoc-members:
   :show-inheritance:

Explainable readouts (``mir.explain``)
--------------------------------------

Which named channel of ``Φ`` carries the signal, and which clonotypes drive it (Theory T7).
``ChannelBuilder.add(..., preserve_magnitude=True)`` scales a magnitude-carrying (sub-probability)
block by one global scalar instead of per-column z-scoring.

.. automodule:: mir.explain
   :members:
   :undoc-members:
   :show-inheritance:

Cohort / digital donor (``mir.cohort``)
---------------------------------------

Fuse per-chain repertoire embeddings into one hash-verified, serialisable donor matrix; batch
residualisation (with optional James–Stein shrinkage), depth and missingness diagnostics, sample
clustering, and incidence biomarkers (Theory T7).

.. automodule:: mir.cohort
   :members:
   :undoc-members:
   :show-inheritance:

Exposure trajectory (``mir.track``)
-----------------------------------

A PhenoPath-style (Campbell & Yau 2018) covariate-disentangled latent trajectory over a per-sample
channel matrix — repertoire-level exposure detection, complementing ``mir.density``'s clone-level
TCRNET/ALICE enrichment.

.. automodule:: mir.track
   :members:
   :undoc-members:
   :show-inheritance:

Generative loop (``mir.generate``)
----------------------------------

A fitted (optionally class-conditional) density over ``RepertoireDescriptor`` vectors: sample new
synthetic donor states, or evolve one along a coordinate via the fitted covariance's conditional
mean — the mechanical half of the generative loop (ROADMAP Phase 2).

.. automodule:: mir.generate
   :members:
   :undoc-members:
   :show-inheritance:

Digital twin (``mir.twin``)
---------------------------

One donor's perturbable, simulatable state — glues ``RepertoireDescriptor`` + an optional
``mir.track`` trajectory position + a ``mir.generate``/``mir.ml.diffusion`` generator into one object.

.. automodule:: mir.twin
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark harness (``mir.bench``)
---------------------------------

``mir.bench.vdjdb``
~~~~~~~~~~~~~~~~~~~

VDJdb loader + antigen subsets.

.. automodule:: mir.bench.vdjdb
   :members:
   :undoc-members:
   :show-inheritance:

``mir.bench.metrics``
~~~~~~~~~~~~~~~~~~~~~

Clustering (DBSCAN/HDBSCAN/OPTICS) + F1 / retention / purity.

.. automodule:: mir.bench.metrics
   :members:
   :undoc-members:
   :show-inheritance:

``mir.bench.theory``
~~~~~~~~~~~~~~~~~~~~

Reproduced supplementary theory (S1–S3, T5–T6, codec losslessness).

.. automodule:: mir.bench.theory
   :members:
   :undoc-members:
   :show-inheritance:

``mir.bench.eval``
~~~~~~~~~~~~~~~~~~

Scorers for the explainable readout: cross-validated AUC, Cox C-index, log-rank — plus
``recovery_report``, the grouped-CV ridge that asks whether each basic repertoire statistic is
carried *inside* the embedding.

.. automodule:: mir.bench.eval
   :members:
   :undoc-members:
   :show-inheritance:

Neural codecs and learned encoders (``mir.ml``)
-----------------------------------------------

``mir.ml.tokenize``
~~~~~~~~~~~~~~~~~~~

CDR3 tokenisation for the neural codecs.

.. automodule:: mir.ml.tokenize
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.encoder``
~~~~~~~~~~~~~~~~~~

Forward encoder: sequence → compact embedding code.

.. automodule:: mir.ml.encoder
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.decoder``
~~~~~~~~~~~~~~~~~~

Inverse decoder: code → sequence.

.. automodule:: mir.ml.decoder
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.train``
~~~~~~~~~~~~~~~~

Training loops + device selection (CUDA → MPS → CPU).

.. automodule:: mir.ml.train
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.codec``
~~~~~~~~~~~~~~~~

Unified encoder+decoder codec with a geometry-anchor term.

.. automodule:: mir.ml.codec
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.bundle``
~~~~~~~~~~~~~~~~~

`CodecBundle` — prototype-hash-verified shipping of a trained codec.

.. automodule:: mir.ml.bundle
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.set_encoder``
~~~~~~~~~~~~~~~~~~~~~~

Learned repertoire track (Set-Transformer / DeepRC attention pooling).

.. automodule:: mir.ml.set_encoder
   :members:
   :undoc-members:
   :show-inheritance:

``mir.ml.diffusion``
~~~~~~~~~~~~~~~~~~~~

Conditional diffusion generator (DDPM/DDIM + classifier-free guidance) over a compact descriptor/code
space — the research half of the generative loop, complementing ``mir.generate``'s linear Gaussian.

.. automodule:: mir.ml.diffusion
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

``mir.aliases``
~~~~~~~~~~~~~~~

Species / locus aliases.

.. automodule:: mir.aliases
   :members:
   :undoc-members:
   :show-inheritance:

``mir.alleles``
~~~~~~~~~~~~~~~

Allele normalisation with default-allele cascade.

.. automodule:: mir.alleles
   :members:
   :undoc-members:
   :show-inheritance:

Command-line interface (``mir``)
--------------------------------

``pip install mirpy-lib`` installs a ``mir`` console script (also ``python -m mir.cli``) with two
commands, one per embedding scale:

Both commands drop non-coding clonotypes (stop codon / legacy out-of-frame ``junction_aa``) before
embedding by default via ``vdjtools.preprocess.filter_functional`` — pass ``--no-filter-functional``
to disable.

``mir embed clonotypes INPUT``
   One repertoire's clonotype table → a per-clonotype TCREMP embedding table (columns ``e0…``).
   Flags: ``--species``, ``--locus`` (inferred when the file has one locus), ``--n-prototypes``,
   ``--mode {vjcdr3,cdr123}``, ``--replicate R`` (prototype draw), ``--pca K`` (compact the table),
   ``--filter-functional``/``--no-filter-functional``, ``--threads``, ``-o`` (``.tsv`` /
   ``.parquet``; default stdout TSV).

``mir embed repertoires INPUT...``
   A dataset of clonotype tables → one repertoire vector ``Φ(S)`` per sample **per chain** on one
   shared basis (columns ``phi0…``; sample id = filename stem). Flags: ``--species``, ``--locus``
   (restrict), ``--n-prototypes``, ``--replicate R``,
   ``--weight {log2p1,duplicate_count,distinct,log1p,anscombe}`` (default ``log2p1``),
   ``--blocks mean,diversity[,second]``, ``--n-rff``, ``--n-rff-second``, ``--n-components``,
   ``--mmd OUT`` (also write the per-chain pairwise unbiased-MMD matrix),
   ``--filter-functional``/``--no-filter-functional``, ``--threads``, ``--seed``, ``-o``.

.. automodule:: mir.cli
   :members: main, build_parser
   :show-inheritance:

