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

The portable signature (``mir.signature``)
------------------------------------------

The hand-off object: a fixed-width, name-addressed, already-standardised vector that anyone who
``pip install mirpy-lib`` can compute from their own AIRR files and drop straight into a model.
``Φ(S)`` above is a fingerprint but not a portable one — its basis is fitted on *your* cohort, so
two collaborators get incomparable vectors. The signature fixes the basis and the scale.

Two halves, concatenated on ``sample_id`` and namespaced so they never collide:

``vsig``
   Statistics of the clone-size vector — diversity, clonality, junction length, segment usage,
   isotype, residue composition, physico-chemistry, Pgen, SHM, locus balance. Computed in
   :mod:`vdjtools.signature`.
``rsig``
   Geometry — every column a linear functional, a norm, or a mixture coefficient of the
   prototype-sum measure. Computed here.

Both share one frozen column contract, imported from
:mod:`vdjtools.signature.layout`: a column name is always ``<sig>:<block>:<locus>:<feature>``
(``-`` for a cross-locus column), and the tiers ``core`` (153) ⊂ ``standard`` (689) ⊂ ``full``
(1404) are exact **index subsets** of one column order.

**Holes are never zeros.** An unsequenced locus, a compartment below its clonotype floor, or a
statistic the sample is too shallow to estimate is ``nan`` plus a ``mask:`` column — because a model
that reads "absent" as "zero" reads an unsequenced chain as biology.

**The rotation is fit-free.** It is the PCA of the bundled prototype panel embedded against itself —
zero samples, so nobody's coordinates move when a reference is refreshed, and it covers all seven
loci. Only location and scale come from data. See :doc:`signature` for which scale reference to use.

**The full API for this package is on its own page** — see :doc:`signature`, which documents
``mir.signature``, ``mir.signature.assemble``, ``mir.signature.blocks``, ``mir.signature.reference``
and ``mir.signature.scale`` alongside the measurements behind each design choice. It is not repeated
here so that every symbol has one canonical entry.

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

``pip install mirpy-lib`` installs a ``mir`` console script (also ``python -m mir.cli``) with four
commands: two embedding scales, plus the portable signature and its column presets.

Both ``embed`` commands drop non-coding clonotypes (stop codon / legacy out-of-frame
``junction_aa``) before embedding, and there is **no flag to disable it**: a stop codon is in
seqtree's alphabet, so an unfiltered frame does not crash — it embeds to a finite, meaningless
distance and contaminates the geometry silently. ``--no-filter-functional`` is refused with a
message pointing at ``vdjtools filter --nonproductive``, which is what to use when the
non-productive fraction is the thing you want.

``mir embed clonotypes INPUT``
   One repertoire's clonotype table → a per-clonotype TCREMP embedding table (columns ``e0…``).
   Flags: ``--species``, ``--locus`` (inferred when the file has one locus), ``--n-prototypes``,
   ``--mode {vjcdr3,cdr123}``, ``--replicate R`` (prototype draw), ``--pca K`` (compact the table),
   ``--threads``, ``-o`` (``.tsv`` / ``.parquet``; default stdout TSV).

``mir embed repertoires INPUT...``
   A dataset of clonotype tables → one repertoire vector ``Φ(S)`` per sample **per chain** on one
   shared basis (columns ``phi0…``; sample id = filename stem). Flags: ``--species``, ``--locus``
   (restrict), ``--n-prototypes``, ``--replicate R``,
   ``--weight {log2p1,duplicate_count,distinct,log1p,anscombe}`` (default ``log2p1``),
   ``--blocks mean,diversity[,second]``, ``--n-rff``, ``--n-rff-second``, ``--n-components``,
   ``--mmd OUT`` (also write the per-chain pairwise unbiased-MMD matrix),
   ``--threads``, ``--seed``, ``-o``.

``mir signature INPUT...``
   AIRR clonotype tables → **one fixed-width named feature vector per sample**, standardised
   against a frozen reference so a downstream model needs no scaler of its own. This is the command
   to send a collaborator. Flags: ``--tier {core,standard,full}``, ``--preset NAME``,
   ``--species``, ``--weight {log2p1,duplicate_count,distinct,log1p,anscombe}``,
   ``--standardize {reference,none}``, ``--scale NAME|PATH`` (which scale reference to
   standardise against — a bundled model name or a path; a named-but-missing one raises rather
   than silently producing an unstandardised matrix), ``--jobs`` / ``-j`` (0 = every core; renamed from ``--threads``),
   ``--describe`` (print the column dictionary and read no input), ``--channels`` (print the
   channel vocabulary and read no input), ``-o``.

   This emits the ``rsig`` half alone — 528 columns at the ``standard`` tier. ``vdjtools
   signature`` emits the ``vsig`` half and reports how many ``rsig`` columns it left to this
   command; the two concatenate on ``sample_id`` for the full 689.
   :doc:`signature` covers the scale references and :doc:`channels` the vocabulary the columns
   group into.

``mir presets [NAME]``
   The named column subsets and their ranking — ``compact`` (86), ``classify`` (615),
   ``transfer`` (550), ``geometry`` (514), ``statistics`` (101), ``bcell`` (271), ``full`` (1404),
   ``nuisance`` (74, ranked *avoid*). With no argument, the whole table; with a name, that preset's
   column list. A preset resolves from the frozen layout alone, so two people choosing the same
   name get the same columns in the same order.

.. automodule:: mir.cli
   :members: main, build_parser
   :show-inheritance:

