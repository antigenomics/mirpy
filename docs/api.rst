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

.. automodule:: mir.repertoire
   :members:
   :undoc-members:
   :show-inheritance:

Explainable readouts (``mir.explain``)
--------------------------------------

Which named channel of ``Φ`` carries the signal, and which clonotypes drive it (Theory T7).

.. automodule:: mir.explain
   :members:
   :undoc-members:
   :show-inheritance:

Cohort / digital donor (``mir.cohort``)
---------------------------------------

Fuse per-chain repertoire embeddings into one hash-verified, serialisable donor matrix; batch
residualisation, sample clustering, and incidence biomarkers (Theory T7).

.. automodule:: mir.cohort
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

Scorers for the explainable readout: cross-validated AUC, Cox C-index, log-rank.

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

``mir embed clonotypes INPUT``
   One repertoire's clonotype table → a per-clonotype TCREMP embedding table (columns ``e0…``).
   Flags: ``--species``, ``--locus`` (inferred when the file has one locus), ``--n-prototypes``,
   ``--mode {vjcdr3,cdr123}``, ``--pca K`` (compact the table), ``--threads``, ``-o`` (``.tsv`` /
   ``.parquet``; default stdout TSV).

``mir embed repertoires INPUT...``
   A dataset of clonotype tables → one repertoire vector ``Φ(S)`` per sample **per chain** on one
   shared basis (columns ``phi0…``; sample id = filename stem). Flags: ``--locus`` (restrict),
   ``--weight {log1p,anscombe,distinct}``, ``--blocks mean,diversity[,second]``, ``--n-rff``,
   ``--n-components``, ``--mmd OUT`` (also write the per-chain pairwise unbiased-MMD matrix),
   ``--threads``, ``--seed``, ``-o``.

.. automodule:: mir.cli
   :members: main, build_parser
   :show-inheritance:

