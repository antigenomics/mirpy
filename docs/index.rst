mirpy
=====

ML-oriented embeddings for immune receptor repertoires (TCR/BCR) — the antigenomics group's
machine-learning / embedding library (PyPI ``mirpy-lib``, import ``mir``).

Pure-Python and built on the antigenomics ecosystem
(`seqtree <https://github.com/antigenomics/seqtree>`_,
`vdjtools <https://github.com/antigenomics/vdjtools>`_,
`arda <https://github.com/antigenomics/arda>`_).

.. note::

   **v3.1.0** — the prototype (TCREMP) clonotype embedding plus the Part-2 tier: neural codecs
   (``mir.ml``), continuous-density background subtraction (``mir.density``, T6), and a sample-level
   repertoire embedding (``mir.repertoire``, T7). Coordinates are arda-native and versioned. The
   classical v1.x/v2 toolkit is frozen on the ``legacy-v2`` branch (``mirpy-lib`` 2.x).

New here? The :doc:`user guide <usage>` has runnable examples for each module; the mathematical
theory (T1–T7) lives in ``THEORY.md`` and recorded benchmark numbers in ``BENCHMARKS.md``.

Quickstart — clonotype embedding
--------------------------------

.. code-block:: python

   from mir.embedding.tcremp import TCREmp

   model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
   X = model.embed(df)      # polars frame (v_call / j_call / junction_aa) -> (N, 3K) float32

Distance in the prototype-embedding space approximates the pairwise alignment distance (Theory T1);
``TCREmp`` computes the junction part via ``seqtree.gapblock`` and adds baked germline V/J distances.

Sample-level (repertoire) embedding
-----------------------------------

.. code-block:: python

   from mir.repertoire import fit_repertoire_space, sample_embedding, mmd_matrix

   space = fit_repertoire_space(model, pooled_clonotypes)   # ONE basis for the cohort
   embs  = [sample_embedding(space, s) for s in samples]    # Φ(S): mean ‖ diversity ‖ second moment
   D     = mmd_matrix(embs, unbiased=True)                  # pairwise repertoire distance (unbiased MMD²)

Capabilities (see the :doc:`API reference <api>`)
-------------------------------------------------

- **Clonotype embedding** — TCREMP prototype embedding (``mir.embedding``) on ``seqtree.gapblock`` +
  baked arda germline distances (``mir.distances``), with PCA denoise and per-chain presets.
- **Density** — graph-free TCRNET/ALICE neighbour enrichment in embedding space, with an
  abundance channel and exact / kdtree / ANN backends (``mir.density``, T6).
- **Repertoire embedding** — one fixed vector per sample: RFF kernel mean, coverage-standardised
  Hill diversity, and a second-moment Fisher block; MMD / HLA-stratified distance and a motif
  witness (``mir.repertoire``, T7).
- **Neural codecs** — forward / inverse / Pgen / unified codecs and a learned repertoire set
  encoder; CUDA → MPS → CPU device selection (``mir.ml``, ``[ml]`` extra).
- **Benchmark harness** — VDJdb clustering + F1/retention and reproduced theory (``mir.bench``).

.. toctree::
   :hidden:

   self
   usage
   api
