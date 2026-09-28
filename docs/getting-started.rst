Getting started
===============

A first pass through mirpy in about fifteen minutes: install it, embed receptors, denoise the
embedding, and measure the distance between two groups of repertoires. Everything here runs offline
against the prototype tables that ship in the wheel, so there is nothing to download.

If you already know what you want, the :doc:`guides <usage>` are organised one task per page,
:doc:`cli` lists every command, and :doc:`math` explains why the embedding is valid.

Install
-------

.. code-block:: bash

   pip install mirpy-lib
   mir --help

Pure-Python wheel. The alignment, Pgen and sampling work is reused from
`seqtree <https://github.com/antigenomics/seqtree>`_ and
`vdjtools <https://github.com/antigenomics/vdjtools>`_, which install with it.

Step 1 -- receptors to work on
------------------------------

Normally you would read your own file, in any format vdjtools understands:

.. code-block:: python

   from vdjtools import io as vio

   sample = vio.read("clones.tsv")     # AIRR, MiXcr, immunoSEQ, Parquet, native -- detected

So that this page runs offline, use the bundled prototype table instead. It is 10,000 real, unique,
productive human TRB receptors, which is a perfectly good set of clonotypes to embed:

.. code-block:: python

   from mir.embedding.prototypes import load_prototypes

   pool = load_prototypes("human", "TRB")
   pool.columns      # ['v_call', 'j_call', 'junction_aa']
   pool.height       # 10000

.. note::

   Rows 0 to 999 of this table **are** the default prototype set at ``n_prototypes=1000``. Embedding
   a receptor against itself gives distance 0, which would flatter any result, so the examples below
   take clonotypes from row 5000 onward -- genuinely held out from the coordinate system they are
   measured in.

Step 2 -- receptors to vectors
------------------------------

.. code-block:: python

   from mir.embedding.tcremp import TCREmp

   model  = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
   sample = pool.slice(5000, 400)          # 400 held-out clonotypes
   X = model.embed(sample)
   X.shape                                 # (400, 3000)

3,000 columns for 1,000 prototypes, because each receptor contributes three distances per prototype:
one for V, one for J, and one for the junction. The junction distance is computed through
``seqtree.gapblock``; the V and J distances are precomputed from the arda germline, so they cost a
lookup rather than an alignment.

Leaving ``n_prototypes`` out uses the per-chain preset instead, which is what you want unless you are
deliberately testing sensitivity -- 2,000 prototypes for human TRB, so 6,000 columns.

Step 3 -- denoise
-----------------

Three thousand columns are correlated by construction: nearby prototypes measure nearly the same
thing. PCA to the preset's dimension keeps the geometry and drops the redundancy:

.. code-block:: python

   from mir.embedding.pca import pca_denoise
   from mir.embedding.presets import get_preset

   preset = get_preset("human", "TRB")     # n_prototypes=2000, n_components=65, recon=260
   Xd = pca_denoise(X, n_components=65)
   Xd.shape                                # (400, 65)

Use the 95-percent-variance dimension (``n_components``) for clustering and visualisation, and the
99-percent one (``n_components_recon``) when reconstructing sequences with a neural codec, where
losing sequence detail matters. See :ref:`which-prototypes` for the full table and for how to test
whether a result depends on which prototypes you drew.

Step 4 -- repertoires, not receptors
------------------------------------

To compare *donors* rather than receptors, each repertoire needs its own single vector. Build two
groups that genuinely differ -- here, repertoires of short junctions against repertoires of long
ones -- so that the distance has something to find:

.. code-block:: python

   import numpy as np
   import polars as pl
   from mir.repertoire import fit_repertoire_space, sample_embedding, mmd_matrix

   pool = pool.with_columns(pl.col("junction_aa").str.len_chars().alias("L"))

   def donor(expr, n, seed):
       """One synthetic repertoire: n clonotypes matching expr, with random clone sizes."""
       rng = np.random.default_rng(seed)
       d = pool.filter(expr).sample(n, seed=seed)
       return d.with_columns(pl.Series("duplicate_count", rng.integers(1, 50, n)))

   samples = ([donor(pl.col("L") <= 13, 300, s) for s in (1, 2)] +     # two short-junction donors
              [donor(pl.col("L") >= 17, 300, s) for s in (3, 4)])      # two long-junction donors

   space = fit_repertoire_space(model, pl.concat(samples))   # ONE basis for the whole cohort
   embs  = [sample_embedding(space, s) for s in samples]
   mmd_matrix(embs, unbiased=True)

.. code-block:: text

          short1  short2   long1   long2
   short1  0.0000  0.0000  0.0410  0.0430
   short2  0.0000  0.0000  0.0542  0.0509
   long1   0.0410  0.0542  0.0000  0.0000
   long2   0.0430  0.0509  0.0000  0.0000

Two donors of the same kind are at distance 0 and two of different kinds are 0.041 to 0.054 apart.
Exact zeros within a group are the right answer rather than a rounding artefact: the unbiased
estimator removes the self-similarity term analytically, so for two samples drawn from the *same*
distribution its estimate is zero up to noise, and a small negative estimate is clamped to zero.

.. important::

   ``fit_repertoire_space`` must be fitted **once**, on the pooled cohort, and every sample embedded
   through that one basis. Two samples embedded through different bases -- or through different
   prototype replicates -- are not comparable, and every container here refuses a prototype-hash
   mismatch rather than mixing them silently.

   Pass ``unbiased=True`` whenever your samples differ in depth or diversity. The biased estimator
   carries a positive self-term of about ``1/n_eff``, which inflates distances for low-diversity
   samples -- so if diversity is itself what you are studying, the bias shows up as signal with the
   wrong sign.

Step 5 -- the same thing from the shell
---------------------------------------

.. code-block:: bash

   # one repertoire -> per-clonotype embedding table, the input to clustering and ML
   mir embed clonotypes sample.tsv --pca 50 -o clonotypes.parquet

   # a set of repertoires -> one fingerprint per sample on a shared basis, plus the MMD matrix
   mir embed repertoires cohort/*.tsv.gz -o phi.tsv --mmd mmd.tsv

Sample ids default to the filename stem and the locus is inferred per file. With several loci the MMD
matrix is written per chain, as ``mmd.TRB.tsv``, ``mmd.TRA.tsv`` and so on. See :doc:`cli`.

Step 6 -- one named vector per sample
-------------------------------------

``Phi(S)`` is a coordinate system fitted on *your* cohort, which is what makes it powerful and also
what makes it incomparable with anyone else's. When you need a vector that a collaborator can
reproduce independently, use a :doc:`signature` instead: fixed, named columns standardised against a
published reference corpus.

.. code-block:: bash

   mir signature --corpus blood cohort/*.tsv.gz -o rsig.tsv

That emits the geometry half. The statistics half comes from vdjtools, and the two join on
``sample_id``:

.. code-block:: bash

   vdjtools signature --corpus blood cohort/*.tsv.gz -o vsig.tsv

Where to go next
----------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Next
     - Page
   * - Cluster antigen-specific receptors, and find enriched neighbourhoods
     - :doc:`usage`
   * - What must be filtered before embedding, and why it is not optional
     - :doc:`preprocessing`
   * - Portable, named, standardised columns
     - :doc:`signature` | :doc:`channels`
   * - Why the prototype embedding is a valid coordinate system
     - :doc:`math`
   * - Every command and flag
     - :doc:`cli`
   * - A term you did not recognise
     - :doc:`glossary`
