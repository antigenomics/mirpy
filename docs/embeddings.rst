Clonotype and sample embeddings
===============================

Choose the output by the unit you want to analyse:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Command
     - One row represents
     - Coordinates
   * - ``mir embed clonotypes``
     - A clonotype, with sample and clonotype IDs
     - Interleaved V/J/junction distances to prototypes
   * - ``mir embed repertoires``
     - A sample, across its observed loci
     - Weighted RFF means and separate channels
   * - ``mir pca``
     - A sample in an existing RFF table
     - PCs fitted on those RFF columns
   * - ``mir embeddings`` / ``mir signature``
     - A sample
     - Existing corpus-standardised rsig features; see :doc:`signature`

The sample path is clonotype coordinates → shared clonotype scaler/PCA → RFF →
weighted sample mean → optional sample PCA. Applying RFF to the mean clonotype
coordinate is a different calculation. Existing rsig features are not an RFF
subset or extension; their corpus rotations cannot be applied to an RFF table.

.. contents::
   :local:
   :depth: 2

``mir embed clonotypes``
------------------------

A list of repertoires in, raw clonotype embeddings out, with ``sample_id`` and
``clonotype_id`` on every row. Mixed loci are accepted; ``locus`` identifies each
row's panel. Supplied clonotype or AIRR sequence IDs are preserved; otherwise
stable within-sample IDs are assigned. Files sharing a filename stem form one
sample. No PCA is fitted unless ``--pca K --locus LOC`` is requested.

.. code-block:: bash

   mir embed clonotypes cohort/*.tsv -o clonotypes.tsv --threads 1
   pip install "mirpy-lib[parquet]"
   mir embed clonotypes cohort/*.tsv -o clonotypes.parquet

Rows are embedded and written in bounded batches. TSV needs only the base package;
streaming Parquet row groups use the optional ``parquet`` extra. Omitting ``--n-prototypes``
uses the established per-chain preset; supplying it overrides the count. ``--mode``, ``--species``, ``--locus`` and ``--replicate``
select the raw coordinate system. Native batch kernels own ``--threads``.

``mir embed repertoires``
-------------------------

A list of repertoires in, one RFF kernel-mean row per sample out. The order is
clonotype coordinates → the shared clonotype basis → RFF → weighted mean.
The default output is unprojected RFF and diversity, with optional second-moment
channels. Sample PCA runs only when requested. Counts and missing-locus indicators
are retained; a missing locus is a hole, not a zero-valued embedding.

.. code-block:: bash

   mir embed repertoires cohort/*.tsv --save-basis basis -o rff.tsv --threads 1
   mir embed repertoires control/*.tsv --basis basis -o control_rff.tsv --threads 1

``--basis`` reuses saved ``RepertoireSpace`` models; load only trusted model files.
Without it, the established seeded clonotype fit is performed one locus at a time,
reading samples in passes instead of retaining the whole cohort. ``--n-components``
controls that clonotype basis. ``--n-rff`` controls the RFF width (default 1024).
``--blocks mean,diversity`` is the default; add ``second`` when needed. All samples
share the same mapping. Each worker retains one sample and bounded numerical blocks;
only sample vectors survive the reduction. ``--threads 1`` starts no worker pool.

``mir pca`` and optional inline PCA
-----------------------------------

Fit on the resulting RFF vectors, fit on a compatible control cohort, or apply a
saved rotation. The implementation is shared with ``vdjtools.signature.corpus.TablePCA``
and the donor identity PCA. Selected RFF columns are projected; diversity,
second-moment, count and presence channels pass through unchanged. Training medians
fill missing values; wholly missing columns use zero. IDs are preserved. The
rotation stores feature names and the shared basis fingerprint, and refuses a
mismatch. It is a NumPy artifact with no executable pickle payload.

.. code-block:: bash

   mir pca rff.tsv --fit rotation.npz --n-components 32 -o pcs.tsv
   mir pca rff.tsv --cohort control_rff.tsv --fit control_rotation.npz -o control_pcs.tsv
   mir pca rff.tsv --rotation rotation.npz -o reused_pcs.tsv

   mir embed repertoires cohort/*.tsv --basis basis --pca 32 -o pcs.tsv
   mir embed repertoires cohort/*.tsv --basis basis --pca-cohort control_rff.tsv --pca 32 -o pcs.tsv
   mir embed repertoires cohort/*.tsv --basis basis --rotation rotation.npz -o pcs.tsv

Use ``--save-rotation`` to save an inline fit and ``--rff-output`` to also keep its
unprojected RFF table. ``--pca`` is sample PCA; ``--n-components`` is the shared
clonotype basis. Neither independently fitted RFF bases nor the older ``rsig``
corpus rotations can be substituted for a rotation fitted to these RFF columns.

Optional comparison and clonotype PCA
-------------------------------------

``mir embed repertoires --mmd FILE`` retains the existing unbiased MMD calculation
from the mean RFF block and effective sample sizes; it writes one distance table per
locus. Pairwise distance output is quadratic in sample count and is memory checked.
``mir embed clonotypes --pca K --locus LOC`` retains the existing within-locus
clonotype PCA option. This opt-in fit requires the clonotype matrix in memory;
the default clonotype output streams without fitting.

Memory use
----------

The repertoire workflow reads and releases one sample per worker and retains only
RFF rows and channels. Shared clonotype-basis fitting retains one selected locus
matrix at a time. Scaler statistics use at most 128 MiB of estimated scratch,
standardization reuses that matrix, and it is released before radius calibration.
Native V/J/junction component batches use a 64 MiB component budget; RFF reduction
uses bounded row batches. SHM evidence and distinct nucleotide clonotypes survive
the embedding reader's column trimming.

Memory still grows with sample-table width, sample count, selected basis-fit rows
and prototype count. Sample PCA requires its compact sample matrix, and MMD
requires a quadratic distance matrix. Visible cgroup limits are checked before
known large numerical allocations; an unusually large input sample can still
require substantial memory to read. Choose fewer workers or reuse a saved basis
when input frames dominate the allocation.

Python tables and deferred readers
----------------------------------

For file cohorts, use zero-argument readers so a worker loads one sample at a time.
Readers return a locus-to-frame mapping and must return fresh frames on every call;
basis fitting reads selected samples in several passes.

.. code-block:: python

   import mir                            # establish runtime limits before Polars/NumPy
   from functools import partial
   from vdjtools.io import read
   from vdjtools.cores import serial_kernels
   from mir.embedding.table import fit_sample_spaces, sample_embeddings, TablePCA
   from mir.embedding.tcremp import TCREmp

   def read_trb(path):
       return {"TRB": read(path, keep=("v_mutations", "v_identity"))}

   readers = [("S1", partial(read_trb, "s1.tsv")),
              ("S2", partial(read_trb, "s2.tsv"))]
   models = {"TRB": TCREmp.from_defaults("human", "TRB", threads=0)}
   with serial_kernels():
       spaces = fit_sample_spaces(readers, models, n_rff=1024)
   table = sample_embeddings(readers, spaces, n_jobs=1)
   rotation = TablePCA.fit(table, n_components=2)
   rotation.save("sample_pca.npz")
   pcs = rotation.transform(table)

The reader must supply productive, valid clonotypes. The CLI applies its mandatory
productive filter; the Python embedding API validates its input rather than quietly
filtering it. See :doc:`preprocessing`. :class:`mir.repertoire.RepertoireSpace`
can save each fitted locus basis; those model files use pickle and must be trusted.
``TablePCA`` is shared with vdjtools, saves non-executable NumPy parameters, and
projects ``phi:`` columns by default. Channels and identifiers pass through.

Two RFF tables are comparable when their full basis and abundance weighting match,
not merely their column names or widths. Preserve ``embedding_space`` when exporting,
joining or selecting a table. Fit a sample rotation once and reuse it for held-out
samples; fitting separate rotations does not make their PC axes comparable.

.. _reference-rotation-bundles:

Reference rotation bundles
--------------------------

``mir.embedding.reference.build_reference_rotations`` fits repeatable raw-sample
readers. It exports two separate rotations: ``features-pca.npz`` for unrotated
vdjtools statistical features, and ``rff-pca.npz`` for mirpy sample RFF means.
It also saves ``feature-columns.txt`` and the per-locus RFF models in ``basis/``.
It does not fit or replace a vsig/rsig signature corpus. Feature matrices are
generated fresh and released after fitting rather than saved as intermediates.

.. code-block:: python

   from mir.embedding.reference import build_reference_rotations

   build_reference_rotations(readers, ["TRB"], "reference", jobs=1)

Each reference bundle has its own shared clonotype basis. Choose a reference
first, embed samples through its saved basis, then apply its RFF rotation:

.. code-block:: bash

   mir embed repertoires samples/*.tsv --basis reference/basis --threads 2 -o rff.tsv
   mir pca rff.tsv --rotation reference/rff-pca.npz -o pcs.tsv

For statistical features, use the bundle's human gene vocabulary, ``log2p1``
abundance weighting and declared shared coverage target. The recipe defaults to
coverage ``0.8``. See `vdjtools' feature guide
<https://docs.isalgo.dev/vdjtools/features.html>`_. A rotation refuses a different
feature identity. Keep unprojected outputs when needed; loading a rotation does
not change the underlying representation.

``bundle.json`` records representations, parameters and SHA256 checksums. Verify
these checksums before loading the trusted pickle basis files. The NumPy rotation
files contain no executable pickle payload. Raw membership, source identifiers
and fit diagnostics belong in private provenance, not the distributed bundle.
