Command reference
=================

``pip install mirpy-lib`` installs one ``mir`` command with embedding, PCA and signature subcommands. This page lists each
one, what it is for, and the options that change an answer rather than only a path.

``mir <command> --help`` is the authoritative list of flags and defaults -- it is generated from the
code, so it cannot drift. This page is the map.

Conventions
-----------

**Input.** Any format :mod:`vdjtools.io` reads: AIRR Rearrangement TSV, native vdjtools, MiXcr,
immunoSEQ, Parquet. Several files or a glob are fine.

**Output.** ``-o/--output`` writes TSV, or Parquet when the path ends in ``.parquet``. Parquet is
worth using for the raw per-clonotype embedding, which is thousands of columns wide. Omit ``-o`` and
the result goes to stdout.

**Sample identity.** The sample ID is the filename up to its first dot; files sharing it form one sample. The locus is inferred per file, or
pin it with ``--locus``, which accepts aliases -- ``beta``, ``T-alpha`` -- and errors on anything it
cannot resolve rather than guessing.

**Threads.** ``--threads 0`` means all allocated cores. Positive counts are capped by the allocation.

**Non-coding clonotypes are dropped, and there is no flag to keep them.** Both ``embed`` commands
filter to productive clonotypes first. This is deliberate: a stop codon is a character in seqtree's
alphabet, so an unfiltered frame does not fail -- it embeds to a finite, meaningless distance and
contaminates the geometry silently. ``--no-filter-functional`` is refused with a pointer to
``vdjtools filter --nonproductive``, which is what to use when the non-productive fraction is itself
the thing you want.

``mir embed clonotypes``
------------------------

A list of repertoires in, raw clonotype embeddings out, with ``sample_id`` and
``clonotype_id`` on every row. Mixed loci are accepted; ``locus`` identifies each
row's panel. Supplied clonotype or AIRR sequence IDs are preserved; otherwise
stable within-sample IDs are assigned. Files sharing a filename stem form one
sample. No PCA is fitted by this command.

.. code-block:: bash

   mir embed clonotypes cohort/*.tsv -o clonotypes.tsv --threads 1
   pip install "mirpy-lib[parquet]"
   mir embed clonotypes cohort/*.tsv -o clonotypes.parquet

Rows are embedded and written in bounded batches. TSV needs only the base package;
streaming Parquet row groups use the optional ``parquet`` extra. ``--n-prototypes``
uses the established per-chain preset. ``--mode``, ``--species``, ``--locus`` and ``--replicate``
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

``mir embeddings`` / ``mir signature``
--------------------------------------

One repertoire in, one row of fixed, named, already-standardised features out -- the **geometry
half** of the portable signature. The statistics half is ``vdjtools signature``, and the two join on
``sample_id``.

.. code-block:: bash

   mir signature --corpus blood cohort/*.tsv.gz -o rsig.parquet
   mir signature --corpus blood --components 32 --describe

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Meaning
   * - ``--corpus``
     - **Required.** A published name -- ``blood``, ``tissue``, ``deep-tcr``, ``blood-uncapped``,
       ``tissue-uncapped``, ``synthetic-blood``, ``synthetic-tissue``, ``naive``, ``memory`` -- or a
       path. Downloaded and digest-verified on first use, then cached. There is no default: a
       signature is comparable to another one only if both were rotated through the same corpus.
   * - ``--weight``
     - Clonotype abundance weighting: ``log2p1`` (default), ``duplicate_count``, ``distinct``, ``log1p`` or ``anscombe``.
   * - ``--components``
     - An integer count or a variance fraction. Truncating a wider rotation downward is exact.
   * - ``--winsorize``
     - ``features``, ``pcs`` or ``none``. Whatever is clamped is reported in
       ``rsig:qc:-:winsor_frac``.
   * - ``--winsor-p``
     - Which stored percentile to clamp at, ``0.01`` or ``0.05``. Defaults to whichever the
       artifact was fitted with; both are stored, so switching needs no refit.
   * - ``--on-duplicate``
     - ``error`` (default) or ``sum``. A frame with no ``junction_nt`` that repeats an amino-acid
       clonotype key cannot say whether those rows are one clonotype or two, so the library refuses
       rather than guessing.
   * - ``--describe``
     - Print the columns **this** invocation emits, and exit.
   * - ``--named``
     - Also emit the reportable raw blocks under their own names (``depth``, ``band``,
       ``band_igh``), beside the rotated components. Values carry their declared transform, not a
       natural scale.
   * - ``--min-clonotypes``
     - The floor below which a locus's whole ``div``/``disp`` family is a hole, default ``5``. A
       dispersion measured on three clonotypes is not a measurement; the geometry itself is still
       computed there.
   * - ``--columns``
     - Restrict the output to a file of column names.
   * - ``-j/--jobs``
     - Worker threads over samples, ``0`` for all cores.

Which corpus to choose, and what the columns mean, are in :doc:`signature` and :doc:`channels`.

``mir corpus``
--------------

Fetch a published corpus artifact, or build a synthetic one.

.. code-block:: bash

   mir corpus --fetch all                              # pre-warm the cache
   mir corpus --corpus synthetic-blood -o sb.npz       # build and fit your own
   mir corpus --smoke -o /tmp/smoke.npz                # a reduced build, minutes not hours

Building a synthetic corpus uses no samples from anybody's cohort: every receptor is drawn from
vdjtools' bundled recombination models, so the artifact is reproducible by anyone who installs the
library and is byte-identical across processes and thread counts. ``--samples``, ``--size``,
``--seed``, ``--loci``, ``--source`` and ``--depth-spread`` control the draw and are all recorded in
the manifest.

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
