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

**Species.** ``--species`` selects the receptor species for embedding, signature computation
and synthetic corpus building; the default is ``human``. Use a basis or corpus fitted for the same species.

**Output.** ``-o/--output`` writes TSV, or Parquet when the path ends in ``.parquet``. Parquet is
worth using for the raw per-clonotype embedding, which is thousands of columns wide. Omit ``-o`` and
the result goes to stdout.

**Sample identity.** The sample ID is the filename up to its first dot; files sharing it form one sample. The locus is inferred per file, or
pin it with ``--locus``, which accepts aliases -- ``beta``, ``T-alpha`` -- and errors on anything it
cannot resolve rather than guessing.

**Threads.** ``--threads 0`` means all allocated cores. Positive counts must fit the allocation; excessive requests raise.

**Non-coding clonotypes are dropped, and there is no flag to keep them.** Both ``embed`` commands
filter to productive clonotypes first. This is deliberate: a stop codon is a character in seqtree's
alphabet, so an unfiltered frame does not fail -- it embeds to a finite, meaningless distance and
contaminates the geometry silently. ``--no-filter-functional`` is refused with a pointer to
``vdjtools filter --nonproductive``, which is what to use when the non-productive fraction is itself
the thing you want.

Embedding and PCA commands
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Command
     - Result
   * - ``mir embed clonotypes INPUT...``
     - Clonotype coordinates with sample/clonotype IDs. ``--pca K --locus LOC`` opts into clonotype PCA.
   * - ``mir embed repertoires INPUT...``
     - Unprojected sample RFF vectors. ``--save-basis`` / ``--basis`` save or reuse the mapping.
   * - ``mir pca TABLE --fit FILE``
     - Fit and save sample PCA; ``--cohort TABLE`` selects compatible fitting rows.
   * - ``mir pca TABLE --rotation FILE``
     - Apply a previously fitted sample rotation.

.. code-block:: bash

   mir embed repertoires samples/*.tsv --save-basis basis --threads 1 -o rff.tsv
   mir pca rff.tsv --fit rotation.npz --n-components 32 -o pcs.tsv

Inline sample PCA uses ``--pca``, ``--pca-cohort`` or ``--rotation``. Keep its RFF
input with ``--rff-output`` and save a fit with ``--save-rotation``.
``--n-components`` on ``embed repertoires`` controls the clonotype basis, not sample
PCA. ``--mmd FILE`` writes one unbiased MMD distance table per locus.

:doc:`embeddings` defines output columns, basis compatibility, channels, memory
budgets, Parquet requirements and the complete fit/reuse workflows.

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
     - Worker threads over samples, ``0`` for allocated cores.

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

Pre-stage reference rotations
-----------------------------

``mir reference NAME --revision COMMIT`` fetches a reference from the published
``isalgo/mirpy_models`` dataset, verifies every manifest checksum, and prints its
local directory. ``COMMIT`` must be the full immutable commit SHA. It does not
load pickle models or fit a rotation. Use the resulting directory with
``--basis``, ``--rotation`` and vdjtools' ``--feature-vocabulary``.

``--cache-dir PATH`` selects a writable Hugging Face cache directory. ``--offline``
uses only a snapshot already staged there. Pre-stage once before workflow fan-out;
workers can use the verified directory as a read-only reference input. Hugging
Face handles per-file download locks and atomic cache publication.
