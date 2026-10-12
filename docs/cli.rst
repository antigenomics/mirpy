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
defaults to 256 per locus. ``--mode``, ``--species``, ``--locus`` and ``--replicate``
select the raw coordinate system. Native batch kernels own ``--threads``.

``mir embed repertoires``
-------------------------

A list of repertoires in, one **raw sample embedding row** per sample out, with
``sample_id``. Each sample is read once, reduced through the existing fixed-prototype
weighted sum, and released. Only compact vectors survive the bounded sample map;
they are assembled into a Polars table. There is no pooled clonotype matrix,
clonotype-level scaler/PCA fit, or random-feature fit.

.. code-block:: bash

   mir embed repertoires cohort/*.tsv -o sample_embeddings.tsv --threads 1

``--n-prototypes`` defaults to 256 per locus; ``--weight`` defaults to ``log2p1``.
The V/J/junction slots are retained, interleaved as in ``TCREmp``: 768 coordinates
per locus, or 5,376 across seven human loci. A 10,000-row float64 raw matrix is
about 410 MiB. Missing loci are NaN with explicit ``present:<locus>`` indicators;
empty samples are retained. ``embedding_space`` fingerprints the ordered panels
and coordinate options. Worker threads share fixed panels and use serial inner
kernels; ``--threads 1`` creates no worker pool.

``mir pca``
------------

Fit/save a rotation from a **sample embedding table**, or apply that same rotation
later. PCA is deliberately downstream of embedding. IDs, counts and other metadata
are retained. Missing coordinates are filled from training medians, and presence
indicators retain missing-locus information. Column and embedding-space checks
reject incompatible input. The artifact is NumPy data, without executable pickle.

.. code-block:: bash

   mir pca sample_embeddings.tsv --fit rotation.npz --n-components 256 -o pcs.tsv
   mir pca future_embeddings.tsv --rotation rotation.npz -o future_pcs.tsv

Fit on training samples and apply the saved rotation to held-out samples. Refitting
on another cohort defines different coordinates. The default dimension is 256,
capped by the number of samples and features; ``--seed`` fixes the solver seed.

The CLI no longer fits the older RFF repertoire basis inside ``embed``. The
standalone Python ``fit_repertoire_space`` / ``sample_embedding`` / ``mmd_matrix``
APIs remain available for that distinct kernel representation.

``mir signature``
-----------------

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
