Command reference
=================

``pip install mirpy-lib`` installs one ``mir`` command with four subcommands. This page lists each
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

**Sample identity.** The sample id defaults to the filename stem. The locus is inferred per file, or
pin it with ``--locus``, which accepts aliases -- ``beta``, ``T-alpha`` -- and errors on anything it
cannot resolve rather than guessing.

**Threads.** ``--threads 0`` means all cores.

**Non-coding clonotypes are dropped, and there is no flag to keep them.** Both ``embed`` commands
filter to productive clonotypes first. This is deliberate: a stop codon is a character in seqtree's
alphabet, so an unfiltered frame does not fail -- it embeds to a finite, meaningless distance and
contaminates the geometry silently. ``--no-filter-functional`` is refused with a pointer to
``vdjtools filter --nonproductive``, which is what to use when the non-productive fraction is itself
the thing you want.

``mir embed clonotypes``
------------------------

One repertoire in, one row per clonotype out -- the input to clustering, visualisation and per-receptor
machine learning.

.. code-block:: bash

   mir embed clonotypes sample.tsv -o clonotypes.parquet
   mir embed clonotypes sample.tsv --pca 50 --n-prototypes 1000 -o clonotypes.parquet

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Meaning
   * - ``--species``
     - ``human`` (default) or ``mouse``.
   * - ``--locus``
     - Pin the chain instead of inferring it. Accepts aliases.
   * - ``--n-prototypes``
     - How many prototypes define the coordinate system. Omitted, the per-chain preset is used.
   * - ``--mode``
     - ``vjcdr3`` (default: three distance blocks, V and J and junction) or ``cdr123``.
   * - ``--replicate``
     - Which disjoint block of prototypes to draw, ``0`` by default. Used to test whether a result
       depends on the draw -- see :ref:`which-prototypes`. Different replicates are **different
       coordinate systems** and their embeddings must never be compared directly.
   * - ``--pca``
     - Denoise to this many components. The preset's 95-percent dimension is the usual choice.
   * - ``--threads``
     - ``0`` for all cores.

``mir embed repertoires``
-------------------------

A set of repertoires in, one fingerprint ``Phi(S)`` per sample per chain out, all on **one shared
basis** so the rows are mutually comparable.

.. code-block:: bash

   mir embed repertoires cohort/*.tsv.gz -o phi.tsv
   mir embed repertoires cohort/*.tsv.gz -o phi.tsv --mmd mmd.tsv --blocks mean,diversity,second

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Meaning
   * - ``--blocks``
     - Which blocks make up the vector: ``mean`` (the random-feature kernel mean), ``diversity``
       (the coverage-standardised Hill profile), ``second`` (the co-occurrence second moment).
       Default ``mean,diversity``.
   * - ``--weight``
     - The clone-size transform: ``log2p1`` (default, concave, so one hyperexpanded clone cannot
       dominate), ``duplicate_count`` (linear in clone size), ``distinct`` (presence only),
       ``log1p``, ``anscombe``.
   * - ``--n-rff`` / ``--n-rff-second``
     - Random-feature dimensions for the kernel mean and the second-moment block.
   * - ``--n-components``
     - PCA dimension of the underlying clonotype space.
   * - ``--mmd``
     - Also write the pairwise MMD distance matrix. Per chain: with several loci this becomes
       ``mmd.TRB.tsv``, ``mmd.TRA.tsv`` and so on; with one locus the name is used as given.
   * - ``--seed``
     - Fixes the random features, and therefore the basis.

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
     - Worker processes over samples, ``0`` for all cores.

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
