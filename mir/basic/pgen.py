"""OLGA-based generation probability and sequence generation for V(D)J recombination.

Wraps the OLGA library to provide:

- Pgen computation (exact and 1-mismatch) with true multicore scaling via a
  persistent process pool whose workers load the OLGA model once and handle
  all subsequent batches without re-initialisation overhead.
- Productive CDR3 sequence generation with full VDJ annotation.
- Parallel pool generation via :meth:`OlgaModel.generate_pool` — the
  recommended entry-point for VDJbet analysis.  Each record stores
  ``log2_pgen`` (not log10) so downstream code never needs a unit conversion.
- V/J gene-usage adjustment (importance sampling) via
  :class:`PgenGeneUsageAdjustment`.

Parallel strategy
-----------------
``compute_pgen_junction_aa_bulk`` uses a ``multiprocessing.Pool`` whose workers
are initialised once with the OLGA model via a pool initializer.  For repeated
calls (e.g. across multiple ALICE samples), the pool is reused so the model is
loaded only once per worker per :class:`OlgaModel` lifetime.  Sequence
generation workers still rebuild a fresh model because the sequence-generation
model is not needed in Pgen workers.
"""

from __future__ import annotations

import multiprocessing
import os
from math import ceil
import math
from typing import TYPE_CHECKING, Iterable

# Use "spawn" on every platform to avoid fork-safety issues (urllib3/requests
# connection-pool threads from huggingface_hub can leave locked mutexes in the
# parent that get inherited by forked children, causing indefinite hangs).
_MP_CTX = multiprocessing.get_context("spawn")
Pool = _MP_CTX.Pool

import numpy as np
import olga.generation_probability as pgen
import olga.load_model as load_model
import olga.sequence_generation as seq_gen
import polars as pl

from mir import get_resource_path
from mir.basic.aliases import LOCUS_TO_OLGA_SUFFIX
from mir.basic import mirseq as _mirseq

if TYPE_CHECKING:
    from mir.basic.gene_usage import GeneUsage

translate_bidi = _mirseq.translate_bidi

# Loci that have a D gene segment in their recombination model.
_D_PRESENT: frozenset[str] = frozenset({"TRB", "TRD", "IGH"})

# Number of terminal positions to skip when computing 1mm pgen.
# The first and last residues of a CDR3 are highly conserved (C-anchor at 0,
# F/W-anchor at the end) and encoded by the V/J genes, so mutating them adds
# noise rather than signal.  Skipping 2 positions on each end reduces OLGA
# calls per sequence from L+1 to L-3 (~30% fewer for a 12-AA CDR3).
# Override with env var MIRPY_PGEN_1MM_SKIP_ENDS=0 to restore full behaviour.
_PGEN_1MM_SKIP_ENDS: int = max(0, int(os.getenv("MIRPY_PGEN_1MM_SKIP_ENDS", "2")))


def _split_n(n: int, k: int) -> list[int]:
    """Split *n* into *k* roughly equal positive chunks."""
    q, r = divmod(n, k)
    return [q + (1 if i < r else 0) for i in range(k) if q + (1 if i < r else 0) > 0]


# ---------------------------------------------------------------------------
# Per-worker state (set by pool initializer, never shared between processes)
# ---------------------------------------------------------------------------

_WORKER_PGEN_MODEL = None  # set by _init_pgen_worker in each child process


def _init_pgen_worker(model_dir: str, is_d_present: bool) -> None:
    """Load the OLGA pgen model once in each pool worker process."""
    global _WORKER_PGEN_MODEL

    params_file = f"{model_dir}/model_params.txt"
    marginals_file = f"{model_dir}/model_marginals.txt"
    v_anchor = f"{model_dir}/V_gene_CDR3_anchors.csv"
    j_anchor = f"{model_dir}/J_gene_CDR3_anchors.csv"

    if is_d_present:
        gd = load_model.GenomicDataVDJ()
        gm = load_model.GenerativeModelVDJ()
    else:
        gd = load_model.GenomicDataVJ()
        gm = load_model.GenerativeModelVJ()

    gd.load_igor_genomic_data(params_file, v_anchor, j_anchor)
    gm.load_and_process_igor_model(marginals_file)

    if is_d_present:
        _WORKER_PGEN_MODEL = pgen.GenerationProbabilityVDJ(gm, gd)
    else:
        _WORKER_PGEN_MODEL = pgen.GenerationProbabilityVJ(gm, gd)


def _compute_exact_batch(seqs: list[str]) -> list[float]:
    """Compute exact amino-acid Pgen for a batch in a pool worker."""
    m = _WORKER_PGEN_MODEL
    return [float(m.compute_aa_CDR3_pgen(s)) for s in seqs]


def _compute_1mm_inner(m, seq: str, skip: int) -> float:
    """Compute 1mm pgen for *seq*, skipping the first/last *skip* positions.

    Uses the same identity as OLGA's ``compute_hamming_dist_1_pgen``:
        P_inner = P(self) + sum_{i in inner} [P(X at i) - P(self)]
    where ``inner = range(skip, L - skip)`` and 'X' is the OLGA wildcard.
    Falls back to the full OLGA call when the sequence is too short.
    """
    L = len(seq)
    if skip == 0 or L <= 2 * skip:
        return float(m.compute_hamming_dist_1_pgen(seq, print_warnings=False))
    # Resolve V/J masks once (None → all genes), matching the internal
    # contract of compute_hamming_dist_1_pgen before it calls compute_CDR3_pgen.
    V_mask, J_mask = m.format_usage_masks(None, None, False)
    p_self = float(m.compute_CDR3_pgen(seq, V_mask, J_mask))
    tot = p_self
    for i in range(skip, L - skip):
        tot += float(m.compute_CDR3_pgen(seq[:i] + "X" + seq[i + 1:], V_mask, J_mask)) - p_self
    return max(0.0, tot)


def _compute_1mm_batch(seqs: list[str]) -> list[float]:
    """Compute 1-mismatch Pgen for a batch in a pool worker."""
    m = _WORKER_PGEN_MODEL
    skip = _PGEN_1MM_SKIP_ENDS
    return [_compute_1mm_inner(m, s, skip) for s in seqs]


# ---------------------------------------------------------------------------
# Workers for sequence generation (still reconstruct the full OlgaModel)
# ---------------------------------------------------------------------------

def _generate_chunk(args: tuple) -> list[str]:
    """Worker: generate CDR3 aa sequences (no Pgen) for generate_sequences_parallel."""
    init_kwargs, n, seed = args
    model = OlgaModel(**init_kwargs, seed=None)
    np.random.seed(seed)
    return [model._sample_cdr3_aa() for _ in range(n)]


# ---------------------------------------------------------------------------
# Attempt-counted generation (for MC Pgen denominator calibration)
# ---------------------------------------------------------------------------

def _gen_one_counted_cdr3_vdj(sg, conserved_J_residues: str = "FVW") -> tuple[str, int]:
    """Return (cdr3_aa, n_attempts) for one productive VDJ event.

    Counts every recombination event tried, including non-productive ones, so
    that sum(n_attempts) / n_productive = 1 / P(productive).  This ratio is
    used to normalise MC Pgen estimates to the OLGA scale.
    """
    n = 0
    while True:
        n += 1
        recomb_events = sg.choose_random_recomb_events()

        V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
        if len(V_seq_full) <= max(recomb_events["delV"], 0):
            continue

        D_seq_full = sg.cutD_genomic_CDR3_segs[recomb_events["D"]]
        J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]

        if len(D_seq_full) < (recomb_events["delDl"] + recomb_events["delDr"]):
            continue
        if len(J_seq_full) < recomb_events["delJ"]:
            continue

        V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
        D_seq = D_seq_full[recomb_events["delDl"] : len(D_seq_full) - recomb_events["delDr"]]
        J_seq = J_seq_full[recomb_events["delJ"] :]

        if (
            len(V_seq) + len(D_seq) + len(J_seq)
            + recomb_events["insVD"] + recomb_events["insDJ"]
        ) % 3 != 0:
            continue

        insVD_seq = seq_gen.rnd_ins_seq(recomb_events["insVD"], sg.C_Rvd, sg.C_first_nt_bias_insVD)
        insDJ_seq = seq_gen.rnd_ins_seq(recomb_events["insDJ"], sg.C_Rdj, sg.C_first_nt_bias_insDJ)[::-1]

        ntseq = V_seq + insVD_seq + D_seq + insDJ_seq + J_seq
        aaseq = translate_bidi(ntseq)

        if "*" in aaseq:
            continue
        if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
            continue

        return aaseq, n


def _gen_one_counted_cdr3_vj(sg, conserved_J_residues: str = "FVW") -> tuple[str, int]:
    """Return (cdr3_aa, n_attempts) for one productive VJ event."""
    n = 0
    while True:
        n += 1
        recomb_events = sg.choose_random_recomb_events()

        V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
        if len(V_seq_full) <= max(recomb_events["delV"], 0):
            continue

        J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]
        if len(J_seq_full) < recomb_events["delJ"]:
            continue

        V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
        J_seq = J_seq_full[recomb_events["delJ"] :]

        if (len(V_seq) + len(J_seq) + recomb_events["insVJ"]) % 3 != 0:
            continue

        insVJ_seq = seq_gen.rnd_ins_seq(recomb_events["insVJ"], sg.C_Rvj, sg.C_first_nt_bias_insVJ)

        ntseq = V_seq + insVJ_seq + J_seq
        aaseq = translate_bidi(ntseq)

        if "*" in aaseq:
            continue
        if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
            continue

        return aaseq, n


def _generate_counted_chunk(args: tuple) -> tuple[list[str], int]:
    """Worker: generate n sequences and return (seqs, n_total_rearrangements).

    n_total_rearrangements = M + K where M is productive sequences and K is
    rejected non-productive events.  Used as denominator for MC Pgen.
    """
    init_kwargs, n, seed = args
    model = OlgaModel(**init_kwargs, seed=None)
    np.random.seed(seed)
    sg = model.seq_gen_model
    _gen = _gen_one_counted_cdr3_vdj if model.is_d_present else _gen_one_counted_cdr3_vj
    seqs: list[str] = []
    total = 0
    for _ in range(n):
        seq, attempts = _gen(sg)
        seqs.append(seq)
        total += attempts
    return seqs, total


def _generate_pool_chunk(args: tuple) -> list[dict]:
    """Worker for :meth:`OlgaModel.generate_pool`.

    Rebuilds a fresh model in the child process (OLGA models are not
    picklable), seeds NumPy, generates *n* sequences, computes Pgen for each,
    and returns annotated dicts with ``log2_pgen``.

    Record keys: ``junction_aa``, ``junction``, ``v_gene``, ``j_gene``,
    ``v_end``, ``j_start``, ``log2_pgen``.
    """
    init_kwargs, n, seed = args
    model = OlgaModel(**init_kwargs, seed=None)
    np.random.seed(seed)
    _gen_one = (
        model._gen_one_vdj_with_meta
        if model.is_d_present
        else model._gen_one_vj_with_meta
    )
    result: list[dict] = []
    for _ in range(n):
        rec = _gen_one()
        p_raw = float(model.pgen_model.compute_aa_CDR3_pgen(rec["junction_aa"]))
        rec["log2_pgen"] = (
            math.log2(p_raw) if p_raw > 0 else float("-inf")
        )
        result.append(rec)
    return result


class OlgaModel:
    """Interface to an OLGA recombination model for generation-probability
    inference and sequence generation.

    Supports all nine built-in models (human TRA/TRB/TRG/TRD/IGH/IGK/IGL,
    mouse TRA/TRB).  A custom model directory can be provided via *model*.

    By default the numpy RNG is seeded with 42 on construction so that
    subsequent generation calls are deterministic.  Pass ``seed=None`` to
    preserve the caller's random state.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        locus: str = "TRB",
        species: str = "human",
        is_d_present: bool | None = None,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            model: Path to a directory containing an OLGA model.
                When ``None``, the built-in model for *locus*/*species* is used.
            locus: Receptor locus (TRA, TRB, TRG, TRD, IGH, IGK, IGL).
            species: Organism (``"human"`` or ``"mouse"``).
            is_d_present: Override D-gene flag; inferred from *locus* when ``None``.
            seed: Seed passed to ``numpy.random.seed`` after model loading.
                ``None`` leaves the numpy RNG state unchanged.
        """
        locus_u = locus.upper()
        species_l = species.lower()

        if model is None:
            try:
                olga_name = LOCUS_TO_OLGA_SUFFIX[locus_u]
            except KeyError:
                raise ValueError(
                    f"Unsupported locus={locus!r}. Supported: {sorted(LOCUS_TO_OLGA_SUFFIX)}"
                )
            model = get_resource_path(f"olga/default_models/{species_l}_{olga_name}")

        if is_d_present is None:
            is_d_present = locus_u in _D_PRESENT

        self.is_d_present: bool = is_d_present

        self._init_kwargs: dict = {
            "model": model,
            "locus": locus,
            "species": species,
            "is_d_present": is_d_present,
        }

        params_file_name    = f"{model}/model_params.txt"
        marginals_file_name = f"{model}/model_marginals.txt"
        V_anchor_pos_file   = f"{model}/V_gene_CDR3_anchors.csv"
        J_anchor_pos_file   = f"{model}/J_gene_CDR3_anchors.csv"

        if self.is_d_present:
            genomic_data     = load_model.GenomicDataVDJ()
            generative_model = load_model.GenerativeModelVDJ()
        else:
            genomic_data     = load_model.GenomicDataVJ()
            generative_model = load_model.GenerativeModelVJ()

        genomic_data.load_igor_genomic_data(
            params_file_name, V_anchor_pos_file, J_anchor_pos_file
        )
        generative_model.load_and_process_igor_model(marginals_file_name)

        self.v_names: list[str] = [x[0] for x in genomic_data.__dict__["genV"]]
        self.j_names: list[str] = [x[0] for x in genomic_data.__dict__["genJ"]]
        self.gen_model = generative_model

        if self.is_d_present:
            self.pgen_model    = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        else:
            self.pgen_model    = pgen.GenerationProbabilityVJ(generative_model, genomic_data)
            self.seq_gen_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)

        if seed is not None:
            np.random.seed(seed)

        # Persistent pool: created on first bulk call, reused across calls.
        self._pgen_pool: Pool | None = None
        self._pgen_pool_n_jobs: int = 0

    def close(self) -> None:
        """Terminate the persistent worker pool and release its resources.

        Call this explicitly when the model will no longer be used.  The
        pool is also closed on garbage collection, but explicit teardown is
        safer and avoids relying on CPython's reference-counting finalizer.
        """
        self._close_pool()

    def __enter__(self) -> "OlgaModel":
        return self

    def __exit__(self, *_: object) -> None:
        self._close_pool()

    def __del__(self) -> None:
        self._close_pool()

    def _close_pool(self) -> None:
        pool = getattr(self, "_pgen_pool", None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
            self._pgen_pool = None
            self._pgen_pool_n_jobs = 0

    def _get_pgen_pool(self, n_jobs: int) -> Pool:
        """Return a persistent pool of *n_jobs* workers, creating it if needed."""
        if self._pgen_pool is None or self._pgen_pool_n_jobs != n_jobs:
            self._close_pool()
            model_dir = self._init_kwargs["model"]
            self._pgen_pool = Pool(
                n_jobs,
                initializer=_init_pgen_worker,
                initargs=(model_dir, self.is_d_present),
            )
            self._pgen_pool_n_jobs = n_jobs
        return self._pgen_pool

    # ------------------------------------------------------------------
    # Pgen computation
    # ------------------------------------------------------------------

    def compute_pgen_junction(self, junction: str) -> float:
        """Return the generation probability for a nucleotide junction sequence.

        Args:
            junction: Nucleotide sequence (CDR3, including conserved C and F/W).
        """
        return self.pgen_model.compute_nt_CDR3_pgen(junction)

    def compute_pgen_junction_aa(self, junction_aa: str) -> float:
        """Return the generation probability for an amino-acid junction sequence.

        Args:
            junction_aa: Amino-acid sequence (CDR3).
        """
        return float(self.pgen_model.compute_aa_CDR3_pgen(junction_aa))

    def compute_pgen_junction_aa_bulk(
        self,
        junction_aas: Iterable[str],
        *,
        max_mismatches: int = 0,
        n_jobs: int = 1,
    ) -> list[float]:
        """Compute Pgen for many amino-acid junctions using a persistent process pool.

        Workers are initialised with the OLGA model once per pool lifetime and
        reused across subsequent calls on the same :class:`OlgaModel` instance,
        eliminating per-call model-loading overhead.

        Args:
            junction_aas: Iterable of CDR3 amino-acid sequences.
            max_mismatches: 0 for exact Pgen, 1 for Hamming-1 neighbourhood sum.
            n_jobs: Number of parallel worker processes.

        Returns:
            List of Pgen values in input order.
        """
        seqs = list(junction_aas)
        if not seqs:
            return []
        if max_mismatches not in {0, 1}:
            raise ValueError("max_mismatches must be 0 or 1")

        batch_fn = _compute_exact_batch if max_mismatches == 0 else _compute_1mm_batch

        if n_jobs <= 1 or len(seqs) < 64:
            m = self.pgen_model
            if max_mismatches == 0:
                return [float(m.compute_aa_CDR3_pgen(s)) for s in seqs]
            skip = _PGEN_1MM_SKIP_ENDS
            return [_compute_1mm_inner(m, s, skip) for s in seqs]

        chunk_size = max(1, ceil(len(seqs) / n_jobs))
        chunks = [seqs[i : i + chunk_size] for i in range(0, len(seqs), chunk_size)]

        pool = self._get_pgen_pool(n_jobs)
        results = pool.map(batch_fn, chunks)
        return [p for chunk in results for p in chunk]

    def compute_pgen_junction_aa_1mm(self, junction_aa: str) -> float:
        """Return the cumulative generation probability allowing one amino-acid mismatch.

        Args:
            junction_aa: Amino-acid sequence (CDR3).
        """
        return float(self.pgen_model.compute_hamming_dist_1_pgen(junction_aa, print_warnings=False))

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _sample_cdr3_aa(self) -> str:
        """Draw one productive CDR3 amino-acid sequence from the model."""
        result = self.seq_gen_model.gen_rnd_prod_CDR3()
        assert result is not None
        return result[1]

    def generate_sequences(self, n: int = 1000, seed: int | None = 42) -> list[str]:
        """Generate *n* productive CDR3 amino-acid sequences.

        Args:
            n: Number of sequences to generate.
            seed: Seed for ``numpy.random`` before sampling.  Pass ``None``
                to continue from the current RNG state.

        Returns:
            List of CDR3 amino-acid strings.
        """
        if seed is not None:
            np.random.seed(seed)
        return [self._sample_cdr3_aa() for _ in range(n)]

    def generate_sequences_parallel(
        self,
        n: int = 1000,
        n_jobs: int = 4,
        seed: int = 42,
    ) -> list[str]:
        """Generate *n* productive CDR3 sequences using *n_jobs* worker processes.

        Args:
            n: Total number of sequences to generate.
            n_jobs: Number of parallel worker processes.
            seed: Base seed; worker *i* uses ``seed + i``.

        Returns:
            List of CDR3 amino-acid strings (length == *n*).
        """
        if n_jobs <= 1:
            return self.generate_sequences(n, seed=seed)

        sizes = _split_n(n, n_jobs)
        args = [
            (self._init_kwargs, size, seed + i)
            for i, size in enumerate(sizes)
        ]
        with Pool(n_jobs) as pool:
            chunks = pool.map(_generate_chunk, args)
        return [seq for chunk in chunks for seq in chunk]

    def generate_sequences_counted(
        self,
        n: int = 1_000_000,
        n_jobs: int = 8,
        seed: int = 42,
    ) -> tuple[list[str], int]:
        """Generate *n* sequences and return ``(sequences, n_total_rearrangements)``.

        ``n_total_rearrangements`` = M + K where M is the number of productive
        sequences and K is the count of rejected non-productive recombination
        events.  ``n_total_rearrangements`` is the correct denominator for
        Monte-Carlo Pgen estimation::

            pgen_mc(seq) = n_matches_in_pool / n_total_rearrangements

        This makes ``pgen_mc`` directly comparable to OLGA analytical Pgen,
        which is defined over all rearrangements (productive *and* non-productive).

        Args:
            n: Number of productive sequences to generate.
            n_jobs: Worker processes for parallel generation.
            seed: Base seed; worker *i* uses ``seed + i``.

        Returns:
            Tuple ``(sequences, n_total_rearrangements)``.
        """
        if n_jobs <= 1:
            np.random.seed(seed)
            sg = self.seq_gen_model
            _gen = _gen_one_counted_cdr3_vdj if self.is_d_present else _gen_one_counted_cdr3_vj
            seqs: list[str] = []
            total = 0
            for _ in range(n):
                seq, attempts = _gen(sg)
                seqs.append(seq)
                total += attempts
            return seqs, total

        sizes = _split_n(n, n_jobs)
        args = [(self._init_kwargs, size, seed + i) for i, size in enumerate(sizes)]
        with Pool(n_jobs) as pool:
            chunks: list[tuple[list[str], int]] = pool.map(_generate_counted_chunk, args)
        seqs_out = [s for chunk_seqs, _ in chunks for s in chunk_seqs]
        n_total = sum(n_t for _, n_t in chunks)
        return seqs_out, n_total

    def generate_pool(
        self,
        n: int = 1_000_000,
        n_jobs: int = 4,
        seed: int = 42,
    ) -> list[dict]:
        """Generate *n* annotated sequences with log₂ Pgen, optionally in parallel.

        This is the primary entry-point for building a
        :class:`~mir.comparative.vdjbet.PgenBinPool`.

        Args:
            n: Total number of sequences to generate.
            n_jobs: Number of parallel worker processes.
            seed: Base RNG seed; worker *i* uses ``seed + i``.

        Returns:
            List of dicts with keys:
            ``junction_aa``, ``junction``, ``v_gene``, ``j_gene``,
            ``v_end``, ``j_start``, ``log2_pgen``.
        """
        if n_jobs <= 1:
            np.random.seed(seed)
            result: list[dict] = []
            _gen_one = (
                self._gen_one_vdj_with_meta
                if self.is_d_present
                else self._gen_one_vj_with_meta
            )
            for _ in range(n):
                rec = _gen_one()
                p_raw = float(self.pgen_model.compute_aa_CDR3_pgen(rec["junction_aa"]))
                rec["log2_pgen"] = (
                    math.log2(p_raw) if p_raw > 0 else float("-inf")
                )
                result.append(rec)
            return result

        sizes = _split_n(n, n_jobs)
        args = [(self._init_kwargs, size, seed + i) for i, size in enumerate(sizes)]
        with Pool(n_jobs) as pool:
            chunks = pool.map(_generate_pool_chunk, args)
        return [rec for chunk in chunks for rec in chunk]

    # ------------------------------------------------------------------
    # Annotated generation (junction_aa + V/J gene + optional Pgen)
    # ------------------------------------------------------------------

    def _gen_one_vdj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """Generate one productive VDJ recombination event with full annotation.

        Returns:
            Dict with keys: junction_aa, junction, v_gene, j_gene, v_end, j_start.
        """
        sg = self.seq_gen_model

        while True:
            recomb_events = sg.choose_random_recomb_events()

            V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
            if len(V_seq_full) <= max(recomb_events["delV"], 0):
                continue

            D_seq_full = sg.cutD_genomic_CDR3_segs[recomb_events["D"]]
            J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]

            if len(D_seq_full) < (recomb_events["delDl"] + recomb_events["delDr"]):
                continue
            if len(J_seq_full) < recomb_events["delJ"]:
                continue

            V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
            D_seq = D_seq_full[recomb_events["delDl"] : len(D_seq_full) - recomb_events["delDr"]]
            J_seq = J_seq_full[recomb_events["delJ"] :]

            if (
                len(V_seq) + len(D_seq) + len(J_seq)
                + recomb_events["insVD"] + recomb_events["insDJ"]
            ) % 3 != 0:
                continue

            insVD_seq = seq_gen.rnd_ins_seq(recomb_events["insVD"], sg.C_Rvd, sg.C_first_nt_bias_insVD)
            insDJ_seq = seq_gen.rnd_ins_seq(recomb_events["insDJ"], sg.C_Rdj, sg.C_first_nt_bias_insDJ)[::-1]

            ntseq = V_seq + insVD_seq + D_seq + insDJ_seq + J_seq
            aaseq = translate_bidi(ntseq)

            if "*" in aaseq:
                continue
            if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
                continue

            v_end   = len(V_seq) // 3
            j_start = len(aaseq) - len(J_seq) // 3 + 1

            return {
                "junction_aa": aaseq,
                "junction":    ntseq,
                "v_gene":      self.v_names[recomb_events["V"]],
                "j_gene":      self.j_names[recomb_events["J"]],
                "v_end":       v_end,
                "j_start":     j_start,
            }

    def _gen_one_vj_with_meta(self, conserved_J_residues: str = "FVW") -> dict:
        """Generate one productive VJ recombination event with full annotation.

        Returns:
            Dict with keys: junction_aa, junction, v_gene, j_gene, v_end, j_start.
        """
        sg = self.seq_gen_model

        while True:
            recomb_events = sg.choose_random_recomb_events()

            V_seq_full = sg.cutV_genomic_CDR3_segs[recomb_events["V"]]
            if len(V_seq_full) <= max(recomb_events["delV"], 0):
                continue

            J_seq_full = sg.cutJ_genomic_CDR3_segs[recomb_events["J"]]
            if len(J_seq_full) < recomb_events["delJ"]:
                continue

            V_seq = V_seq_full[: len(V_seq_full) - recomb_events["delV"]]
            J_seq = J_seq_full[recomb_events["delJ"] :]

            if (len(V_seq) + len(J_seq) + recomb_events["insVJ"]) % 3 != 0:
                continue

            insVJ_seq = seq_gen.rnd_ins_seq(recomb_events["insVJ"], sg.C_Rvj, sg.C_first_nt_bias_insVJ)

            ntseq = V_seq + insVJ_seq + J_seq
            aaseq = translate_bidi(ntseq)

            if "*" in aaseq:
                continue
            if aaseq[0] != "C" or aaseq[-1] not in conserved_J_residues:
                continue

            v_end   = len(V_seq) // 3
            j_start = len(aaseq) - len(J_seq) // 3 + 1

            return {
                "junction_aa": aaseq,
                "junction":    ntseq,
                "v_gene":      self.v_names[recomb_events["V"]],
                "j_gene":      self.j_names[recomb_events["J"]],
                "v_end":       v_end,
                "j_start":     j_start,
            }

    def generate_sequences_with_meta(
        self,
        n: int = 1000,
        pgens: bool = True,
        seed: int | None = 42,
        pgen_adjustment=None,
    ) -> list[dict]:
        """Generate *n* productive sequences with full annotation.

        Args:
            n: Number of sequences to generate.
            pgens: Whether to compute and attach generation probabilities.
            seed: Seed for ``numpy.random`` before sampling.
            pgen_adjustment: Optional :class:`PgenGeneUsageAdjustment`.

        Returns:
            List of annotation dicts, one per generated sequence.
        """
        if seed is not None:
            np.random.seed(seed)
        _gen_one = self._gen_one_vdj_with_meta if self.is_d_present else self._gen_one_vj_with_meta
        locus = self._init_kwargs.get("locus", "")
        res = []
        for _ in range(n):
            rec = _gen_one()
            if pgens:
                p_raw = float(self.pgen_model.compute_aa_CDR3_pgen(rec["junction_aa"]))
                rec["pgen_raw"] = p_raw
                if pgen_adjustment is not None and p_raw > 0:
                    p_adj = pgen_adjustment.adjust_pgen(locus, rec["v_gene"], rec["j_gene"], p_raw)
                    rec["pgen"] = math.log10(p_adj) if p_adj > 0 else float("-inf")
                else:
                    rec["pgen"] = math.log10(p_raw) if p_raw > 0 else float("-inf")
            res.append(rec)
        return res

    def compute_usage_cache(
        self,
        n: int = 100_000,
        *,
        seed: int = 42,
        n_jobs: int = 1,
    ) -> "GeneUsage":
        """Estimate V-J gene usage by a cold run of *n* OLGA samples.

        Args:
            n: Number of sequences to sample.
            seed: numpy RNG seed.
            n_jobs: Number of worker processes for synthetic generation.

        Returns:
            GeneUsage with clonotype counts per V-J pair for this locus.
        """
        from mir.basic.gene_usage import GeneUsage

        if int(n_jobs) > 1:
            records = self.generate_pool(n=n, n_jobs=int(n_jobs), seed=seed)
        else:
            records = self.generate_sequences_with_meta(n, pgens=False, seed=seed)
        locus = self._init_kwargs.get("locus", "")
        usage_df = pl.from_dicts(
            [{"v_gene": str(r.get("v_gene", "")), "j_gene": str(r.get("j_gene", ""))} for r in records]
        )
        return GeneUsage.from_dataframe(usage_df, locus=locus)


# ---------------------------------------------------------------------------
# Pgen gene-usage adjustment (importance sampling)
# ---------------------------------------------------------------------------

class PgenGeneUsageAdjustment:
    """Adjusts OLGA generation probabilities to match a target V-J gene usage.

    For each generated sequence with genes ``(v, j)``, multiplies its Pgen
    by::

        target_vj_fraction(v, j) / olga_vj_fraction(v, j)

    Parameters
    ----------
    target:
        :class:`~mir.basic.gene_usage.GeneUsage` describing the target gene
        usage (e.g. computed from a real sample).
    cache_size:
        Number of OLGA sequences used to estimate the model's native gene
        usage.
    seed:
        RNG seed for the OLGA cache run.
    """

    def __init__(
        self,
        target: "GeneUsage",
        *,
        cache_size: int = 100_000,
        seed: int = 42,
        olga_n_jobs: int = 1,
        count: str = "count_rearrangement",
        pseudocount: float = 1.0,
        reference: "GeneUsage | None" = None,
    ) -> None:
        from mir.basic.gene_usage import GeneUsage as _GeneUsage

        self._target = target
        self._cache_size = cache_size
        self._seed = seed
        self._olga_n_jobs = max(1, int(olga_n_jobs))
        self._count = count
        self._pseudocount = pseudocount
        self._olga_cache: dict[str, _GeneUsage] = {}
        if reference is not None:
            for locus in reference.loci:
                self._olga_cache[locus] = reference
        self._factor_cache: dict[str, dict[tuple[str, str], float]] = {}

    def _get_olga_cache(self, locus: str):
        if locus not in self._olga_cache:
            model = OlgaModel(locus=locus, seed=self._seed)
            self._olga_cache[locus] = model.compute_usage_cache(
                self._cache_size,
                seed=self._seed,
                n_jobs=self._olga_n_jobs,
            )
        return self._olga_cache[locus]

    def _get_factor_cache(self, locus: str) -> dict[tuple[str, str], float]:
        if locus not in self._factor_cache:
            olga = self._get_olga_cache(locus)
            factors = self._target.correction_factors(
                olga,
                locus,
                scope="vj",
                count=self._count,
                pseudocount=self._pseudocount,
            )
            self._factor_cache[locus] = {
                key: float(val)
                for key, val in factors.items()
                if isinstance(key, tuple) and len(key) == 2
            }
        return self._factor_cache[locus]

    def factor(self, locus: str, v: str, j: str) -> float:
        """Pgen adjustment factor for (locus, v, j).

        Args:
            locus: IMGT locus code (e.g. ``"TRB"``).
            v: V-gene name.
            j: J-gene name.
        """
        v_base = v.split("*")[0]
        j_base = j.split("*")[0]
        pair = (v_base, j_base)
        return self._get_factor_cache(locus).get(pair, 1.0)

    def adjust_pgen(self, locus: str, v: str, j: str, pgen: float) -> float:
        """Return ``pgen * factor(locus, v, j)``.

        Args:
            locus: IMGT locus code.
            v: V-gene name.
            j: J-gene name.
            pgen: Raw generation probability (linear, not log).
        """
        return pgen * self.factor(locus, v, j)


_GENE_USAGE_PROB_CACHE: dict[tuple[str, str, int], dict[str, dict[object, float]]] = {}


# ---------------------------------------------------------------------------
# Monte-Carlo Pgen pool
# ---------------------------------------------------------------------------

_AAS = "ACDEFGHIKLMNPQRSTVWY"


class McPgenPool:
    """Monte-Carlo generation-probability pool backed by tcrtrie.

    Stores a large set of productive CDR3 sequences (synthetic or real) and
    answers Pgen queries by counting exact or Hamming-1 matches.

    **Synthetic pools** (built via :meth:`build_synthetic`) track the total
    number of recombination attempts M + K, where M is the productive count
    and K is the number of rejected non-productive events.  Using M + K as
    the denominator makes ``pgen_mc`` directly comparable to OLGA analytical
    Pgen, which is defined over all rearrangements::

        pgen_mc(seq) = n_matches / n_total_rearrangements

    **Real-repertoire pools** (built via :meth:`build_real`) use the control
    size as the denominator.  The resulting ``pgen_mc`` estimates the
    probability of *observing* a sequence in that specific individual's
    repertoire, which includes thymic selection effects (Q-factor).

    Relationship to TCRNET
    ----------------------
    TCRNET counts Hamming-1 neighbors in a control repertoire and reports a
    binomial enrichment p-value.  :class:`McPgenPool` with a large synthetic
    control is the same neighbor-count operation, but converts the count to
    an absolute Pgen estimate by dividing by the pool size.  ALICE =
    TCRNET with a large synthetic background plus analytical Pgen fallback
    for sparse sequences.
    """

    def __init__(
        self,
        sequences: list[str],
        n_total: int,
        *,
        skip_ends: int = 2,
        locus: str = "TRB",
        species: str = "human",
    ) -> None:
        from collections import Counter
        try:
            from tcrtrie import Trie as _Trie
        except ImportError as exc:  # pragma: no cover
            raise ImportError("tcrtrie is required for McPgenPool") from exc

        self.n_productive: int = len(sequences)
        self.n_total: int = n_total
        self.skip_ends: int = skip_ends
        self.locus: str = locus
        self.species: str = species
        self.p_productive: float = self.n_productive / max(1, self.n_total)

        self._counter: Counter = Counter(sequences)
        self._unique_seqs: list[str] = list(self._counter.keys())
        n_u = len(self._unique_seqs)
        self._trie = _Trie(
            sequences=self._unique_seqs,
            vGenes=[""] * n_u,
            jGenes=[""] * n_u,
        )

    # ------------------------------------------------------------------
    # Single-sequence queries
    # ------------------------------------------------------------------

    def pgen_exact(self, seq: str) -> float:
        """Estimate exact-match Pgen for *seq*."""
        return self._counter.get(seq, 0) / self.n_total

    def pgen_1mm(self, seq: str) -> float:
        """Estimate 1-mismatch Pgen for *seq* (inner positions only)."""
        return self.pgen_1mm_bulk([seq])[0]

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def pgen_exact_bulk(self, seqs: list[str]) -> list[float]:
        """Bulk exact-match Pgen estimation.

        Args:
            seqs: CDR3 amino-acid sequences.

        Returns:
            Pgen estimates in input order.
        """
        inv = 1.0 / self.n_total
        return [self._counter.get(s, 0) * inv for s in seqs]

    def pgen_1mm_bulk(
        self,
        seqs: list[str],
        n_jobs: int = 1,
    ) -> list[float]:
        """Bulk Hamming-1 Pgen estimation via tcrtrie + inner-position filter.

        Searches the pool for sequences within Hamming distance 1 of each
        query, keeps only matches where the mismatch falls inside the inner
        window (positions ``[skip_ends, L - skip_ends)``), and sums
        pool-sequence counts.

        Args:
            seqs: CDR3 amino-acid sequences.
            n_jobs: Thread count passed to tcrtrie's batch search.

        Returns:
            Pgen estimates in input order.
        """
        if not seqs:
            return []

        inv = 1.0 / self.n_total
        skip = self.skip_ends
        counter = self._counter
        unique_seqs = self._unique_seqs

        # Batched Hamming-1 search (substitutions only; no indels)
        all_hits = self._trie.SearchIndicesForAll(
            seqs,
            maxSubstitution=1,
            maxInsertion=0,
            maxDeletion=0,
            numThreads=max(1, n_jobs),
        )

        results: list[float] = []
        for seq, hits in zip(seqs, all_hits):
            total = 0
            L = len(seq)
            for idx, dist in hits:
                neighbor = unique_seqs[idx]
                cnt = counter[neighbor]
                if dist == 0:
                    total += cnt
                else:
                    # Find mismatch position; keep only inner-window mismatches
                    for i, (a, b) in enumerate(zip(seq, neighbor)):
                        if a != b:
                            if skip <= i < L - skip:
                                total += cnt
                            break
            results.append(total * inv)
        return results

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def build_synthetic(
        cls,
        n: int = 10_000_000,
        *,
        locus: str = "TRB",
        species: str = "human",
        n_jobs: int = 8,
        seed: int = 42,
        skip_ends: int = 2,
    ) -> "McPgenPool":
        """Build a synthetic Pgen pool, tracking non-productive rejections.

        The pool is generated by OLGA and the attempt counter tracks all
        recombination events (productive + non-productive), giving the correct
        denominator for Pgen estimation on the OLGA scale.

        Args:
            n: Number of productive sequences to generate.
            locus: Receptor locus (e.g. ``"TRB"``).
            species: ``"human"`` or ``"mouse"``.
            n_jobs: Worker processes for generation.
            seed: Base RNG seed.
            skip_ends: Terminal positions to skip for 1mm Pgen (default 2).

        Returns:
            Fully constructed :class:`McPgenPool`.
        """
        model = OlgaModel(locus=locus, species=species, seed=None)
        seqs, n_total = model.generate_sequences_counted(n, n_jobs=n_jobs, seed=seed)
        model.close()
        return cls(seqs, n_total, skip_ends=skip_ends, locus=locus, species=species)

    @classmethod
    def build_real(
        cls,
        sequences: list[str],
        *,
        skip_ends: int = 2,
        locus: str = "TRB",
        species: str = "human",
    ) -> "McPgenPool":
        """Build a pool from real-repertoire sequences.

        For real controls ``n_total = len(sequences)`` (no non-productive
        correction).  Resulting Pgen estimates include thymic selection effects
        (Q-factor); use alongside OLGA analytical Pgen to measure Q.

        Args:
            sequences: CDR3 amino-acid strings from the control repertoire.
            skip_ends: Terminal positions to skip for 1mm Pgen.
            locus: Receptor locus.
            species: Organism.
        """
        return cls(sequences, len(sequences), skip_ends=skip_ends, locus=locus, species=species)


_MC_POOL_CACHE: dict[tuple[str, str, int, int, int], McPgenPool] = {}


def get_or_build_mc_pool(
    *,
    locus: str = "TRB",
    species: str = "human",
    n: int = 10_000_000,
    seed: int = 42,
    skip_ends: int = 2,
    n_jobs: int = 8,
) -> McPgenPool:
    """Return a cached synthetic :class:`McPgenPool`, building it if necessary.

    Pool is keyed by ``(locus, species, n, seed, skip_ends)`` and cached
    in-process for the session lifetime.  Subsequent calls with the same
    parameters return the same pool instantly.

    Args:
        locus: Receptor locus (e.g. ``"TRB"``).
        species: ``"human"`` or ``"mouse"``.
        n: Pool size (productive sequences).
        seed: OLGA generation seed.
        skip_ends: Terminal positions to skip for 1mm Pgen.
        n_jobs: Workers used *only* when the pool must be built.

    Returns:
        Cached or newly built :class:`McPgenPool`.
    """
    key = (locus, species, n, seed, skip_ends)
    pool = _MC_POOL_CACHE.get(key)
    if pool is None:
        pool = McPgenPool.build_synthetic(
            n, locus=locus, species=species, n_jobs=n_jobs, seed=seed, skip_ends=skip_ends,
        )
        _MC_POOL_CACHE[key] = pool
    return pool


def clear_mc_pool_cache() -> None:
    """Clear the in-process MC pool cache, freeing memory."""
    _MC_POOL_CACHE.clear()


def compute_gene_usage_probabilities_from_control_df(
    control_df: "pl.DataFrame",
) -> dict[str, dict[object, float]]:
    """Estimate OLGA V/J/VJ probabilities from a synthetic control table."""
    from mir.common.alleles import allele_to_major

    required = {"v_gene", "j_gene"}
    missing = required.difference(control_df.columns)
    if missing:
        raise ValueError(f"control_df missing required columns: {sorted(missing)}")

    df = (
        control_df.select(["v_gene", "j_gene"])
        .with_columns([
            pl.col("v_gene").cast(pl.Utf8).map_elements(
                lambda x: allele_to_major(str(x or "")), return_dtype=pl.Utf8
            ),
            pl.col("j_gene").cast(pl.Utf8).map_elements(
                lambda x: allele_to_major(str(x or "")), return_dtype=pl.Utf8
            ),
        ])
        .filter((pl.col("v_gene") != "") & (pl.col("j_gene") != ""))
    )
    total = len(df)
    if total == 0:
        return {"v": {}, "j": {}, "vj": {}}

    v_vc = df["v_gene"].value_counts(sort=False)
    p_v = {row[0]: float(row[1]) / total for row in v_vc.iter_rows()}

    j_vc = df["j_gene"].value_counts(sort=False)
    p_j = {row[0]: float(row[1]) / total for row in j_vc.iter_rows()}

    vj_counts = df.group_by(["v_gene", "j_gene"]).agg(pl.len().alias("count"))
    p_vj = {(row[0], row[1]): float(row[2]) / total for row in vj_counts.iter_rows()}

    return {"v": p_v, "j": p_j, "vj": p_vj}


def get_olga_gene_usage_probabilities(
    *,
    species: str,
    locus: str,
    synthetic_n: int,
    n_jobs: int | None = None,
    control_manager=None,
    control_kwargs: dict | None = None,
) -> dict[str, dict[object, float]]:
    """Load or compute cached OLGA V/J/VJ probabilities for a locus."""
    from mir.common.control import ControlManager

    manager = control_manager or ControlManager()
    kwargs = dict(control_kwargs or {})
    kwargs.setdefault("n", int(synthetic_n))
    if n_jobs is not None:
        kwargs.setdefault("n_jobs", max(1, int(n_jobs)))
    kwargs.setdefault("progress", False)
    control_df = manager.ensure_and_load_control_df(
        "synthetic",
        species,
        locus,
        **kwargs,
    )
    return compute_gene_usage_probabilities_from_control_df(control_df)
