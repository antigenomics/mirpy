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
import pandas as pd

from mir import get_resource_path
from mir.basic.aliases import LOCUS_TO_OLGA_SUFFIX
from mir.basic import mirseq as _mirseq

if TYPE_CHECKING:
    from mir.basic.gene_usage import GeneUsage

translate_bidi = _mirseq.translate_bidi

# Loci that have a D gene segment in their recombination model.
_D_PRESENT: frozenset[str] = frozenset({"TRB", "TRD", "IGH"})


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


def _compute_1mm_batch(seqs: list[str]) -> list[float]:
    """Compute 1-mismatch Pgen for a batch in a pool worker."""
    m = _WORKER_PGEN_MODEL
    return [float(m.compute_hamming_dist_1_pgen(s, print_warnings=False)) for s in seqs]


# ---------------------------------------------------------------------------
# Workers for sequence generation (still reconstruct the full OlgaModel)
# ---------------------------------------------------------------------------

def _generate_chunk(args: tuple) -> list[str]:
    """Worker: generate CDR3 aa sequences (no Pgen) for generate_sequences_parallel."""
    init_kwargs, n, seed = args
    model = OlgaModel(**init_kwargs, seed=None)
    np.random.seed(seed)
    return [model._sample_cdr3_aa() for _ in range(n)]


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
            # Use this instance's model directly in the main process.
            m = self.pgen_model
            if max_mismatches == 0:
                return [float(m.compute_aa_CDR3_pgen(s)) for s in seqs]
            return [float(m.compute_hamming_dist_1_pgen(s, print_warnings=False)) for s in seqs]

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
        usage_df = pd.DataFrame.from_records(records, columns=["v_gene", "j_gene"])
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


def compute_gene_usage_probabilities_from_control_df(
    control_df: pd.DataFrame,
) -> dict[str, dict[object, float]]:
    """Estimate OLGA V/J/VJ probabilities from a synthetic control table."""
    from mir.common.alleles import allele_to_major

    required = {"v_gene", "j_gene"}
    missing = required.difference(control_df.columns)
    if missing:
        raise ValueError(f"control_df missing required columns: {sorted(missing)}")

    df = control_df.loc[:, ["v_gene", "j_gene"]].copy()
    df["v_gene"] = df["v_gene"].map(lambda x: allele_to_major(str(x or "")))
    df["j_gene"] = df["j_gene"].map(lambda x: allele_to_major(str(x or "")))
    df = df[(df["v_gene"] != "") & (df["j_gene"] != "")]
    total = len(df)
    if total == 0:
        return {"v": {}, "j": {}, "vj": {}}

    p_v = (df["v_gene"].value_counts(sort=False) / total).to_dict()
    p_j = (df["j_gene"].value_counts(sort=False) / total).to_dict()
    p_vj = (df.groupby(["v_gene", "j_gene"], sort=False).size() / total).to_dict()

    return {
        "v": {k: float(v) for k, v in p_v.items()},
        "j": {k: float(v) for k, v in p_j.items()},
        "vj": {k: float(v) for k, v in p_vj.items()},
    }


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
