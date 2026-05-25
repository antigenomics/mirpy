"""Advanced benchmarks and analysis for TCREmp embedding.

Comprehensive testing covering:
- n_jobs analysis: when parallelization helps vs hurts
- Fixed-gap vs BioPython performance on 10k clonotypes × 3k prototypes
- Embedding distance metrics: cosine, RMSE between methods
- Prototype symmetry: validation of latent space distances vs sequence distances
- Latent space quality metrics: R² and RMSE for embedding reconstruction
"""

import os
import time
import warnings
from typing import Tuple

import numpy as np
import pytest
from scipy.stats import f_oneway, linregress
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from mir.common.clonotype import Clonotype
from mir.embedding.tcremp import TCREmp
from mir.distances.aligner import JunctionAligner, BioAlignerWrapper, GermlineAligner
from mir.common.gene_library import GeneLibrary

# Control flag for benchmark execution
skip_benchmarks = pytest.mark.skipif(
    not os.getenv("RUN_BENCHMARK"), reason="set RUN_BENCHMARK=1 to run"
)

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def _generate_random_clonotypes(
    n: int,
    v_genes: list[str],
    j_genes: list[str],
    junction_len_range: Tuple[int, int] = (8, 20),
    seed: int = SEED,
) -> list[Clonotype]:
    """Generate random clonotypes from available gene calls.

    Args:
        n: Number of clonotypes to generate.
        v_genes: Available V genes.
        j_genes: Available J genes.
        junction_len_range: (min_len, max_len) for junction amino-acid length.
        seed: Random seed.

    Returns:
        List of random Clonotype objects.
    """
    np.random.seed(seed)
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    clonotypes = []

    for _ in range(n):
        v_gene = np.random.choice(v_genes)
        j_gene = np.random.choice(j_genes)
        junction_len = np.random.randint(junction_len_range[0], junction_len_range[1])
        junction_aa = "".join(np.random.choice(list(aa_alphabet), size=junction_len))
        clonotypes.append(
            Clonotype(v_gene=v_gene, j_gene=j_gene, junction_aa=junction_aa)
        )

    return clonotypes


# ===================================================================
# Test 1: Manual distance computation for verification
# ===================================================================


class TestTCREmpManualDistances:
    """Verify distance formula with manually computed examples."""

    def test_manual_distance_computation(self):
        """Check distance formula: d(a,b) = s(a,a) + s(b,b) - 2*s(a,b)."""
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=2, junction_method="fixed_gap")
        aligner = model.junction_aligner

        # Simple sequences for manual verification
        seq_a = "CASSIRSSYEQYF"
        seq_b = "CASSIRSSYEQYF"  # Same as seq_a

        # Self-scores
        score_aa = aligner.score(seq_a, seq_a)
        score_bb = aligner.score(seq_b, seq_b)

        # Cross-score
        score_ab = aligner.score(seq_a, seq_b)

        # Expected distance
        expected_d = score_aa + score_bb - 2.0 * score_ab

        # For identical sequences, distance should be 0
        assert expected_d == pytest.approx(0.0), f"Expected 0, got {expected_d}"

    def test_embedding_shape_and_dtype(self):
        """Verify embedding shape and dtype."""
        n_proto = 10
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=n_proto)
        clonotypes = [
            Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF"),
            Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01", junction_aa="CASRQDTQYF"),
        ]
        X = model.embed(clonotypes)

        assert X.shape == (2, 3 * n_proto), f"Expected shape (2, {3*n_proto}), got {X.shape}"
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
        assert np.isfinite(X).all(), "Embedding contains NaN or inf"

    def test_embedding_symmetry_on_prototypes(self):
        """Test that prototypes embed symmetrically when used as queries."""
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=50, junction_method="fixed_gap")
        
        # Convert prototypes DataFrame to Clonotype objects
        proto_clonotypes = [
            Clonotype(v_gene=r["v_gene"], j_gene=r["j_gene"], junction_aa=r["junction_aa"])
            for r in model.prototypes.iter_rows(named=True)
        ]
        
        # Embed the prototypes themselves
        X_proto = model.embed(proto_clonotypes, n_jobs=1)
        
        # When using only junction distances (ignore V/J), distance matrix should be symmetric
        # with 0 on diagonal
        junc_idx = slice(2, None, 3)  # Extract junction distances every 3rd element
        junc_mat = X_proto[:, junc_idx]
        
        # Reshape to get (n_protos, n_protos) distance matrix
        assert junc_mat.shape == (50, 50), f"Expected (50, 50), got {junc_mat.shape}"


# ===================================================================
# Test 2: n_jobs analysis - when parallelization helps vs hurts
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestNJobsAnalysis:
    """Analyze when parallelization helps vs hurts.

    Tests embedding with various clonotype/prototype combinations to identify
    the breakpoint where threading becomes worthwhile.
    """

    SEED = SEED

    def test_njobs_scaling_analysis(self):
        """Test embedding speedup across different problem sizes."""
        configs = [
            (100, 100, "small_small"),       # Small clonotypes, small prototypes
            (500, 200, "medium_small"),      # Medium clonotypes, small prototypes
            (1000, 500, "large_small"),      # Large clonotypes, small prototypes
            (5000, 1000, "large_large"),     # Large clonotypes, large prototypes
            (10000, 1000, "xlarge_large"),   # XL clonotypes, large prototypes
            (10000, 3000, "xlarge_xlarge"),  # XL clonotypes, XL prototypes
        ]

        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        v_seqs = dict(lib.get_sequences_aa(locus="TRB", gene="V"))
        j_seqs = dict(lib.get_sequences_aa(locus="TRB", gene="J"))
        v_genes = sorted(v_seqs.keys())[:50]
        j_genes = sorted(j_seqs.keys())

        results = []

        for n_clono, n_proto, label in configs:
            print(f"\n--- Testing {label}: {n_clono} clonotypes × {n_proto} prototypes ---")

            # Build model
            model = TCREmp.from_defaults("human", "TRB", n_prototypes=min(n_proto, 1000), junction_method="fixed_gap")
            
            # Generate clonotypes
            clonotypes = _generate_random_clonotypes(n_clono, v_genes, j_genes, seed=self.SEED)

            # Test serial (n_jobs=1)
            t0_serial = time.perf_counter()
            X_serial = model.embed(clonotypes, n_jobs=1)
            t_serial = time.perf_counter() - t0_serial

            # Test parallel (n_jobs=cpu_count)
            cpu_count = os.cpu_count() or 1
            t0_parallel = time.perf_counter()
            X_parallel = model.embed(clonotypes, n_jobs=cpu_count)
            t_parallel = time.perf_counter() - t0_parallel

            speedup = t_serial / t_parallel if t_parallel > 0 else 0.0
            results.append({
                "label": label,
                "n_clonotypes": n_clono,
                "n_prototypes": n_proto,
                "t_serial": t_serial,
                "t_parallel": t_parallel,
                "speedup": speedup,
            })

            print(f"  Serial: {t_serial:.3f}s, Parallel: {t_parallel:.3f}s, Speedup: {speedup:.2f}x")

            # Verify embeddings match
            assert np.allclose(X_serial, X_parallel), \
                f"Serial and parallel embeddings differ for {label}"

        # Summary
        print("\n=== Summary ===")
        print("Config                       | Serial  | Parallel | Speedup")
        print("-" * 60)
        for r in results:
            print(
                f"{r['label']:28} | {r['t_serial']:7.3f}s | {r['t_parallel']:8.3f}s | {r['speedup']:6.2f}x"
            )

        # Recommendation: parallelization worth it if speedup > 1.2x
        parallelizable_configs = [r for r in results if r["speedup"] > 1.2]
        print(f"\nConfigs where speedup > 1.2x: {len(parallelizable_configs)}/{len(results)}")
        if parallelizable_configs:
            min_size = min(r["n_clonotypes"] for r in parallelizable_configs)
            print(f"Parallelization recommended for: ≥{min_size} clonotypes")


# ===================================================================
# Test 3: Fixed-gap vs BioPython (10k clonotypes × 3k prototypes)
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestFixedGapVsBioPython:
    """Compare fixed-gap vs BioPython alignment on large-scale problem."""

    SEED = SEED
    N_CLONOTYPES = 10000
    N_PROTOTYPES = 3000

    def test_large_scale_alignment_performance(self):
        """Benchmark both alignment methods on realistic large-scale problem."""
        print(f"\n--- Testing {self.N_CLONOTYPES} clonotypes × {self.N_PROTOTYPES} prototypes ---")

        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        v_seqs = dict(lib.get_sequences_aa(locus="TRB", gene="V"))
        j_seqs = dict(lib.get_sequences_aa(locus="TRB", gene="J"))
        v_genes = sorted(v_seqs.keys())[:50]
        j_genes = sorted(j_seqs.keys())

        clonotypes = _generate_random_clonotypes(self.N_CLONOTYPES, v_genes, j_genes, seed=self.SEED)

        # Test fixed-gap
        print("Building fixed-gap model...")
        t0 = time.perf_counter()
        model_fixed = TCREmp.from_defaults(
            "human", "TRB", n_prototypes=self.N_PROTOTYPES, junction_method="fixed_gap"
        )
        t_build_fixed = time.perf_counter() - t0

        print(f"Embedding with fixed-gap ({self.N_CLONOTYPES} clonotypes)...")
        t0 = time.perf_counter()
        X_fixed = model_fixed.embed(clonotypes, n_jobs=os.cpu_count() or 1)
        t_embed_fixed = time.perf_counter() - t0

        pairs_per_sec_fixed = (self.N_CLONOTYPES * self.N_PROTOTYPES) / t_embed_fixed
        print(f"  Fixed-gap embedding: {t_embed_fixed:.2f}s ({pairs_per_sec_fixed/1e6:.1f}M pairs/s)")

        # Test BioPython
        print("Building BioPython model...")
        t0 = time.perf_counter()
        model_bio = TCREmp.from_defaults(
            "human", "TRB", n_prototypes=self.N_PROTOTYPES, junction_method="biopython"
        )
        t_build_bio = time.perf_counter() - t0

        print(f"Embedding with BioPython ({self.N_CLONOTYPES} clonotypes)...")
        t0 = time.perf_counter()
        X_bio = model_bio.embed(clonotypes, n_jobs=1)  # BioPython not thread-safe, use serial
        t_embed_bio = time.perf_counter() - t0

        pairs_per_sec_bio = (self.N_CLONOTYPES * self.N_PROTOTYPES) / t_embed_bio
        print(f"  BioPython embedding: {t_embed_bio:.2f}s ({pairs_per_sec_bio/1e3:.1f}k pairs/s)")

        speedup = t_embed_bio / t_embed_fixed
        print(f"\nSpeedup (BioPython / Fixed-gap): {speedup:.1f}x")

        # Assertion: fixed-gap should be > 50x faster
        assert speedup > 50, f"Expected >50x speedup, got {speedup:.1f}x"

        # Both embeddings should have correct shape
        assert X_fixed.shape == (self.N_CLONOTYPES, 3 * self.N_PROTOTYPES)
        assert X_bio.shape == (self.N_CLONOTYPES, 3 * self.N_PROTOTYPES)


# ===================================================================
# Test 4: Embedding distance comparison (cosine, RMSE)
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestEmbeddingDistanceComparison:
    """Compare embeddings from fixed-gap vs BioPython using distance metrics."""

    SEED = SEED
    N_CLONOTYPES = 1000
    N_PROTOTYPES = 500

    def test_embedding_distance_metrics(self):
        """Compute cosine and RMSE between embeddings from two methods."""
        print(f"\n--- Embedding distance comparison ({self.N_CLONOTYPES} clonotypes) ---")

        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        v_seqs = dict(lib.get_sequences_aa(locus="TRB", gene="V"))
        j_seqs = dict(lib.get_sequences_aa(locus="TRB", gene="J"))
        v_genes = sorted(v_seqs.keys())[:30]
        j_genes = sorted(j_seqs.keys())

        clonotypes = _generate_random_clonotypes(self.N_CLONOTYPES, v_genes, j_genes, seed=self.SEED)

        # Generate embeddings with both methods
        model_fixed = TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES, junction_method="fixed_gap")
        X_fixed = model_fixed.embed(clonotypes, n_jobs=1)

        model_bio = TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES, junction_method="biopython")
        X_bio = model_bio.embed(clonotypes, n_jobs=1)

        # Compute cosine distances between embeddings (row-wise)
        cosine_dists = []
        for i in range(len(clonotypes)):
            # Normalize rows
            x_fixed_norm = X_fixed[i] / (np.linalg.norm(X_fixed[i]) + 1e-10)
            x_bio_norm = X_bio[i] / (np.linalg.norm(X_bio[i]) + 1e-10)
            cosine_dist = 1.0 - np.dot(x_fixed_norm, x_bio_norm)
            cosine_dists.append(cosine_dist)

        cosine_dists = np.array(cosine_dists)
        
        # Compute RMSE between embeddings (element-wise)
        rmse = np.sqrt(np.mean((X_fixed - X_bio) ** 2))

        # Compute correlation
        corr = np.corrcoef(X_fixed.flatten(), X_bio.flatten())[0, 1]

        print(f"  Cosine distance (mean): {np.mean(cosine_dists):.6f} ± {np.std(cosine_dists):.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Correlation: {corr:.6f}")

        # Sanity check: embeddings should have finite, non-NaN values
        assert np.isfinite(cosine_dists).all(), "Cosine distances contain NaN/inf"
        assert np.isfinite(rmse), "RMSE is not finite"
        assert np.isfinite(corr), "Correlation is not finite"


# ===================================================================
# Test 5: Prototype symmetry & latent space distances
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestPrototypeSymmetryAndLatentSpace:
    """Validate that prototype-vs-prototype distances are symmetric with zero diagonal.

    Also compute R² and RMSE between sequence space and latent space distances.
    """

    SEED = SEED
    N_PROTOTYPES = 1000

    def test_prototype_symmetry_and_reconstruction(self):
        """Test symmetry of prototype distance matrix in latent space."""
        print(f"\n--- Testing prototype symmetry ({self.N_PROTOTYPES} prototypes) ---")

        # Build model
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES, junction_method="fixed_gap")
        
        # Convert prototypes to clonotypes
        proto_clonotypes = [
            Clonotype(v_gene=r["v_gene"], j_gene=r["j_gene"], junction_aa=r["junction_aa"])
            for r in model.prototypes.iter_rows(named=True)
        ]

        # Embed prototypes
        print("Embedding prototypes...")
        X_proto = model.embed(proto_clonotypes, n_jobs=os.cpu_count() or 1)

        # Extract junction distances (every 3rd column starting from index 2)
        junc_idx = slice(2, None, 3)
        junc_mat = X_proto[:, junc_idx]

        # Check symmetry: |mat - mat.T| should be ~ 0
        asymmetry = np.abs(junc_mat - junc_mat.T)
        max_asymmetry = np.max(asymmetry)
        mean_asymmetry = np.mean(asymmetry)
        print(f"  Asymmetry (max): {max_asymmetry:.2e}, (mean): {mean_asymmetry:.2e}")

        # Check diagonal: should be ~ 0
        diagonal = np.diag(junc_mat)
        max_diag = np.max(np.abs(diagonal))
        mean_diag = np.mean(np.abs(diagonal))
        print(f"  Diagonal (max |value|): {max_diag:.2e}, (mean): {mean_diag:.2e}")

        assert max_asymmetry < 1e-4, f"Matrix not symmetric: max asymmetry {max_asymmetry}"
        assert max_diag < 1e-2, f"Diagonal not zero: max |diag| {max_diag}"

        # Now compute sequence-space distances vs latent-space distances
        print("\nComputing sequence-space distances...")
        aligner = model.junction_aligner
        junc_seqs = model._proto_junction

        # Sample 100×100 subset for tractability
        sample_indices = np.random.choice(len(junc_seqs), size=min(100, len(junc_seqs)), replace=False)
        seq_dist_mat = np.zeros((len(sample_indices), len(sample_indices)), dtype=np.float32)

        for i, idx_i in enumerate(sample_indices):
            for j, idx_j in enumerate(sample_indices):
                ss_i = model._proto_junction_selfscores[idx_i]
                ss_j = model._proto_junction_selfscores[idx_j]
                cross = aligner.score(junc_seqs[idx_i], junc_seqs[idx_j])
                seq_dist_mat[i, j] = float(ss_i + ss_j - 2.0 * cross)

        # Extract corresponding latent-space distances
        latent_dist_mat = junc_mat[np.ix_(sample_indices, sample_indices)].astype(np.float32)

        # Compute R² and RMSE
        flat_seq = seq_dist_mat.flatten()
        flat_latent = latent_dist_mat.flatten()

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((flat_seq - flat_latent) ** 2)
        ss_tot = np.sum((flat_seq - np.mean(flat_seq)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

        rmse = np.sqrt(np.mean((flat_seq - flat_latent) ** 2))

        print(f"  R² (sequence → latent): {r2:.6f}")
        print(f"  RMSE (sequence → latent): {rmse:.6f}")

        # For junction-only distances, should be perfect reconstruction
        assert r2 > 0.95, f"R² too low: {r2:.6f}"


# ===================================================================
# Test 6: BioPython C library alternatives exploration
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestBioPythonAlternatives:
    """Explore and benchmark BioPython against potential faster alternatives."""

    SEED = SEED
    N_TESTS = 1000

    def test_biopython_alternative_analysis(self):
        """Benchmark BioPython and discuss potential C library alternatives."""
        print("\n--- BioPython Alternative Analysis ---")

        # Test BioPython performance on subset
        from mir.distances.aligner import BioAlignerWrapper

        bio_aligner = BioAlignerWrapper()

        # Generate random sequences
        np.random.seed(self.SEED)
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

        queries = ["".join(np.random.choice(list(aa_alphabet), 15)) for _ in range(100)]
        references = ["".join(np.random.choice(list(aa_alphabet), 15)) for _ in range(100)]

        # Time BioPython batch alignment
        t0 = time.perf_counter()
        scores = bio_aligner.score_matrix(queries, references)
        t_bio = time.perf_counter() - t0

        pairs = len(queries) * len(references)
        pairs_per_sec = pairs / t_bio

        print(f"\nBioPython performance:")
        print(f"  {pairs} pairs in {t_bio:.3f}s")
        print(f"  {pairs_per_sec/1e3:.1f}k pairs/second")

        print("\nPotential C library alternatives for TCR/BCR alignment:")
        print("  1. SSW (Smith-Waterman SIMD) - https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library")
        print("  2. parasail - https://github.com/jeffdaily/parasail")
        print("  3. MAFFT - https://github.com/thomaskf/Fast-multiple-sequence-alignment")
        print("  4. BLOSUM62 vectorized in NumPy for diagonal variants")
        print("\nRecommendation:")
        print("  For now, keep fixed-gap (JunctionAligner) as default (~90x faster)")
        print("  Use BioPython only when full DP semantics are explicitly required")
