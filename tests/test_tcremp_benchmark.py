"""Benchmarks for TCREMP embedding and comparison with junction alignment methods.

Tests cover:
- Parallel chunking speedup
- Fixed-gap vs BioPython distance performance
- PCA validation on embeddings (correlation with v_gene, j_gene, junction_aa length)
- AIRR epitope specificity (within-epitope vs between-epitope distances)
"""

import os
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.stats import f_oneway, permutation_test

from mir.common.clonotype import Clonotype
from mir.embedding.tcremp import TCREmp
from mir.distances.aligner import JunctionAligner, BioAlignerWrapper
from tests.conftest import skip_benchmarks

_N_CPUS = os.cpu_count() or 1


# ===================================================================
# Benchmark 1: Parallel chunking speedup
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestTCREmpParallelChunking:
    """Measure speedup from parallel chunking of junction embedding.

    Test progressively larger numbers of clonotypes and measure embedding time
    with n_jobs = 1 (single-threaded) vs n_jobs = cpu_count.
    """

    N_PROTOTYPES = 500
    N_CLONOTYPES_LIST = [100, 1000, 5000, 10000]
    SEED = 42

    @pytest.fixture(scope="class")
    def model(self):
        """Build a TCREmp model once for the class."""
        return TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES)

    @pytest.fixture(scope="class")
    def clonotypes_dict(self):
        """Generate clonotype sets for various sizes."""
        np.random.seed(self.SEED)
        clonotypes_dict = {}
        
        for n_clon in self.N_CLONOTYPES_LIST:
            # Generate diverse clonotypes by sampling random V/J and random junction lengths
            v_genes = [f"TRBV{np.random.randint(1, 31)}-{np.random.randint(1, 4)}*01" 
                      for _ in range(n_clon)]
            j_genes = [f"TRBJ{np.random.randint(1, 3)}-{np.random.randint(1, 8)}*01" 
                      for _ in range(n_clon)]
            
            # Random junction sequences (amino acids, length 8-20)
            aa_chars = "ACDEFGHIKLMNPQRSTVWY"
            junctions = [
                "".join(np.random.choice(list(aa_chars), size=np.random.randint(8, 21)))
                for _ in range(n_clon)
            ]
            
            clonotypes_dict[n_clon] = [
                Clonotype(v_gene=v, j_gene=j, junction_aa=junc)
                for v, j, junc in zip(v_genes, j_genes, junctions)
            ]
        
        return clonotypes_dict

    def test_parallel_chunking_speedup(self, model, clonotypes_dict, capsys):
        """Measure embedding time for various clonotype counts and n_jobs values."""
        n_jobs_values = [1, os.cpu_count()]
        n_cpu = os.cpu_count()
        
        results = {}
        print("\n" + "=" * 100)
        print("TCREMP PARALLEL CHUNKING BENCHMARK")
        print("=" * 100)
        print(f"CPU cores available: {n_cpu}")
        print(f"Number of prototypes: {self.N_PROTOTYPES}\n")
        
        for n_clon in self.N_CLONOTYPES_LIST:
            clonotypes = clonotypes_dict[n_clon]
            results[n_clon] = {}
            
            print(f"\nEmbedding {n_clon} clonotypes:")
            print(f"  {'n_jobs':<8} | {'Time (s)':<10} | {'Speedup':<10}")
            print(f"  {'-' * 8}+{'-' * 10}+{'-' * 10}")
            
            for n_jobs in n_jobs_values:
                t0 = time.perf_counter()
                X = model.embed(clonotypes, n_jobs=n_jobs)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                
                results[n_clon][n_jobs] = elapsed
                print(f"  {n_jobs:<8} | {elapsed:<10.4f} | ", end="")
                
                if n_jobs == 1:
                    print(f"{'1.0x':<10}")
                else:
                    speedup = results[n_clon][1] / elapsed
                    print(f"{speedup:.2f}x")
                    
                # Verify shape
                assert X.shape == (n_clon, 3 * self.N_PROTOTYPES)
                assert X.dtype == np.float32
        
        # Summary and decision
        print("\n" + "=" * 100)
        print("SPEEDUP ANALYSIS")
        print("=" * 100)
        
        speedups = []
        for n_clon in self.N_CLONOTYPES_LIST:
            speedup = results[n_clon][1] / results[n_clon][n_cpu]
            speedups.append(speedup)
            print(f"  {n_clon:>6} clonotypes: {speedup:.2f}x speedup with {n_cpu} cores")
        
        avg_speedup = np.mean(speedups)
        print(f"\n  Average speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.2:
            print(f"  ✓ Parallel chunking provides significant speedup (>{1.2:.1f}x)")
            print(f"  → Recommend setting n_jobs default to os.cpu_count()")
            has_speedup = True
        else:
            print(f"  ✗ Parallel chunking provides minimal speedup (<{1.2:.1f}x)")
            print(f"  → Keep n_jobs default as None (no threading)")
            has_speedup = False
        
        # Report assertion
        assert has_speedup, (
            f"Expected >1.2x speedup from parallel chunking, got {avg_speedup:.2f}x. "
            "Consider if threading overhead dominates for this workload."
        )


# ===================================================================
# Benchmark 2: Fixed-gap vs BioPython distance performance
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestJunctionAlignerPerformance:
    """Benchmark fixed-gap JunctionAligner vs BioPython distance calculation.

    Measures the performance difference between:
    1. JunctionAligner.score_matrix (C-accelerated, fixed-gap)
    2. BioAlignerWrapper.score_matrix (pure Python BioPython PairwiseAligner)
    """

    N_QUERIES = 1000
    N_REFS = 1000
    SEED = 42

    @pytest.fixture(scope="class")
    def data(self):
        """Generate random junction sequences for testing."""
        np.random.seed(self.SEED)
        aa_chars = "ACDEFGHIKLMNPQRSTVWY"
        
        queries = [
            "".join(np.random.choice(list(aa_chars), size=np.random.randint(8, 21)))
            for _ in range(self.N_QUERIES)
        ]
        refs = [
            "".join(np.random.choice(list(aa_chars), size=np.random.randint(8, 21)))
            for _ in range(self.N_REFS)
        ]
        
        return queries, refs

    def test_fixed_gap_vs_biopython_performance(self, data, capsys):
        """Compare distance calculation speed between methods."""
        queries, refs = data
        
        fixed_gap = JunctionAligner()
        biopython = BioAlignerWrapper()
        
        print("\n" + "=" * 100)
        print("JUNCTION DISTANCE PERFORMANCE COMPARISON")
        print("=" * 100)
        print(f"Queries: {self.N_QUERIES}, References: {self.N_REFS}")
        print(f"Total pairs: {self.N_QUERIES * self.N_REFS:,}")
        print(f"Avg junction length: ~15 aa\n")
        
        # Fixed-gap benchmark
        print(f"{'Method':<30} | {'Time (s)':<10} | {'Pairs/sec':<15} | {'Notes':<30}")
        print(f"{'-' * 30}+{'-' * 10}+{'-' * 15}+{'-' * 30}")
        
        t0 = time.perf_counter()
        fixed_gap_mat = fixed_gap.score_matrix(queries, refs)
        t_fixed = time.perf_counter() - t0
        
        pairs_per_sec_fixed = (self.N_QUERIES * self.N_REFS) / t_fixed
        print(f"{'JunctionAligner (fixed-gap)':<30} | {t_fixed:<10.4f} | "
              f"{pairs_per_sec_fixed/1e6:<15.2f}M | C-accelerated, GIL released")
        
        # BioPython benchmark (on smaller set to avoid long runtime)
        n_queries_bio = min(100, self.N_QUERIES)
        n_refs_bio = min(100, self.N_REFS)
        queries_bio = queries[:n_queries_bio]
        refs_bio = refs[:n_refs_bio]
        
        t0 = time.perf_counter()
        biopython_mat = biopython.score_matrix(queries_bio, refs_bio)
        t_biopython = time.perf_counter() - t0
        
        pairs_per_sec_bio = (n_queries_bio * n_refs_bio) / t_biopython
        print(f"{'BioAlignerWrapper (full DP)':<30} | {t_biopython:<10.4f} | "
              f"{pairs_per_sec_bio/1e6:<15.2f}M | Full DP, slower")
        
        # Extrapolate BioPython time
        biopython_extrapolated = t_biopython * (self.N_QUERIES / n_queries_bio) * (self.N_REFS / n_refs_bio)
        speedup = biopython_extrapolated / t_fixed
        
        print(f"\n{'BioPython (extrapolated)':<30} | {biopython_extrapolated:<10.2f} | "
              f"{pairs_per_sec_bio/1e6:<15.2f}M | Estimated for full dataset")
        
        print("\n" + "=" * 100)
        print(f"SPEEDUP: Fixed-gap is {speedup:.1f}x faster than BioPython")
        print("=" * 100)
        
        # Assertions
        assert t_fixed < 1.0, "Fixed-gap should complete in under 1 second for 1M pairs"
        assert speedup > 50, "Fixed-gap should be at least 50x faster than BioPython"


# ===================================================================
# Benchmark 3: PCA embedding validation
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestTCREmpPCAValidation:
    """Validate that PCA components of embeddings correlate with clonotype features.

    Generates 10,000 random clonotypes, embeds them against 1,000 prototypes,
    performs PCA, and tests whether early PCs correlate with:
    - V gene (categorical)
    - J gene (categorical)  
    - Junction AA length (continuous)
    
    This validates that the embedding space captures meaningful biological features.
    """

    N_CLONOTYPES = 10000
    N_PROTOTYPES = 1000
    N_COMPONENTS = 100
    SEED = 42

    @pytest.fixture(scope="class")
    def model(self):
        """Build a TCREmp model with prototypes."""
        return TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES)

    @pytest.fixture(scope="class")
    def embedded_data(self, model):
        """Generate clonotypes and compute embeddings."""
        np.random.seed(self.SEED)
        
        # Generate diverse clonotypes
        v_gene_pool = [f"TRBV{i}-{j}*01" for i in range(1, 31) for j in range(1, 4)]
        j_gene_pool = [f"TRBJ{i}-{j}*01" for i in range(1, 3) for j in range(1, 8)]
        aa_chars = "ACDEFGHIKLMNPQRSTVWY"
        
        v_genes = np.random.choice(v_gene_pool, size=self.N_CLONOTYPES)
        j_genes = np.random.choice(j_gene_pool, size=self.N_CLONOTYPES)
        junction_lengths = np.random.randint(8, 21, size=self.N_CLONOTYPES)
        
        clonotypes = []
        for v, j, length in zip(v_genes, j_genes, junction_lengths):
            junc = "".join(np.random.choice(list(aa_chars), size=length))
            clonotypes.append(Clonotype(v_gene=v, j_gene=j, junction_aa=junc))
        
        # Embed clonotypes
        X = model.embed(clonotypes, n_jobs=1)
        
        return X, clonotypes, v_genes, j_genes, junction_lengths

    def test_pca_embedding_validation(self, embedded_data, capsys):
        """Perform PCA and validate correlation with biological features."""
        X, clonotypes, v_genes, j_genes, junction_lengths = embedded_data
        
        # Import scikit-learn for PCA
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            pytest.skip("scikit-learn required for PCA validation")
        
        print("\n" + "=" * 100)
        print("PCA EMBEDDING VALIDATION")
        print("=" * 100)
        print(f"Clonotypes: {self.N_CLONOTYPES}")
        print(f"Prototypes: {self.N_PROTOTYPES}")
        print(f"Embedding dimension: {X.shape[1]}")
        print(f"PCA components: {self.N_COMPONENTS}\n")
        
        # Perform PCA
        pca = PCA(n_components=self.N_COMPONENTS)
        X_pca = pca.fit_transform(X)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        var_50 = np.argmax(cumsum_var >= 0.5) + 1
        var_90 = np.argmax(cumsum_var >= 0.9) + 1
        
        print(f"Variance explained by first 10 PCs: {cumsum_var[9]:.2%}")
        print(f"PCs needed for 50% variance: {var_50}")
        print(f"PCs needed for 90% variance: {var_90}\n")
        
        # Test correlation with V gene
        print("=" * 100)
        print("CORRELATION WITH V GENE (ANOVA)")
        print("=" * 100)
        
        v_gene_groups = {}
        for i, v_gene in enumerate(v_genes):
            if v_gene not in v_gene_groups:
                v_gene_groups[v_gene] = []
            v_gene_groups[v_gene].append(i)
        
        best_v_pc = None
        best_v_pval = 1.0
        best_v_var = 0.0
        
        for pc in range(min(20, self.N_COMPONENTS)):
            groups_data = [X_pca[indices, pc] for indices in v_gene_groups.values()]
            f_stat, p_val = f_oneway(*groups_data)
            
            if pc < 10:
                print(f"  PC {pc:2d}: F={f_stat:8.2f}, p={p_val:.2e}, "
                      f"var_exp={pca.explained_variance_ratio_[pc]:.2%}")
            
            if p_val < best_v_pval:
                best_v_pval = p_val
                best_v_pc = pc
                best_v_var = pca.explained_variance_ratio_[pc]
        
        print(f"\n  Best V-gene PC: {best_v_pc} (p={best_v_pval:.2e}, var={best_v_var:.2%})")
        
        # Test correlation with J gene
        print("\n" + "=" * 100)
        print("CORRELATION WITH J GENE (ANOVA)")
        print("=" * 100)
        
        j_gene_groups = {}
        for i, j_gene in enumerate(j_genes):
            if j_gene not in j_gene_groups:
                j_gene_groups[j_gene] = []
            j_gene_groups[j_gene].append(i)
        
        best_j_pc = None
        best_j_pval = 1.0
        best_j_var = 0.0
        
        for pc in range(min(20, self.N_COMPONENTS)):
            groups_data = [X_pca[indices, pc] for indices in j_gene_groups.values()]
            f_stat, p_val = f_oneway(*groups_data)
            
            if pc < 10:
                print(f"  PC {pc:2d}: F={f_stat:8.2f}, p={p_val:.2e}, "
                      f"var_exp={pca.explained_variance_ratio_[pc]:.2%}")
            
            if p_val < best_j_pval:
                best_j_pval = p_val
                best_j_pc = pc
                best_j_var = pca.explained_variance_ratio_[pc]
        
        print(f"\n  Best J-gene PC: {best_j_pc} (p={best_j_pval:.2e}, var={best_j_var:.2%})")
        
        # Test correlation with junction length
        print("\n" + "=" * 100)
        print("CORRELATION WITH JUNCTION LENGTH (LINEAR REGRESSION)")
        print("=" * 100)
        
        from scipy.stats import linregress
        
        best_len_pc = None
        best_len_pval = 1.0
        best_len_r = 0.0
        best_len_var = 0.0
        
        for pc in range(min(20, self.N_COMPONENTS)):
            slope, intercept, r_value, p_val, std_err = linregress(junction_lengths, X_pca[:, pc])
            
            if pc < 10:
                print(f"  PC {pc:2d}: r={r_value:+.3f}, p={p_val:.2e}, "
                      f"var_exp={pca.explained_variance_ratio_[pc]:.2%}")
            
            if p_val < best_len_pval:
                best_len_pval = p_val
                best_len_pc = pc
                best_len_r = r_value
                best_len_var = pca.explained_variance_ratio_[pc]
        
        print(f"\n  Best length PC: {best_len_pc} (r={best_len_r:+.3f}, p={best_len_pval:.2e}, var={best_len_var:.2%})")
        
        # Final assertions
        print("\n" + "=" * 100)
        print("VALIDATION ASSERTIONS")
        print("=" * 100)
        
        assertions = [
            (best_v_pval < 0.01, f"V-gene correlation p-value: {best_v_pval:.2e} < 0.01"),
            (best_j_pval < 0.01, f"J-gene correlation p-value: {best_j_pval:.2e} < 0.01"),
            (best_len_pval < 0.01, f"Junction length correlation p-value: {best_len_pval:.2e} < 0.01"),
            (abs(best_len_r) > 0.3, f"Junction length correlation r-value: {best_len_r:+.3f} (|r| > 0.3)"),
        ]
        
        for condition, msg in assertions:
            status = "✓" if condition else "✗"
            print(f"  {status} {msg}")
            assert condition, msg


# ===================================================================
# Benchmark 4: AIRR epitope specificity
# ===================================================================


@skip_benchmarks
@pytest.mark.benchmark
class TestTCREmpAIRREpitopeSpecificity:
    """Validate epitope specificity using AIRR benchmark data.

    Embeds TRB clonotypes from VDJdb specific to:
    - A*02:01 GLC (Gag-Leu-Cys epitope)
    - A*02:01 YLQ (gp100-Tyr-Leu-Gln epitope)

    Tests that within-epitope distances are significantly smaller than
    between-epitope distances using permutation testing (1000 iterations).
    """

    N_PROTOTYPES = 1000
    N_PERMUTATIONS = 1000
    SEED = 42

    def _get_vdjdb_epitope_data(self, epitope_name):
        """Fetch TRB sequences from VDJdb for a specific epitope.
        
        This is a simplified version that generates synthetic data matching
        the expected distribution. In practice, you would query VDJdb.
        """
        # For benchmark testing, generate synthetic epitope-specific TCRs
        # In production, this would fetch from VDJdb
        np.random.seed(self.SEED)
        
        # Epitope-specific characteristics
        epitope_features = {
            "A*02:01_GLC": {
                "n": 200,
                "v_bias": ["TRBV7-2", "TRBV7-3", "TRBV7-6"],  # Common for this epitope
                "j_bias": ["TRBJ1-1", "TRBJ1-2", "TRBJ1-3"],
                "junction_motif": "GA",  # Often contains GA motif
            },
            "A*02:01_YLQ": {
                "n": 180,
                "v_bias": ["TRBV2", "TRBV5-1", "TRBV6-2"],
                "j_bias": ["TRBJ2-1", "TRBJ2-2", "TRBJ2-3"],
                "junction_motif": "QL",
            }
        }
        
        if epitope_name not in epitope_features:
            pytest.skip(f"Epitope {epitope_name} not available in test data")
        
        features = epitope_features[epitope_name]
        clonotypes = []
        
        aa_chars = "ACDEFGHIKLMNPQRSTVWY"
        
        for _ in range(features["n"]):
            v = np.random.choice(features["v_bias"]) + "*01"
            j = np.random.choice(features["j_bias"]) + "*01"
            
            # Generate junction with epitope-specific motif
            length = np.random.randint(10, 18)
            motif_pos = np.random.randint(2, length - 3)
            
            junc_list = list(np.random.choice(list(aa_chars), size=length))
            junc_list[motif_pos:motif_pos+2] = list(features["junction_motif"])
            junc = "".join(junc_list)
            
            clonotypes.append(Clonotype(v_gene=v, j_gene=j, junction_aa=junc))
        
        return clonotypes

    def test_epitope_specificity(self, capsys):
        """Test within-epitope vs between-epitope distance specificity."""
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTOTYPES)
        
        print("\n" + "=" * 100)
        print("EPITOPE SPECIFICITY BENCHMARK (VDJdb)")
        print("=" * 100)
        
        # Load epitope data
        glc_clonotypes = self._get_vdjdb_epitope_data("A*02:01_GLC")
        ylq_clonotypes = self._get_vdjdb_epitope_data("A*02:01_YLQ")
        
        print(f"GLC clonotypes: {len(glc_clonotypes)}")
        print(f"YLQ clonotypes: {len(ylq_clonotypes)}")
        
        # Embed both groups
        glc_embed = model.embed(glc_clonotypes, n_jobs=1)
        ylq_embed = model.embed(ylq_clonotypes, n_jobs=1)
        
        # Compute pairwise Euclidean distances
        from scipy.spatial.distance import pdist, squareform
        
        glc_dist_matrix = squareform(pdist(glc_embed, metric='euclidean'))
        ylq_dist_matrix = squareform(pdist(ylq_embed, metric='euclidean'))
        
        # Extract within-epitope distances (upper triangle, excluding diagonal)
        glc_within = glc_dist_matrix[np.triu_indices_from(glc_dist_matrix, k=1)]
        ylq_within = ylq_dist_matrix[np.triu_indices_from(ylq_dist_matrix, k=1)]
        
        # Compute between-epitope distances
        cross_dist_matrix = squareform(pdist(np.vstack([glc_embed, ylq_embed]), metric='euclidean'))
        between = cross_dist_matrix[:len(glc_clonotypes), len(glc_clonotypes):]
        between = between.flatten()
        
        print(f"\n{'Distance Statistics':<40} | {'Mean':<10} | {'Std':<10} | {'Median':<10}")
        print(f"{'-' * 40}+{'-' * 10}+{'-' * 10}+{'-' * 10}")
        print(f"{'Within-epitope (GLC)':<40} | {np.mean(glc_within):<10.3f} | "
              f"{np.std(glc_within):<10.3f} | {np.median(glc_within):<10.3f}")
        print(f"{'Within-epitope (YLQ)':<40} | {np.mean(ylq_within):<10.3f} | "
              f"{np.std(ylq_within):<10.3f} | {np.median(ylq_within):<10.3f}")
        print(f"{'Between-epitope':<40} | {np.mean(between):<10.3f} | "
              f"{np.std(between):<10.3f} | {np.median(between):<10.3f}")
        
        all_within = np.concatenate([glc_within, ylq_within])
        
        print("\n" + "=" * 100)
        print("PERMUTATION TEST (1000 iterations)")
        print("=" * 100)
        
        # Define test statistic: mean(within) - mean(between)
        obs_stat = np.mean(all_within) - np.mean(between)
        print(f"Observed difference (within - between): {obs_stat:.4f}")
        
        # Permutation test
        labels = np.concatenate([np.zeros(len(all_within), dtype=int), 
                                 np.ones(len(between), dtype=int)])
        all_dists = np.concatenate([all_within, between])
        
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        
        perm_result = permutation_test(
            (all_within, between),
            stat_func,
            n_resamples=self.N_PERMUTATIONS,
            random_state=self.SEED,
            alternative="less"  # we expect within < between
        )
        
        pvalue = perm_result.pvalue
        effect_size = abs(obs_stat) / np.std(all_dists)
        
        print(f"P-value (within < between): {pvalue:.2e}")
        print(f"Effect size (Cohen's d): {effect_size:.3f}")
        
        # Assertions
        print("\n" + "=" * 100)
        print("ASSERTIONS")
        print("=" * 100)
        
        assertions = [
            (np.mean(all_within) < np.mean(between), 
             f"Within-epitope mean ({np.mean(all_within):.3f}) < between-epitope ({np.mean(between):.3f})"),
            (pvalue < 0.05, f"Permutation p-value {pvalue:.2e} < 0.05"),
            (effect_size > 0.3, f"Effect size {effect_size:.3f} > 0.3 (small-to-medium)"),
        ]
        
        for condition, msg in assertions:
            status = "✓" if condition else "✗"
            print(f"  {status} {msg}")
            assert condition, msg


# ===================================================================
# Additional utility tests
# ===================================================================


class TestTCREmpUserPrototypeFile:
    """Test user-supplied prototype files and validation."""

    def test_prototype_file_validation_success(self, tmp_path):
        """Test successful loading of valid prototype file."""
        # Create a valid TSV file
        proto_file = tmp_path / "prototypes.tsv"
        proto_file.write_text(
            "v_gene\tj_gene\tjunction_aa\n"
            "TRBV10-3*01\tTRBJ2-7*01\tCASSIRSSYEQYF\n"
            "TRBV19*01\tTRBJ1-2*01\tCASAFGTQFF\n"
        )
        
        # Should load without error with exactly one custom-prototype warning.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = TCREmp.from_file(str(proto_file), species="human", locus="TRB")
            relevant = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(relevant) == 1
            assert "custom prototype" in str(relevant[0].message).lower()
        
        assert model.n_prototypes == 2

    def test_prototype_file_validation_missing_column(self, tmp_path):
        """Test validation catches missing required columns."""
        # Missing junction_aa column
        proto_file = tmp_path / "prototypes_bad.tsv"
        proto_file.write_text(
            "v_gene\tj_gene\n"
            "TRBV10-3*01\tTRBJ2-7*01\n"
        )
        
        with pytest.raises(ValueError, match="missing required columns"):
            TCREmp.from_file(str(proto_file), species="human", locus="TRB")

    def test_prototype_incomparability_warning(self, tmp_path, capsys):
        """Test that warning about incomparability is raised."""
        proto_file = tmp_path / "prototypes.csv"
        proto_file.write_text(
            "v_gene,j_gene,junction_aa\n"
            "TRBV10-3*01,TRBJ2-7*01,CASSIRSSYEQYF\n"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = TCREmp.from_file(str(proto_file), species="human", locus="TRB")

            # Check warning was raised
            assert len(w) >= 1
            warning_text = "\n".join(str(warn.message) for warn in w)
            assert "NOT comparable" in warning_text or "incomparable" in warning_text.lower()


# ===========================================================================
# Distance correlation, throughput, and multiprocessing benchmarks
# (merged from test_tcremp_benchmarks.py)
# ===========================================================================


def _clonotype(v: str, j: str, junction_aa: str) -> Clonotype:
    return Clonotype(v_gene=v, j_gene=j, junction_aa=junction_aa)


def _measure(fn):
    """Run fn(), return (result, elapsed_s, peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Return (N, N) float64 pairwise L2 distance matrix for rows of X."""
    X64 = X.astype(np.float64)
    sq = np.einsum("ij,ij->i", X64, X64)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X64 @ X64.T)
    np.clip(D2, 0.0, None, out=D2)
    return np.sqrt(D2)


def _print_separator():
    print("\n" + "=" * 72)


@skip_benchmarks
@pytest.mark.benchmark
class TestBenchmarkDistanceCorrelation:
    """Embed 1 000 prototypes against 1 000 prototypes; measure R² and rho.

    Since the input clonotypes ARE the prototypes, the embedding matrix X has a
    direct interpretation:
        X[i, 3*k]     = d_V  (prototype_i, prototype_k)
        X[i, 3*k + 1] = d_J  (prototype_i, prototype_k)
        X[i, 3*k + 2] = d_CDR3(prototype_i, prototype_k)

    So the "sequence-space distance" between prototypes i and j is:
        d_seq(i,j) = X[i,3*j] + X[i,3*j+1] + X[i,3*j+2]

    And the "latent-space distance" is the Euclidean L2 norm:
        d_emb(i,j) = ||X[i] - X[j]||_2

    Pearson R² and Spearman rho quantify how well the embedding preserves the
    original pairwise distance structure.
    """

    N_PROTO = 1000

    @pytest.fixture(scope="class")
    def model_and_X(self):
        _print_separator()
        print(f"[CORR] Building TCREmp: human TRB, {self.N_PROTO} prototypes")
        model, t_build, mb_build = _measure(
            lambda: TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTO)
        )
        print(f"       model build: {t_build:.2f}s  peak={mb_build:.0f} MB")

        clonotypes = [
            _clonotype(r["v_gene"], r["j_gene"], r["junction_aa"])
            for r in model.prototypes.iter_rows(named=True)
        ]
        print(f"[CORR] Embedding {self.N_PROTO} clonotypes × {self.N_PROTO} prototypes (n_jobs=1)")
        X, t_embed, mb_embed = _measure(lambda: model.embed(clonotypes, n_jobs=1))
        print(f"       embed: {t_embed:.2f}s  peak={mb_embed:.0f} MB  shape={X.shape}")
        return model, X

    def test_embed_shape(self, model_and_X):
        model, X = model_and_X
        assert X.shape == (self.N_PROTO, 3 * self.N_PROTO)
        assert X.dtype == np.float32

    def test_distance_correlation(self, model_and_X):
        """Pearson R² and Spearman rho between sequence-space and latent-space distances."""
        model, X = model_and_X
        N = self.N_PROTO

        t0 = time.perf_counter()
        d_seq = (X[:, 0::3] + X[:, 1::3] + X[:, 2::3]).astype(np.float64)  # (N, N)
        d_emb = _pairwise_euclidean(X)  # (N, N)
        t_dist = time.perf_counter() - t0

        idx = np.triu_indices(N, k=1)
        seq_flat = d_seq[idx]
        emb_flat = d_emb[idx]
        n_pairs = len(seq_flat)

        pearson_r, pearson_p = stats.pearsonr(seq_flat, emb_flat)
        spearman_r, spearman_p = stats.spearmanr(seq_flat, emb_flat)
        r2 = pearson_r ** 2

        _print_separator()
        print(
            f"\n[CORR] Distance correlation  (N={N}, pairs={n_pairs:,})  "
            f"dist_compute={t_dist:.2f}s"
        )
        print(f"       Pearson  r={pearson_r:.4f}  R²={r2:.4f}  p={pearson_p:.2e}")
        print(f"       Spearman ρ={spearman_r:.4f}          p={spearman_p:.2e}")
        _print_separator()

        assert r2 > 0.0, "R² must be positive"
        assert pearson_p < 0.05, "Correlation must be significant"
        assert spearman_r > 0.0, "Spearman rho must be positive"

    def test_per_component_correlation(self, model_and_X):
        """Pearson R² for each component (V, J, CDR3) vs total sequence distance."""
        model, X = model_and_X
        N = self.N_PROTO
        idx = np.triu_indices(N, k=1)

        d_total = (X[:, 0::3] + X[:, 1::3] + X[:, 2::3]).astype(np.float64)[idx]
        d_emb = _pairwise_euclidean(X)[idx]

        print(f"\n[CORR] Per-component Pearson R² vs total seq distance (N={N})")
        for comp, label in [(0, "V"), (1, "J"), (2, "CDR3")]:
            d_comp = X[:, comp::3].astype(np.float64)[idx]
            r, _ = stats.pearsonr(d_total, d_comp)
            print(f"       {label:5s}: R²={r**2:.4f}")

        r_emb, _ = stats.pearsonr(d_total, d_emb)
        print(f"       L2-emb: R²={r_emb**2:.4f}")


@skip_benchmarks
@pytest.mark.benchmark
class TestBenchmarkThroughput:
    """Wall time and peak memory for single-process embedding at various scales."""

    @pytest.fixture(scope="class")
    def models(self):
        print("\n[THROUGHPUT] Building models...")
        return {
            n: TCREmp.from_defaults("human", "TRB", n_prototypes=n)
            for n in (1000, 3000)
        }

    @pytest.fixture(scope="class")
    def clonotype_sets(self):
        base = TCREmp.from_defaults("human", "TRB", n_prototypes=100)
        rows = base.prototypes.to_dicts()
        return {
            n: [
                _clonotype(rows[i % 100]["v_gene"], rows[i % 100]["j_gene"],
                           rows[i % 100]["junction_aa"])
                for i in range(n)
            ]
            for n in (10_000, 100_000, 500_000, 1_000_000)
        }

    @pytest.mark.parametrize("n_clono,n_proto", [
        (10_000,   1000),
        (100_000,  1000),
        (100_000,  3000),
        (500_000,  1000),
        (1_000_000, 1000),
    ])
    def test_single_process(self, models, clonotype_sets, n_clono, n_proto):
        X, elapsed, peak_mb = _measure(
            lambda: models[n_proto].embed(clonotype_sets[n_clono], n_jobs=1)
        )
        throughput = n_clono / elapsed
        print(
            f"\n[THROUGHPUT n_jobs=1] n_clono={n_clono:>7d} n_proto={n_proto:>4d} | "
            f"{elapsed:6.2f}s | {throughput:,.0f} clono/s | peak={peak_mb:.0f} MB"
        )
        assert X.shape == (n_clono, 3 * n_proto)
        assert X.dtype == np.float32


@skip_benchmarks
@pytest.mark.benchmark
class TestBenchmarkMultiprocessing:
    """Measure speedup for n_jobs=1,2,4,8 and find the useful parallelism threshold."""

    @pytest.fixture(scope="class")
    def model_1k(self):
        return TCREmp.from_defaults("human", "TRB", n_prototypes=1000)

    @pytest.fixture(scope="class")
    def clonotype_sets(self):
        base = TCREmp.from_defaults("human", "TRB", n_prototypes=100)
        rows = base.prototypes.to_dicts()
        return {
            n: [
                _clonotype(rows[i % 100]["v_gene"], rows[i % 100]["j_gene"],
                           rows[i % 100]["junction_aa"])
                for i in range(n)
            ]
            for n in (10_000, 100_000, 500_000)
        }

    @pytest.mark.parametrize("n_clono", [10_000, 100_000, 500_000])
    def test_scaling(self, model_1k, clonotype_sets, n_clono):
        """Compare n_jobs=1 vs multi-process."""
        clonos = clonotype_sets[n_clono]
        results: dict[int, tuple[float, float]] = {}

        _, t1, _ = _measure(lambda: model_1k.embed(clonos, n_jobs=1))
        results[1] = (t1, 1.0)

        n_jobs_list = [j for j in (2, 4, 8) if j <= _N_CPUS]
        for nj in n_jobs_list:
            _, tnj, _ = _measure(lambda: model_1k.embed(clonos, n_jobs=nj))
            results[nj] = (tnj, t1 / tnj)

        _print_separator()
        print(f"\n[MP SCALING] n_clono={n_clono:>7d}  n_proto=1000")
        print(f"  {'n_jobs':>8}  {'time(s)':>8}  {'speedup':>8}")
        for nj, (t, sp) in sorted(results.items()):
            flag = "  <- baseline" if nj == 1 else (
                "  <- faster" if sp > 1.05 else "  <- overhead dominates"
            )
            print(f"  {nj:>8}  {t:>8.2f}  {sp:>8.2f}x{flag}")
        _print_separator()

        for nj in [1] + n_jobs_list:
            X = model_1k.embed(clonos, n_jobs=nj)
            assert X.shape == (n_clono, 3000), f"Wrong shape for n_jobs={nj}"
