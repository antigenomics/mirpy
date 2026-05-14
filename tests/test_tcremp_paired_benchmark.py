"""Benchmarks for paired-chain TCREmp embedding on VDJdb full records."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mir.common.parser import VDJdbFullPairedParser
from mir.embedding.tcremp import PairedTCREmp

skip_benchmarks = pytest.mark.skipif(
    not os.getenv("RUN_BENCHMARK"), reason="set RUN_BENCHMARK=1 to run"
)

ASSETS = Path(__file__).parent / "assets"
_VDJDB_FULL_FILE = ASSETS / "vdjdb_full.txt.gz"


@skip_benchmarks
@pytest.mark.benchmark
class TestPairedTCREmpPerformance:
    """Compare per-record embedding cost for single-chain and paired VDJdb inputs."""

    N_RECORDS = 2000
    N_PROTOTYPES = 200

    @pytest.fixture(scope="class")
    def paired_records(self):
        parser = VDJdbFullPairedParser()
        sample = parser.parse_file(
            _VDJDB_FULL_FILE,
            sample_id="vdjdb_full_human",
            species="HomoSapiens",
        )
        return sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes[: self.N_RECORDS]

    @pytest.fixture(scope="class")
    def model(self):
        return PairedTCREmp.from_defaults(
            species="human",
            locus_pair="TRA_TRB",
            n_prototypes=self.N_PROTOTYPES,
        )

    def test_single_vs_paired_embedding_speed(self, paired_records, model):
        tra = [pair.clonotype1 if pair.clonotype1.locus == "TRA" else pair.clonotype2 for pair in paired_records]
        trb = [pair.clonotype1 if pair.clonotype1.locus == "TRB" else pair.clonotype2 for pair in paired_records]

        t0 = time.perf_counter()
        x_tra = model.chain1_model.embed(tra)
        t_tra = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_trb = model.chain2_model.embed(trb)
        t_trb = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_pair = model.embed(paired_records)
        t_pair = time.perf_counter() - t0

        n_records = len(paired_records)
        print("\nPaired TCREmp benchmark")
        print(f"records: {n_records:,}   prototypes/chain: {self.N_PROTOTYPES}")
        print(f"TRA single:  {t_tra:.3f}s   {1e3 * t_tra / max(n_records, 1):.3f} ms/record")
        print(f"TRB single:  {t_trb:.3f}s   {1e3 * t_trb / max(n_records, 1):.3f} ms/record")
        print(f"paired:      {t_pair:.3f}s   {1e3 * t_pair / max(n_records, 1):.3f} ms/record")
        print(f"paired / (TRA+TRB): {t_pair / max(t_tra + t_trb, 1e-9):.3f}x")

        assert x_tra.shape == (n_records, model.chain1_model.embedding_dim)
        assert x_trb.shape == (n_records, model.chain2_model.embedding_dim)
        assert x_pair.shape == (n_records, model.embedding_dim)
        assert t_pair > 0.0