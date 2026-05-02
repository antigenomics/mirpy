"""Benchmark for control-backed bag-of-k-mers profile generation.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_bag_of_kmers_benchmark.py -s
"""

from __future__ import annotations

import time
from pathlib import Path

from mir.common.control import ControlManager
from mir.embedding.bag_of_kmers import (
    BagOfKmersParams,
    build_control_kmer_profile,
    ensure_control_kmer_profile,
)
from tests.conftest import skip_benchmarks


@skip_benchmarks
def test_bag_of_kmers_real_human_trb_k3_cas_position0(tmp_path: Path, capsys) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")
    params = BagOfKmersParams(use_v=False, k=3, gapped=False, reduced_alphabet=False)

    t0 = time.perf_counter()
    profile_mem = build_control_kmer_profile(
        mgr,
        control_type="real",
        species="human",
        locus="TRB",
        params=params,
        control_kwargs={"dataset_repo": "isalgo/airr_control", "overwrite": False},
        cache=False,
    )
    elapsed_mem = time.perf_counter() - t0

    t0 = time.perf_counter()
    profile_cached = ensure_control_kmer_profile(
        mgr,
        control_type="real",
        species="human",
        locus="TRB",
        params=params,
        control_kwargs={"dataset_repo": "isalgo/airr_control", "overwrite": False},
        cache=True,
        overwrite=False,
    )
    elapsed_cached = time.perf_counter() - t0

    token_stats = profile_mem.token_stats
    pos_stats = profile_mem.position_stats

    top = token_stats.sort_values("n", ascending=False).head(10)
    cas_pos = pos_stats[pos_stats["token"] == "CAS"].sort_values("count", ascending=False)
    top_cas_pos = int(cas_pos.iloc[0]["pos"]) if len(cas_pos) else -1

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("bag_of_kmers benchmark: real human TRB k=3")
        print(f"profile: {profile_mem.metadata['profile_name']}")
        print(f"runtime (in-memory default): {elapsed_mem:.2f}s")
        print(f"runtime (with cache write):  {elapsed_cached:.2f}s")
        print(f"total kmers T: {profile_mem.metadata['total_kmers']}")
        print("top tokens:")
        for row in top.itertuples(index=False):
            print(f"  {row.token}\t{row.n}")
        print(f"CAS dominant position: {top_cas_pos}")
        print("=" * 76)

    assert len(token_stats) > 0
    assert profile_mem.metadata["cache_enabled"] is False
    assert profile_cached.metadata["cache_enabled"] is True
    assert top.iloc[0]["token"] == "CAS"
    assert top_cas_pos == 0
