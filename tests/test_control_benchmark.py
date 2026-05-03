"""Benchmark tests for control generation/download workflows.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_control_benchmark.py -s
"""

from __future__ import annotations

import time
from pathlib import Path

from mir.common.control import ControlManager
from tests.conftest import benchmark_max_seconds, benchmark_scale, skip_benchmarks


@skip_benchmarks
def test_generate_and_fetch_controls_synthetic_and_real(capsys, tmp_path: Path) -> None:
    """Benchmark both synthetic and real control workflows with diagnostics.

    Synthetic:
    - human TRB
    - mouse TRA

    Real (HuggingFace airr_control):
    - human TRB
    - mouse TRA

    Reports verbosity, timing, paths, manifest diagnostics, and cache-hit timings.
    """
    control_dir = tmp_path / "controls_benchmark"
    mgr = ControlManager(control_dir=control_dir)

    # Scale workload by benchmark knob while keeping it non-trivial.
    n = max(5_000, int(round(30_000 * benchmark_scale(default=1.0))))

    # Synthetic: human TRB
    t0 = time.perf_counter()
    rec_h_trb = mgr.ensure_synthetic_control(
        "human",
        "TRB",
        n=n,
        overwrite=True,
        seed=42,
        chunk_size=5_000,
        progress=True,
    )
    t_h_trb = time.perf_counter() - t0

    # Synthetic: mouse TRA
    t0 = time.perf_counter()
    rec_m_tra = mgr.ensure_synthetic_control(
        "mouse",
        "TRA",
        n=n,
        overwrite=True,
        seed=43,
        chunk_size=5_000,
        progress=True,
    )
    t_m_tra = time.perf_counter() - t0

    # Capture and check verbosity.
    out = capsys.readouterr().out
    assert "Generated synthetic control human/TRB" in out
    assert "Generated synthetic control mouse/TRA" in out

    # Load generated synthetic controls and compute diagnostics.
    df_h_trb = mgr.load_control_df("synthetic", "human", "TRB", n=n)
    df_m_tra = mgr.load_control_df("synthetic", "mouse", "TRA", n=n)

    p_h_trb = Path(rec_h_trb.path)
    p_m_tra = Path(rec_m_tra.path)
    assert p_h_trb.exists()
    assert p_m_tra.exists()

    size_h_syn = p_h_trb.stat().st_size
    size_m_syn = p_m_tra.stat().st_size

    # Real: human TRB (download + build), then cache hit timing.
    t0 = time.perf_counter()
    rec_real_h_trb = mgr.ensure_real_control("human", "TRB", overwrite=True)
    t_real_h_trb_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = mgr.ensure_real_control("human", "TRB", overwrite=False)
    t_real_h_trb_cache = time.perf_counter() - t0
    df_real_h_trb = mgr.load_control_df("real", "human", "TRB")

    # Real: mouse TRA (download + build), then cache hit timing.
    t0 = time.perf_counter()
    rec_real_m_tra = mgr.ensure_real_control("mouse", "TRA", overwrite=True)
    t_real_m_tra_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = mgr.ensure_real_control("mouse", "TRA", overwrite=False)
    t_real_m_tra_cache = time.perf_counter() - t0
    df_real_m_tra = mgr.load_control_df("real", "mouse", "TRA")

    p_real_h_trb = Path(rec_real_h_trb.path)
    p_real_m_tra = Path(rec_real_m_tra.path)
    assert p_real_h_trb.exists()
    assert p_real_m_tra.exists()
    size_h_real = p_real_h_trb.stat().st_size
    size_m_real = p_real_m_tra.stat().st_size

    manifest = mgr.load_manifest()
    records = manifest.get("records", {})

    # Report diagnostics in benchmark output.
    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Control generation/download benchmark")
        print(f"Control root folder: {mgr.control_dir}")
        print(f"Manifest path:       {mgr.manifest_path}")
        print("Synthetic controls")
        print(f"  Runtime human/TRB: {t_h_trb:.2f}s")
        print(f"  Runtime mouse/TRA: {t_m_tra:.2f}s")
        print(f"  Rows human/TRB:    {len(df_h_trb)}")
        print(f"  Rows mouse/TRA:    {len(df_m_tra)}")
        print(f"  File human/TRB:    {p_h_trb} ({size_h_syn} bytes)")
        print(f"  File mouse/TRA:    {p_m_tra} ({size_m_syn} bytes)")
        print("Real controls (HuggingFace)")
        print(f"  Build human/TRB:   {t_real_h_trb_build:.2f}s | cache hit: {t_real_h_trb_cache:.2f}s")
        print(f"  Build mouse/TRA:   {t_real_m_tra_build:.2f}s | cache hit: {t_real_m_tra_cache:.2f}s")
        print(f"  Rows human/TRB:    {len(df_real_h_trb)}")
        print(f"  Rows mouse/TRA:    {len(df_real_m_tra)}")
        print(f"  File human/TRB:    {p_real_h_trb} ({size_h_real} bytes)")
        print(f"  File mouse/TRA:    {p_real_m_tra} ({size_m_real} bytes)")
        print(f"Manifest records:    {len(records)}")
        print(
            "Synthetic human/TRB unique V/J: "
            f"{df_h_trb['v_gene'].nunique()} / {df_h_trb['j_gene'].nunique()}"
        )
        print(
            "Synthetic mouse/TRA unique V/J: "
            f"{df_m_tra['v_gene'].nunique()} / {df_m_tra['j_gene'].nunique()}"
        )
        print(
            "Real human/TRB unique V/J:      "
            f"{df_real_h_trb['v_gene'].nunique()} / {df_real_h_trb['j_gene'].nunique()}"
        )
        print(
            "Real mouse/TRA unique V/J:      "
            f"{df_real_m_tra['v_gene'].nunique()} / {df_real_m_tra['j_gene'].nunique()}"
        )
        print("=" * 76)

    # Sanity assertions.
    assert len(df_h_trb) == n
    assert len(df_m_tra) == n
    assert len(df_real_h_trb) > 0
    assert len(df_real_m_tra) > 0
    assert {"duplicate_count", "junction", "junction_aa", "v_gene", "j_gene", "log2_pgen"}.issubset(df_h_trb.columns)
    assert {"duplicate_count", "junction", "junction_aa", "v_gene", "j_gene", "log2_pgen"}.issubset(df_m_tra.columns)
    assert set(df_real_h_trb.columns) == {"duplicate_count", "junction", "junction_aa", "v_gene", "j_gene"}
    assert set(df_real_m_tra.columns) == {"duplicate_count", "junction", "junction_aa", "v_gene", "j_gene"}
    assert (df_h_trb["duplicate_count"] >= 1).all()
    assert (df_m_tra["duplicate_count"] >= 1).all()
    assert (df_real_h_trb["duplicate_count"] >= 1).all()
    assert (df_real_m_tra["duplicate_count"] >= 1).all()

    max_s = benchmark_max_seconds(default=300.0)
    assert t_h_trb < max_s, f"human/TRB control generation is too slow: {t_h_trb:.2f}s >= {max_s:.2f}s"
    assert t_m_tra < max_s, f"mouse/TRA control generation is too slow: {t_m_tra:.2f}s >= {max_s:.2f}s"
    assert (
        t_real_h_trb_cache <= t_real_h_trb_build
    ), "Expected cache-hit real-control retrieval to be faster than build for human/TRB"
    assert (
        t_real_m_tra_cache <= t_real_m_tra_build
    ), "Expected cache-hit real-control retrieval to be faster than build for mouse/TRA"
