"""Optional parity checks between mirpy single-cell counting and scirpy loading.

These tests are integration-style and are skipped unless scirpy is installed
and dcode/10x_vdj_v1 benchmark assets are present locally.
"""

from __future__ import annotations

import polars as pl
import pytest

from mir.common.single_cell import load_10x_vdj_v1_donor
from tests.prepare_airr_benchmark_data import ensure_test_data
from tests.sc_helpers import discover_dcode_donor_pair as _discover_donor_pair


@pytest.mark.integration
def test_scirpy_parity_for_tra_trb_presence_counts() -> None:
    """Compare mirpy and scirpy TRA/TRB presence quadrants on one donor.

    Exact counts are expected to differ because mirpy currently builds counts
    from consensus-linked chains, while scirpy includes all cell-associated
    chains in the AIRR object. This test checks for qualitative parity and
    bounded divergence on the dominant TRA+/TRB+ quadrant.
    """
    ensure_test_data(force=False, verbose=False)
    donor_files = _discover_donor_pair()
    if donor_files is None:
        pytest.skip("No local dcode/10x_vdj_v1 donor assets found")

    all_contig, consensus = donor_files

    donor = load_10x_vdj_v1_donor(
        consensus_annotations_path=consensus,
        all_contig_annotations_path=all_contig,
    )

    mir_counts = (
        donor.chain_multiplicity.filter(pl.col("locus_pair") == "TRA_TRB")
        .group_by(["n_chain1", "m_chain2"])
        .agg(pl.sum("cell_count").alias("cells"))
    )
    mir_presence = (
        mir_counts.with_columns(
            pl.when(pl.col("n_chain1") > 0).then(pl.lit("TRA+")).otherwise(pl.lit("TRA-")).alias("tra"),
            pl.when(pl.col("m_chain2") > 0).then(pl.lit("TRB+")).otherwise(pl.lit("TRB-")).alias("trb"),
        )
        .group_by(["tra", "trb"])
        .agg(pl.sum("cells").alias("cells"))
        .sort(["tra", "trb"])
    )

    scirpy = pytest.importorskip("scirpy")
    _ = pytest.importorskip("scanpy")
    _ = pytest.importorskip("awkward")

    adata = scirpy.io.read_10x_vdj(all_contig, filtered=False)
    airr = adata.obsm["airr"]

    import awkward as ak

    n_tra = ak.to_numpy(ak.sum(airr.locus == "TRA", axis=1))
    n_trb = ak.to_numpy(ak.sum(airr.locus == "TRB", axis=1))
    sc_df = pl.DataFrame({"n_tra": n_tra, "n_trb": n_trb})

    sc_presence = (
        sc_df.with_columns(
            pl.when(pl.col("n_tra") > 0).then(pl.lit("TRA+")).otherwise(pl.lit("TRA-")).alias("tra"),
            pl.when(pl.col("n_trb") > 0).then(pl.lit("TRB+")).otherwise(pl.lit("TRB-")).alias("trb"),
        )
        .group_by(["tra", "trb"])
        .len()
        .rename({"len": "cells"})
        .sort(["tra", "trb"])
    )

    mir_map = {
        (r["tra"], r["trb"]): int(r["cells"])
        for r in mir_presence.to_dicts()
    }
    sc_map = {
        (r["tra"], r["trb"]): int(r["cells"])
        for r in sc_presence.to_dicts()
    }

    # Both loaders should preserve the major positive quadrant and orphan bins.
    for quadrant in (("TRA+", "TRB+"), ("TRA+", "TRB-"), ("TRA-", "TRB+")):
        assert mir_map.get(quadrant, 0) > 0
        assert sc_map.get(quadrant, 0) > 0

    # Dominant TRA+/TRB+ counts should be reasonably close (within 25%).
    dominant_mir = mir_map[("TRA+", "TRB+")]
    dominant_sc = sc_map[("TRA+", "TRB+")]
    rel_gap = abs(dominant_mir - dominant_sc) / max(dominant_mir, dominant_sc)
    assert rel_gap < 0.25

    # scirpy may retain an explicit TRA-/TRB- bin that mirpy currently omits.
    assert sc_map.get(("TRA-", "TRB-"), 0) >= mir_map.get(("TRA-", "TRB-"), 0)
