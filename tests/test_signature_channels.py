"""The channel bridge: the layout names the column groups, mir.explain ablates them.

`channel_spec` is the only place the two meet, so what is tested here is that the map it hands
over is the one `mir.explain` expects — indices into the matrix the caller actually has, and an
`attributable` set that is declared by the layout rather than guessed from a name.
"""
from __future__ import annotations

import numpy as np
import pytest

from mir.explain import ChannelSpec, channel_report
from mir.signature import channel_spec, columns


def test_channel_spec_is_a_channelspec_over_the_whole_tier():
    spec = channel_spec("standard")
    assert isinstance(spec, ChannelSpec)
    assert spec.width == len(columns("standard"))


def test_only_the_geometry_blocks_are_attributable():
    """Attributability is a clonotype pre-image, not a naming convention: a Hill number has none."""
    spec = channel_spec("standard")
    assert spec.attributable == {"rsig:contrast", "rsig:phiv", "rsig:phij", "rsig:phic"}


def test_indices_follow_the_column_list_given_not_the_full_tier():
    """The usual call is `channel_spec(columns=frame.columns[1:])` — sample_id dropped."""
    cols = columns("core")[10:40]
    spec = channel_spec(columns=cols)
    assert spec.width == len(cols)
    assert max(i for v in spec.columns_by_name.values() for i in v) == len(cols) - 1


def test_per_locus_keys_keep_their_attributability():
    spec = channel_spec("core", per_locus=True)
    assert "rsig:contrast:TRB" in spec.attributable
    assert "vsig:div:TRB" not in spec.attributable


def test_a_report_runs_end_to_end_on_a_signature_shaped_matrix():
    """The point of the bridge: an ablation over signature columns names a channel, not an index."""
    cols = columns("core")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, len(cols)))
    spec = channel_spec("core")
    rep = channel_report(X, spec, lambda B: float(B.shape[1]), base=0.0)
    assert rep.best in spec.names
    assert set(rep.frame()["channel"]) == set(spec.names)


def test_an_unknown_channel_raises_rather_than_returning_nothing():
    with pytest.raises(ValueError):
        channel_spec("core").columns("vsig:not_a_channel")
