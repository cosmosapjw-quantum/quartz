"""Tests for the Stage 7 / C6 trace p_flip channel."""

from __future__ import annotations

import numpy as np

from quartz.phase15_trace import (
    TRACE_CACHE_SCHEMA_VERSION,
    build_trace_artifact,
    load_cached_trace,
    store_cached_trace,
)


def test_build_trace_artifact_records_p_flips_and_bumps_schema():
    assert TRACE_CACHE_SCHEMA_VERSION >= 6
    art = build_trace_artifact(
        [8, 16, 32],
        [np.array([0.7, 0.3]), np.array([0.6, 0.4]), np.array([0.9, 0.1])],
        [1.0, 2.0, 3.0],
        source="fresh",
        trace_p_flips=[0.4, 0.2, 0.05],
        checkpoint_id="C01", position_id="P0007",
    )
    assert art["trace_cache_schema_version"] == TRACE_CACHE_SCHEMA_VERSION
    assert art["trace_p_flips"] == [0.4, 0.2, 0.05]
    assert len(art["trace_p_flips"]) == len(art["trace_budgets"])
    # Stage 7: bundles self-identify for the O6 join
    assert art["checkpoint_id"] == "C01"
    assert art["position_id"] == "P0007"


def test_missing_p_flip_pads_none_backcompat():
    art = build_trace_artifact(
        [8, 16],
        [np.array([0.5, 0.5]), np.array([0.8, 0.2])],
        [1.0, 1.0],
        source="fresh",
    )
    # no trace_p_flips passed => None-padded, one per budget
    assert art["trace_p_flips"] == [None, None]
    # a partial list keeps its Nones
    art2 = build_trace_artifact(
        [8, 16], [np.array([0.5, 0.5]), np.array([0.8, 0.2])], [1.0, 1.0],
        source="fresh", trace_p_flips=[None, 0.1],
    )
    assert art2["trace_p_flips"] == [None, 0.1]


def test_trace_p_flips_roundtrip_through_cache(tmp_path):
    art = build_trace_artifact(
        [8, 16], [np.array([0.7, 0.3]), np.array([0.9, 0.1])], [1.0, 2.0],
        source="fresh", trace_p_flips=[0.3, 0.05],
    )
    store_cached_trace(tmp_path, "k1", art)
    loaded = load_cached_trace(tmp_path, "k1")
    assert loaded is not None
    assert loaded["trace_p_flips"] == [0.3, 0.05]
    assert loaded["trace_cache_schema_version"] == TRACE_CACHE_SCHEMA_VERSION
