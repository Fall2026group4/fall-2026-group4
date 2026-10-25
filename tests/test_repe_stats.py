"""Tests for src/repe/stats.py (the random-control statistics). Needs only numpy."""

import numpy as np

from src.repe.stats import bh_adjust, min_attainable_p, rank_summary


def test_smallest_attainable_p_shrinks_with_more_controls():
    assert round(min_attainable_p(20), 3) == 0.048   # Review #5: 1/21
    assert round(min_attainable_p(100), 4) == 0.0099  # Review #6: 1/101
    assert min_attainable_p(200) < 0.005             # the Review #6 target


def test_bh_adjust_is_monotone_and_never_below_raw_p():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    adj = bh_adjust(p)
    assert np.all(adj >= p)
    assert np.all(adj <= 1.0)
    assert list(np.argsort(adj, kind="stable")) == list(np.argsort(p, kind="stable"))


def _directional(own_value, n_random, seed=0):
    rng = np.random.default_rng(seed)
    d = {"own": np.full(40, own_value) + rng.normal(0, 0.01, 40)}
    for i in range(n_random):
        d[f"random_{i}"] = rng.normal(0, 0.05, 40)
    return d


def test_strong_own_direction_ranks_first_with_the_floor_p_value():
    rng = np.random.default_rng(1)
    d = _directional(own_value=5.0, n_random=200)
    out = rank_summary(d, "own", rng, np.ones(40, dtype=bool))
    assert out["own_rank"] == 1
    assert out["n_random"] == 200
    assert abs(out["p_value"] - 1 / 201) < 1e-12
    assert out["p_value"] < 0.005


def test_zero_effect_direction_is_not_significant():
    rng = np.random.default_rng(2)
    d = _directional(own_value=0.0, n_random=200, seed=3)
    out = rank_summary(d, "own", rng, np.ones(40, dtype=bool))
    assert out["p_value"] > 0.05
    assert out["ci_low"] <= out["own_directional"] <= out["ci_high"]


def test_twenty_controls_cannot_reach_p_below_0_048():
    rng = np.random.default_rng(4)
    d = _directional(own_value=5.0, n_random=20)
    out = rank_summary(d, "own", rng, np.ones(40, dtype=bool))
    assert out["own_rank"] == 1
    assert out["p_value"] >= 1 / 21 - 1e-12
