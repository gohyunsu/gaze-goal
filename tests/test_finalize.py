from __future__ import annotations

import numpy as np

from goalexit.audit.finalize import (
    _bootstrap_draws,
    reader_cluster_bootstrap,
    reader_cluster_contrast,
)


def test_reader_cluster_bootstrap_is_deterministic_and_goal_stratified() -> None:
    goals = {"o0": 0, "o1": 0, "h0": 1, "h1": 1}
    stats = {
        "o0": {"goal": 0, "correct": 3, "total": 4},
        "o1": {"goal": 0, "correct": 1, "total": 2},
        "h0": {"goal": 1, "correct": 4, "total": 4},
        "h1": {"goal": 1, "correct": 1, "total": 2},
    }
    readers_a, draws_a = _bootstrap_draws(goals, 100, 17)
    readers_b, draws_b = _bootstrap_draws(goals, 100, 17)
    assert readers_a == readers_b == {0: ["o0", "o1"], 1: ["h0", "h1"]}
    assert all(np.array_equal(draws_a[goal], draws_b[goal]) for goal in (0, 1))
    result = reader_cluster_bootstrap(stats, readers_a, draws_a, seed=17)
    assert result["pooled_balanced_accuracy"] == (4 / 6 + 5 / 6) / 2
    assert result["replicates"] == 100
    assert result["stratified_by_reader_goal"] is True


def test_reader_cluster_contrast_uses_paired_draws() -> None:
    goals = {"o0": 0, "o1": 0, "h0": 1, "h1": 1}
    stats = {
        reader: {"goal": goal, "correct": 1, "total": 2}
        for reader, goal in goals.items()
    }
    readers, draws = _bootstrap_draws(goals, 50, 19)
    result = reader_cluster_contrast(stats, stats, readers, draws, seed=19)
    assert result["pooled_balanced_accuracy_difference"] == 0.0
    assert result["ci95_percentile"] == [0.0, 0.0]
    assert result["paired_reader_draws"] is True
