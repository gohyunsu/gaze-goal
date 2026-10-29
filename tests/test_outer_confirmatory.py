from __future__ import annotations

import pandas as pd

from goalexit.eval.final_evaluation import fixed_epoch_from_selection
from goalexit.eval.outer_analysis import _crossed_bootstrap, _reader_cluster_bootstrap


def _synthetic_outer_frame() -> pd.DataFrame:
    rows = []
    for goal, readers in ((0, ("o0", "o1", "o2")), (1, ("h0", "h1", "h2"))):
        for reader in readers:
            for article in ("a0", "a1", "a2"):
                rows.append(
                    {
                        "reader_id": reader,
                        "article_id": article,
                        "goal_index": goal,
                        "d": 0.1 if goal == 0 else 0.9,
                        "b1": 0.9 if goal == 0 else 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_fixed_epoch_is_integer_median_of_three_selected_epochs() -> None:
    selection = {"selected_results": [{"selected_epoch": 9}, {"selected_epoch": 2}, {"selected_epoch": 5}]}
    assert fixed_epoch_from_selection(selection) == 5


def test_confirmatory_bootstraps_are_paired_and_deterministic() -> None:
    frame = _synthetic_outer_frame()
    first = _reader_cluster_bootstrap(frame, "d", "b1", replicates=100, seed=41)
    second = _reader_cluster_bootstrap(frame, "d", "b1", replicates=100, seed=41)
    assert first == second
    assert first["point_difference"] == 1.0
    assert first["ci_95_percentile"] == [1.0, 1.0]
    crossed = _crossed_bootstrap(frame, "d", "b1", replicates=100, seed=42)
    assert crossed["ci_95_percentile"] == [1.0, 1.0]
