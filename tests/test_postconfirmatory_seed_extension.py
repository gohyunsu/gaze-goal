from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from goalexit.eval.postconfirmatory_ablation import (
    _is_teacher_cache_variant,
    _variant_spec,
)
from goalexit.eval.postconfirmatory_seed_extension import (
    _mean_crossed_bootstrap,
    _mean_reader_bootstrap,
    _variant_map,
)


def _selection_state() -> dict:
    selection = {
        "selected_learning_rate": 0.001,
        "selected_alpha": 0.5,
        "selected_candidate_id": "candidate",
        "selected_results": [
            {"selected_epoch": 2},
            {"selected_epoch": 3},
            {"selected_epoch": 4},
        ],
    }
    return {"selections": {"r0_a0": {"b1": selection, "teacher": selection, "d": selection}}}


def test_extension_variants_pair_each_condition_by_seed() -> None:
    config = {"base_seed": 13, "label_smoothing": 0.1}
    state = _selection_state()
    expected = {
        "label_smoothing_s29": ("ls", 29),
        "gaze_teacher_s29": ("gaze_teacher", 29),
        "gaze_kd_s47": ("d_gaze", 47),
    }
    for variant, (family, seed) in expected.items():
        spec = _variant_spec(variant, config, state, "r0_a0")
        assert spec["family"] == family
        assert spec["seed"] == seed
    assert _variant_map(29) == {
        "gru": "gru_s29",
        "label_smoothing": "label_smoothing_s29",
        "gaze_teacher": "gaze_teacher_s29",
        "gaze_kd": "gaze_kd_s29",
        "text_gaze_teacher": "text_gaze_teacher_s29",
        "text_gaze_kd": "text_gaze_kd_s29",
    }
    assert _is_teacher_cache_variant("gaze_teacher_s29")
    assert _is_teacher_cache_variant("text_gaze_teacher_s47")
    assert not _is_teacher_cache_variant("gaze_kd_s29")


def test_extension_runtime_outputs_do_not_dirty_source_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    probe = "experiments/postconfirmatory_seed_extension/probe.json"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", probe],
        cwd=root,
        check=False,
    )
    assert result.returncode == 0


def test_multiseed_bootstraps_are_paired_and_deterministic() -> None:
    rows = []
    for goal, readers in ((0, ("o0", "o1")), (1, ("h0", "h1"))):
        for reader in readers:
            for article in ("a0", "a1"):
                row = {"reader_id": reader, "article_id": article, "goal_index": goal}
                for seed in (13, 29, 47):
                    row[f"left_{seed}"] = 0.1 if goal == 0 else 0.9
                    row[f"right_{seed}"] = 0.9 if goal == 0 else 0.1
                rows.append(row)
    frame = pd.DataFrame(rows)
    pairs = tuple((f"left_{seed}", f"right_{seed}") for seed in (13, 29, 47))
    reader = _mean_reader_bootstrap(frame, pairs, replicates=50, seed=7)
    assert reader["point_mean_difference"] == 1.0
    assert reader["ci_95_percentile"] == [1.0, 1.0]
    assert reader == _mean_reader_bootstrap(frame, pairs, replicates=50, seed=7)
    crossed = _mean_crossed_bootstrap(frame, pairs, replicates=50, seed=8)
    assert crossed["ci_95_percentile"] == [1.0, 1.0]
