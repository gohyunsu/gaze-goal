from __future__ import annotations

import numpy as np
import pandas as pd

from goalexit.eval.postfeedback_strong_gru import (
    _candidate_id,
    _class_mean_logits,
    _selection,
)


def _result(ba: float, nll: float, parameters: int, epoch: int = 3) -> dict:
    return {
        "metrics": {"FULL": {"balanced_accuracy": ba, "nll": nll}},
        "run_config": {"learning_rate": 0.001, "alpha": None},
        "parameter_counts": {"trainable": parameters},
        "selected_epoch": epoch,
    }


def test_bce_selection_prefers_validation_score_before_capacity() -> None:
    small_id = _candidate_id("bce", "small", 0.001, None)
    wide_id = _candidate_id("bce", "wide", 0.001, None)
    candidates = {
        small_id: [_result(0.60, 0.7, 10) for _ in range(3)],
        wide_id: [_result(0.61, 0.8, 20) for _ in range(3)],
    }
    architectures = {
        small_id: ("small", {"hidden_size": 32}),
        wide_id: ("wide", {"hidden_size": 64}),
    }
    selected = _selection("bce", candidates, architectures)
    assert selected["selected_architecture_id"] == "wide"


def test_class_mean_targets_remove_trial_correspondence() -> None:
    targets = pd.DataFrame(
        {"trial_id": ["a", "b", "c", "d"], "goal_index": [0, 0, 1, 1]}
    )
    observed = _class_mean_logits(
        {"a": -3.0, "b": -1.0, "c": 1.0, "d": 5.0},
        targets,
        temperature=2.0,
    )
    assert observed["a"] == observed["b"]
    assert observed["c"] == observed["d"]
    teacher = np.asarray([-3.0, -1.0, 1.0, 5.0])
    softened = 1.0 / (1.0 + np.exp(-teacher / 2.0))
    assert np.isclose(1.0 / (1.0 + np.exp(-observed["a"] / 2.0)), softened[:2].mean())
    assert np.isclose(1.0 / (1.0 + np.exp(-observed["c"] / 2.0)), softened[2:].mean())


def test_class_mean_targets_reject_nonpositive_temperature() -> None:
    targets = pd.DataFrame({"trial_id": ["a", "b"], "goal_index": [0, 1]})
    try:
        _class_mean_logits({"a": -1.0, "b": 1.0}, targets, temperature=0.0)
    except ValueError as error:
        assert "temperature" in str(error).lower()
    else:
        raise AssertionError("Expected nonpositive temperature to be rejected")
