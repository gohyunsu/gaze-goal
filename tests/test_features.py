from __future__ import annotations

import numpy as np
import pandas as pd

from goalexit.features.extract import b0_prefix_features, decision_lengths, observed_length, student_prefix
from goalexit.features.schema import B0_FEATURES, STUDENT_BASE_FEATURES, STUDENT_FEATURES, audit_feature_names


def _frame(length: int, future_offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(length, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "duration_ms": 100 + index,
            "x_norm": index / 10,
            "y_norm": index / 20,
            "dx_norm": index / 30,
            "dy_norm": -index / 40,
            "line_change": (index % 7 == 0).astype(float),
        }
    )
    frame["log_fixation_duration"] = np.log1p(frame["duration_ms"])
    amplitude = np.hypot(frame["dx_norm"], frame["dy_norm"])
    frame["log_saccade_amplitude"] = np.log1p(amplitude)
    angle = np.arctan2(frame["dy_norm"], frame["dx_norm"])
    frame["saccade_angle_sin"] = np.sin(angle)
    frame["saccade_angle_cos"] = np.cos(angle)
    frame.loc[0, ["saccade_angle_sin", "saccade_angle_cos"]] = [0.0, 1.0]
    if length > 5:
        frame.loc[5:, ["duration_ms", "x_norm", "y_norm", "dx_norm", "dy_norm"]] += future_offset
    return frame


def test_feature_allowlists_are_exact_and_leakage_free() -> None:
    audit = audit_feature_names()
    assert audit["status"] == "pass"
    assert audit["violations"] == []
    assert len(STUDENT_FEATURES) == 10
    assert len(B0_FEATURES) == 22


def test_prefix_features_are_causal_and_position_uses_observed_length_only() -> None:
    short = _frame(5)
    long = _frame(30, future_offset=10000)
    np.testing.assert_allclose(
        b0_prefix_features(short, 5), b0_prefix_features(long, 5), rtol=0, atol=0
    )
    short_base = short[STUDENT_BASE_FEATURES].to_numpy(np.float32)
    long_base = long[STUDENT_BASE_FEATURES].to_numpy(np.float32)
    np.testing.assert_allclose(student_prefix(short_base, 5), student_prefix(long_base, 5))
    np.testing.assert_array_equal(student_prefix(short_base, 5)[:, -1], np.linspace(0, 1, 5))


def test_short_fixed_prefix_is_missing_and_decision_times_include_natural_end() -> None:
    assert observed_length(7, 10) is None
    assert observed_length(7, "FULL") == 7
    assert decision_lengths(4) == [4]
    assert decision_lengths(12) == [5, 10, 12]
    assert decision_lengths(15) == [5, 10, 15]
