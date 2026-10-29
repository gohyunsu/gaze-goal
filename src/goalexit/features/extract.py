from __future__ import annotations

import numpy as np
import pandas as pd

from goalexit.features.schema import B0_FEATURES, PREFIXES, STUDENT_FEATURES


def observed_length(full_length: int, checkpoint: int | str) -> int | None:
    if full_length < 1:
        raise ValueError("A trial must contain at least one fixation")
    if checkpoint == "FULL":
        return full_length
    value = int(checkpoint)
    if value < 1:
        raise ValueError("A checkpoint must be positive")
    return value if full_length >= value else None


def student_prefix(base: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(base, dtype=np.float32)
    if values.ndim != 2 or length < 1 or length > len(values):
        raise ValueError("Invalid causal student prefix")
    positions = np.arange(length, dtype=np.float32) / max(1, length - 1)
    result = np.column_stack([values[:length], positions]).astype(np.float32)
    if result.shape[1] != len(STUDENT_FEATURES) or not np.isfinite(result).all():
        raise ValueError("Invalid student event vector")
    return result


def b0_prefix_features(sequence: pd.DataFrame, length: int) -> np.ndarray:
    if length < 1 or length > len(sequence):
        raise ValueError("Invalid causal B0 prefix")
    prefix = sequence.iloc[:length]
    duration = prefix["duration_ms"].to_numpy(np.float64)
    dx = prefix["dx_norm"].to_numpy(np.float64)
    dy = prefix["dy_norm"].to_numpy(np.float64)
    amplitude = np.hypot(dx, dy)
    x = prefix["x_norm"].to_numpy(np.float64)
    y = prefix["y_norm"].to_numpy(np.float64)
    values = {
        "observed_fixation_count": float(length),
        "total_dwell_ms": float(duration.sum()),
        "mean_fixation_duration_ms": float(duration.mean()),
        "std_fixation_duration_ms": float(duration.std(ddof=0)),
        "median_fixation_duration_ms": float(np.median(duration)),
        "p25_fixation_duration_ms": float(np.quantile(duration, 0.25)),
        "p75_fixation_duration_ms": float(np.quantile(duration, 0.75)),
        "p90_fixation_duration_ms": float(np.quantile(duration, 0.90)),
        "mean_saccade_amplitude": float(amplitude.mean()),
        "std_saccade_amplitude": float(amplitude.std(ddof=0)),
        "median_saccade_amplitude": float(np.median(amplitude)),
        "p90_saccade_amplitude": float(np.quantile(amplitude, 0.90)),
        "mean_absolute_dx_norm": float(np.abs(dx).mean()),
        "mean_absolute_dy_norm": float(np.abs(dy).mean()),
        "net_dx_norm": float(dx.sum()),
        "net_dy_norm": float(dy.sum()),
        "forward_movement_proportion": float((dx > 0).mean()),
        "backward_movement_proportion": float((dx < 0).mean()),
        "line_change_rate": float(prefix["line_change"].mean()),
        "x_spatial_dispersion": float(x.std(ddof=0)),
        "y_spatial_dispersion": float(y.std(ddof=0)),
        "path_length_norm": float(amplitude.sum()),
    }
    if list(values) != B0_FEATURES:
        raise AssertionError("B0 feature order drift")
    output = np.asarray(list(values.values()), dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("B0 features contain non-finite values")
    return output


def all_b0_checkpoints(sequence: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for checkpoint in PREFIXES:
        length = observed_length(len(sequence), checkpoint)
        if length is not None:
            result[str(checkpoint)] = b0_prefix_features(sequence, length)
    return result


def decision_lengths(full_length: int) -> list[int]:
    if full_length < 1:
        raise ValueError("A trial must contain at least one fixation")
    if full_length < 5:
        return [full_length]
    values = list(range(5, full_length + 1, 5))
    if values[-1] != full_length:
        values.append(full_length)
    return values
