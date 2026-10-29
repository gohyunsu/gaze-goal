from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NumericPreprocessor:
    feature_names: tuple[str, ...]
    continuous_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    training_trial_count: int
    training_observation_count: int

    @property
    def continuous_positions(self) -> np.ndarray:
        return np.asarray(
            [self.feature_names.index(name) for name in self.continuous_names],
            dtype=np.int64,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        if result.ndim != 2 or result.shape[1] != len(self.feature_names):
            raise ValueError(f"Unexpected preprocessing shape {result.shape}")
        positions = self.continuous_positions
        block = result[:, positions].astype(np.float64)
        missing = ~np.isfinite(block)
        if missing.any():
            block[missing] = np.take(self.medians, np.where(missing)[1])
        block = (block - self.means) / self.scales
        result[:, positions] = block.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Preprocessed values contain non-finite numbers")
        return result

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "continuous_names": list(self.continuous_names),
            "continuous_medians": self.medians.tolist(),
            "continuous_means_after_imputation": self.means.tolist(),
            "continuous_population_scales": self.scales.tolist(),
            "training_trial_count": self.training_trial_count,
            "training_observation_count": self.training_observation_count,
            "fit_scope": "current inner training trial support only",
            "validation_or_outer_statistics_used": False,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "NumericPreprocessor":
        return cls(
            feature_names=tuple(value["feature_names"]),
            continuous_names=tuple(value["continuous_names"]),
            medians=np.asarray(value["continuous_medians"], dtype=np.float64),
            means=np.asarray(value["continuous_means_after_imputation"], dtype=np.float64),
            scales=np.asarray(value["continuous_population_scales"], dtype=np.float64),
            training_trial_count=int(value["training_trial_count"]),
            training_observation_count=int(value["training_observation_count"]),
        )


def fit_numeric_preprocessor(
    values: np.ndarray,
    feature_names: list[str],
    continuous_names: list[str],
    *,
    training_trial_count: int,
) -> NumericPreprocessor:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(feature_names) or len(array) == 0:
        raise ValueError("Invalid preprocessing fit matrix")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Duplicate feature name")
    if not set(continuous_names).issubset(feature_names):
        raise ValueError("Continuous feature absent from input")
    positions = np.asarray([feature_names.index(name) for name in continuous_names])
    block = array[:, positions].copy()
    block[~np.isfinite(block)] = np.nan
    medians = np.nanmedian(block, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("Training feature is entirely missing")
    missing = np.isnan(block)
    if missing.any():
        block[missing] = np.take(medians, np.where(missing)[1])
    means = block.mean(axis=0)
    scales = block.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    return NumericPreprocessor(
        feature_names=tuple(feature_names),
        continuous_names=tuple(continuous_names),
        medians=medians,
        means=means,
        scales=scales,
        training_trial_count=training_trial_count,
        training_observation_count=len(array),
    )
