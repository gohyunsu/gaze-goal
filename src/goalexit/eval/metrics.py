from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    balanced_accuracy: float
    auroc: float
    macro_f1: float
    brier: float
    nll: float
    ece: float
    macro_by_article_balanced_accuracy: float
    confusion_counts: tuple[tuple[int, int], tuple[int, int]]
    trial_count: int

    def to_dict(self) -> dict:
        value = asdict(self)
        value["confusion_counts"] = [list(row) for row in self.confusion_counts]
        return value


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    confidence = np.where(prediction == 1, probability, 1.0 - probability)
    correct = prediction == target
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(result)


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    recalls: list[float] = []
    for label in (0, 1):
        mask = target == label
        if not mask.any():
            return float("nan")
        recalls.append(float((prediction[mask] == label).mean()))
    return float(np.mean(recalls))


def evaluate_binary(
    target: np.ndarray,
    probability: np.ndarray,
    article_ids: np.ndarray,
    *,
    ece_bins: int = 15,
) -> BinaryMetrics:
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    article_ids = np.asarray(article_ids).astype(str)
    if probability.shape != (len(target),):
        raise ValueError("Binary probabilities must be one-dimensional")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Invalid binary probabilities")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Evaluation population must contain both classes")
    prediction = (probability >= 0.5).astype(np.int64)
    confusion = np.asarray(
        [
            [int(((target == truth) & (prediction == predicted)).sum()) for predicted in (0, 1)]
            for truth in (0, 1)
        ]
    )
    article_scores = []
    for article in sorted(set(article_ids)):
        mask = article_ids == article
        score = _balanced_accuracy(target[mask], prediction[mask])
        if np.isfinite(score):
            article_scores.append(score)
    if not article_scores:
        raise ValueError("No article has both target classes")
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return BinaryMetrics(
        balanced_accuracy=_balanced_accuracy(target, prediction),
        auroc=float(roc_auc_score(target, probability)),
        macro_f1=float(f1_score(target, prediction, labels=[0, 1], average="macro")),
        brier=float(np.mean((probability - target) ** 2)),
        nll=float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))),
        ece=expected_calibration_error(target, probability, bins=ece_bins),
        macro_by_article_balanced_accuracy=float(np.mean(article_scores)),
        confusion_counts=(
            tuple(int(value) for value in confusion[0]),
            tuple(int(value) for value in confusion[1]),
        ),
        trial_count=len(target),
    )


def aggregate_metrics(rows: list[dict], metric: str = "balanced_accuracy") -> dict:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite aggregate metric {metric}")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "evaluations": len(values),
    }


def paired_contrast(left: list[dict], right: list[dict]) -> dict:
    left_map = {
        (row["context_id"], int(row["inner_holdout"])): float(row["metrics"]["balanced_accuracy"])
        for row in left
    }
    right_map = {
        (row["context_id"], int(row["inner_holdout"])): float(row["metrics"]["balanced_accuracy"])
        for row in right
    }
    if set(left_map) != set(right_map) or len(left_map) != 75:
        raise ValueError("Paired contrast does not cover identical 75 evaluations")
    differences = np.asarray([left_map[key] - right_map[key] for key in sorted(left_map)])
    return {
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "q25_difference": float(np.quantile(differences, 0.25)),
        "q75_difference": float(np.quantile(differences, 0.75)),
        "positive_direction_count": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "evaluations": len(differences),
    }
