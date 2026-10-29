from __future__ import annotations

import numpy as np
import pytest

from goalexit.eval.metrics import evaluate_binary
from goalexit.eval.outer_guard import OuterEvaluationSealedError, require_outer_authorization


def test_binary_metrics_are_finite_and_exact_for_perfect_predictions() -> None:
    target = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.2, 0.8])
    articles = np.array(["a", "a", "b", "b"])
    metrics = evaluate_binary(target, probability, articles)
    assert metrics.balanced_accuracy == 1.0
    assert metrics.auroc == 1.0
    assert metrics.macro_by_article_balanced_accuracy == 1.0
    assert np.isfinite(list(metrics.to_dict().values())[:7]).all()


def test_outer_evaluation_is_sealed_without_lock(tmp_path) -> None:
    with pytest.raises(OuterEvaluationSealedError, match="final_goalexit_lock.yaml is absent"):
        require_outer_authorization(tmp_path)
