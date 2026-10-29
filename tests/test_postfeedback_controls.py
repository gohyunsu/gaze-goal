from __future__ import annotations

import pandas as pd

from goalexit.eval.postconfirmatory_ablation import _class_conditional_shuffle


def test_class_conditional_shuffle_is_distribution_preserving_derangement() -> None:
    logits = {f"t{index}": float(index) for index in range(8)}
    targets = pd.DataFrame(
        {
            "trial_id": list(logits),
            "goal_index": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    shuffled, mapping_hash = _class_conditional_shuffle(
        logits, targets, context_id="r0_a0", seed=13
    )
    assert len(mapping_hash) == 64
    assert all(shuffled[key] != logits[key] for key in logits)
    for label in (0, 1):
        ids = targets.loc[targets["goal_index"].eq(label), "trial_id"]
        assert sorted(shuffled[key] for key in ids) == sorted(logits[key] for key in ids)


def test_class_conditional_shuffle_is_deterministic() -> None:
    logits = {f"t{index}": float(index) for index in range(6)}
    targets = pd.DataFrame(
        {"trial_id": list(logits), "goal_index": [0, 0, 0, 1, 1, 1]}
    )
    first = _class_conditional_shuffle(logits, targets, context_id="r1_a2", seed=29)
    second = _class_conditional_shuffle(logits, targets, context_id="r1_a2", seed=29)
    assert first == second
