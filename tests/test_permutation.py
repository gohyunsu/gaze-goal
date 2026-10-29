from __future__ import annotations

import numpy as np
import pandas as pd

from goalexit.audit.permutation import _permuted_targets
from goalexit.utils import canonical_json_sha256


def test_permuted_reader_mapping_is_canonical_json_serializable() -> None:
    targets = pd.DataFrame(
        {
            "trial_id": ["t0", "t1", "t2", "t3"],
            "goal_index": [0, 0, 1, 1],
        }
    )
    metadata = pd.DataFrame(
        {"reader_id": ["r0", "r0", "r1", "r1"]},
        index=["t0", "t1", "t2", "t3"],
    )

    permuted, mapping = _permuted_targets(
        targets,
        metadata,
        targets["trial_id"].tolist(),
        np.random.default_rng(13),
    )

    assert all(type(value) is int for value in mapping.values())
    assert canonical_json_sha256(mapping)
    assert set(permuted["goal_index"]) == {0, 1}
