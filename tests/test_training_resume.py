from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from goalexit.train.engine import _load_resume_state, _save_resume_state
from goalexit.train import run_development
from goalexit.train.run_development import _validate_teacher_cache_entry
from goalexit.utils import canonical_json_sha256, sha256


def test_epoch_resume_state_is_atomic_identity_bound_and_restores_rng(tmp_path) -> None:
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    path = tmp_path / "partial_training_state.pt"
    payload = {
        "schema": "goalexit-epoch-resume-v1",
        "run_identity_sha256": "frozen-run",
        "next_epoch": 4,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {},
        "best_state_dict": {"weight": torch.tensor([0.5])},
        "best_score": (0.6, -0.7),
        "best_epoch": 2,
        "stale_epochs": 1,
        "history": [{"epoch": 1}],
        "resume_events": 0,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    _save_resume_state(path, payload)
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    expected = torch.rand(5)
    torch.manual_seed(999)
    loaded = _load_resume_state(path, "frozen-run", torch.device("cpu"))
    assert loaded is not None
    torch.set_rng_state(loaded["torch_rng_state"])
    torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)
    assert loaded["next_epoch"] == 4

    with pytest.raises(RuntimeError, match="different frozen run identity"):
        _load_resume_state(path, "different-run", torch.device("cpu"))


def test_teacher_cache_entry_requires_exact_hash_membership_and_provenance(tmp_path) -> None:
    cache_path = tmp_path / "teacher.parquet"
    checkpoint_path = tmp_path / "teacher.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    training_ids = ["trial-b", "trial-a"]
    pd.DataFrame(
        {"trial_id": training_ids, "teacher_logit": [0.25, -0.5]}
    ).to_parquet(cache_path, index=False)
    entry = {
        "context_id": "r0_a0",
        "inner_holdout": 0,
        "training_trials": 2,
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "source_teacher_checkpoint_path": str(checkpoint_path),
        "source_teacher_checkpoint_sha256": sha256(checkpoint_path),
        "source_teacher_run_identity_sha256": "run-id",
        "cache_path": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "outer_rows_touched": False,
    }
    arguments = {
        "context_id": "r0_a0",
        "inner": 0,
        "training_ids": training_ids,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256(checkpoint_path),
        "run_identity_sha256": "run-id",
        "cache_path": cache_path,
    }
    _validate_teacher_cache_entry(entry, **arguments)

    bad = dict(entry, source_teacher_run_identity_sha256="wrong")
    with pytest.raises(ValueError, match="provenance mismatch"):
        _validate_teacher_cache_entry(bad, **arguments)

    pd.DataFrame(
        {"trial_id": ["trial-a"], "teacher_logit": [0.1]}
    ).to_parquet(cache_path, index=False)
    entry["cache_sha256"] = sha256(cache_path)
    with pytest.raises(ValueError, match="membership mismatch"):
        _validate_teacher_cache_entry(entry, **arguments)


def test_development_stage_uses_spawn_process_context(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    class FakeFuture:
        def __init__(self, context_id: str) -> None:
            self.context_id = context_id

        def result(self):
            return {"context_id": self.context_id, "families": {}}

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context) -> None:
            observed["max_workers"] = max_workers
            observed["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def submit(self, _function, _config, index, _families, _inherited):
            return FakeFuture(f"r{index // 5}_a{index % 5}")

    monkeypatch.setattr(run_development, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        run_development,
        "load_yaml",
        lambda _path: {"parallel_workers": 4},
    )
    monkeypatch.setattr(
        run_development.concurrent.futures,
        "ProcessPoolExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        run_development.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )

    result = run_development._run_stage(
        tmp_path / "config.yaml",
        ("d",),
    )
    assert len(result) == 25
    assert observed == {"max_workers": 4, "start_method": "spawn"}
