from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from goalexit.data.dataset import GoalExitStore
from goalexit.features.schema import B0_FEATURES, STUDENT_FEATURES
from goalexit.models.goalexit import PrivilegedTeacher, make_deployment_model, parameter_counts
from goalexit.train.engine import (
    _length_batches,
    _shuffled_batches,
    _student_batch,
    _teacher_batch,
    _training_loss,
    predict_deployment,
    predict_teacher,
)
from goalexit.utils import deterministic_seed, save_json, sha256


def smooth_binary_targets(target: torch.Tensor, epsilon: float) -> torch.Tensor:
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("Label smoothing must be in [0, 1)")
    return target * (1.0 - epsilon) + 0.5 * epsilon


def _save_fixed_resume(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_fixed_resume(
    path: Path, expected_run_identity: str, device: torch.device
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "goalexit-fixed-refit-resume-v1":
        raise RuntimeError(f"Unrecognized fixed-refit resume schema at {path}")
    if payload.get("run_identity_sha256") != expected_run_identity:
        raise RuntimeError("Fixed-refit state belongs to a different frozen run identity")
    return payload


def fit_fixed_epochs(
    *,
    family: str,
    training_ids: list[str],
    prediction_ids: list[str],
    training_targets: pd.DataFrame,
    prediction_metadata: pd.DataFrame,
    store: GoalExitStore,
    learning_rate: float,
    fixed_epochs: int,
    seed: int,
    weight_decay: float,
    batch_size: int,
    gradient_clip: float,
    dropout: float,
    device: torch.device,
    run_directory: Path,
    checkpoint_path: Path,
    prediction_path: Path,
    run_config: dict[str, Any],
    text_cache: np.ndarray | None = None,
    text_index: dict[str, int] | None = None,
    teacher_training_logits: dict[str, float] | None = None,
    alpha: float | None = None,
    temperature: float = 2.0,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    if family not in {"b0", "b1", "teacher", "t_text", "d", "gaze_teacher", "d_gaze", "ls"}:
        raise ValueError(f"Unauthorized fixed-refit family {family}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("Label smoothing must be in [0, 1)")
    if family != "ls" and label_smoothing != 0.0:
        raise ValueError("Label smoothing is only valid for the LS family")
    if fixed_epochs < 1:
        raise ValueError("Fixed refit requires at least one epoch")
    if set(training_ids) & set(prediction_ids):
        raise PermissionError("Fixed-refit training and prediction supports overlap")
    deterministic_seed(seed)
    if family in {"teacher", "t_text", "gaze_teacher"}:
        if text_cache is None or text_index is None:
            raise ValueError("Teacher fixed refit lacks frozen text cache")
        preprocessor = store.fit_preprocessor(family, training_ids)
        train_gaze = store.transformed_teacher_gaze(training_ids, preprocessor)
        model: nn.Module = PrivilegedTeacher(
            maximum_words=int(text_cache.shape[1]),
            text_only=family == "t_text",
            gaze_only=family == "gaze_teacher",
            dropout=dropout,
        ).to(device)
        train_lengths = {key: len(train_gaze[key]) for key in training_ids}
    else:
        preprocessor = store.fit_preprocessor(family, training_ids)
        if family == "b0":
            train_b0 = store.transformed_b0(training_ids, preprocessor)
            model = make_deployment_model("b0", len(B0_FEATURES), dropout=dropout).to(device)
            train_lengths = {key: 1 for key in training_ids}
        else:
            train_student = store.transformed_student(training_ids, preprocessor)
            model = make_deployment_model(family, len(STUDENT_FEATURES), dropout=dropout).to(device)
            train_lengths = {key: len(train_student[key]) for key in training_ids}
    if family in {"d", "d_gaze"}:
        if teacher_training_logits is None or set(teacher_training_logits) != set(training_ids):
            raise ValueError("D fixed refit requires exact training-only teacher logits")
        if alpha is None:
            raise ValueError("D fixed refit lacks frozen alpha")
    target_map = training_targets.set_index("trial_id")["goal_index"].astype(float).to_dict()
    if set(target_map) != set(training_ids):
        raise ValueError("Fixed-refit target support mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    resume_path = run_directory / "partial_training_state.pt"
    start_epoch = 1
    history: list[dict[str, float | int]] = []
    resume_events = 0
    resume = _load_fixed_resume(
        resume_path, str(run_config["run_identity_sha256"]), device
    )
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        history = list(resume["history"])
        start_epoch = int(resume["next_epoch"])
        resume_events = int(resume.get("resume_events", 0)) + 1
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device=device)
    for epoch in range(start_epoch, fixed_epochs + 1):
        model.train()
        batches = (
            _shuffled_batches(len(training_ids), batch_size, seed, epoch)
            if family == "b0"
            else _length_batches(training_ids, train_lengths, batch_size, seed, epoch)
        )
        losses: list[float] = []
        for positions in batches:
            optimizer.zero_grad(set_to_none=True)
            if family == "b0":
                selected = [training_ids[int(position)] for position in positions]
                values = np.stack([train_b0[key]["FULL"] for key in selected])
                logits = model(torch.from_numpy(values).to(device))
            elif family in {"teacher", "t_text", "gaze_teacher"}:
                text, gaze_values, mask, selected = _teacher_batch(
                    training_ids,
                    train_gaze,
                    store.paragraph_ids,
                    text_cache,
                    text_index,
                    positions,
                    device,
                )
                logits = model(text, gaze_values, mask)
            else:
                values, lengths, selected, _ = _student_batch(
                    training_ids, train_student, positions, device, "FULL"
                )
                logits = model(values, lengths)
            target = torch.as_tensor(
                [target_map[key] for key in selected], dtype=torch.float32, device=device
            )
            if family == "ls":
                target = smooth_binary_targets(target, label_smoothing)
            teacher = (
                torch.as_tensor(
                    [teacher_training_logits[key] for key in selected],
                    dtype=torch.float32,
                    device=device,
                )
                if teacher_training_logits is not None
                else None
            )
            loss = _training_loss(
                family, logits, target, teacher, alpha, temperature
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses))})
        _save_fixed_resume(
            resume_path,
            {
                "schema": "goalexit-fixed-refit-resume-v1",
                "run_identity_sha256": run_config["run_identity_sha256"],
                "next_epoch": epoch + 1,
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "resume_events": resume_events,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device).cpu()
                    if device.type == "cuda"
                    else None
                ),
            },
        )
    if len(history) != fixed_epochs:
        raise RuntimeError("Fixed-refit history does not reach the frozen epoch count")
    model.eval()
    preprocessor_record = preprocessor.to_dict()
    preprocessor_record["fit_scope"] = "current outer population-training trial support only"
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "goalexit-outer-checkpoint-v1",
        "family": family,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "preprocessor": preprocessor_record,
        "maximum_words": int(text_cache.shape[1]) if family in {"teacher", "t_text", "gaze_teacher"} else None,
        "text_only": family == "t_text",
        "gaze_only": family == "gaze_teacher",
        "run_config": run_config,
        "fixed_epochs": fixed_epochs,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path)
    frame = (
        predict_teacher(
            model,
            prediction_ids,
            store,
            preprocessor,
            text_cache,
            text_index,
            batch_size=batch_size,
            device=device,
        )
        if family in {"teacher", "t_text", "gaze_teacher"}
        else predict_deployment(
            model,
            family,
            prediction_ids,
            store,
            preprocessor,
            "FULL",
            batch_size=batch_size,
            device=device,
        )
    )
    frame = frame.rename(columns={"probability_h": "probability"})
    if "probability" not in frame or "probability_h" in frame:
        raise RuntimeError("Fixed-refit prediction probability schema is invalid")
    metadata = prediction_metadata.set_index("trial_id").loc[frame["trial_id"]]
    frame["reader_id"] = metadata["reader_id"].astype(str).to_numpy()
    frame["article_id"] = metadata["article_id"].astype(str).to_numpy()
    frame["full_fixation_count"] = metadata["full_fixation_count"].to_numpy(np.int32)
    for key in (
        "git_commit",
        "confirmatory_config_sha256",
        "canonical_data_sha256",
        "split_manifest_sha256",
        "lock_sha256",
        "context_id",
        "model_family",
        "seed",
        "run_identity_sha256",
    ):
        frame[key] = run_config[key]
    frame["checkpoint_sha256"] = checkpoint_hash
    frame["outer_labels_accessed"] = False
    frame["outer_scientific_metrics_computed"] = False
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(prediction_path, index=False, compression="zstd")
    result = {
        "schema": "goalexit-outer-refit-result-v1",
        "status": "complete",
        "family": family,
        "fixed_epochs": fixed_epochs,
        "epoch_resume_enabled": True,
        "resume_events": resume_events,
        "history": history,
        "parameter_counts": parameter_counts(model),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "prediction_rows": len(frame),
        "preprocessor": preprocessor_record,
        "run_config": run_config,
        "outer_labels_accessed": False,
        "outer_scientific_metrics_computed": False,
    }
    result_path = run_directory / "result.json"
    save_json(result_path, result)
    result["result_path"] = str(result_path)
    result["result_sha256"] = sha256(result_path)
    resume_path.unlink(missing_ok=True)
    return result
