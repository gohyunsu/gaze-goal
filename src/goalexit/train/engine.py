from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from goalexit.data.dataset import GoalExitStore
from goalexit.data.preprocessing import NumericPreprocessor
from goalexit.eval.metrics import evaluate_binary
from goalexit.features.extract import observed_length, student_prefix
from goalexit.features.schema import B0_FEATURES, PREFIXES, STUDENT_FEATURES
from goalexit.models.goalexit import PrivilegedTeacher, make_deployment_model, parameter_counts
from goalexit.utils import deterministic_seed, save_json, sha256


def bernoulli_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
    reduction: str = "mean",
) -> torch.Tensor:
    hard = nn.functional.binary_cross_entropy_with_logits(
        student_logits, target, reduction="none"
    )
    teacher = torch.sigmoid(teacher_logits / temperature).clamp(1e-7, 1 - 1e-7)
    student = torch.sigmoid(student_logits / temperature).clamp(1e-7, 1 - 1e-7)
    kl = teacher * (torch.log(teacher) - torch.log(student)) + (1 - teacher) * (
        torch.log1p(-teacher) - torch.log1p(-student)
    )
    value = (1 - alpha) * hard + alpha * temperature**2 * kl
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    raise ValueError(f"Unsupported reduction {reduction}")


def _shuffled_batches(size: int, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    positions = np.arange(size, dtype=np.int64)
    np.random.default_rng(seed + epoch * 1_000_003 + 17).shuffle(positions)
    return [positions[start : start + batch_size] for start in range(0, size, batch_size)]


def _length_batches(
    ids: list[str], lengths: dict[str, int], batch_size: int, seed: int, epoch: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + epoch * 1_000_003 + 17)
    positions = np.arange(len(ids), dtype=np.int64)
    rng.shuffle(positions)
    values = np.asarray([lengths[key] for key in ids], dtype=np.int32)
    batches: list[np.ndarray] = []
    for start in range(0, len(positions), batch_size * 8):
        pool = positions[start : start + batch_size * 8]
        pool = pool[np.argsort(values[pool], kind="stable")]
        batches.extend(
            pool[index : index + batch_size] for index in range(0, len(pool), batch_size)
        )
    rng.shuffle(batches)
    return batches


def _student_batch(
    ids: list[str],
    transformed: dict[str, np.ndarray],
    positions: np.ndarray,
    device: torch.device,
    checkpoint: int | str = "FULL",
) -> tuple[torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    selected_ids: list[str] = []
    arrays: list[np.ndarray] = []
    lengths: list[int] = []
    original_positions: list[int] = []
    for position in positions:
        key = ids[int(position)]
        length = observed_length(len(transformed[key]), checkpoint)
        if length is None:
            continue
        selected_ids.append(key)
        arrays.append(student_prefix(transformed[key], length))
        lengths.append(length)
        original_positions.append(int(position))
    if not arrays:
        raise ValueError("Student batch has no eligible sequences")
    maximum = max(lengths)
    padded = np.zeros((len(arrays), maximum, len(STUDENT_FEATURES)), dtype=np.float32)
    for index, (array, length) in enumerate(zip(arrays, lengths)):
        padded[index, :length] = array
    return (
        torch.from_numpy(padded).to(device),
        torch.as_tensor(lengths, dtype=torch.long, device=device),
        selected_ids,
        np.asarray(original_positions, dtype=np.int64),
    )


def _teacher_batch(
    ids: list[str],
    gaze: dict[str, np.ndarray],
    paragraph_ids: dict[str, str],
    text_cache: np.ndarray,
    text_index: dict[str, int],
    positions: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    selected = [ids[int(position)] for position in positions]
    lengths = [len(gaze[key]) for key in selected]
    maximum = max(lengths)
    text = np.zeros((len(selected), maximum, 768), dtype=np.float32)
    gaze_batch = np.zeros((len(selected), maximum, 8), dtype=np.float32)
    mask = np.ones((len(selected), maximum), dtype=bool)
    for index, (key, length) in enumerate(zip(selected, lengths)):
        paragraph_index = text_index[paragraph_ids[key]]
        text[index, :length] = np.asarray(text_cache[paragraph_index, :length], dtype=np.float32)
        gaze_batch[index, :length] = gaze[key]
        mask[index, :length] = False
    return (
        torch.from_numpy(text).to(device),
        torch.from_numpy(gaze_batch).to(device),
        torch.from_numpy(mask).to(device),
        selected,
    )


def _is_better(metrics: dict[str, Any], best: tuple[float, float] | None) -> bool:
    score = (float(metrics["balanced_accuracy"]), -float(metrics["nll"]))
    return best is None or score > best


@torch.no_grad()
def predict_deployment(
    model: nn.Module,
    family: str,
    ids: list[str],
    store: GoalExitStore,
    preprocessor: NumericPreprocessor,
    checkpoint: int | str,
    *,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    eligible = [
        key for key in ids if observed_length(len(store.student_base[key]), checkpoint) is not None
    ]
    logits = np.empty(len(eligible), dtype=np.float64)
    observed = np.empty(len(eligible), dtype=np.int32)
    if family == "b0":
        transformed = store.transformed_b0(eligible, preprocessor)
        for start in range(0, len(eligible), batch_size):
            batch_ids = eligible[start : start + batch_size]
            values = np.stack([transformed[key][str(checkpoint)] for key in batch_ids])
            logits[start : start + len(batch_ids)] = (
                model(torch.from_numpy(values).to(device)).cpu().numpy()
            )
            observed[start : start + len(batch_ids)] = [
                len(store.student_base[key])
                if checkpoint == "FULL"
                else int(checkpoint)
                for key in batch_ids
            ]
    else:
        transformed = store.transformed_student(eligible, preprocessor)
        lengths = {
            key: int(observed_length(len(transformed[key]), checkpoint)) for key in eligible
        }
        order = np.argsort([lengths[key] for key in eligible], kind="stable")
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            values, batch_lengths, _, original = _student_batch(
                eligible, transformed, positions, device, checkpoint
            )
            logits[original] = model(values, batch_lengths).cpu().numpy()
            observed[original] = batch_lengths.cpu().numpy()
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return pd.DataFrame(
        {
            "trial_id": eligible,
            "prefix": str(checkpoint),
            "observed_fixations": observed,
            "logit": logits,
            "probability_h": probability,
        }
    )


@torch.no_grad()
def predict_teacher(
    model: PrivilegedTeacher,
    ids: list[str],
    store: GoalExitStore,
    preprocessor: NumericPreprocessor,
    text_cache: np.ndarray,
    text_index: dict[str, int],
    *,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    gaze = store.transformed_teacher_gaze(ids, preprocessor)
    lengths = {key: len(gaze[key]) for key in ids}
    order = np.argsort([lengths[key] for key in ids], kind="stable")
    logits = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(order), batch_size):
        positions = order[start : start + batch_size]
        text, gaze_values, mask, _ = _teacher_batch(
            ids,
            gaze,
            store.paragraph_ids,
            text_cache,
            text_index,
            positions,
            device,
        )
        logits[positions] = model(text, gaze_values, mask).cpu().numpy()
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return pd.DataFrame(
        {
            "trial_id": ids,
            "prefix": "FULL",
            "observed_fixations": [len(store.student_base[key]) for key in ids],
            "logit": logits,
            "probability_h": probability,
        }
    )


def _metrics_for_frame(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    ece_bins: int,
) -> dict[str, Any]:
    target = targets.set_index("trial_id").loc[frame["trial_id"]]
    meta = metadata.set_index("trial_id").loc[frame["trial_id"]]
    return evaluate_binary(
        target["goal_index"].to_numpy(np.int64),
        frame["probability_h"].to_numpy(np.float64),
        meta["article_id"].astype(str).to_numpy(),
        ece_bins=ece_bins,
    ).to_dict()


def _attach_prediction_columns(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    target = targets.set_index("trial_id").loc[frame["trial_id"]]
    meta = metadata.set_index("trial_id").loc[frame["trial_id"]]
    output = frame.copy()
    output["target_goal_index"] = target["goal_index"].to_numpy(np.int8)
    output["reader_id"] = meta["reader_id"].astype(str).to_numpy()
    output["article_id"] = meta["article_id"].astype(str).to_numpy()
    output["full_fixation_count"] = meta["full_fixation_count"].to_numpy(np.int32)
    for key, value in provenance.items():
        output[key] = value
    output["outer_test_evaluated"] = False
    return output


def _training_loss(
    family: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    alpha: float | None,
    temperature: float,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if family in {"d", "a", "d_gaze"}:
        if teacher_logits is None or alpha is None:
            raise ValueError("Distilled model lacks teacher targets")
        return bernoulli_distillation_loss(
            logits,
            teacher_logits,
            targets,
            alpha=alpha,
            temperature=temperature,
            reduction=reduction,
        )
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction=reduction
    )


def _save_resume_state(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist one completed epoch of an official candidate run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_resume_state(
    path: Path, expected_run_identity: str, device: torch.device
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "goalexit-epoch-resume-v1":
        raise RuntimeError(f"Unrecognized candidate resume schema at {path}")
    if payload.get("run_identity_sha256") != expected_run_identity:
        raise RuntimeError(
            "Candidate resume state belongs to a different frozen run identity; "
            "quarantine it before continuing"
        )
    return payload


def train_candidate(
    *,
    family: str,
    training_ids: list[str],
    validation_ids: list[str],
    training_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    store: GoalExitStore,
    learning_rate: float,
    seed: int,
    weight_decay: float,
    maximum_epochs: int,
    patience: int,
    minimum_delta: float,
    batch_size: int,
    gradient_clip: float,
    dropout: float,
    ece_bins: int,
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
) -> dict[str, Any]:
    deterministic_seed(seed)
    if family in {"teacher", "t_text"}:
        if text_cache is None or text_index is None:
            raise ValueError("Teacher lacks frozen RoBERTa cache")
        preprocessor = store.fit_preprocessor(family, training_ids)
        train_gaze = store.transformed_teacher_gaze(training_ids, preprocessor)
        maximum_words = int(text_cache.shape[1])
        model: nn.Module = PrivilegedTeacher(
            maximum_words=maximum_words,
            text_only=family == "t_text",
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
    target_map = training_targets.set_index("trial_id")["goal_index"].astype(float).to_dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume_events = 0
    resume_path = run_directory / "partial_training_state.pt"
    resume = _load_resume_state(
        resume_path, str(run_config["run_identity_sha256"]), device
    )
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        best_state = resume["best_state_dict"]
        saved_score = resume["best_score"]
        best_score = tuple(float(value) for value in saved_score) if saved_score else None
        best_epoch = int(resume["best_epoch"])
        stale = int(resume["stale_epochs"])
        history = list(resume["history"])
        start_epoch = int(resume["next_epoch"])
        resume_events = int(resume.get("resume_events", 0)) + 1
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if stale >= patience:
            start_epoch = maximum_epochs + 1
    for epoch in range(start_epoch, maximum_epochs + 1):
        model.train()
        batches = (
            _shuffled_batches(len(training_ids), batch_size, seed, epoch)
            if family == "b0"
            else _length_batches(training_ids, train_lengths, batch_size, seed, epoch)
        )
        epoch_losses: list[float] = []
        for positions in batches:
            optimizer.zero_grad(set_to_none=True)
            if family == "b0":
                selected = [training_ids[int(position)] for position in positions]
                values = np.stack([train_b0[key]["FULL"] for key in selected])
                logits = model(torch.from_numpy(values).to(device))
                target = torch.as_tensor(
                    [target_map[key] for key in selected], dtype=torch.float32, device=device
                )
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
            elif family in {"teacher", "t_text"}:
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
                target = torch.as_tensor(
                    [target_map[key] for key in selected], dtype=torch.float32, device=device
                )
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            elif family == "a":
                selected_batch = [training_ids[int(position)] for position in positions]
                per_trial = torch.zeros(len(selected_batch), dtype=torch.float32, device=device)
                counts = torch.zeros(len(selected_batch), dtype=torch.float32, device=device)
                for checkpoint in PREFIXES:
                    eligible_local = np.asarray(
                        [
                            index
                            for index, key in enumerate(selected_batch)
                            if observed_length(len(train_student[key]), checkpoint) is not None
                        ],
                        dtype=np.int64,
                    )
                    if not len(eligible_local):
                        continue
                    values, lengths, selected, _ = _student_batch(
                        selected_batch,
                        train_student,
                        eligible_local,
                        device,
                        checkpoint,
                    )
                    logits = model(values, lengths)
                    target = torch.as_tensor(
                        [target_map[key] for key in selected], dtype=torch.float32, device=device
                    )
                    teacher = torch.as_tensor(
                        [teacher_training_logits[key] for key in selected],
                        dtype=torch.float32,
                        device=device,
                    )
                    losses = _training_loss(
                        family,
                        logits,
                        target,
                        teacher,
                        alpha,
                        temperature,
                        reduction="none",
                    )
                    local_tensor = torch.as_tensor(eligible_local, dtype=torch.long, device=device)
                    per_trial.index_add_(0, local_tensor, losses)
                    counts.index_add_(0, local_tensor, torch.ones_like(losses))
                if (counts == 0).any():
                    raise ValueError("Anytime trial has no authorized prefix")
                loss = (per_trial / counts).mean()
            else:
                values, lengths, selected, _ = _student_batch(
                    training_ids, train_student, positions, device, "FULL"
                )
                logits = model(values, lengths)
                target = torch.as_tensor(
                    [target_map[key] for key in selected], dtype=torch.float32, device=device
                )
                teacher = (
                    torch.as_tensor(
                        [teacher_training_logits[key] for key in selected],
                        dtype=torch.float32,
                        device=device,
                    )
                    if teacher_training_logits is not None
                    else None
                )
                loss = _training_loss(family, logits, target, teacher, alpha, temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if family in {"teacher", "t_text"}:
            validation_frame = predict_teacher(
                model,
                validation_ids,
                store,
                preprocessor,
                text_cache,
                text_index,
                batch_size=batch_size,
                device=device,
            )
        else:
            validation_frame = predict_deployment(
                model,
                family,
                validation_ids,
                store,
                preprocessor,
                "FULL",
                batch_size=batch_size,
                device=device,
            )
        metrics = _metrics_for_frame(
            validation_frame,
            validation_targets,
            validation_metadata,
            ece_bins=ece_bins,
        )
        history.append({"epoch": epoch, "training_loss": float(np.mean(epoch_losses)), **metrics})
        improved = _is_better(metrics, best_score)
        if improved:
            best_score = (float(metrics["balanced_accuracy"]), -float(metrics["nll"]))
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        _save_resume_state(
            resume_path,
            {
                "schema": "goalexit-epoch-resume-v1",
                "run_identity_sha256": run_config["run_identity_sha256"],
                "next_epoch": epoch + 1,
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale,
                "history": history,
                "resume_events": resume_events,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
            },
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No candidate checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema": "goalexit-checkpoint-v1",
        "family": family,
        "model_state_dict": best_state,
        "preprocessor": preprocessor.to_dict(),
        "maximum_words": int(text_cache.shape[1]) if family in {"teacher", "t_text"} else None,
        "text_only": family == "t_text",
        "run_config": run_config,
        "selected_epoch": best_epoch,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path)
    provenance = {
        key: run_config[key]
        for key in (
            "git_commit",
            "config_sha256",
            "canonical_data_sha256",
            "split_manifest_sha256",
            "context_id",
            "inner_holdout",
            "model_family",
            "candidate_id",
            "seed",
        )
    }
    provenance["checkpoint_sha256"] = checkpoint_hash
    frames: list[pd.DataFrame] = []
    metrics_by_prefix: dict[str, Any] = {}
    report_prefixes = (
        PREFIXES if family in {"b0", "b1", "a"} else ("FULL",)
    )
    for checkpoint in report_prefixes:
        frame = (
            predict_teacher(
                model,
                validation_ids,
                store,
                preprocessor,
                text_cache,
                text_index,
                batch_size=batch_size,
                device=device,
            )
            if family in {"teacher", "t_text"}
            else predict_deployment(
                model,
                family,
                validation_ids,
                store,
                preprocessor,
                checkpoint,
                batch_size=batch_size,
                device=device,
            )
        )
        metrics_by_prefix[str(checkpoint)] = _metrics_for_frame(
            frame,
            validation_targets,
            validation_metadata,
            ece_bins=ece_bins,
        )
        frames.append(
            _attach_prediction_columns(
                frame, validation_targets, validation_metadata, provenance
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    prediction_hash = sha256(prediction_path)
    metrics_path = run_directory / "metrics.json"
    result = {
        "schema": "goalexit-candidate-result-v1",
        "status": "complete",
        "family": family,
        "selected_epoch": best_epoch,
        "epochs_ran": len(history),
        "epoch_resume_enabled": True,
        "resume_events": resume_events,
        "metrics": metrics_by_prefix,
        "history": history,
        "parameter_counts": parameter_counts(model),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_hash,
        "prediction_rows": len(predictions),
        "preprocessor": preprocessor.to_dict(),
        "run_config": run_config,
        "outer_test_evaluated": False,
    }
    save_json(metrics_path, result)
    result["metrics_path"] = str(metrics_path)
    result["metrics_sha256"] = sha256(metrics_path)
    resume_path.unlink(missing_ok=True)
    return result


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, NumericPreprocessor, dict[str, Any]]:
    value = torch.load(path, map_location=device, weights_only=False)
    family = str(value["family"])
    if family in {"teacher", "t_text", "gaze_teacher"}:
        model: nn.Module = PrivilegedTeacher(
            maximum_words=int(value["maximum_words"]),
            text_only=bool(value["text_only"]),
            gaze_only=bool(value.get("gaze_only", False)),
            dropout=float(value["run_config"]["dropout"]),
        )
    else:
        deployment_family = "b1" if family in {"ls", "d_gaze"} else family
        model = make_deployment_model(
            deployment_family,
            len(B0_FEATURES) if family == "b0" else len(STUDENT_FEATURES),
            dropout=float(value["run_config"]["dropout"]),
        )
    model.load_state_dict(value["model_state_dict"])
    model.to(device).eval()
    return model, NumericPreprocessor.from_dict(value["preprocessor"]), value
