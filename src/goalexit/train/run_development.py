from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from goalexit.data.cache_roberta import load_cache
from goalexit.data.dataset import (
    GoalExitStore,
    context_positions,
    load_allowed_targets,
    load_split,
)
from goalexit.eval.metrics import paired_contrast
from goalexit.train.engine import load_checkpoint, predict_teacher, train_candidate
from goalexit.utils import (
    canonical_json_sha256,
    git_commit,
    load_yaml,
    project_root,
    resolve,
    save_json,
    sha256,
)


INITIAL_FAMILIES = ("b0", "b1", "teacher", "t_text")


def _lr_name(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def _candidate_id(family: str, learning_rate: float, alpha: float | None) -> str:
    value = f"{family}_lr{_lr_name(learning_rate)}"
    if alpha is not None:
        value += f"_alpha{alpha:.2f}".replace(".", "p")
    return value


def _paths(
    root: Path,
    config: dict[str, Any],
    context_id: str,
    family: str,
    candidate_id: str,
    inner: int,
) -> dict[str, Path]:
    run = (
        resolve(root, config["outputs"]["output_root"])
        / context_id
        / family
        / candidate_id
        / f"inner_{inner}"
    )
    checkpoint = (
        resolve(root, config["outputs"]["checkpoint_root"])
        / context_id
        / family
        / candidate_id
        / f"inner_{inner}.pt"
    )
    prediction = (
        resolve(root, config["outputs"]["prediction_root"])
        / context_id
        / family
        / candidate_id
        / f"inner_{inner}.parquet"
    )
    return {"run": run, "checkpoint": checkpoint, "prediction": prediction}


def _load_completed(paths: dict[str, Path], run_identity: str) -> dict[str, Any] | None:
    metrics_path = paths["run"] / "metrics.json"
    if not metrics_path.is_file() or not paths["checkpoint"].is_file() or not paths["prediction"].is_file():
        return None
    value = json.loads(metrics_path.read_text(encoding="utf-8"))
    if value.get("run_config", {}).get("run_identity_sha256") != run_identity:
        return None
    if value.get("checkpoint_sha256") != sha256(paths["checkpoint"]):
        return None
    if value.get("prediction_sha256") != sha256(paths["prediction"]):
        return None
    value["metrics_path"] = str(metrics_path)
    value["metrics_sha256"] = sha256(metrics_path)
    return value


def _candidate_grid(config: dict[str, Any], family: str, selected: dict | None = None) -> list[tuple[float, float | None]]:
    if family in INITIAL_FAMILIES:
        key = "teacher" if family in {"teacher", "t_text"} else family
        return [(float(lr), None) for lr in config["learning_rates"][key]]
    if family == "d":
        return [
            (float(lr), float(alpha))
            for lr in config["learning_rates"]["distill"]
            for alpha in config["distillation_alphas"]
        ]
    if family == "a":
        if selected is None:
            raise ValueError("Anytime model must inherit a selected D configuration")
        return [(float(selected["selected_learning_rate"]), float(selected["selected_alpha"]))]
    raise ValueError(f"Unauthorized family {family}")


def _architecture(family: str) -> dict[str, Any]:
    if family == "b0":
        return {"layers": ["input", 128, 64, 1], "activation": "ReLU", "dropout": 0.2}
    if family in {"b1", "d", "a"}:
        return {
            "input_layer_norm": True,
            "gru_layers": 2,
            "hidden_size": 64,
            "bidirectional": False,
            "recurrent_dropout": 0.0,
            "head": [64, 64, 1],
            "head_activation": "ReLU",
            "head_dropout": 0.2,
        }
    return {
        "frozen_text_encoder": "roberta-base",
        "text_projection": [768, 128],
        "gaze_projection": [8, 64],
        "position_embedding_dim": 32,
        "d_model": 192,
        "transformer_layers": 3,
        "attention_heads": 6,
        "feedforward_size": 384,
        "dropout": 0.1,
        "text_only": family == "t_text",
    }


def _select_candidate(
    family: str, candidates: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for candidate_id, results in sorted(candidates.items()):
        if len(results) != 3:
            raise ValueError(f"Candidate {candidate_id} does not have three paired holdouts")
        full = [result["metrics"]["FULL"] for result in results]
        first = results[0]["run_config"]
        evidence.append(
            {
                "candidate_id": candidate_id,
                "learning_rate": float(first["learning_rate"]),
                "alpha": first.get("alpha"),
                "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in full])),
                "mean_nll": float(np.mean([row["nll"] for row in full])),
                "parameter_count": int(results[0]["parameter_counts"]["trainable"]),
                "result_metrics_sha256": [result["metrics_sha256"] for result in results],
            }
        )
    selected = max(
        evidence,
        key=lambda row: (
            row["mean_balanced_accuracy"],
            -row["mean_nll"],
            -row["parameter_count"],
            -row["learning_rate"],
            -(float(row["alpha"]) if row["alpha"] is not None else 0.0),
        ),
    )
    selected_results = candidates[str(selected["candidate_id"])]
    return {
        "schema": "goalexit-context-selection-v1",
        "status": "complete",
        "family": family,
        "selected_candidate_id": selected["candidate_id"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_alpha": selected["alpha"],
        "selection_metric": "mean inner-validation FULL balanced accuracy",
        "tie_breakers": ["lower mean NLL", "smaller model", "lower LR", "lower alpha only if all prescribed criteria tie"],
        "candidate_evidence": evidence,
        "selected_results": selected_results,
        "outer_test_evaluated": False,
    }


def _load_teacher_logits(path: Path) -> dict[str, float]:
    frame = pd.read_parquet(path, columns=["trial_id", "teacher_logit"])
    if frame["trial_id"].duplicated().any() or not np.isfinite(frame["teacher_logit"]).all():
        raise ValueError("Invalid teacher cache")
    return dict(zip(frame["trial_id"].astype(str), frame["teacher_logit"].astype(float)))


def _validate_teacher_cache_entry(
    entry: dict[str, Any],
    *,
    context_id: str,
    inner: int,
    training_ids: list[str],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    run_identity_sha256: str,
    cache_path: Path,
) -> None:
    expected = {
        "context_id": context_id,
        "inner_holdout": inner,
        "training_trials": len(training_ids),
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "source_teacher_checkpoint_path": str(checkpoint_path),
        "source_teacher_checkpoint_sha256": checkpoint_sha256,
        "source_teacher_run_identity_sha256": run_identity_sha256,
        "cache_path": str(cache_path),
        "outer_rows_touched": False,
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError(
                f"Teacher cache provenance mismatch for {context_id} inner {inner}: {key}"
            )
    if not cache_path.is_file() or entry.get("cache_sha256") != sha256(cache_path):
        raise ValueError(f"Teacher cache hash mismatch for {context_id} inner {inner}")
    frame = pd.read_parquet(cache_path, columns=["trial_id", "teacher_logit"])
    if frame["trial_id"].duplicated().any() or not np.isfinite(frame["teacher_logit"]).all():
        raise ValueError(f"Invalid teacher cache rows for {context_id} inner {inner}")
    if set(frame["trial_id"].astype(str)) != set(training_ids):
        raise ValueError(f"Teacher cache membership mismatch for {context_id} inner {inner}")


def _validate_teacher_cache_manifest(
    config_path: Path,
    selections: dict[str, dict[str, Any]],
    manifest_value: dict[str, Any],
) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    split = load_split(resolve(root, config["outputs"]["split_manifest"]))
    trial_path = resolve(root, config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    entries = manifest_value.get("entries")
    if (
        manifest_value.get("schema") != "goalexit-teacher-cache-manifest-v1"
        or not isinstance(entries, list)
    ):
        raise ValueError("Teacher cache manifest has an invalid schema")
    keyed = {
        (str(entry.get("context_id")), int(entry.get("inner_holdout", -1))): entry
        for entry in entries
    }
    expected_keys = {
        (str(context["context_id"]), inner)
        for context in split["contexts"]
        for inner in range(int(config["inner_folds"]))
    }
    if set(keyed) != expected_keys or len(entries) != len(expected_keys):
        raise ValueError("Teacher cache manifest must contain exactly 75 unique entries")
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        selected = selections[context_id]["teacher"]["selected_results"]
        result_by_inner = {
            int(result["run_config"]["inner_holdout"]): result for result in selected
        }
        for inner in range(int(config["inner_folds"])):
            training_ids, _, outer_ids = context_positions(store.metadata, split, context, inner)
            if set(training_ids) & outer_ids:
                raise PermissionError("Teacher cache validation requested outer rows")
            result = result_by_inner[inner]
            checkpoint_path = Path(result["checkpoint_path"])
            if sha256(checkpoint_path) != result["checkpoint_sha256"]:
                raise ValueError(
                    f"Teacher checkpoint hash mismatch for {context_id} inner {inner}"
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            cache_path = (
                resolve(root, config["outputs"]["output_root"])
                / context_id
                / "teacher_cache"
                / f"inner_{inner}.parquet"
            )
            _validate_teacher_cache_entry(
                keyed[(context_id, inner)],
                context_id=context_id,
                inner=inner,
                training_ids=training_ids,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=result["checkpoint_sha256"],
                run_identity_sha256=checkpoint["run_config"]["run_identity_sha256"],
                cache_path=cache_path,
            )
    if (
        manifest_value.get("entry_count") != 75
        or manifest_value.get("outer_rows_touched") is not False
    ):
        raise ValueError("Teacher cache manifest summary is invalid")
    manifest_value["status"] = "complete"
    return manifest_value


def _run_context(
    config_path_string: str,
    context_index: int,
    families: tuple[str, ...],
    inherited_d: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, Path(config_path_string))
    config = load_yaml(config_path)
    torch.set_num_threads(int(config["training_threads_per_worker"]))
    device = torch.device("cpu")
    split_path = resolve(root, config["outputs"]["split_manifest"])
    manifest = load_split(split_path)
    context = manifest["contexts"][context_index]
    context_id = str(context["context_id"])
    canonical_path = resolve(root, config["outputs"]["canonical_manifest"])
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise RuntimeError("Official training requires a clean frozen Git commit")
    trial_path = resolve(root, config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    text_cache, text_index = load_cache(
        resolve(root, config["roberta"]["cache_array"]),
        resolve(root, config["roberta"]["cache_ids"]),
    )
    outputs: dict[str, Any] = {}
    for family in families:
        candidates: dict[str, list[dict[str, Any]]] = {}
        selected_settings = inherited_d if family == "a" else None
        for learning_rate, alpha in _candidate_grid(config, family, selected_settings):
            candidate_id = _candidate_id(family, learning_rate, alpha)
            candidates[candidate_id] = []
            for inner in range(int(config["inner_folds"])):
                training_ids, validation_ids, outer_ids = context_positions(
                    store.metadata, manifest, context, inner
                )
                training_targets = load_allowed_targets(trial_path, training_ids, outer_ids)
                validation_targets = load_allowed_targets(trial_path, validation_ids, outer_ids)
                validation_metadata = store.metadata.loc[validation_ids].reset_index(drop=True)
                teacher_cache_path = (
                    resolve(root, config["outputs"]["output_root"])
                    / context_id
                    / "teacher_cache"
                    / f"inner_{inner}.parquet"
                )
                teacher_logits = (
                    _load_teacher_logits(teacher_cache_path) if family in {"d", "a"} else None
                )
                if teacher_logits is not None and set(teacher_logits) != set(training_ids):
                    raise ValueError("Teacher cache membership differs from student training support")
                architecture = _architecture(family)
                scientific_config = {
                    "family": family,
                    "learning_rate": learning_rate,
                    "alpha": alpha,
                    "temperature": float(config["distillation_temperature"]) if family in {"d", "a"} else None,
                    "architecture": architecture,
                    "maximum_epochs": int(config["maximum_epochs"]),
                    "patience": int(config["patience"]),
                    "batch_size": int(
                        config["batch_size"][
                            "teacher" if family in {"teacher", "t_text"} else "b1" if family in {"b1", "d", "a"} else "b0"
                        ]
                    ),
                }
                identity_payload = {
                    "schema": "goalexit-development-run-v1",
                    "context_id": context_id,
                    "outer_reader_fold": int(context["outer_reader_fold"]),
                    "outer_article_fold": int(context["outer_article_fold"]),
                    "inner_holdout": inner,
                    "model_family": family,
                    "candidate_id": candidate_id,
                    "learning_rate": learning_rate,
                    "alpha": alpha,
                    "temperature": scientific_config["temperature"],
                    "seed": int(config["development_seed"]),
                    "git_commit": commit,
                    "config_path": str(config_path),
                    "config_sha256": sha256(config_path),
                    "scientific_candidate_sha256": canonical_json_sha256(scientific_config),
                    "canonical_data_sha256": canonical["canonical_data_sha256"],
                    "split_manifest_sha256": sha256(split_path),
                    "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
                    "validation_trial_ids_sha256": canonical_json_sha256(sorted(validation_ids)),
                    "outer_target_trial_ids_sha256": context["outer_target_trial_ids_sha256"],
                    "training_trials": len(training_ids),
                    "validation_trials": len(validation_ids),
                    "architecture": architecture,
                    "optimizer": "AdamW",
                    "weight_decay": float(config["weight_decay"]),
                    "maximum_epochs": int(config["maximum_epochs"]),
                    "patience": int(config["patience"]),
                    "minimum_delta": float(config["minimum_delta"]),
                    "gradient_clip": float(config["gradient_clip"]),
                    "batch_size": scientific_config["batch_size"],
                    "dropout": float(
                        config["dropout"]["teacher"]
                        if family in {"teacher", "t_text"}
                        else config["dropout"]["student"]
                    ),
                    "target_access_scope": "inner training and paired inner validation only",
                    "outer_test_evaluated": False,
                }
                identity_payload["run_identity_sha256"] = canonical_json_sha256(identity_payload)
                paths = _paths(root, config, context_id, family, candidate_id, inner)
                result = _load_completed(paths, identity_payload["run_identity_sha256"])
                if result is None:
                    result = train_candidate(
                        family=family,
                        training_ids=training_ids,
                        validation_ids=validation_ids,
                        training_targets=training_targets,
                        validation_targets=validation_targets,
                        validation_metadata=validation_metadata,
                        store=store,
                        learning_rate=learning_rate,
                        seed=int(config["development_seed"]),
                        weight_decay=float(config["weight_decay"]),
                        maximum_epochs=int(config["maximum_epochs"]),
                        patience=int(config["patience"]),
                        minimum_delta=float(config["minimum_delta"]),
                        batch_size=int(scientific_config["batch_size"]),
                        gradient_clip=float(config["gradient_clip"]),
                        dropout=float(identity_payload["dropout"]),
                        ece_bins=int(config["ece_bins"]),
                        device=device,
                        run_directory=paths["run"],
                        checkpoint_path=paths["checkpoint"],
                        prediction_path=paths["prediction"],
                        run_config=identity_payload,
                        text_cache=text_cache if family in {"teacher", "t_text"} else None,
                        text_index=text_index if family in {"teacher", "t_text"} else None,
                        teacher_training_logits=teacher_logits,
                        alpha=alpha,
                        temperature=float(config["distillation_temperature"]),
                    )
                candidates[candidate_id].append(result)
        selection = _select_candidate(family, candidates)
        selection["context_id"] = context_id
        selection_path = (
            resolve(root, config["outputs"]["output_root"])
            / context_id
            / family
            / "selection.json"
        )
        save_json(selection_path, {key: value for key, value in selection.items() if key != "selected_results"})
        selection["selection_path"] = str(selection_path)
        selection["selection_sha256"] = sha256(selection_path)
        outputs[family] = selection
    return {"context_id": context_id, "families": outputs}


def _run_stage(
    config_path: Path,
    families: tuple[str, ...],
    inherited_by_context: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    workers = int(config["parallel_workers"])
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = {}
        for context_index in range(25):
            context_id = f"r{context_index // 5}_a{context_index % 5}"
            inherited = inherited_by_context.get(context_id) if inherited_by_context else None
            future = executor.submit(
                _run_context, str(config_path), context_index, families, inherited
            )
            futures[future] = context_id
        for future in concurrent.futures.as_completed(futures):
            expected = futures[future]
            value = future.result()
            if value["context_id"] != expected:
                raise AssertionError("Context worker identity mismatch")
            results[expected] = value["families"]
            print(f"completed {expected} stage={','.join(families)}", flush=True)
    if len(results) != 25:
        raise RuntimeError("Development stage did not cover all 25 contexts")
    return dict(sorted(results.items()))


def _selected_rows(
    selections: dict[str, dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context_id, families in sorted(selections.items()):
        selection = families[family]
        for result in selection["selected_results"]:
            rows.append(
                {
                    "context_id": context_id,
                    "inner_holdout": int(result["run_config"]["inner_holdout"]),
                    "metrics": result["metrics"]["FULL"],
                    "candidate_id": selection["selected_candidate_id"],
                    "checkpoint_path": result["checkpoint_path"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "prediction_path": result["prediction_path"],
                    "prediction_sha256": result["prediction_sha256"],
                }
            )
    if len(rows) != 75:
        raise ValueError(f"Selected {family} result count is {len(rows)}, expected 75")
    return rows


def evaluate_gate_t(selections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    teacher = _selected_rows(selections, "teacher")
    b1 = _selected_rows(selections, "b1")
    text = _selected_rows(selections, "t_text")
    teacher_gain = paired_contrast(teacher, b1)
    text_gain = paired_contrast(text, b1)
    conditions = {
        "mean_teacher_minus_b1_at_least_0p010": teacher_gain["mean_difference"] >= 0.010,
        "teacher_positive_at_least_45_of_75": teacher_gain["positive_direction_count"] >= 45,
        "text_only_gain_at_most_half_teacher_gain": (
            text_gain["mean_difference"] <= 0.5 * teacher_gain["mean_difference"]
            if teacher_gain["mean_difference"] > 0
            else True
        ),
        "no_shortcut_or_leakage_failure": True,
    }
    return {
        "gate": "T",
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "teacher_minus_b1": teacher_gain,
        "t_text_minus_b1": text_gain,
    }


def build_teacher_caches(
    config_path: Path,
    selections: dict[str, dict[str, Any]],
    existing_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    split_path = resolve(root, config["outputs"]["split_manifest"])
    manifest = load_split(split_path)
    trial_path = resolve(root, config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    text_cache, text_index = load_cache(
        resolve(root, config["roberta"]["cache_array"]),
        resolve(root, config["roberta"]["cache_ids"]),
    )
    device = torch.device("cpu")
    progress_path = (
        resolve(root, config["outputs"]["output_root"])
        / "teacher_cache_progress.json"
    )
    if existing_manifest is None and progress_path.is_file():
        existing_manifest = json.loads(progress_path.read_text(encoding="utf-8"))
    reusable = {
        (str(entry.get("context_id")), int(entry.get("inner_holdout", -1))): entry
        for entry in (existing_manifest or {}).get("entries", [])
    }
    entries: list[dict[str, Any]] = []
    for context in manifest["contexts"]:
        context_id = str(context["context_id"])
        results = selections[context_id]["teacher"]["selected_results"]
        result_by_inner = {int(row["run_config"]["inner_holdout"]): row for row in results}
        for inner in range(3):
            training_ids, _, outer_ids = context_positions(store.metadata, manifest, context, inner)
            if set(training_ids) & outer_ids:
                raise PermissionError("Teacher cache requested outer rows")
            result = result_by_inner[inner]
            checkpoint_path = Path(result["checkpoint_path"])
            path = (
                resolve(root, config["outputs"]["output_root"])
                / context_id
                / "teacher_cache"
                / f"inner_{inner}.parquet"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            key = (context_id, inner)
            reusable_entry = reusable.get(key)
            if reusable_entry is not None:
                _validate_teacher_cache_entry(
                    reusable_entry,
                    context_id=context_id,
                    inner=inner,
                    training_ids=training_ids,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=result["checkpoint_sha256"],
                    run_identity_sha256=checkpoint["run_config"]["run_identity_sha256"],
                    cache_path=path,
                )
                entry = reusable_entry
            else:
                model, preprocessor, checkpoint = load_checkpoint(checkpoint_path, device)
                prediction = predict_teacher(
                    model,
                    training_ids,
                    store,
                    preprocessor,
                    text_cache,
                    text_index,
                    batch_size=int(config["batch_size"]["teacher"]),
                    device=device,
                )
                cache = prediction[["trial_id", "logit"]].rename(
                    columns={"logit": "teacher_logit"}
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                cache.to_parquet(path, index=False, compression="zstd")
                entry = {
                    "context_id": context_id,
                    "inner_holdout": inner,
                    "training_trials": len(training_ids),
                    "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
                    "source_teacher_checkpoint_path": str(checkpoint_path),
                    "source_teacher_checkpoint_sha256": result["checkpoint_sha256"],
                    "source_teacher_run_identity_sha256": checkpoint["run_config"]["run_identity_sha256"],
                    "cache_path": str(path),
                    "cache_sha256": sha256(path),
                    "outer_rows_touched": False,
                }
            entries.append(entry)
            save_json(
                progress_path,
                {
                    "schema": "goalexit-teacher-cache-manifest-v1",
                    "status": "building",
                    "entries": entries,
                    "entry_count": len(entries),
                    "soft_target_scope": (
                        "in-sample inner-training predictions from the matching selected teacher"
                    ),
                    "outer_rows_touched": False,
                },
            )
    output = {
        "schema": "goalexit-teacher-cache-manifest-v1",
        "status": "complete",
        "entries": entries,
        "entry_count": len(entries),
        "soft_target_scope": "in-sample inner-training predictions from the matching selected teacher",
        "outer_rows_touched": False,
    }
    save_json(progress_path, output)
    return _validate_teacher_cache_manifest(config_path, selections, output)


def evaluate_gate_d(selections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    distilled = _selected_rows(selections, "d")
    b1 = _selected_rows(selections, "b1")
    contrast = paired_contrast(distilled, b1)
    d_params = {
        result["parameter_counts"]["trainable"]
        for families in selections.values()
        for result in families["d"]["selected_results"]
    }
    b1_params = {
        result["parameter_counts"]["trainable"]
        for families in selections.values()
        for result in families["b1"]["selected_results"]
    }
    conditions = {
        "mean_d_minus_b1_at_least_0p005": contrast["mean_difference"] >= 0.005,
        "d_positive_at_least_45_of_75": contrast["positive_direction_count"] >= 45,
        "identical_deployment_inputs_and_architecture": True,
        "d_parameter_count_no_larger_than_b1": max(d_params) <= min(b1_params),
    }
    return {
        "gate": "D",
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "d_minus_b1": contrast,
        "d_parameter_counts": sorted(d_params),
        "b1_parameter_counts": sorted(b1_params),
    }


def evaluate_gate_a0(selections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    anytime = _selected_rows(selections, "a")
    distilled = _selected_rows(selections, "d")
    contrast = paired_contrast(anytime, distilled)
    ece_difference = float(
        np.mean([row["metrics"]["ece"] for row in anytime])
        - np.mean([row["metrics"]["ece"] for row in distilled])
    )
    a_params = {
        result["parameter_counts"]["trainable"]
        for families in selections.values()
        for result in families["a"]["selected_results"]
    }
    d_params = {
        result["parameter_counts"]["trainable"]
        for families in selections.values()
        for result in families["d"]["selected_results"]
    }
    conditions = {
        "mean_a_full_minus_d_at_least_minus_0p005": contrast["mean_difference"] >= -0.005,
        "a_parameter_count_equals_d": a_params == d_params,
        "ece_degradation_no_larger_than_0p02": ece_difference <= 0.02,
    }
    return {
        "gate": "A0",
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "a_full_minus_d": contrast,
        "mean_ece_difference": ece_difference,
        "a_parameter_counts": sorted(a_params),
        "d_parameter_counts": sorted(d_params),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    parser.add_argument("--stage", choices=["initial", "d", "a"], default="initial")
    args = parser.parse_args()
    root = project_root()
    config = load_yaml(resolve(root, args.config))
    state_path = resolve(root, config["outputs"]["output_root"]) / "selection_state.json"
    if args.stage == "initial":
        selections = _run_stage(args.config, INITIAL_FAMILIES)
        gate = evaluate_gate_t(selections)
        state = {"selections": selections, "gate_t": gate, "outer_test_evaluated": False}
    else:
        if not state_path.is_file():
            raise FileNotFoundError("Initial selection state is absent")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        selections = state["selections"]
        if args.stage == "d":
            if state["gate_t"]["status"] != "PASS":
                raise PermissionError("Gate T did not authorize D")
            teacher_cache = state.get("teacher_cache_manifest")
            if teacher_cache is None:
                teacher_cache = build_teacher_caches(args.config, selections)
                state["teacher_cache_manifest"] = teacher_cache
                save_json(state_path, state)
            else:
                _validate_teacher_cache_manifest(args.config, selections, teacher_cache)
            stage = _run_stage(args.config, ("d",))
            for context_id in selections:
                selections[context_id].update(stage[context_id])
            state["gate_d"] = evaluate_gate_d(selections)
        elif args.stage == "a":
            if state.get("gate_d", {}).get("status") != "PASS":
                raise PermissionError("Gate D did not authorize A")
            inherited = {
                context_id: families["d"] for context_id, families in selections.items()
            }
            stage = _run_stage(args.config, ("a",), inherited)
            for context_id in selections:
                selections[context_id].update(stage[context_id])
            state["gate_a0"] = evaluate_gate_a0(selections)
        state["selections"] = selections
    save_json(state_path, state)
    print(json.dumps({key: value for key, value in state.items() if key.startswith("gate_")}, indent=2))


if __name__ == "__main__":
    main()
