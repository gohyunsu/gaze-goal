from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from goalexit.data.dataset import (
    GoalExitStore,
    context_positions,
    load_allowed_targets,
    load_split,
    outer_context_positions,
)
from goalexit.eval.final_evaluation import fixed_epoch_from_selection
from goalexit.eval.metrics import evaluate_binary
from goalexit.eval.postconfirmatory_seed_extension import (
    _mean_crossed_bootstrap,
    _mean_reader_bootstrap,
)
from goalexit.train.engine import train_candidate
from goalexit.train.fixed_refit import fit_fixed_epochs
from goalexit.train.run_development import _load_teacher_logits
from goalexit.utils import (
    canonical_json_sha256,
    git_commit,
    load_yaml,
    project_root,
    resolve,
    save_json,
    sha256,
)


SEEDS = (13, 29, 47)
OUTER_VARIANTS = tuple(
    f"{name}_s{seed}"
    for seed in SEEDS
    for name in ("tuned_bce", "tuned_kd", "class_mean_kd")
)


def _lr_name(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def _candidate_id(
    family: str, architecture_id: str, learning_rate: float, alpha: float | None
) -> str:
    value = f"{family}_{architecture_id}_lr{_lr_name(learning_rate)}"
    if alpha is not None:
        value += f"_alpha{alpha:.2f}".replace(".", "p")
    return value


def _inner_paths(
    root: Path,
    config: dict[str, Any],
    context_id: str,
    family: str,
    candidate_id: str,
    inner: int,
) -> dict[str, Path]:
    base = resolve(root, config["outputs"]["output_root"])
    run = base / context_id / family / candidate_id / f"inner_{inner}"
    return {
        "run": run,
        "checkpoint": resolve(root, config["outputs"]["checkpoint_root"])
        / context_id
        / family
        / candidate_id
        / f"inner_{inner}.pt",
        "prediction": resolve(root, config["outputs"]["prediction_root"])
        / context_id
        / family
        / candidate_id
        / f"inner_{inner}.parquet",
    }


def _outer_paths(
    root: Path, config: dict[str, Any], context_id: str, variant: str
) -> dict[str, Path]:
    return {
        "run": resolve(root, config["outputs"]["output_root"])
        / context_id
        / "outer"
        / variant,
        "checkpoint": resolve(root, config["outputs"]["checkpoint_root"])
        / context_id
        / f"{variant}.pt",
        "prediction": resolve(root, config["outputs"]["prediction_root"])
        / context_id
        / f"{variant}.parquet",
    }


def _load_complete(
    paths: dict[str, Path], run_identity: str, expected_ids: list[str]
) -> dict[str, Any] | None:
    result_path = paths["run"] / (
        "metrics.json" if "inner_" in paths["run"].name else "result.json"
    )
    if not all(
        path.is_file()
        for path in (result_path, paths["checkpoint"], paths["prediction"])
    ):
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("run_config", {}).get("run_identity_sha256") != run_identity:
        return None
    if result.get("checkpoint_sha256") != sha256(paths["checkpoint"]):
        return None
    if result.get("prediction_sha256") != sha256(paths["prediction"]):
        return None
    available = pd.read_parquet(paths["prediction"])
    frame = (
        available.loc[available["prefix"].astype(str) == "FULL", ["trial_id"]]
        if "prefix" in available
        else available[["trial_id"]]
    )
    if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
        expected_ids
    ):
        raise ValueError("Completed prediction membership mismatch")
    result["result_path"] = str(result_path)
    result["result_sha256"] = sha256(result_path)
    return result


def _selection(
    family: str,
    candidates: dict[str, list[dict[str, Any]]],
    architecture_by_candidate: dict[str, tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for candidate_id, results in sorted(candidates.items()):
        if len(results) != 3:
            raise ValueError(f"Candidate {candidate_id} lacks three paired holdouts")
        full = [result["metrics"]["FULL"] for result in results]
        run = results[0]["run_config"]
        architecture_id, architecture = architecture_by_candidate[candidate_id]
        evidence.append(
            {
                "candidate_id": candidate_id,
                "architecture_id": architecture_id,
                "architecture": architecture,
                "learning_rate": float(run["learning_rate"]),
                "alpha": run.get("alpha"),
                "mean_balanced_accuracy": float(
                    np.mean([row["balanced_accuracy"] for row in full])
                ),
                "mean_nll": float(np.mean([row["nll"] for row in full])),
                "parameter_count": int(results[0]["parameter_counts"]["trainable"]),
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
    return {
        "schema": "goalexit-postfeedback-strong-gru-selection-v1",
        "family": family,
        "selected_candidate_id": selected["candidate_id"],
        "selected_architecture_id": selected["architecture_id"],
        "selected_architecture": selected["architecture"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_alpha": selected["alpha"],
        "selection_metric": "mean inner-validation full-trial balanced accuracy",
        "tie_breakers": [
            "lower mean NLL",
            "smaller model",
            "lower learning rate",
            "lower alpha",
        ],
        "candidate_evidence": evidence,
        "selected_results": candidates[str(selected["candidate_id"])],
        "outer_test_evaluated": False,
    }


def _inner_identity(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    canonical: dict[str, Any],
    split_path: Path,
    context: dict[str, Any],
    inner: int,
    family: str,
    candidate_id: str,
    architecture_id: str,
    architecture: dict[str, Any],
    learning_rate: float,
    alpha: float | None,
    training_ids: list[str],
    validation_ids: list[str],
    device: torch.device,
) -> dict[str, Any]:
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise RuntimeError("Strong-GRU execution requires a clean committed tree")
    value: dict[str, Any] = {
        "schema": "goalexit-postfeedback-strong-gru-inner-run-v1",
        "post_hoc": True,
        "outer_results_known_when_designed": True,
        "context_id": str(context["context_id"]),
        "inner_holdout": inner,
        "model_family": family,
        "candidate_id": candidate_id,
        "architecture_id": architecture_id,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "temperature": float(config["distillation_temperature"]) if family == "d" else None,
        "seed": int(config["development_seed"]),
        "optimizer": "AdamW",
        "weight_decay": float(config["weight_decay"]),
        "maximum_epochs": int(config["maximum_epochs"]),
        "patience": int(config["patience"]),
        "minimum_delta": float(config["minimum_delta"]),
        "gradient_clip": float(config["gradient_clip"]),
        "batch_size": int(config["batch_size"]),
        "dropout": float(config["head_dropout"]),
        "training_device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(device),
        "host": platform.node(),
        "git_commit": commit,
        "config_sha256": sha256(config_path),
        "confirmatory_config_sha256": sha256(config_path),
        "protocol_sha256": sha256(resolve(root, config["protocol"])),
        "lock_sha256": sha256(resolve(root, config["protocol"])),
        "canonical_data_sha256": canonical["canonical_data_sha256"],
        "split_manifest_sha256": sha256(split_path),
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "validation_trial_ids_sha256": canonical_json_sha256(sorted(validation_ids)),
        "outer_test_evaluated": False,
    }
    value["run_identity_sha256"] = canonical_json_sha256(value)
    return value


def _save_selection_state(path: Path, selections: dict[str, Any]) -> None:
    completed = sorted(
        family
        for family in ("bce", "kd")
        if all(family in values for values in selections.values())
    )
    save_json(
        path,
        {
            "schema": "goalexit-postfeedback-strong-gru-selection-state-v1",
            "post_hoc": True,
            "completed_stages": completed,
            "selections": dict(sorted(selections.items())),
            "outer_test_evaluated": False,
        },
    )


def run_inner(config_path: Path, stage: str) -> None:
    root = project_root()
    resolved = resolve(root, config_path)
    config = load_yaml(resolved)
    split_path = resolve(root, config["split_manifest"])
    split = load_split(split_path)
    dev_config = load_yaml(resolve(root, config["development_config"]))
    canonical = json.loads(
        resolve(root, config["canonical_manifest"]).read_text(encoding="utf-8")
    )
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    state_path = resolve(root, config["outputs"]["selection_state"])
    selections = (
        json.loads(state_path.read_text(encoding="utf-8"))["selections"]
        if state_path.is_file()
        else {}
    )
    device = torch.device("cuda:0")
    if config.get("device") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Strong-GRU search requires the configured CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(int(config["training_threads"]))
    architectures = {
        str(key): dict(value) for key, value in config["architectures"].items()
    }
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        selections.setdefault(context_id, {})
        if stage == "kd" and "bce" not in selections[context_id]:
            raise FileNotFoundError(f"BCE selection missing for {context_id}")
        search_architectures = (
            architectures
            if stage == "bce"
            else {
                str(selections[context_id]["bce"]["selected_architecture_id"]): dict(
                    selections[context_id]["bce"]["selected_architecture"]
                )
            }
        )
        family = "b1" if stage == "bce" else "d"
        candidates: dict[str, list[dict[str, Any]]] = {}
        candidate_architectures: dict[str, tuple[str, dict[str, Any]]] = {}
        for architecture_id, architecture in search_architectures.items():
            alphas = (None,) if stage == "bce" else tuple(
                float(value) for value in config["distillation_alphas"]
            )
            for learning_rate in map(float, config["learning_rates"]):
                for alpha in alphas:
                    candidate_id = _candidate_id(
                        stage, architecture_id, learning_rate, alpha
                    )
                    candidate_architectures[candidate_id] = (
                        architecture_id,
                        architecture,
                    )
                    results: list[dict[str, Any]] = []
                    for inner in range(int(config["inner_folds"])):
                        training_ids, validation_ids, outer_ids = context_positions(
                            store.metadata, split, context, inner
                        )
                        training_targets = load_allowed_targets(
                            resolve(root, dev_config["outputs"]["trial_table"]),
                            training_ids,
                            outer_ids,
                        )
                        validation_targets = load_allowed_targets(
                            resolve(root, dev_config["outputs"]["trial_table"]),
                            validation_ids,
                            outer_ids,
                        )
                        run_config = _inner_identity(
                            root=root,
                            config_path=resolved,
                            config=config,
                            canonical=canonical,
                            split_path=split_path,
                            context=context,
                            inner=inner,
                            family=family,
                            candidate_id=candidate_id,
                            architecture_id=architecture_id,
                            architecture=architecture,
                            learning_rate=learning_rate,
                            alpha=alpha,
                            training_ids=training_ids,
                            validation_ids=validation_ids,
                            device=device,
                        )
                        paths = _inner_paths(
                            root, config, context_id, stage, candidate_id, inner
                        )
                        result = _load_complete(
                            paths, run_config["run_identity_sha256"], validation_ids
                        )
                        if result is None:
                            teacher_logits = None
                            if stage == "kd":
                                cache = (
                                    resolve(root, config["inner_teacher_cache_root"])
                                    / context_id
                                    / "teacher_cache"
                                    / f"inner_{inner}.parquet"
                                )
                                teacher_logits = _load_teacher_logits(cache)
                                if set(teacher_logits) != set(training_ids):
                                    raise ValueError("Inner teacher-cache support mismatch")
                            result = train_candidate(
                                family=family,
                                training_ids=training_ids,
                                validation_ids=validation_ids,
                                training_targets=training_targets,
                                validation_targets=validation_targets,
                                validation_metadata=store.metadata.loc[
                                    validation_ids
                                ].reset_index(drop=True),
                                store=store,
                                learning_rate=learning_rate,
                                seed=int(config["development_seed"]),
                                weight_decay=float(config["weight_decay"]),
                                maximum_epochs=int(config["maximum_epochs"]),
                                patience=int(config["patience"]),
                                minimum_delta=float(config["minimum_delta"]),
                                batch_size=int(config["batch_size"]),
                                gradient_clip=float(config["gradient_clip"]),
                                dropout=float(config["head_dropout"]),
                                ece_bins=int(config["ece_bins"]),
                                device=device,
                                run_directory=paths["run"],
                                checkpoint_path=paths["checkpoint"],
                                prediction_path=paths["prediction"],
                                run_config=run_config,
                                teacher_training_logits=teacher_logits,
                                alpha=alpha,
                                temperature=float(config["distillation_temperature"]),
                                student_architecture=architecture,
                            )
                        results.append(result)
                    candidates[candidate_id] = results
        selections[context_id][stage] = _selection(
            stage, candidates, candidate_architectures
        )
        _save_selection_state(state_path, selections)
        print(f"selected {stage} {context_id}", flush=True)


def _outer_teacher_cache(
    root: Path, config: dict[str, Any], context_id: str, seed: int
) -> Path:
    if seed == int(config["base_seed"]):
        return (
            resolve(root, config["base_teacher_cache_root"])
            / context_id
            / "text_gaze_teacher.parquet"
        )
    return (
        resolve(root, config["extension_teacher_cache_root"])
        / context_id
        / f"text_gaze_teacher_s{seed}.parquet"
    )


def _class_mean_logits(
    teacher_logits: dict[str, float],
    training_targets: pd.DataFrame,
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    target = training_targets.set_index("trial_id")["goal_index"].astype(int).to_dict()
    if set(target) != set(teacher_logits):
        raise ValueError("Class-mean targets and teacher logits differ in support")
    softened = {
        key: float(1.0 / (1.0 + np.exp(-value / temperature)))
        for key, value in teacher_logits.items()
    }
    mean_probabilities = {
        label: float(
            np.mean(
                [softened[key] for key, value in target.items() if value == label]
            )
        )
        for label in (0, 1)
    }
    epsilon = 1e-7
    mean_logits = {
        label: float(
            temperature
            * np.log(
                np.clip(probability, epsilon, 1.0 - epsilon)
                / (1.0 - np.clip(probability, epsilon, 1.0 - epsilon))
            )
        )
        for label, probability in mean_probabilities.items()
    }
    return {key: mean_logits[value] for key, value in target.items()}


def _outer_identity(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    canonical: dict[str, Any],
    split_path: Path,
    context: dict[str, Any],
    variant: str,
    family: str,
    selection: dict[str, Any],
    seed: int,
    fixed_epochs: int,
    training_ids: list[str],
    prediction_ids: list[str],
    teacher_cache_hash: str | None,
    target_transform: str,
    device: torch.device,
) -> dict[str, Any]:
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise RuntimeError("Strong-GRU execution requires a clean committed tree")
    value: dict[str, Any] = {
        "schema": "goalexit-postfeedback-strong-gru-outer-run-v1",
        "post_hoc": True,
        "outer_results_known_when_designed": True,
        "context_id": str(context["context_id"]),
        "model_family": variant,
        "training_family": family,
        "selected_candidate_id": selection["selected_candidate_id"],
        "architecture_id": selection["selected_architecture_id"],
        "architecture": selection["selected_architecture"],
        "learning_rate": float(selection["selected_learning_rate"]),
        "alpha": selection.get("selected_alpha"),
        "temperature": float(config["distillation_temperature"]) if family == "d" else None,
        "fixed_epochs": fixed_epochs,
        "epoch_rule": "integer median of three selected inner-validation epochs",
        "teacher_target_transform": target_transform,
        "teacher_cache_sha256": teacher_cache_hash,
        "seed": seed,
        "optimizer": "AdamW",
        "weight_decay": float(config["weight_decay"]),
        "gradient_clip": float(config["gradient_clip"]),
        "batch_size": int(config["batch_size"]),
        "dropout": float(config["head_dropout"]),
        "training_device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(device),
        "host": platform.node(),
        "git_commit": commit,
        "config_sha256": sha256(config_path),
        "confirmatory_config_sha256": sha256(config_path),
        "protocol_sha256": sha256(resolve(root, config["protocol"])),
        "lock_sha256": sha256(resolve(root, config["protocol"])),
        "canonical_data_sha256": canonical["canonical_data_sha256"],
        "split_manifest_sha256": sha256(split_path),
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "prediction_trial_ids_sha256": canonical_json_sha256(sorted(prediction_ids)),
        "outer_labels_accessed": False,
        "outer_scientific_metrics_computed": False,
    }
    value["run_identity_sha256"] = canonical_json_sha256(value)
    return value


def run_outer(config_path: Path) -> None:
    root = project_root()
    resolved = resolve(root, config_path)
    config = load_yaml(resolved)
    split_path = resolve(root, config["split_manifest"])
    split = load_split(split_path)
    dev_config = load_yaml(resolve(root, config["development_config"]))
    canonical = json.loads(
        resolve(root, config["canonical_manifest"]).read_text(encoding="utf-8")
    )
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    selection_state = json.loads(
        resolve(root, config["outputs"]["selection_state"]).read_text(encoding="utf-8")
    )
    if selection_state.get("completed_stages") != ["bce", "kd"]:
        raise ValueError("BCE and KD selections must be complete before outer refits")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Strong-GRU outer refits require CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(int(config["training_threads"]))
    state_path = resolve(root, config["outputs"]["outer_state"])
    state = (
        json.loads(state_path.read_text(encoding="utf-8")).get("results", {})
        if state_path.is_file()
        else {}
    )
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        training_ids, prediction_ids = outer_context_positions(
            store.metadata, split, context
        )
        training_targets = load_allowed_targets(
            resolve(root, dev_config["outputs"]["trial_table"]),
            training_ids,
            set(prediction_ids),
        )
        prediction_metadata = store.metadata.loc[prediction_ids].reset_index(drop=True)
        selected = selection_state["selections"][context_id]
        if selected["kd"]["selected_architecture"] != selected["bce"]["selected_architecture"]:
            raise ValueError("KD architecture was not inherited from BCE")
        state.setdefault(context_id, {})
        for seed in SEEDS:
            teacher_path = _outer_teacher_cache(root, config, context_id, seed)
            teacher_logits = _load_teacher_logits(teacher_path)
            if set(teacher_logits) != set(training_ids):
                raise ValueError("Outer teacher-cache support mismatch")
            for name in ("tuned_bce", "tuned_kd", "class_mean_kd"):
                family = "b1" if name == "tuned_bce" else "d"
                selection = selected["bce" if family == "b1" else "kd"]
                fixed_epochs = fixed_epoch_from_selection(selection)
                variant = f"{name}_s{seed}"
                target_transform = "none"
                logits = None
                cache_hash = None
                if family == "d":
                    logits = teacher_logits
                    cache_hash = sha256(teacher_path)
                    target_transform = "identity"
                    if name == "class_mean_kd":
                        logits = _class_mean_logits(
                            teacher_logits,
                            training_targets,
                            float(config["distillation_temperature"]),
                        )
                        target_transform = "class_conditional_mean_soft_probability"
                run_config = _outer_identity(
                    root=root,
                    config_path=resolved,
                    config=config,
                    canonical=canonical,
                    split_path=split_path,
                    context=context,
                    variant=variant,
                    family=family,
                    selection=selection,
                    seed=seed,
                    fixed_epochs=fixed_epochs,
                    training_ids=training_ids,
                    prediction_ids=prediction_ids,
                    teacher_cache_hash=cache_hash,
                    target_transform=target_transform,
                    device=device,
                )
                paths = _outer_paths(root, config, context_id, variant)
                result = _load_complete(
                    paths, run_config["run_identity_sha256"], prediction_ids
                )
                if result is None:
                    result = fit_fixed_epochs(
                        family=family,
                        training_ids=training_ids,
                        prediction_ids=prediction_ids,
                        training_targets=training_targets,
                        prediction_metadata=prediction_metadata,
                        store=store,
                        learning_rate=float(selection["selected_learning_rate"]),
                        fixed_epochs=fixed_epochs,
                        seed=seed,
                        weight_decay=float(config["weight_decay"]),
                        batch_size=int(config["batch_size"]),
                        gradient_clip=float(config["gradient_clip"]),
                        dropout=float(config["head_dropout"]),
                        device=device,
                        run_directory=paths["run"],
                        checkpoint_path=paths["checkpoint"],
                        prediction_path=paths["prediction"],
                        run_config=run_config,
                        teacher_training_logits=logits,
                        alpha=selection.get("selected_alpha"),
                        temperature=float(config["distillation_temperature"]),
                        student_architecture=selection["selected_architecture"],
                    )
                state[context_id][variant] = result
        save_json(
            state_path,
            {
                "schema": "goalexit-postfeedback-strong-gru-outer-state-v1",
                "post_hoc": True,
                "completed_variants": sorted(
                    {variant for values in state.values() for variant in values}
                ),
                "results": dict(sorted(state.items())),
                "outer_labels_accessed_during_fit": False,
            },
        )
        print(f"completed outer {context_id}", flush=True)


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def analyze(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    dev_config = load_yaml(resolve(root, config["development_config"]))
    split = load_split(resolve(root, config["split_manifest"]))
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    state = json.loads(
        resolve(root, config["outputs"]["outer_state"]).read_text(encoding="utf-8")
    )
    if state.get("completed_variants") != sorted(OUTER_VARIANTS):
        raise ValueError("Strong-GRU outer state is incomplete")
    labels = pd.read_parquet(
        resolve(root, dev_config["outputs"]["trial_table"]),
        columns=["trial_id", "goal_index", "reader_id", "article_id"],
    )
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in OUTER_VARIANTS}
    audit: list[dict[str, Any]] = []
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        _, target_ids = outer_context_positions(store.metadata, split, context)
        for variant in OUTER_VARIANTS:
            result = state["results"][context_id][variant]
            path = Path(result["prediction_path"])
            frame = pd.read_parquet(path, columns=["trial_id", "probability"])
            if frame["trial_id"].duplicated().any() or set(
                frame["trial_id"].astype(str)
            ) != set(target_ids):
                raise ValueError(f"Prediction membership mismatch: {context_id}/{variant}")
            if sha256(path) != result["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {context_id}/{variant}")
            pieces[variant].append(frame.rename(columns={"probability": variant}))
            audit.append(
                {
                    "context_id": context_id,
                    "variant": variant,
                    "prediction_sha256": result["prediction_sha256"],
                }
            )
    frame = labels.copy()
    for variant, values in pieces.items():
        incoming = pd.concat(values, ignore_index=True)
        if incoming["trial_id"].duplicated().any() or len(incoming) != len(labels):
            raise ValueError(f"Prediction partition is not exact: {variant}")
        frame = frame.merge(incoming, on="trial_id", validate="one_to_one")
    target = frame["goal_index"].to_numpy(np.int64)
    articles = frame["article_id"].astype(str).to_numpy()
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        values = {
            name: evaluate_binary(
                target, frame[f"{name}_s{seed}"].to_numpy(), articles
            ).to_dict()
            for name in ("tuned_bce", "tuned_kd", "class_mean_kd")
        }
        per_seed.append(
            {
                "seed": seed,
                **values,
                "kd_minus_bce": values["tuned_kd"]["balanced_accuracy"]
                - values["tuned_bce"]["balanced_accuracy"],
                "kd_minus_class_mean": values["tuned_kd"]["balanced_accuracy"]
                - values["class_mean_kd"]["balanced_accuracy"],
            }
        )
    pairs = {
        "tuned_kd_minus_tuned_bce": tuple(
            (f"tuned_kd_s{seed}", f"tuned_bce_s{seed}") for seed in SEEDS
        ),
        "tuned_kd_minus_class_mean_kd": tuple(
            (f"tuned_kd_s{seed}", f"class_mean_kd_s{seed}") for seed in SEEDS
        ),
        "class_mean_kd_minus_tuned_bce": tuple(
            (f"class_mean_kd_s{seed}", f"tuned_bce_s{seed}") for seed in SEEDS
        ),
    }
    contrasts: dict[str, Any] = {}
    for index, (name, pair) in enumerate(pairs.items()):
        contrasts[name] = {
            "reader_cluster_bootstrap": _mean_reader_bootstrap(
                frame,
                pair,
                replicates=int(config["bootstrap"]["reader_cluster_replicates"]),
                seed=int(config["bootstrap"]["seed"]) + index,
            ),
            "reader_article_crossed_bootstrap": _mean_crossed_bootstrap(
                frame,
                pair,
                replicates=int(config["bootstrap"]["reader_article_replicates"]),
                seed=int(config["bootstrap"]["seed"]) + 100 + index,
            ),
        }
    selections = json.loads(
        resolve(root, config["outputs"]["selection_state"]).read_text(encoding="utf-8")
    )["selections"]
    architecture_counts: dict[str, int] = {}
    for value in selections.values():
        key = str(value["bce"]["selected_architecture_id"])
        architecture_counts[key] = architecture_counts.get(key, 0) + 1
    result = {
        "schema": "goalexit-postfeedback-strong-gru-results-v1",
        "status": "complete",
        "post_hoc": True,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "contrasts": contrasts,
        "bce_selected_architecture_counts": architecture_counts,
        "audit": audit,
    }
    save_json(resolve(root, config["outputs"]["metrics"]), result)
    resolve(root, config["outputs"]["report"]).write_text(
        _report(result), encoding="utf-8"
    )
    return result


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Strong GRU and matched distillation results",
        "",
        "**Status:** complete; post hoc.",
        "",
        "The GRU architecture was selected using BCE only. The selected architecture was then fixed for the matched KD search.",
        "",
        "| Seed | Tuned BCE | Tuned KD | Class-mean KD | KD - BCE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['tuned_bce']['balanced_accuracy']:.4f} | "
            f"{row['tuned_kd']['balanced_accuracy']:.4f} | "
            f"{row['class_mean_kd']['balanced_accuracy']:.4f} | "
            f"{row['kd_minus_bce']:+.4f} |"
        )
    lines.extend(["", "| Contrast | Mean delta BA | Reader 95% CI |", "|---|---:|---:|"])
    for name, value in result["contrasts"].items():
        summary = value["reader_cluster_bootstrap"]
        low, high = summary["ci_95_percentile"]
        lines.append(
            f"| {name.replace('_', ' ')} | {summary['point_mean_difference']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] |"
        )
    lines.extend(["", "Selected BCE architectures: " + json.dumps(result["bce_selected_architecture_counts"], sort_keys=True), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/postfeedback_strong_gru.yaml")
    )
    parser.add_argument(
        "--stage", choices=("bce", "kd", "outer", "analyze"), required=True
    )
    args = parser.parse_args()
    if args.stage in {"bce", "kd"}:
        run_inner(args.config, args.stage)
    elif args.stage == "outer":
        run_outer(args.config)
    else:
        result = analyze(args.config)
        summary = result["contrasts"]["tuned_kd_minus_tuned_bce"][
            "reader_cluster_bootstrap"
        ]
        print(f"tuned_kd_minus_tuned_bce={summary['point_mean_difference']:.6f}")


if __name__ == "__main__":
    main()
