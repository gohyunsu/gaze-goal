from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from goalexit.data.cache_roberta import load_cache
from goalexit.data.dataset import (
    GoalExitStore,
    load_allowed_targets,
    load_split,
    outer_context_positions,
)
from goalexit.eval.final_evaluation import fixed_epoch_from_selection
from goalexit.eval.metrics import evaluate_binary
from goalexit.eval.outer_analysis import _crossed_bootstrap, _reader_cluster_bootstrap
from goalexit.train.engine import load_checkpoint, predict_teacher
from goalexit.train.fixed_refit import fit_fixed_epochs
from goalexit.train.run_development import _architecture
from goalexit.utils import (
    canonical_json_sha256,
    git_commit,
    load_yaml,
    project_root,
    resolve,
    save_json,
    sha256,
)


TEACHER_VARIANTS = (
    "text_gaze_teacher",
    "text_gaze_teacher_s29",
    "text_gaze_teacher_s47",
    "text_only",
    "gaze_teacher",
)
STUDENT_VARIANTS = (
    "stats",
    "gru_s13",
    "label_smoothing",
    "gaze_kd",
    "text_gaze_kd_s13",
    "gru_s29",
    "text_gaze_kd_s29",
    "gru_s47",
    "text_gaze_kd_s47",
)
ALL_VARIANTS = (*TEACHER_VARIANTS, *STUDENT_VARIANTS)
ORIGINAL_TO_RERUN = {
    "b0": "stats",
    "b1": "gru_s13",
    "teacher": "text_gaze_teacher",
    "t_text": "text_only",
    "d": "text_gaze_kd_s13",
}


def _is_teacher_cache_variant(variant: str) -> bool:
    return (
        variant == "gaze_teacher"
        or variant.startswith("gaze_teacher_s")
        or variant == "text_gaze_teacher"
        or variant.startswith("text_gaze_teacher_s")
    )


def _paths(
    root: Path, config: dict[str, Any], context_id: str, variant: str
) -> dict[str, Path]:
    return {
        "run": resolve(root, config["outputs"]["output_root"]) / context_id / variant,
        "checkpoint": resolve(root, config["outputs"]["checkpoint_root"])
        / context_id
        / f"{variant}.pt",
        "prediction": resolve(root, config["outputs"]["prediction_root"])
        / context_id
        / f"{variant}.parquet",
    }


def _variant_spec(
    variant: str, config: dict[str, Any], selection_state: dict[str, Any], context_id: str
) -> dict[str, Any]:
    selections = selection_state["selections"][context_id]
    if variant == "stats":
        family, inherited, seed = "b0", "b0", int(config["base_seed"])
    elif variant.startswith("gru_s"):
        family, inherited, seed = "b1", "b1", int(variant.removeprefix("gru_s"))
    elif variant == "text_gaze_teacher":
        family, inherited, seed = "teacher", "teacher", int(config["base_seed"])
    elif variant.startswith("text_gaze_teacher_s"):
        family, inherited, seed = (
            "teacher",
            "teacher",
            int(variant.removeprefix("text_gaze_teacher_s")),
        )
    elif variant == "text_only":
        family, inherited, seed = "t_text", "t_text", int(config["base_seed"])
    elif variant.startswith("text_gaze_kd_s"):
        family, inherited, seed = "d", "d", int(variant.removeprefix("text_gaze_kd_s"))
    elif variant.startswith("shuffled_text_gaze_kd_s"):
        family, inherited, seed = (
            "d",
            "d",
            int(variant.removeprefix("shuffled_text_gaze_kd_s")),
        )
    elif variant == "gaze_teacher":
        family, inherited, seed = "gaze_teacher", "teacher", int(config["base_seed"])
    elif variant.startswith("gaze_teacher_s"):
        family, inherited, seed = (
            "gaze_teacher",
            "teacher",
            int(variant.removeprefix("gaze_teacher_s")),
        )
    elif variant == "label_smoothing":
        family, inherited, seed = "ls", "b1", int(config["base_seed"])
    elif variant.startswith("label_smoothing_s"):
        family, inherited, seed = (
            "ls",
            "b1",
            int(variant.removeprefix("label_smoothing_s")),
        )
    elif variant == "gaze_kd":
        family, inherited, seed = "d_gaze", "d", int(config["base_seed"])
    elif variant.startswith("gaze_kd_s"):
        family, inherited, seed = (
            "d_gaze",
            "d",
            int(variant.removeprefix("gaze_kd_s")),
        )
    else:
        raise ValueError(f"Unknown post-confirmatory variant {variant}")
    selection = selections[inherited]
    return {
        "variant": variant,
        "family": family,
        "inherited_family": inherited,
        "selection": selection,
        "seed": seed,
        "learning_rate": float(selection["selected_learning_rate"]),
        "alpha": (
            float(selection["selected_alpha"])
            if selection.get("selected_alpha") is not None
            else None
        ),
        "fixed_epochs": fixed_epoch_from_selection(selection),
        "label_smoothing": (
            float(config["label_smoothing"])
            if variant == "label_smoothing" or variant.startswith("label_smoothing_s")
            else 0.0
        ),
        "teacher_target_transform": (
            "class_conditional_cyclic_shuffle"
            if variant.startswith("shuffled_text_gaze_kd_s")
            else "identity"
        ),
    }


def _run_identity(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    dev_config: dict[str, Any],
    canonical: dict[str, Any],
    split_path: Path,
    protocol_path: Path,
    context: dict[str, Any],
    spec: dict[str, Any],
    training_ids: list[str],
    prediction_ids: list[str],
    teacher_cache_sha256: str | None,
    device: torch.device,
) -> dict[str, Any]:
    family = str(spec["family"])
    architecture_family = (
        "teacher"
        if family == "gaze_teacher"
        else "d"
        if family == "d_gaze"
        else "b1"
        if family == "ls"
        else family
    )
    teacher_family = family in {"teacher", "t_text", "gaze_teacher"}
    batch_key = "teacher" if teacher_family else "b0" if family == "b0" else "b1"
    dropout_key = "teacher" if teacher_family else "student"
    architecture = _architecture(architecture_family)
    if teacher_family:
        architecture["text_only"] = family == "t_text"
        architecture["gaze_only"] = family == "gaze_teacher"
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise RuntimeError("Post-confirmatory execution requires a clean committed tree")
    value: dict[str, Any] = {
        "schema": "goalexit-postconfirmatory-run-v1",
        "postconfirmatory": True,
        "outer_results_known_when_designed": True,
        "context_id": str(context["context_id"]),
        "outer_reader_fold": int(context["outer_reader_fold"]),
        "outer_article_fold": int(context["outer_article_fold"]),
        "model_family": str(spec["variant"]),
        "training_family": family,
        "inherited_selection_family": str(spec["inherited_family"]),
        "selected_candidate_id": str(spec["selection"]["selected_candidate_id"]),
        "learning_rate": float(spec["learning_rate"]),
        "alpha": spec["alpha"],
        "label_smoothing": float(spec["label_smoothing"]),
        "temperature": (
            float(dev_config["distillation_temperature"])
            if family in {"d", "d_gaze"}
            else None
        ),
        "fixed_epochs": int(spec["fixed_epochs"]),
        "epoch_rule": "integer median of the inherited three inner selected epochs",
        "architecture": architecture,
        "optimizer": "AdamW",
        "weight_decay": float(dev_config["weight_decay"]),
        "gradient_clip": float(dev_config["gradient_clip"]),
        "batch_size": int(dev_config["batch_size"][batch_key]),
        "dropout": float(dev_config["dropout"][dropout_key]),
        "seed": int(spec["seed"]),
        "training_device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(device),
        "host": platform.node(),
        "git_commit": commit,
        "confirmatory_config_sha256": sha256(
            resolve(root, config["confirmatory_config"])
        ),
        "ablation_config_sha256": sha256(config_path),
        "lock_sha256": sha256(protocol_path),
        "canonical_data_sha256": canonical["canonical_data_sha256"],
        "split_manifest_sha256": sha256(split_path),
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "prediction_trial_ids_sha256": canonical_json_sha256(sorted(prediction_ids)),
        "training_trials": len(training_ids),
        "prediction_trials": len(prediction_ids),
        "teacher_cache_sha256": teacher_cache_sha256,
        "teacher_target_transform": str(spec["teacher_target_transform"]),
        "outer_labels_accessed": False,
        "outer_scientific_metrics_computed": False,
    }
    value["run_identity_sha256"] = canonical_json_sha256(value)
    return value


def _class_conditional_shuffle(
    teacher_logits: dict[str, float],
    training_targets: pd.DataFrame,
    *,
    context_id: str,
    seed: int,
) -> tuple[dict[str, float], str]:
    target_map = (
        training_targets.set_index("trial_id")["goal_index"].astype(int).to_dict()
    )
    if set(target_map) != set(teacher_logits):
        raise ValueError("Teacher logits and targets differ before shuffling")
    shuffled: dict[str, float] = {}
    mapping: dict[str, str] = {}
    for label in (0, 1):
        ids = sorted(key for key, value in target_map.items() if value == label)
        if len(ids) < 2:
            raise ValueError("Class-conditional shuffle requires at least two trials")
        shuffle_seed = int(
            canonical_json_sha256(
                {
                    "context_id": context_id,
                    "seed": seed,
                    "label": label,
                    "transform": "class_conditional_cyclic_shuffle",
                }
            )[:16],
            16,
        )
        rng = np.random.default_rng(shuffle_seed)
        order = list(np.asarray(ids)[rng.permutation(len(ids))])
        for position, trial_id in enumerate(order):
            source_id = str(order[(position + 1) % len(order)])
            trial_id = str(trial_id)
            shuffled[trial_id] = float(teacher_logits[source_id])
            mapping[trial_id] = source_id
    if any(trial_id == source_id for trial_id, source_id in mapping.items()):
        raise AssertionError("Class-conditional cyclic shuffle retained an identity mapping")
    before = {
        label: sorted(teacher_logits[key] for key, value in target_map.items() if value == label)
        for label in (0, 1)
    }
    after = {
        label: sorted(shuffled[key] for key, value in target_map.items() if value == label)
        for label in (0, 1)
    }
    if before != after:
        raise AssertionError("Shuffle changed a class-conditional logit distribution")
    return shuffled, canonical_json_sha256(mapping)


def _load_complete(
    paths: dict[str, Path], run_identity: str, prediction_ids: list[str]
) -> dict[str, Any] | None:
    result_path = paths["run"] / "result.json"
    if not all(path.is_file() for path in (result_path, paths["checkpoint"], paths["prediction"])):
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if value.get("run_config", {}).get("run_identity_sha256") != run_identity:
        return None
    if value.get("checkpoint_sha256") != sha256(paths["checkpoint"]):
        return None
    if value.get("prediction_sha256") != sha256(paths["prediction"]):
        return None
    frame = pd.read_parquet(paths["prediction"])
    if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
        prediction_ids
    ):
        raise ValueError("Completed ablation prediction membership mismatch")
    value["result_path"] = str(result_path)
    value["result_sha256"] = sha256(result_path)
    return value


def _teacher_cache_path(
    root: Path, config: dict[str, Any], context_id: str, teacher_variant: str
) -> Path:
    return (
        resolve(root, config["outputs"]["teacher_cache_root"])
        / context_id
        / f"{teacher_variant}.parquet"
    )


def _write_teacher_cache(
    *,
    root: Path,
    config: dict[str, Any],
    dev_config: dict[str, Any],
    context_id: str,
    training_ids: list[str],
    store: GoalExitStore,
    text_cache: np.ndarray,
    text_index: dict[str, int],
    result: dict[str, Any],
    device: torch.device,
    teacher_variant: str,
) -> None:
    if not _is_teacher_cache_variant(teacher_variant):
        raise ValueError(f"Unauthorized teacher-cache source {teacher_variant}")
    path = _teacher_cache_path(root, config, context_id, teacher_variant)
    manifest_path = path.with_suffix(".json")
    model, preprocessor, checkpoint = load_checkpoint(Path(result["checkpoint_path"]), device)
    expected_gaze_only = teacher_variant == "gaze_teacher" or teacher_variant.startswith(
        "gaze_teacher_s"
    )
    if bool(checkpoint.get("gaze_only")) != expected_gaze_only:
        raise ValueError("Teacher checkpoint modality does not match cache variant")
    if bool(checkpoint.get("text_only")):
        raise ValueError("Text-only checkpoints cannot provide KD targets")
    frame = predict_teacher(
        model,
        training_ids,
        store,
        preprocessor,
        text_cache,
        text_index,
        batch_size=int(dev_config["batch_size"]["teacher"]),
        device=device,
    )[["trial_id", "logit"]].rename(columns={"logit": "teacher_logit"})
    if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
        training_ids
    ):
        raise ValueError("Gaze-teacher cache support mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    save_json(
        manifest_path,
        {
            "schema": "goalexit-postconfirmatory-teacher-cache-v1",
            "context_id": context_id,
            "teacher_variant": teacher_variant,
            "training_trials": len(training_ids),
            "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
            "source_checkpoint_sha256": result["checkpoint_sha256"],
            "cache_sha256": sha256(path),
            "postconfirmatory": True,
            "outer_labels_accessed": False,
        },
    )


def _teacher_logits(
    root: Path,
    config: dict[str, Any],
    context_id: str,
    training_ids: list[str],
    teacher_variant: str,
) -> tuple[dict[str, float], str]:
    path = _teacher_cache_path(root, config, context_id, teacher_variant)
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Teacher cache is absent for {context_id}/{teacher_variant}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_sha256") != sha256(path):
        raise ValueError(f"Teacher cache hash mismatch for {context_id}/{teacher_variant}")
    if manifest.get("teacher_variant") != teacher_variant:
        raise ValueError("Teacher-cache manifest variant mismatch")
    frame = pd.read_parquet(path, columns=["trial_id", "teacher_logit"])
    if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
        training_ids
    ):
        raise ValueError(
            f"Teacher cache membership mismatch for {context_id}/{teacher_variant}"
        )
    return (
        dict(zip(frame["trial_id"].astype(str), frame["teacher_logit"].astype(float))),
        sha256(path),
    )


def _run_context(
    config_path_string: str,
    context_index: int,
    variants: tuple[str, ...],
    device_string: str,
) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, config_path_string)
    config = load_yaml(config_path)
    dev_config = load_yaml(resolve(root, config["development_config"]))
    selection_state = json.loads(
        resolve(root, config["development_state"]).read_text(encoding="utf-8")
    )
    split_path = resolve(root, config["split_manifest"])
    split = load_split(split_path)
    context = split["contexts"][context_index]
    context_id = str(context["context_id"])
    canonical = json.loads(
        resolve(root, config["canonical_manifest"]).read_text(encoding="utf-8")
    )
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    training_ids, prediction_ids = outer_context_positions(store.metadata, split, context)
    training_targets = load_allowed_targets(
        resolve(root, dev_config["outputs"]["trial_table"]),
        training_ids,
        set(prediction_ids),
    )
    prediction_metadata = store.metadata.loc[prediction_ids].reset_index(drop=True)
    torch.set_num_threads(int(config["training_threads_per_worker"]))
    device = torch.device(device_string)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Uniform ablation suite requires the configured CUDA device")
    torch.cuda.set_device(device)
    need_text = any(
        variant in TEACHER_VARIANTS or _is_teacher_cache_variant(variant)
        for variant in variants
    )
    text_cache: np.ndarray | None = None
    text_index: dict[str, int] | None = None
    if need_text:
        text_cache, text_index = load_cache(
            resolve(root, dev_config["roberta"]["cache_array"]),
            resolve(root, dev_config["roberta"]["cache_ids"]),
        )
    protocol_path = resolve(root, config["protocol"])
    outputs: dict[str, Any] = {}
    for variant in variants:
        spec = _variant_spec(variant, config, selection_state, context_id)
        teacher_logits = None
        teacher_cache_hash = None
        if variant == "gaze_kd":
            teacher_logits, teacher_cache_hash = _teacher_logits(
                root, config, context_id, training_ids, "gaze_teacher"
            )
        elif variant.startswith("gaze_kd_s"):
            seed = int(variant.removeprefix("gaze_kd_s"))
            teacher_logits, teacher_cache_hash = _teacher_logits(
                root, config, context_id, training_ids, f"gaze_teacher_s{seed}"
            )
        elif variant.startswith("text_gaze_kd_s"):
            seed = int(variant.removeprefix("text_gaze_kd_s"))
            teacher_variant = (
                "text_gaze_teacher"
                if seed == int(config["base_seed"])
                else f"text_gaze_teacher_s{seed}"
            )
            teacher_logits, teacher_cache_hash = _teacher_logits(
                root, config, context_id, training_ids, teacher_variant
            )
        elif variant.startswith("shuffled_text_gaze_kd_s"):
            seed = int(variant.removeprefix("shuffled_text_gaze_kd_s"))
            teacher_variant = (
                "text_gaze_teacher"
                if seed == int(config["base_seed"])
                else f"text_gaze_teacher_s{seed}"
            )
            teacher_logits, teacher_cache_hash = _teacher_logits(
                root, config, context_id, training_ids, teacher_variant
            )
            teacher_logits, shuffle_hash = _class_conditional_shuffle(
                teacher_logits,
                training_targets,
                context_id=context_id,
                seed=seed,
            )
        run_config = _run_identity(
            root=root,
            config_path=config_path,
            config=config,
            dev_config=dev_config,
            canonical=canonical,
            split_path=split_path,
            protocol_path=protocol_path,
            context=context,
            spec=spec,
            training_ids=training_ids,
            prediction_ids=prediction_ids,
            teacher_cache_sha256=teacher_cache_hash,
            device=device,
        )
        if variant.startswith("shuffled_text_gaze_kd_s"):
            run_config["teacher_target_mapping_sha256"] = shuffle_hash
            run_config["run_identity_sha256"] = canonical_json_sha256(
                {key: value for key, value in run_config.items() if key != "run_identity_sha256"}
            )
        paths = _paths(root, config, context_id, variant)
        result = _load_complete(paths, run_config["run_identity_sha256"], prediction_ids)
        if result is None:
            family = str(spec["family"])
            is_teacher = family in {"teacher", "t_text", "gaze_teacher"}
            batch_key = "teacher" if is_teacher else "b0" if family == "b0" else "b1"
            result = fit_fixed_epochs(
                family=family,
                training_ids=training_ids,
                prediction_ids=prediction_ids,
                training_targets=training_targets,
                prediction_metadata=prediction_metadata,
                store=store,
                learning_rate=float(spec["learning_rate"]),
                fixed_epochs=int(spec["fixed_epochs"]),
                seed=int(spec["seed"]),
                weight_decay=float(dev_config["weight_decay"]),
                batch_size=int(dev_config["batch_size"][batch_key]),
                gradient_clip=float(dev_config["gradient_clip"]),
                dropout=float(dev_config["dropout"]["teacher" if is_teacher else "student"]),
                device=device,
                run_directory=paths["run"],
                checkpoint_path=paths["checkpoint"],
                prediction_path=paths["prediction"],
                run_config=run_config,
                text_cache=text_cache if is_teacher else None,
                text_index=text_index if is_teacher else None,
                teacher_training_logits=teacher_logits,
                alpha=spec["alpha"],
                temperature=float(dev_config["distillation_temperature"]),
                label_smoothing=float(spec["label_smoothing"]),
            )
        if _is_teacher_cache_variant(variant):
            assert text_cache is not None and text_index is not None
            _write_teacher_cache(
                root=root,
                config=config,
                dev_config=dev_config,
                context_id=context_id,
                training_ids=training_ids,
                store=store,
                text_cache=text_cache,
                text_index=text_index,
                result=result,
                device=device,
                teacher_variant=variant,
            )
        outputs[variant] = result
    return {"context_id": context_id, "variants": outputs}


def _run_shard(
    config_path_string: str,
    context_indices: tuple[int, ...],
    variants: tuple[str, ...],
    device_string: str,
) -> list[dict[str, Any]]:
    return [
        _run_context(config_path_string, index, variants, device_string)
        for index in context_indices
    ]


def run_stage(config_path: Path, variants: tuple[str, ...]) -> dict[str, Any]:
    root = project_root()
    resolved = resolve(root, config_path)
    config = load_yaml(resolved)
    if not config.get("postconfirmatory") or not config.get("outer_results_known_when_designed"):
        raise ValueError("Ablation status must be explicitly post-confirmatory")
    split = load_split(resolve(root, config["split_manifest"]))
    if len(split["contexts"]) != int(config["outer_contexts"]):
        raise ValueError("Post-confirmatory context count mismatch")
    if config.get("device") != "cuda":
        raise ValueError("Controlled ablations require a uniform CUDA environment")
    gpu_ids = tuple(int(value) for value in config["gpu_ids"])
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must contain unique CUDA device indices")
    shards = {
        gpu_id: tuple(range(offset, len(split["contexts"]), len(gpu_ids)))
        for offset, gpu_id in enumerate(gpu_ids)
    }
    results: dict[str, Any] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(gpu_ids), mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(
                _run_shard,
                str(config_path),
                indices,
                variants,
                f"cuda:{gpu_id}",
            ): gpu_id
            for gpu_id, indices in shards.items()
        }
        for future in concurrent.futures.as_completed(futures):
            gpu_id = futures[future]
            for result in future.result():
                context_id = result["context_id"]
                results[context_id] = result["variants"]
                print(
                    f"completed {context_id} gpu={gpu_id} variants={','.join(variants)}",
                    flush=True,
                )
    state_path = resolve(root, config["outputs"]["output_root"]) / "fit_state.json"
    existing: dict[str, Any] = {}
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8")).get("results", {})
    for context_id, context_results in results.items():
        existing.setdefault(context_id, {}).update(context_results)
    save_json(
        state_path,
        {
            "schema": "goalexit-postconfirmatory-fit-state-v1",
            "postconfirmatory": True,
            "outer_results_known_when_designed": True,
            "completed_variants": sorted(
                {variant for context in existing.values() for variant in context}
            ),
            "results": dict(sorted(existing.items())),
            "outer_labels_accessed_during_fit": False,
        },
    )
    return results


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def _contrast_summary(
    frame: pd.DataFrame,
    context_rows: list[dict[str, Any]],
    left: str,
    right: str,
    config: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    target = frame["goal_index"].to_numpy(np.int64)
    differences = np.asarray([row[left] - row[right] for row in context_rows])
    return {
        "left": left,
        "right": right,
        "pooled_difference": _balanced_accuracy(target, frame[left].to_numpy())
        - _balanced_accuracy(target, frame[right].to_numpy()),
        "context_mean_difference": float(differences.mean()),
        "positive_contexts": int((differences > 0).sum()),
        "contexts": len(differences),
        "reader_cluster_bootstrap": _reader_cluster_bootstrap(
            frame,
            left,
            right,
            replicates=int(config["bootstrap"]["reader_cluster_replicates"]),
            seed=int(config["bootstrap"]["seed"]) + offset,
        ),
        "reader_article_crossed_bootstrap": _crossed_bootstrap(
            frame,
            left,
            right,
            replicates=int(config["bootstrap"]["reader_article_replicates"]),
            seed=int(config["bootstrap"]["seed"]) + 100 + offset,
        ),
    }


def analyze(config_path: Path) -> dict[str, Any]:
    root = project_root()
    resolved = resolve(root, config_path)
    config = load_yaml(resolved)
    dev_config = load_yaml(resolve(root, config["development_config"]))
    split = load_split(resolve(root, config["split_manifest"]))
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    state_path = resolve(root, config["outputs"]["output_root"]) / "fit_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("completed_variants") != sorted(ALL_VARIANTS):
        raise ValueError("Post-confirmatory fit state is incomplete")
    frames: dict[str, list[pd.DataFrame]] = {variant: [] for variant in ALL_VARIANTS}
    audit_entries: list[dict[str, Any]] = []
    trial_to_context: dict[str, str] = {}
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        _, target_ids = outer_context_positions(store.metadata, split, context)
        trial_to_context.update({trial_id: context_id for trial_id in target_ids})
        context_state = state["results"].get(context_id, {})
        if set(context_state) != set(ALL_VARIANTS):
            raise ValueError(f"Uniform rerun variant set is incomplete for {context_id}")
        for variant in ALL_VARIANTS:
            result = context_state[variant]
            prediction_path = Path(result["prediction_path"])
            checkpoint_path = Path(result["checkpoint_path"])
            result_path = Path(result["result_path"])
            if sha256(prediction_path) != result["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {context_id}/{variant}")
            if sha256(checkpoint_path) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {context_id}/{variant}")
            if sha256(result_path) != result["result_sha256"]:
                raise ValueError(f"Result hash mismatch: {context_id}/{variant}")
            frame = pd.read_parquet(prediction_path, columns=["trial_id", "probability"])
            if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
                target_ids
            ):
                raise ValueError(f"Prediction membership mismatch: {context_id}/{variant}")
            frames[variant].append(frame)
            audit_entries.append(
                {
                    "context_id": context_id,
                    "variant": variant,
                    "prediction_rows": len(frame),
                    "prediction_sha256": result["prediction_sha256"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "result_sha256": result["result_sha256"],
                }
            )
    expected_ids = set(store.metadata["trial_id"].astype(str))
    combined: dict[str, pd.DataFrame] = {}
    for variant, pieces in frames.items():
        frame = pd.concat(pieces, ignore_index=True)
        if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != expected_ids:
            raise ValueError(f"Combined prediction partition mismatch for {variant}")
        combined[variant] = frame
    labels = pd.read_parquet(
        resolve(root, dev_config["outputs"]["trial_table"]),
        columns=["trial_id", "goal_index", "reader_id", "article_id"],
    )
    frame = labels.copy()
    for variant in ALL_VARIANTS:
        incoming = combined[variant].rename(columns={"probability": variant})
        frame = frame.merge(incoming, on="trial_id", how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError("Post-confirmatory label join is not exact")
    target = frame["goal_index"].to_numpy(np.int64)
    articles = frame["article_id"].astype(str).to_numpy()
    pooled = {
        variant: evaluate_binary(target, frame[variant].to_numpy(), articles).to_dict()
        for variant in ALL_VARIANTS
    }
    frame["context_id"] = frame["trial_id"].map(trial_to_context)
    context_rows: list[dict[str, Any]] = []
    for context_id, rows in frame.groupby("context_id", sort=True):
        row: dict[str, Any] = {"context_id": str(context_id), "trials": len(rows)}
        local_target = rows["goal_index"].to_numpy(np.int64)
        for variant in ALL_VARIANTS:
            row[variant] = _balanced_accuracy(local_target, rows[variant].to_numpy())
        context_rows.append(row)
    contrast_pairs = (
        ("text_gaze_kd_minus_gru", "text_gaze_kd_s13", "gru_s13"),
        ("label_smoothing_minus_gru", "label_smoothing", "gru_s13"),
        ("gaze_kd_minus_gru", "gaze_kd", "gru_s13"),
        ("text_gaze_kd_minus_gaze_kd", "text_gaze_kd_s13", "gaze_kd"),
        ("text_gaze_teacher_minus_gaze_teacher", "text_gaze_teacher", "gaze_teacher"),
    )
    contrasts = {
        name: _contrast_summary(frame, context_rows, left, right, config, index)
        for index, (name, left, right) in enumerate(contrast_pairs)
    }
    seed_sensitivity: dict[str, Any] = {}
    for seed in (int(config["base_seed"]), *map(int, config["additional_seeds"])):
        gru = f"gru_s{seed}"
        kd = f"text_gaze_kd_s{seed}"
        name = f"seed_{seed}"
        seed_sensitivity[name] = _contrast_summary(
            frame, context_rows, kd, gru, config, 10 + seed
        )
        seed_sensitivity[name]["gru_balanced_accuracy"] = pooled[gru]["balanced_accuracy"]
        seed_sensitivity[name]["kd_balanced_accuracy"] = pooled[kd]["balanced_accuracy"]
        teacher = (
            "text_gaze_teacher"
            if seed == int(config["base_seed"])
            else f"text_gaze_teacher_s{seed}"
        )
        seed_sensitivity[name]["teacher_balanced_accuracy"] = pooled[teacher][
            "balanced_accuracy"
        ]
    seed_deltas = [
        row["pooled_difference"] for row in seed_sensitivity.values()
    ]
    original = json.loads(
        resolve(root, config["confirmatory_metrics"]).read_text(encoding="utf-8")
    )["pooled_metrics"]
    replication = {
        original_name: {
            "original_balanced_accuracy": float(original[original_name]["balanced_accuracy"]),
            "gpu_rerun_balanced_accuracy": float(pooled[rerun_name]["balanced_accuracy"]),
            "rerun_minus_original": float(
                pooled[rerun_name]["balanced_accuracy"]
                - original[original_name]["balanced_accuracy"]
            ),
        }
        for original_name, rerun_name in ORIGINAL_TO_RERUN.items()
    }
    audit = {
        "schema": "goalexit-postconfirmatory-ablation-audit-v1",
        "status": "PASS",
        "postconfirmatory": True,
        "contexts": len(split["contexts"]),
        "uniform_rerun_variants": list(ALL_VARIANTS),
        "validated_new_results": len(audit_entries),
        "expected_new_results": len(split["contexts"]) * len(ALL_VARIANTS),
        "trials_per_variant": len(expected_ids),
        "hashes_verified": True,
        "prediction_partitions_exact": True,
        "entries": audit_entries,
    }
    audit_path = resolve(root, config["outputs"]["audit"])
    save_json(audit_path, audit)
    output = {
        "schema": "goalexit-postconfirmatory-ablation-results-v1",
        "status": "PASS",
        "postconfirmatory": True,
        "outer_results_known_when_designed": True,
        "pooled_metrics": pooled,
        "contrasts": contrasts,
        "seed_sensitivity": seed_sensitivity,
        "original_cpu_to_uniform_gpu_replication": replication,
        "seed_delta_summary": {
            "seeds": [int(config["base_seed"]), *map(int, config["additional_seeds"])],
            "mean_difference": float(np.mean(seed_deltas)),
            "minimum_difference": float(np.min(seed_deltas)),
            "maximum_difference": float(np.max(seed_deltas)),
            "positive_seeds": int(np.sum(np.asarray(seed_deltas) > 0)),
        },
        "context_metrics": context_rows,
        "outer_trials": len(frame),
        "audit_sha256": sha256(audit_path),
    }
    metrics_path = resolve(root, config["outputs"]["metrics"])
    save_json(metrics_path, output)
    _write_report(resolve(root, config["outputs"]["report"]), output)
    return output


def _write_report(path: Path, result: dict[str, Any]) -> None:
    pooled = result["pooled_metrics"]
    lines = [
        "# Post-confirmatory ablation report",
        "",
        "**Status: PASS**",
        "",
        "These analyses were designed after the original outer results were known.",
        "They are mechanism and robustness evidence, not a second confirmatory test.",
        "",
        "## Pooled performance",
        "",
        "| Variant | BA | AUROC | Macro F1 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "stats",
        "gru_s13",
        "label_smoothing",
        "gaze_teacher",
        "gaze_kd",
        "text_gaze_teacher",
        "text_only",
        "text_gaze_kd_s13",
        "gru_s29",
        "text_gaze_kd_s29",
        "gru_s47",
        "text_gaze_kd_s47",
    )
    for variant in order:
        row = pooled[variant]
        lines.append(
            f"| {variant} | {row['balanced_accuracy']:.4f} | {row['auroc']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['nll']:.4f} | {row['brier']:.4f} | "
            f"{row['ece']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Ablation contrasts",
            "",
            "| Contrast | Pooled BA delta | Reader 95% CI | Crossed 95% CI | Positive contexts |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in result["contrasts"].items():
        reader = row["reader_cluster_bootstrap"]["ci_95_percentile"]
        crossed = row["reader_article_crossed_bootstrap"]["ci_95_percentile"]
        lines.append(
            f"| {name} | {row['pooled_difference']:+.4f} | "
            f"[{reader[0]:.4f}, {reader[1]:.4f}] | "
            f"[{crossed[0]:.4f}, {crossed[1]:.4f}] | "
            f"{row['positive_contexts']}/{row['contexts']} |"
        )
    lines.extend(
        [
            "",
            "## Seed sensitivity",
            "",
            "| Seed | Teacher BA | GRU BA | Text--gaze KD BA | Delta | Reader 95% CI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in result["seed_sensitivity"].items():
        reader = row["reader_cluster_bootstrap"]["ci_95_percentile"]
        seed = name.removeprefix("seed_")
        lines.append(
            f"| {seed} | {row['teacher_balanced_accuracy']:.4f} | "
            f"{row['gru_balanced_accuracy']:.4f} | "
            f"{row['kd_balanced_accuracy']:.4f} | {row['pooled_difference']:+.4f} | "
            f"[{reader[0]:.4f}, {reader[1]:.4f}] |"
        )
    summary = result["seed_delta_summary"]
    lines.extend(
        [
            "",
            f"The KD difference is positive for {summary['positive_seeds']}/3 seeds; "
            f"mean delta {summary['mean_difference']:+.4f}, range "
            f"[{summary['minimum_difference']:+.4f}, {summary['maximum_difference']:+.4f}].",
            "",
            "## Original CPU to uniform GPU replication",
            "",
            "| Original family | Original CPU BA | Uniform GPU rerun BA | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, row in result["original_cpu_to_uniform_gpu_replication"].items():
        lines.append(
            f"| {family} | {row['original_balanced_accuracy']:.4f} | "
            f"{row['gpu_rerun_balanced_accuracy']:.4f} | "
            f"{row['rerun_minus_original']:+.4f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-confirmatory GoalExit ablations")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/postconfirmatory_ablation.yaml")
    )
    parser.add_argument(
        "--stage", choices=("teachers", "students", "analyze"), required=True
    )
    args = parser.parse_args()
    if args.stage == "teachers":
        run_stage(args.config, TEACHER_VARIANTS)
    elif args.stage == "students":
        run_stage(args.config, STUDENT_VARIANTS)
    else:
        analyze(args.config)


if __name__ == "__main__":
    main()
