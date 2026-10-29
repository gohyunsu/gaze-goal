from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from goalexit.data.cache_roberta import load_cache
from goalexit.data.dataset import GoalExitStore, load_allowed_targets, load_split, outer_context_positions
from goalexit.eval.outer_guard import validate_kd_outer_lock
from goalexit.train.engine import load_checkpoint, predict_teacher
from goalexit.train.fixed_refit import fit_fixed_epochs
from goalexit.train.run_development import _architecture
from goalexit.utils import canonical_json_sha256, load_yaml, project_root, resolve, save_json, sha256


INITIAL_FAMILIES = ("b0", "b1", "teacher", "t_text")


def fixed_epoch_from_selection(selection: dict[str, Any]) -> int:
    epochs = sorted(int(row["selected_epoch"]) for row in selection["selected_results"])
    if len(epochs) != 3 or epochs[0] < 1:
        raise ValueError("A fixed refit epoch requires exactly three valid selected epochs")
    return epochs[1]


def _paths(root: Path, config: dict[str, Any], context_id: str, family: str) -> dict[str, Path]:
    return {
        "run": resolve(root, config["outputs"]["output_root"]) / context_id / family,
        "checkpoint": resolve(root, config["outputs"]["checkpoint_root"]) / context_id / f"{family}.pt",
        "prediction": resolve(root, config["outputs"]["prediction_root"]) / context_id / f"{family}.parquet",
    }


def _load_complete(
    paths: dict[str, Path], run_identity: str, prediction_ids: list[str]
) -> dict[str, Any] | None:
    result_path = paths["run"] / "result.json"
    if not all(path.is_file() for path in (result_path, paths["checkpoint"], paths["prediction"])):
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if value.get("schema") != "goalexit-outer-refit-result-v1":
        return None
    if value.get("run_config", {}).get("run_identity_sha256") != run_identity:
        return None
    if value.get("checkpoint_sha256") != sha256(paths["checkpoint"]):
        return None
    if value.get("prediction_sha256") != sha256(paths["prediction"]):
        return None
    if value.get("outer_labels_accessed") is not False:
        raise PermissionError("A pre-analysis result claims outer-label access")
    if value.get("outer_scientific_metrics_computed") is not False:
        raise PermissionError("A pre-analysis result claims outer metric computation")
    frame = pd.read_parquet(paths["prediction"])
    if frame["trial_id"].duplicated().any():
        raise ValueError("Duplicate trial in an outer prediction file")
    if set(frame["trial_id"].astype(str)) != set(prediction_ids):
        raise ValueError("Completed outer prediction membership differs from the lock")
    if not np.isfinite(frame[["logit", "probability"]].to_numpy()).all():
        raise ValueError("Non-finite target-free outer prediction")
    value["result_path"] = str(result_path)
    value["result_sha256"] = sha256(result_path)
    return value


def _teacher_cache(
    *,
    root: Path,
    config: dict[str, Any],
    dev_config: dict[str, Any],
    context_id: str,
    training_ids: list[str],
    store: GoalExitStore,
    text_cache: np.ndarray,
    text_index: dict[str, int],
    teacher_result: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    path = resolve(root, config["outputs"]["teacher_cache_root"]) / f"{context_id}.parquet"
    manifest_path = path.with_suffix(".json")
    expected = {
        "schema": "goalexit-outer-teacher-cache-v1",
        "context_id": context_id,
        "training_trials": len(training_ids),
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "source_teacher_checkpoint_path": teacher_result["checkpoint_path"],
        "source_teacher_checkpoint_sha256": teacher_result["checkpoint_sha256"],
        "source_teacher_run_identity_sha256": teacher_result["run_config"]["run_identity_sha256"],
        "outer_rows_touched": False,
        "outer_labels_accessed": False,
    }
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                raise ValueError(f"Outer teacher-cache provenance mismatch: {key}")
        if manifest.get("cache_sha256") != sha256(path):
            raise ValueError("Outer teacher-cache hash mismatch")
    else:
        if path.exists() or manifest_path.exists():
            raise RuntimeError("Incomplete outer teacher-cache artifact pair")
        checkpoint_path = Path(teacher_result["checkpoint_path"])
        if sha256(checkpoint_path) != teacher_result["checkpoint_sha256"]:
            raise ValueError("Outer teacher checkpoint hash mismatch")
        model, preprocessor, checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint["run_config"]["run_identity_sha256"] != expected["source_teacher_run_identity_sha256"]:
            raise ValueError("Outer teacher checkpoint identity mismatch")
        prediction = predict_teacher(
            model, training_ids, store, preprocessor, text_cache, text_index,
            batch_size=int(dev_config["batch_size"]["teacher"]), device=device,
        )
        frame = prediction[["trial_id", "logit"]].rename(columns={"logit": "teacher_logit"})
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        manifest = dict(expected)
        manifest["cache_path"] = str(path)
        manifest["cache_sha256"] = sha256(path)
        save_json(manifest_path, manifest)
    frame = pd.read_parquet(path, columns=["trial_id", "teacher_logit"])
    if frame["trial_id"].duplicated().any():
        raise ValueError("Duplicate trial in outer teacher cache")
    if set(frame["trial_id"].astype(str)) != set(training_ids):
        raise ValueError("Outer teacher-cache membership mismatch")
    if not np.isfinite(frame["teacher_logit"]).all():
        raise ValueError("Non-finite outer teacher logit")
    logits = dict(zip(frame["trial_id"].astype(str), frame["teacher_logit"].astype(float)))
    return logits, manifest


def _run_identity(
    *, root: Path, config_path: Path, config: dict[str, Any], dev_config: dict[str, Any],
    lock: dict[str, Any], canonical: dict[str, Any], split_path: Path, context: dict[str, Any],
    family: str, selection: dict[str, Any], training_ids: list[str], prediction_ids: list[str],
    teacher_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_key = "teacher" if family in {"teacher", "t_text"} else "b1" if family in {"b1", "d"} else "b0"
    dropout_key = "teacher" if family in {"teacher", "t_text"} else "student"
    value: dict[str, Any] = {
        "schema": "goalexit-outer-refit-run-v1",
        "context_id": str(context["context_id"]),
        "outer_reader_fold": int(context["outer_reader_fold"]),
        "outer_article_fold": int(context["outer_article_fold"]),
        "model_family": family,
        "selected_candidate_id": selection["selected_candidate_id"],
        "learning_rate": float(selection["selected_learning_rate"]),
        "alpha": selection.get("selected_alpha"),
        "fixed_epochs": fixed_epoch_from_selection(selection),
        "selected_inner_epochs": [int(row["selected_epoch"]) for row in selection["selected_results"]],
        "epoch_rule": config["epoch_rule"],
        "temperature": float(dev_config["distillation_temperature"]) if family == "d" else None,
        "architecture": _architecture(family),
        "optimizer": "AdamW",
        "weight_decay": float(dev_config["weight_decay"]),
        "gradient_clip": float(dev_config["gradient_clip"]),
        "batch_size": int(dev_config["batch_size"][batch_key]),
        "dropout": float(dev_config["dropout"][dropout_key]),
        "seed": int(config["seed"]),
        "git_commit": lock["execution_commit"],
        "implementation_commit": lock["implementation_commit"],
        "lock_sha256": lock["lock_sha256"],
        "confirmatory_config_sha256": sha256(config_path),
        "development_state_sha256": sha256(resolve(root, config["development_state"])),
        "canonical_data_sha256": canonical["canonical_data_sha256"],
        "split_manifest_sha256": sha256(split_path),
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "prediction_trial_ids_sha256": canonical_json_sha256(sorted(prediction_ids)),
        "training_trials": len(training_ids),
        "prediction_trials": len(prediction_ids),
        "target_access_scope": "population training labels only; outer target metadata only",
        "outer_labels_accessed": False,
        "outer_scientific_metrics_computed": False,
    }
    if teacher_manifest is not None:
        value["teacher_cache_sha256"] = teacher_manifest["cache_sha256"]
        value["teacher_checkpoint_sha256"] = teacher_manifest["source_teacher_checkpoint_sha256"]
    value["run_identity_sha256"] = canonical_json_sha256(value)
    return value


def _run_context(config_path_string: str, context_index: int, families: tuple[str, ...]) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, config_path_string)
    config = load_yaml(config_path)
    lock = validate_kd_outer_lock(config_path, root)
    dev_config = load_yaml(resolve(root, config["development_config"]))
    selection_state = json.loads(resolve(root, config["development_state"]).read_text(encoding="utf-8"))
    split_path = resolve(root, config["split_manifest"])
    split = load_split(split_path)
    context = split["contexts"][context_index]
    context_id = str(context["context_id"])
    canonical = json.loads(resolve(root, config["canonical_manifest"]).read_text(encoding="utf-8"))
    trial_path = resolve(root, dev_config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path, resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    training_ids, prediction_ids = outer_context_positions(store.metadata, split, context)
    training_targets = load_allowed_targets(trial_path, training_ids, set(prediction_ids))
    prediction_metadata = store.metadata.loc[prediction_ids].reset_index(drop=True)
    torch.set_num_threads(int(config["training_threads_per_worker"]))
    device = torch.device("cpu")
    needs_text = any(family in {"teacher", "t_text", "d"} for family in families)
    text_cache: np.ndarray | None = None
    text_index: dict[str, int] | None = None
    if needs_text:
        text_cache, text_index = load_cache(
            resolve(root, dev_config["roberta"]["cache_array"]),
            resolve(root, dev_config["roberta"]["cache_ids"]),
        )
    outputs: dict[str, Any] = {}
    for family in families:
        selection = selection_state["selections"][context_id][family]
        learning_rate = float(selection["selected_learning_rate"])
        alpha = float(selection["selected_alpha"]) if selection.get("selected_alpha") is not None else None
        fixed_epochs = fixed_epoch_from_selection(selection)
        paths = _paths(root, config, context_id, family)
        teacher_logits = None
        teacher_manifest = None
        if family == "d":
            teacher_paths = _paths(root, config, context_id, "teacher")
            teacher_selection = selection_state["selections"][context_id]["teacher"]
            teacher_identity = _run_identity(
                root=root, config_path=config_path, config=config, dev_config=dev_config,
                lock=lock, canonical=canonical, split_path=split_path, context=context,
                family="teacher", selection=teacher_selection, training_ids=training_ids,
                prediction_ids=prediction_ids,
            )
            teacher_result = _load_complete(
                teacher_paths, teacher_identity["run_identity_sha256"], prediction_ids
            )
            if teacher_result is None:
                raise RuntimeError(f"D requires a complete refit teacher for {context_id}")
            assert text_cache is not None and text_index is not None
            teacher_logits, teacher_manifest = _teacher_cache(
                root=root, config=config, dev_config=dev_config, context_id=context_id,
                training_ids=training_ids, store=store, text_cache=text_cache,
                text_index=text_index, teacher_result=teacher_result, device=device,
            )
        run_config = _run_identity(
            root=root, config_path=config_path, config=config, dev_config=dev_config,
            lock=lock, canonical=canonical, split_path=split_path, context=context,
            family=family, selection=selection, training_ids=training_ids,
            prediction_ids=prediction_ids, teacher_manifest=teacher_manifest,
        )
        result = _load_complete(paths, run_config["run_identity_sha256"], prediction_ids)
        if result is None:
            batch_key = "teacher" if family in {"teacher", "t_text"} else "b1" if family in {"b1", "d"} else "b0"
            dropout_key = "teacher" if family in {"teacher", "t_text"} else "student"
            result = fit_fixed_epochs(
                family=family, training_ids=training_ids, prediction_ids=prediction_ids,
                training_targets=training_targets, prediction_metadata=prediction_metadata,
                store=store, learning_rate=learning_rate, fixed_epochs=fixed_epochs,
                seed=int(config["seed"]), weight_decay=float(dev_config["weight_decay"]),
                batch_size=int(dev_config["batch_size"][batch_key]),
                gradient_clip=float(dev_config["gradient_clip"]),
                dropout=float(dev_config["dropout"][dropout_key]), device=device,
                run_directory=paths["run"], checkpoint_path=paths["checkpoint"],
                prediction_path=paths["prediction"], run_config=run_config,
                text_cache=text_cache if family in {"teacher", "t_text"} else None,
                text_index=text_index if family in {"teacher", "t_text"} else None,
                teacher_training_logits=teacher_logits, alpha=alpha,
                temperature=float(dev_config["distillation_temperature"]),
            )
        outputs[family] = result
    return {"context_id": context_id, "families": outputs}


def run_stage(config_path: Path, families: tuple[str, ...]) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    validate_kd_outer_lock(config_path, root)
    split = load_split(resolve(root, config["split_manifest"]))
    if len(split["contexts"]) != int(config["outer_contexts"]):
        raise ValueError("Frozen outer context count mismatch")
    results: dict[str, Any] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(config["parallel_workers"]), mp_context=mp.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(_run_context, str(config_path), index, families): str(context["context_id"])
            for index, context in enumerate(split["contexts"])
        }
        for future in concurrent.futures.as_completed(futures):
            expected = futures[future]
            result = future.result()
            if result["context_id"] != expected:
                raise AssertionError("Outer context worker identity mismatch")
            results[expected] = result["families"]
            print(f"completed {expected} stage={','.join(families)}", flush=True)
    if len(results) != int(config["outer_contexts"]):
        raise RuntimeError("Outer fixed-refit stage did not cover all contexts")
    state_path = resolve(root, config["outputs"]["output_root"]) / "fit_state.json"
    existing: dict[str, Any] = {}
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8")).get("results", {})
    for context_id, families_result in results.items():
        existing.setdefault(context_id, {}).update(families_result)
    save_json(
        state_path,
        {"schema": "goalexit-outer-fit-state-v1",
         "completed_families": sorted({family for context in existing.values() for family in context}),
         "results": dict(sorted(existing.items())), "outer_labels_accessed": False,
         "outer_scientific_metrics_computed": False},
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen GoalExit confirmatory outer refit")
    parser.add_argument("--config", type=Path, default=Path("configs/confirmatory_kd.yaml"))
    parser.add_argument("--stage", choices=["initial", "d"], required=True)
    args = parser.parse_args()
    run_stage(args.config, INITIAL_FAMILIES if args.stage == "initial" else ("d",))


if __name__ == "__main__":
    main()
