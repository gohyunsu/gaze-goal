from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from goalexit.data.dataset import GoalExitStore, context_positions, load_allowed_targets, load_split
from goalexit.train.engine import train_candidate
from goalexit.utils import (
    canonical_json_sha256,
    git_commit,
    load_yaml,
    project_root,
    resolve,
    save_json,
    sha256,
)


def _permuted_targets(
    targets,
    metadata,
    ids: list[str],
    rng: np.random.Generator,
):
    frame = targets.set_index("trial_id").loc[ids].reset_index()
    readers = metadata.loc[ids, "reader_id"].astype(str)
    reader_goal = (
        frame.assign(reader_id=readers.to_numpy())[["reader_id", "goal_index"]]
        .drop_duplicates()
        .sort_values("reader_id")
    )
    if reader_goal.groupby("reader_id")["goal_index"].nunique().ne(1).any():
        raise ValueError("Permutation unit is not reader-constant")
    shuffled = reader_goal["goal_index"].to_numpy(copy=True)
    rng.shuffle(shuffled)
    mapping = {
        reader_id: int(goal_index)
        for reader_id, goal_index in zip(
            reader_goal["reader_id"].astype(str), shuffled
        )
    }
    frame["goal_index"] = readers.map(mapping).to_numpy(np.int8)
    return frame, mapping


def run(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, config_path)
    config = load_yaml(config_path)
    state_path = resolve(root, config["outputs"]["output_root"]) / "selection_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    split_path = resolve(root, config["outputs"]["split_manifest"])
    split = load_split(split_path)
    canonical = json.loads(
        resolve(root, config["outputs"]["canonical_manifest"]).read_text(encoding="utf-8")
    )
    trial_path = resolve(root, config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    context_map = {context["context_id"]: context for context in split["contexts"]}
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise RuntimeError("Permutation sanity requires the frozen clean implementation")
    rows: list[dict[str, Any]] = []
    base_seed = int(config["permutation_sanity"]["seed"])
    for position, (context_id, inner_value) in enumerate(config["permutation_sanity"]["evaluations"]):
        inner = int(inner_value)
        context = context_map[str(context_id)]
        training, validation, outer = context_positions(store.metadata, split, context, inner)
        all_ids = training + validation
        all_targets = load_allowed_targets(trial_path, all_ids, outer)
        rng = np.random.default_rng(base_seed + position)
        permuted, mapping = _permuted_targets(all_targets, store.metadata, all_ids, rng)
        train_target = permuted.set_index("trial_id").loc[training].reset_index()
        validation_target = permuted.set_index("trial_id").loc[validation].reset_index()
        if set(validation_target["goal_index"]) != {0, 1}:
            raise ValueError("Fixed reader permutation produced a single-class validation split")
        selected = state["selections"][context_id]["b1"]
        learning_rate = float(selected["selected_learning_rate"])
        candidate_id = f"reader_permutation_{context_id}_i{inner}"
        run_config = {
            "schema": "goalexit-reader-permutation-run-v1",
            "git_commit": commit,
            "config_sha256": sha256(config_path),
            "canonical_data_sha256": canonical["canonical_data_sha256"],
            "split_manifest_sha256": sha256(split_path),
            "context_id": context_id,
            "inner_holdout": inner,
            "model_family": "b1",
            "candidate_id": candidate_id,
            "seed": base_seed + position,
            "learning_rate": learning_rate,
            "dropout": float(config["dropout"]["student"]),
            "reader_level_permutation": True,
            "permuted_reader_mapping_sha256": canonical_json_sha256(mapping),
            "run_identity_sha256": canonical_json_sha256(
                {
                    "context": context_id,
                    "inner": inner,
                    "seed": base_seed + position,
                    "learning_rate": learning_rate,
                    "mapping": mapping,
                    "commit": commit,
                }
            ),
        }
        output = resolve(root, config["outputs"]["output_root"]) / "sanity" / candidate_id
        storage = root / "research_storage/permutation" / candidate_id
        result = train_candidate(
            family="b1",
            training_ids=training,
            validation_ids=validation,
            training_targets=train_target,
            validation_targets=validation_target,
            validation_metadata=store.metadata.loc[validation].reset_index(drop=True),
            store=store,
            learning_rate=learning_rate,
            seed=base_seed + position,
            weight_decay=float(config["weight_decay"]),
            maximum_epochs=int(config["maximum_epochs"]),
            patience=int(config["patience"]),
            minimum_delta=float(config["minimum_delta"]),
            batch_size=int(config["batch_size"]["b1"]),
            gradient_clip=float(config["gradient_clip"]),
            dropout=float(config["dropout"]["student"]),
            ece_bins=int(config["ece_bins"]),
            device=torch.device("cpu"),
            run_directory=output,
            checkpoint_path=storage / "checkpoint.pt",
            prediction_path=storage / "predictions.parquet",
            run_config=run_config,
        )
        observed_result = next(
            value
            for value in selected["selected_results"]
            if int(value["run_config"]["inner_holdout"]) == inner
        )
        rows.append(
            {
                "context_id": context_id,
                "inner_holdout": inner,
                "permutation_seed": base_seed + position,
                "permuted_reader_mapping_sha256": canonical_json_sha256(mapping),
                "permuted_balanced_accuracy": result["metrics"]["FULL"]["balanced_accuracy"],
                "observed_selected_b1_balanced_accuracy": observed_result["metrics"]["FULL"]["balanced_accuracy"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "prediction_sha256": result["prediction_sha256"],
                "outer_rows_touched": False,
            }
        )
    mean_permuted = float(np.mean([row["permuted_balanced_accuracy"] for row in rows]))
    mean_observed = float(np.mean([row["observed_selected_b1_balanced_accuracy"] for row in rows]))
    maximum = float(config["permutation_sanity"]["chance_tolerance_max_mean_ba"])
    manifest = {
        "schema": "goalexit-reader-label-permutation-v1",
        "status": "pass" if mean_permuted <= maximum else "fail",
        "permutation_unit": "reader",
        "fixed_evaluations": config["permutation_sanity"]["evaluations"],
        "criterion": f"mean permuted BA <= {maximum:.2f}",
        "mean_permuted_balanced_accuracy": mean_permuted,
        "mean_observed_selected_b1_balanced_accuracy": mean_observed,
        "observed_minus_permuted": mean_observed - mean_permuted,
        "evaluations": rows,
        "outer_rows_touched": False,
    }
    save_json(resolve(root, config["outputs"]["permutation_sanity"]), manifest)
    print(
        f"status={manifest['status']} observed={mean_observed:.6f} permuted={mean_permuted:.6f}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
