from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalexit.data.dataset import GoalExitStore, load_split, outer_context_positions
from goalexit.eval.postconfirmatory_ablation import _paths, run_stage
from goalexit.eval.postconfirmatory_seed_extension import (
    _mean_crossed_bootstrap,
    _mean_reader_bootstrap,
)
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


SEEDS = (13, 29, 47)
VARIANTS = tuple(f"shuffled_text_gaze_kd_s{seed}" for seed in SEEDS)


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def analyze(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    dev_config = load_yaml(resolve(root, config["development_config"]))
    base_config = load_yaml(resolve(root, config["base_ablation_config"]))
    split = load_split(resolve(root, config["split_manifest"]))
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    state_path = resolve(root, config["outputs"]["output_root"]) / "fit_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("completed_variants") != sorted(VARIANTS):
        raise ValueError("Shuffled-KD fit state is incomplete")
    labels = pd.read_parquet(
        resolve(root, dev_config["outputs"]["trial_table"]),
        columns=["trial_id", "goal_index", "reader_id", "article_id"],
    )
    frames: dict[str, list[pd.DataFrame]] = {}
    audit: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name in ("gru", "original", "shuffled"):
            frames[f"{name}_{seed}"] = []
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        _, target_ids = outer_context_positions(store.metadata, split, context)
        for seed in SEEDS:
            sources = {
                f"gru_{seed}": resolve(root, base_config["outputs"]["prediction_root"])
                / context_id
                / f"gru_s{seed}.parquet",
                f"original_{seed}": resolve(
                    root, base_config["outputs"]["prediction_root"]
                )
                / context_id
                / f"text_gaze_kd_s{seed}.parquet",
                f"shuffled_{seed}": _paths(
                    root, config, context_id, f"shuffled_text_gaze_kd_s{seed}"
                )["prediction"],
            }
            for name, path in sources.items():
                frame = pd.read_parquet(path, columns=["trial_id", "probability"])
                if frame["trial_id"].duplicated().any() or set(
                    frame["trial_id"].astype(str)
                ) != set(target_ids):
                    raise ValueError(f"Prediction membership mismatch: {context_id}/{name}")
                frames[name].append(frame.rename(columns={"probability": name}))
                audit.append(
                    {
                        "context_id": context_id,
                        "variant": name,
                        "prediction_sha256": sha256(path),
                    }
                )
    frame = labels.copy()
    for name, pieces in frames.items():
        incoming = pd.concat(pieces, ignore_index=True)
        if incoming["trial_id"].duplicated().any() or len(incoming) != len(labels):
            raise ValueError(f"Prediction partition is not exact: {name}")
        frame = frame.merge(incoming, on="trial_id", validate="one_to_one")
    target = frame["goal_index"].to_numpy(np.int64)
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        values = {
            name: _balanced_accuracy(target, frame[f"{name}_{seed}"].to_numpy())
            for name in ("gru", "original", "shuffled")
        }
        per_seed.append(
            {
                "seed": seed,
                **values,
                "original_minus_gru": values["original"] - values["gru"],
                "shuffled_minus_gru": values["shuffled"] - values["gru"],
                "original_minus_shuffled": values["original"] - values["shuffled"],
            }
        )
    pairs = {
        "original_minus_gru": tuple(
            (f"original_{seed}", f"gru_{seed}") for seed in SEEDS
        ),
        "shuffled_minus_gru": tuple(
            (f"shuffled_{seed}", f"gru_{seed}") for seed in SEEDS
        ),
        "original_minus_shuffled": tuple(
            (f"original_{seed}", f"shuffled_{seed}") for seed in SEEDS
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
    result = {
        "schema": "goalexit-postfeedback-shuffled-kd-v1",
        "status": "complete",
        "post_hoc": True,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "contrasts": contrasts,
        "audit": audit,
    }
    save_json(resolve(root, config["outputs"]["metrics"]), result)
    resolve(root, config["outputs"]["report"]).write_text(
        _report(result), encoding="utf-8"
    )
    return result


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Post hoc shuffled-target KD results",
        "",
        "**Status:** complete; post hoc.",
        "",
        "| Seed | GRU | Original KD | Shuffled KD | Original - shuffled |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['gru']:.4f} | {row['original']:.4f} | {row['shuffled']:.4f} | {row['original_minus_shuffled']:+.4f} |"
        )
    lines.extend(["", "| Contrast | Mean delta BA | Reader 95% CI |", "|---|---:|---:|"])
    for name, row in result["contrasts"].items():
        summary = row["reader_cluster_bootstrap"]
        low, high = summary["ci_95_percentile"]
        lines.append(
            f"| {name.replace('_', ' ')} | {summary['point_mean_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/postfeedback_shuffled_kd.yaml")
    )
    parser.add_argument("--stage", choices=("students", "analyze"), required=True)
    args = parser.parse_args()
    if args.stage == "students":
        run_stage(args.config, VARIANTS)
    else:
        result = analyze(args.config)
        print(
            "original_minus_shuffled="
            f"{result['contrasts']['original_minus_shuffled']['reader_cluster_bootstrap']['point_mean_difference']:.6f}"
        )


if __name__ == "__main__":
    main()
