from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalexit.data.dataset import GoalExitStore, load_split, outer_context_positions
from goalexit.eval.metrics import evaluate_binary
from goalexit.eval.postconfirmatory_ablation import ALL_VARIANTS, _paths, run_stage
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


EXTENSION_TEACHERS = ("gaze_teacher_s29", "gaze_teacher_s47")
EXTENSION_STUDENTS = (
    "label_smoothing_s29",
    "label_smoothing_s47",
    "gaze_kd_s29",
    "gaze_kd_s47",
)
EXTENSION_VARIANTS = (*EXTENSION_TEACHERS, *EXTENSION_STUDENTS)
SEEDS = (13, 29, 47)


def _variant_map(seed: int) -> dict[str, str]:
    suffix = "" if seed == 13 else f"_s{seed}"
    return {
        "gru": f"gru_s{seed}",
        "label_smoothing": "label_smoothing" if seed == 13 else f"label_smoothing_s{seed}",
        "gaze_teacher": "gaze_teacher" if seed == 13 else f"gaze_teacher_s{seed}",
        "gaze_kd": "gaze_kd" if seed == 13 else f"gaze_kd_s{seed}",
        "text_gaze_teacher": "text_gaze_teacher" + suffix,
        "text_gaze_kd": f"text_gaze_kd_s{seed}",
    }


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def _mean_reader_bootstrap(
    frame: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    readers: list[dict[str, Any]] = []
    for reader_id, rows in frame.groupby("reader_id", sort=True):
        labels = rows["goal_index"].unique()
        if len(labels) != 1:
            raise ValueError(f"Reader {reader_id} spans multiple goals")
        target = rows["goal_index"].to_numpy(np.int64)
        row: dict[str, Any] = {
            "reader_id": str(reader_id),
            "goal": int(labels[0]),
            "trials": len(rows),
        }
        for index, (left, right) in enumerate(pairs):
            row[f"left_{index}"] = int(((rows[left].to_numpy() >= 0.5) == target).sum())
            row[f"right_{index}"] = int(((rows[right].to_numpy() >= 0.5) == target).sum())
        readers.append(row)
    table = pd.DataFrame(readers)
    groups = {
        label: table.loc[table["goal"] == label].reset_index(drop=True)
        for label in (0, 1)
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        per_pair = np.zeros(len(pairs), dtype=np.float64)
        for label in (0, 1):
            group = groups[label]
            sampled = group.iloc[rng.integers(0, len(group), size=len(group))]
            denominator = sampled["trials"].sum()
            for index in range(len(pairs)):
                per_pair[index] += 0.5 * (
                    sampled[f"left_{index}"].sum() / denominator
                    - sampled[f"right_{index}"].sum() / denominator
                )
        draws[replicate] = per_pair.mean()
    target = frame["goal_index"].to_numpy(np.int64)
    points = [
        _balanced_accuracy(target, frame[left].to_numpy())
        - _balanced_accuracy(target, frame[right].to_numpy())
        for left, right in pairs
    ]
    return {
        "point_mean_difference": float(np.mean(points)),
        "seed_differences": [float(value) for value in points],
        "replicates": replicates,
        "seed": seed,
        "cluster": "reader, sampled within goal; seed-specific BA differences averaged per draw",
        "ci_95_percentile": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_mean": float(draws.mean()),
        "probability_difference_le_zero": float(np.mean(draws <= 0)),
    }


def _mean_crossed_bootstrap(
    frame: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    reader_table = frame[["reader_id", "goal_index"]].drop_duplicates()
    if reader_table["reader_id"].duplicated().any():
        raise ValueError("Crossed bootstrap requires one goal per reader")
    reader_groups = {
        label: sorted(
            reader_table.loc[reader_table["goal_index"] == label, "reader_id"].astype(str)
        )
        for label in (0, 1)
    }
    readers = sorted(frame["reader_id"].astype(str).unique())
    articles = sorted(frame["article_id"].astype(str).unique())
    reader_index = {value: index for index, value in enumerate(readers)}
    article_index = {value: index for index, value in enumerate(articles)}
    row_readers = frame["reader_id"].astype(str).map(reader_index).to_numpy()
    row_articles = frame["article_id"].astype(str).map(article_index).to_numpy()
    target = frame["goal_index"].to_numpy(np.int64)
    correct = [
        (
            (frame[left].to_numpy() >= 0.5) == target,
            (frame[right].to_numpy() >= 0.5) == target,
        )
        for left, right in pairs
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        reader_weights = np.zeros(len(readers), dtype=np.int64)
        for label in (0, 1):
            group = reader_groups[label]
            counts = np.bincount(
                rng.integers(0, len(group), size=len(group)), minlength=len(group)
            )
            for position, reader_id in enumerate(group):
                reader_weights[reader_index[reader_id]] = counts[position]
        article_weights = np.bincount(
            rng.integers(0, len(articles), size=len(articles)), minlength=len(articles)
        )
        weights = reader_weights[row_readers] * article_weights[row_articles]
        per_pair = np.zeros(len(pairs), dtype=np.float64)
        for label in (0, 1):
            mask = target == label
            denominator = weights[mask].sum()
            if denominator == 0:
                raise RuntimeError("Degenerate crossed bootstrap draw")
            for index, (left_correct, right_correct) in enumerate(correct):
                per_pair[index] += 0.5 * (
                    np.sum(weights[mask] * left_correct[mask]) / denominator
                    - np.sum(weights[mask] * right_correct[mask]) / denominator
                )
        draws[replicate] = per_pair.mean()
    return {
        "replicates": replicates,
        "seed": seed,
        "cluster": (
            "independent reader and article multiplicities; "
            "seed-specific BA differences averaged per draw"
        ),
        "ci_95_percentile": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_mean": float(draws.mean()),
        "probability_difference_le_zero": float(np.mean(draws <= 0)),
    }


def _load_prediction(
    root: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    context_id: str,
    variant: str,
    expected_seed: int,
    target_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = state["results"][context_id][variant]
    paths = _paths(root, config, context_id, variant)
    result_path = paths["run"] / "result.json"
    if result.get("status") != "complete":
        raise ValueError(f"Incomplete result: {context_id}/{variant}")
    run = result.get("run_config", {})
    expected_run = {
        "context_id": context_id,
        "model_family": variant,
        "seed": expected_seed,
        "training_device": "cuda:0",
        "torch_version": "2.6.0+cu118",
        "cuda_version": "11.8",
        "gpu_name": "NVIDIA L40S",
        "outer_labels_accessed": False,
        "outer_scientific_metrics_computed": False,
    }
    for key, value in expected_run.items():
        if run.get(key) != value:
            raise ValueError(f"Run provenance mismatch: {context_id}/{variant}/{key}")
    if sha256(paths["checkpoint"]) != result["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {context_id}/{variant}")
    if sha256(paths["prediction"]) != result["prediction_sha256"]:
        raise ValueError(f"Prediction hash mismatch: {context_id}/{variant}")
    if sha256(result_path) != result["result_sha256"]:
        raise ValueError(f"Result hash mismatch: {context_id}/{variant}")
    frame = pd.read_parquet(paths["prediction"], columns=["trial_id", "probability"])
    if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
        target_ids
    ):
        raise ValueError(f"Prediction membership mismatch: {context_id}/{variant}")
    return frame, {
        "context_id": context_id,
        "variant": variant,
        "prediction_rows": len(frame),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "prediction_sha256": result["prediction_sha256"],
        "result_sha256": result["result_sha256"],
    }


def analyze(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    base_config = load_yaml(resolve(root, config["base_ablation_config"]))
    dev_config = load_yaml(resolve(root, config["development_config"]))
    split = load_split(resolve(root, config["split_manifest"]))
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    base_state = json.loads(
        (resolve(root, base_config["outputs"]["output_root"]) / "fit_state.json").read_text(
            encoding="utf-8"
        )
    )
    extension_state = json.loads(
        (resolve(root, config["outputs"]["output_root"]) / "fit_state.json").read_text(
            encoding="utf-8"
        )
    )
    if extension_state.get("completed_variants") != sorted(EXTENSION_VARIANTS):
        raise ValueError("Seed-extension fit state is incomplete")
    if base_state.get("completed_variants") != sorted(ALL_VARIANTS):
        raise ValueError("Base ablation fit state is incomplete")

    maps = {seed: _variant_map(seed) for seed in SEEDS}
    base_variants = sorted(
        {
            variant
            for mapping in maps.values()
            for variant in mapping.values()
            if variant not in EXTENSION_VARIANTS
        }
    )
    variant_seeds = {
        variant: seed
        for seed, mapping in maps.items()
        for variant in mapping.values()
    }
    pieces: dict[str, list[pd.DataFrame]] = {
        variant: [] for variant in (*base_variants, *EXTENSION_VARIANTS)
    }
    entries: list[dict[str, Any]] = []
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        _, target_ids = outer_context_positions(store.metadata, split, context)
        for variant in base_variants:
            frame, entry = _load_prediction(
                root,
                base_config,
                base_state,
                context_id,
                variant,
                variant_seeds[variant],
                target_ids,
            )
            pieces[variant].append(frame)
            entry["source"] = "base_ablation"
            entries.append(entry)
        for variant in EXTENSION_VARIANTS:
            frame, entry = _load_prediction(
                root,
                config,
                extension_state,
                context_id,
                variant,
                variant_seeds[variant],
                target_ids,
            )
            pieces[variant].append(frame)
            entry["source"] = "seed_extension"
            entries.append(entry)

        def result_for(variant: str) -> dict[str, Any]:
            state = extension_state if variant in EXTENSION_VARIANTS else base_state
            return state["results"][context_id][variant]

        for seed, mapping in maps.items():
            students = [
                result_for(mapping[name])
                for name in ("gru", "label_smoothing", "gaze_kd", "text_gaze_kd")
            ]
            if any(
                row["parameter_counts"] != students[0]["parameter_counts"]
                for row in students[1:]
            ):
                raise ValueError(f"Student capacity mismatch: {context_id}/seed-{seed}")
            architectures = [row["run_config"]["architecture"] for row in students]
            if any(value != architectures[0] for value in architectures[1:]):
                raise ValueError(f"Student architecture mismatch: {context_id}/seed-{seed}")
            gaze_kd = students[2]["run_config"]
            text_gaze_kd = students[3]["run_config"]
            for key in ("learning_rate", "alpha", "temperature", "fixed_epochs"):
                if gaze_kd.get(key) != text_gaze_kd.get(key):
                    raise ValueError(
                        f"KD setting mismatch: {context_id}/seed-{seed}/{key}"
                    )

    expected_ids = set(store.metadata["trial_id"].astype(str))
    labels = pd.read_parquet(
        resolve(root, dev_config["outputs"]["trial_table"]),
        columns=["trial_id", "goal_index", "reader_id", "article_id"],
    )
    frame = labels.copy()
    for variant, variant_pieces in pieces.items():
        combined = pd.concat(variant_pieces, ignore_index=True)
        if combined["trial_id"].duplicated().any() or set(
            combined["trial_id"].astype(str)
        ) != expected_ids:
            raise ValueError(f"Combined partition mismatch: {variant}")
        frame = frame.merge(
            combined.rename(columns={"probability": variant}),
            on="trial_id",
            how="inner",
            validate="one_to_one",
        )
    if len(frame) != len(labels):
        raise ValueError("Seed-extension label join is not exact")

    target = frame["goal_index"].to_numpy(np.int64)
    articles = frame["article_id"].astype(str).to_numpy()
    per_seed: dict[str, Any] = {}
    contrast_defs = {
        "label_smoothing_minus_gru": ("label_smoothing", "gru"),
        "gaze_kd_minus_gru": ("gaze_kd", "gru"),
        "text_gaze_kd_minus_gru": ("text_gaze_kd", "gru"),
        "text_gaze_kd_minus_gaze_kd": ("text_gaze_kd", "gaze_kd"),
        "text_gaze_teacher_minus_gaze_teacher": (
            "text_gaze_teacher",
            "gaze_teacher",
        ),
    }
    for seed, mapping in maps.items():
        metrics = {
            name: evaluate_binary(target, frame[variant].to_numpy(), articles).to_dict()
            for name, variant in mapping.items()
        }
        contrasts = {
            name: float(
                metrics[left]["balanced_accuracy"] - metrics[right]["balanced_accuracy"]
            )
            for name, (left, right) in contrast_defs.items()
        }
        per_seed[str(seed)] = {"metrics": metrics, "contrasts": contrasts}

    aggregate: dict[str, Any] = {}
    for offset, (name, (left_name, right_name)) in enumerate(contrast_defs.items()):
        pairs = tuple(
            (maps[seed][left_name], maps[seed][right_name]) for seed in SEEDS
        )
        reader = _mean_reader_bootstrap(
            frame,
            pairs,
            replicates=int(config["bootstrap"]["reader_cluster_replicates"]),
            seed=int(config["bootstrap"]["seed"]) + offset,
        )
        crossed = _mean_crossed_bootstrap(
            frame,
            pairs,
            replicates=int(config["bootstrap"]["reader_article_replicates"]),
            seed=int(config["bootstrap"]["seed"]) + 100 + offset,
        )
        deltas = np.asarray(reader["seed_differences"])
        aggregate[name] = {
            "mean_pooled_difference": float(deltas.mean()),
            "minimum_seed_difference": float(deltas.min()),
            "maximum_seed_difference": float(deltas.max()),
            "positive_seeds": int((deltas > 0).sum()),
            "seeds": len(deltas),
            "reader_cluster_bootstrap": reader,
            "reader_article_crossed_bootstrap": crossed,
        }

    mean_balanced_accuracy = {
        condition: float(
            np.mean(
                [
                    per_seed[str(seed)]["metrics"][condition]["balanced_accuracy"]
                    for seed in SEEDS
                ]
            )
        )
        for condition in maps[13]
    }
    extension_entries = [row for row in entries if row["source"] == "seed_extension"]
    audit = {
        "schema": "goalexit-postconfirmatory-seed-extension-audit-v1",
        "status": "PASS",
        "postconfirmatory": True,
        "contexts": len(split["contexts"]),
        "extension_variants": list(EXTENSION_VARIANTS),
        "validated_extension_results": len(extension_entries),
        "expected_extension_results": len(split["contexts"]) * len(EXTENSION_VARIANTS),
        "base_results_revalidated": len(entries) - len(extension_entries),
        "trials_per_variant": len(expected_ids),
        "hashes_verified": True,
        "prediction_partitions_exact": True,
        "student_capacity_matched_within_seed": True,
        "kd_settings_matched_within_seed": True,
        "uniform_software_and_gpu_model": True,
        "entries": entries,
    }
    audit_path = resolve(root, config["outputs"]["audit"])
    save_json(audit_path, audit)
    output = {
        "schema": "goalexit-postconfirmatory-seed-extension-results-v1",
        "status": "PASS",
        "postconfirmatory": True,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "aggregate_contrasts": aggregate,
        "outer_trials": len(frame),
        "audit_sha256": sha256(audit_path),
    }
    metrics_path = resolve(root, config["outputs"]["metrics"])
    save_json(metrics_path, output)
    _write_report(resolve(root, config["outputs"]["report"]), output)
    return output


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Post-confirmatory seed-balanced extension report",
        "",
        "**Status: PASS**",
        "",
        "All four student conditions and both teacher modalities use the same three seeds.",
        "",
        "## Seed-specific balanced accuracy and matched differences",
        "",
        (
            "| Seed | GRU | Label smoothing | Gaze-only KD | Text--gaze KD | "
            "LS $\\Delta$ | Gaze KD $\\Delta$ | Text--gaze KD $\\Delta$ |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in result["seeds"]:
        row = result["per_seed"][str(seed)]
        metrics = row["metrics"]
        contrasts = row["contrasts"]
        lines.append(
            f"| {seed} | {metrics['gru']['balanced_accuracy']:.4f} | "
            f"{metrics['label_smoothing']['balanced_accuracy']:.4f} | "
            f"{metrics['gaze_kd']['balanced_accuracy']:.4f} | "
            f"{metrics['text_gaze_kd']['balanced_accuracy']:.4f} | "
            f"{contrasts['label_smoothing_minus_gru']:+.4f} | "
            f"{contrasts['gaze_kd_minus_gru']:+.4f} | "
            f"{contrasts['text_gaze_kd_minus_gru']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed mean contrasts",
            "",
            (
                "| Contrast | Mean $\\Delta$BA | Seed range | Reader 95% CI | "
                "Crossed 95% CI | Positive seeds |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in result["aggregate_contrasts"].items():
        reader = row["reader_cluster_bootstrap"]["ci_95_percentile"]
        crossed = row["reader_article_crossed_bootstrap"]["ci_95_percentile"]
        lines.append(
            f"| {name} | {row['mean_pooled_difference']:+.4f} | "
            f"[{row['minimum_seed_difference']:+.4f}, {row['maximum_seed_difference']:+.4f}] | "
            f"[{reader[0]:+.4f}, {reader[1]:+.4f}] | "
            f"[{crossed[0]:+.4f}, {crossed[1]:+.4f}] | "
            f"{row['positive_seeds']}/{row['seeds']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed-balanced extension of post-confirmatory GoalExit ablations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/postconfirmatory_seed_extension.yaml"),
    )
    parser.add_argument(
        "--stage", choices=("teachers", "students", "analyze"), required=True
    )
    args = parser.parse_args()
    if args.stage == "teachers":
        run_stage(args.config, EXTENSION_TEACHERS)
    elif args.stage == "students":
        run_stage(args.config, EXTENSION_STUDENTS)
    else:
        analyze(args.config)


if __name__ == "__main__":
    main()
