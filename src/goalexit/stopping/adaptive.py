from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

from goalexit.data.dataset import GoalExitStore, context_positions, load_allowed_targets, load_split
from goalexit.data.preprocessing import NumericPreprocessor
from goalexit.eval.metrics import evaluate_binary
from goalexit.features.extract import decision_lengths, student_prefix
from goalexit.features.schema import STUDENT_FEATURES
from goalexit.train.engine import load_checkpoint
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


@torch.no_grad()
def predict_trace(
    model: torch.nn.Module,
    preprocessor: NumericPreprocessor,
    store: GoalExitStore,
    ids: list[str],
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    transformed = store.transformed_student(ids, preprocessor)
    target = targets.set_index("trial_id")
    meta = metadata.set_index("trial_id")
    examples = [
        (position, length)
        for position, trial_id in enumerate(ids)
        for length in decision_lengths(len(transformed[trial_id]))
    ]
    examples.sort(key=lambda item: item[1])
    records: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        maximum = max(length for _, length in batch)
        values = np.zeros((len(batch), maximum, len(STUDENT_FEATURES)), dtype=np.float32)
        lengths = []
        for index, (trial_position, length) in enumerate(batch):
            trial_id = ids[trial_position]
            values[index, :length] = student_prefix(transformed[trial_id], length)
            lengths.append(length)
        logits = model(
            torch.from_numpy(values).to(device),
            torch.as_tensor(lengths, dtype=torch.long, device=device),
        ).cpu().numpy()
        for index, (trial_position, length) in enumerate(batch):
            trial_id = ids[trial_position]
            records.append(
                {
                    "trial_id": trial_id,
                    "reader_id": str(meta.loc[trial_id, "reader_id"]),
                    "article_id": str(meta.loc[trial_id, "article_id"]),
                    "target_goal_index": int(target.loc[trial_id, "goal_index"]),
                    "observed_fixations": int(length),
                    "full_fixation_count": int(meta.loc[trial_id, "full_fixation_count"]),
                    "logit": float(logits[index]),
                    "is_natural_end": bool(length == len(transformed[trial_id])),
                }
            )
    frame = pd.DataFrame(records).sort_values(
        ["trial_id", "observed_fixations"], kind="stable"
    )
    if not np.isfinite(frame["logit"]).all():
        raise ValueError("Non-finite anytime trace")
    return frame


def fit_temperature(full_rows: pd.DataFrame) -> dict[str, float]:
    logits = full_rows["logit"].to_numpy(np.float64)
    target = full_rows["target_goal_index"].to_numpy(np.float64)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -50, 50)))
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        return float(-np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability)))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature calibration failed")
    return {
        "temperature": float(np.exp(result.x)),
        "nll": float(result.fun),
        "fit_rows": len(full_rows),
    }


def stop_trace(trace: pd.DataFrame, threshold: float, temperature: float) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in trace.groupby("trial_id", sort=False):
        group = group.sort_values("observed_fixations", kind="stable").copy()
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(group["logit"].to_numpy(np.float64) / temperature, -50, 50))
        )
        prediction = (probability >= 0.5).astype(np.int8)
        confidence = np.maximum(probability, 1 - probability)
        selected = len(group) - 1
        for index in range(1, len(group)):
            if confidence[index] >= threshold and prediction[index] == prediction[index - 1]:
                selected = index
                break
        row = group.iloc[selected].copy()
        row["calibrated_probability_h"] = float(probability[selected])
        row["predicted_goal_index"] = int(prediction[selected])
        row["confidence"] = float(confidence[selected])
        row["threshold"] = float(threshold)
        row["stopped_early"] = bool(
            int(row["observed_fixations"]) < int(row["full_fixation_count"])
        )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _policy_metrics(frame: pd.DataFrame, ece_bins: int) -> dict[str, Any]:
    metric = evaluate_binary(
        frame["target_goal_index"].to_numpy(np.int64),
        frame["calibrated_probability_h"].to_numpy(np.float64),
        frame["article_id"].astype(str).to_numpy(),
        ece_bins=ece_bins,
    ).to_dict()
    fraction = frame["observed_fixations"] / frame["full_fixation_count"]
    metric.update(
        {
            "mean_observed_fixations": float(frame["observed_fixations"].mean()),
            "median_observed_fixations": float(frame["observed_fixations"].median()),
            "p90_observed_fixations": float(frame["observed_fixations"].quantile(0.9)),
            "mean_observed_fraction": float(fraction.mean()),
            "median_observed_fraction": float(fraction.median()),
            "mean_acquisition_reduction": float(1 - fraction.mean()),
            "median_acquisition_reduction": float(1 - fraction.median()),
        }
    )
    return metric


def _select_operating_point(
    evidence: list[dict[str, Any]], full_mean_ba: float, tolerance: float
) -> dict[str, Any] | None:
    eligible = [row for row in evidence if row["mean_balanced_accuracy"] >= full_mean_ba - tolerance]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row["mean_observed_fraction"],
            row["median_observed_fraction"],
            -row["mean_balanced_accuracy"],
            -row["threshold"],
        ),
    )


def run(config_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    if state.get("gate_a0", {}).get("status") != "PASS":
        raise PermissionError("Gate A0 did not authorize adaptive stopping")
    split = load_split(resolve(root, config["outputs"]["split_manifest"]))
    trial_path = resolve(root, config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    device = torch.device("cpu")
    context_records: list[dict[str, Any]] = []
    selected_frames: dict[str, list[pd.DataFrame]] = {"AP1": [], "AP2": [], "FULL": []}
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        selection = state["selections"][context_id]["a"]
        result_by_inner = {
            int(row["run_config"]["inner_holdout"]): row
            for row in selection["selected_results"]
        }
        traces: list[pd.DataFrame] = []
        trace_sources = []
        for inner in range(3):
            _, validation, outer = context_positions(store.metadata, split, context, inner)
            target = load_allowed_targets(trial_path, validation, outer)
            metadata = store.metadata.loc[validation].reset_index(drop=True)
            result = result_by_inner[inner]
            model, preprocessor, checkpoint = load_checkpoint(Path(result["checkpoint_path"]), device)
            trace = predict_trace(
                model,
                preprocessor,
                store,
                validation,
                target,
                metadata,
                batch_size=int(config["batch_size"]["b1"]),
                device=device,
            )
            trace["context_id"] = context_id
            trace["inner_holdout"] = inner
            trace["checkpoint_sha256"] = result["checkpoint_sha256"]
            trace["outer_test_evaluated"] = False
            path = (
                resolve(root, config["outputs"]["prediction_root"])
                / context_id
                / "a"
                / str(selection["selected_candidate_id"])
                / f"inner_{inner}_trace.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            trace.to_parquet(path, index=False, compression="zstd")
            trace_sources.append({"inner_holdout": inner, "path": str(path), "sha256": sha256(path)})
            traces.append(trace)
        combined = pd.concat(traces, ignore_index=True)
        full_rows = combined.loc[combined["is_natural_end"]].copy()
        calibration = fit_temperature(full_rows)
        temperature = float(calibration["temperature"])
        full_by_inner = []
        full_frames = []
        for inner, trace in enumerate(traces):
            full = trace.loc[trace["is_natural_end"]].copy()
            full["calibrated_probability_h"] = 1.0 / (
                1.0 + np.exp(-np.clip(full["logit"] / temperature, -50, 50))
            )
            full["predicted_goal_index"] = (full["calibrated_probability_h"] >= 0.5).astype(int)
            full["threshold"] = 1.0
            full["stopped_early"] = False
            full_by_inner.append(_policy_metrics(full, int(config["ece_bins"])))
            full_frames.append(full)
        full_mean_ba = float(np.mean([row["balanced_accuracy"] for row in full_by_inner]))
        grid: list[dict[str, Any]] = []
        stopped_by_threshold: dict[float, list[pd.DataFrame]] = {}
        for threshold in map(float, config["threshold_grid"]):
            frames = [stop_trace(trace, threshold, temperature) for trace in traces]
            stopped_by_threshold[threshold] = frames
            inner_metrics = [_policy_metrics(frame, int(config["ece_bins"])) for frame in frames]
            combined_stopped = pd.concat(frames, ignore_index=True)
            fraction = combined_stopped["observed_fixations"] / combined_stopped["full_fixation_count"]
            grid.append(
                {
                    "threshold": threshold,
                    "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in inner_metrics])),
                    "median_balanced_accuracy": float(np.median([row["balanced_accuracy"] for row in inner_metrics])),
                    "mean_observed_fixations": float(combined_stopped["observed_fixations"].mean()),
                    "median_observed_fixations": float(combined_stopped["observed_fixations"].median()),
                    "p90_observed_fixations": float(combined_stopped["observed_fixations"].quantile(0.9)),
                    "mean_observed_fraction": float(fraction.mean()),
                    "median_observed_fraction": float(fraction.median()),
                    "mean_acquisition_reduction": float(1 - fraction.mean()),
                    "median_acquisition_reduction": float(1 - fraction.median()),
                }
            )
        ap1 = _select_operating_point(grid, full_mean_ba, 0.010)
        ap2 = _select_operating_point(grid, full_mean_ba, 0.020)
        context_record = {
            "context_id": context_id,
            "temperature_calibration": calibration,
            "full_mean_balanced_accuracy": full_mean_ba,
            "grid": grid,
            "AP1": ap1,
            "AP2": ap2,
            "trace_sources": trace_sources,
        }
        context_records.append(context_record)
        selected_frames["FULL"].extend(full_frames)
        for name, selected in (("AP1", ap1), ("AP2", ap2)):
            if selected is None:
                continue
            for inner, frame in enumerate(stopped_by_threshold[float(selected["threshold"])]):
                frame = frame.copy()
                frame["operating_point"] = name
                selected_frames[name].append(frame)
    summary: dict[str, Any] = {}
    for name, frames in selected_frames.items():
        if not frames:
            summary[name] = None
            continue
        evaluation_metrics = [_policy_metrics(frame, int(config["ece_bins"])) for frame in frames]
        combined = pd.concat(frames, ignore_index=True)
        metrics = _policy_metrics(combined, int(config["ece_bins"]))
        metrics["mean_evaluation_balanced_accuracy"] = float(
            np.mean([row["balanced_accuracy"] for row in evaluation_metrics])
        )
        metrics["median_evaluation_balanced_accuracy"] = float(
            np.median([row["balanced_accuracy"] for row in evaluation_metrics])
        )
        summary[name] = metrics
    if summary["AP1"] is None:
        conditions = {"ap1_exists": False}
    else:
        full_ba = float(summary["FULL"]["mean_evaluation_balanced_accuracy"])
        ap1_ba = float(summary["AP1"]["mean_evaluation_balanced_accuracy"])
        full_combined = pd.concat(selected_frames["FULL"], ignore_index=True)
        ap1_combined = pd.concat(selected_frames["AP1"], ignore_index=True)
        subgroup_losses = {}
        for goal in (0, 1):
            full_goal = full_combined.loc[full_combined["target_goal_index"].eq(goal)]
            ap1_goal = ap1_combined.loc[ap1_combined["target_goal_index"].eq(goal)]
            full_recall = float((full_goal["predicted_goal_index"] == goal).mean())
            ap1_recall = float((ap1_goal["predicted_goal_index"] == goal).mean())
            subgroup_losses[str(goal)] = full_recall - ap1_recall
        positive_contexts = sum(
            record["AP1"] is not None and record["AP1"]["mean_acquisition_reduction"] > 0
            for record in context_records
        )
        conditions = {
            "ap1_exists": True,
            "ap1_ba_within_0p010": ap1_ba >= full_ba - 0.010,
            "mean_acquisition_reduction_at_least_0p30": summary["AP1"]["mean_acquisition_reduction"] >= 0.30,
            "median_acquisition_reduction_at_least_0p25": summary["AP1"]["median_acquisition_reduction"] >= 0.25,
            "positive_reduction_at_least_20_contexts": positive_contexts >= 20,
            "no_goal_subgroup_loses_more_than_0p03": max(subgroup_losses.values()) <= 0.03,
        }
    gate = {
        "gate": "E",
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
    }
    output = {
        "schema": "goalexit-adaptive-stopping-v1",
        "status": "complete",
        "thresholds": list(map(float, config["threshold_grid"])),
        "stability_rule": "two consecutive identical predicted classes",
        "context_records": context_records,
        "summary": summary,
        "gate_e": gate,
        "outer_test_evaluated": False,
    }
    path = resolve(root, config["outputs"]["output_root"]) / "adaptive_stopping.json"
    save_json(path, output)
    output["path"] = str(path)
    output["sha256"] = sha256(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    root = project_root()
    config = load_yaml(resolve(root, args.config))
    state_path = resolve(root, config["outputs"]["output_root"]) / "selection_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("Development selection state is absent")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    adaptive = run(args.config, state)
    state["adaptive"] = adaptive
    state["gate_e"] = adaptive["gate_e"]
    save_json(state_path, state)
    print(json.dumps(adaptive["gate_e"], indent=2))


if __name__ == "__main__":
    main()
