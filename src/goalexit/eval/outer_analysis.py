from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalexit.data.dataset import GoalExitStore, load_split, outer_context_positions
from goalexit.eval.metrics import evaluate_binary
from goalexit.eval.outer_guard import validate_kd_outer_lock
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


FAMILIES = ("b0", "b1", "teacher", "t_text", "d")


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def _reader_cluster_bootstrap(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    readers = []
    for reader_id, rows in frame.groupby("reader_id", sort=True):
        labels = rows["goal_index"].unique()
        if len(labels) != 1:
            raise ValueError(f"Reader {reader_id} spans multiple goals")
        target = rows["goal_index"].to_numpy(np.int64)
        readers.append(
            {
                "reader_id": str(reader_id),
                "goal": int(labels[0]),
                "trials": len(rows),
                "left_correct": int(((rows[left].to_numpy() >= 0.5) == target).sum()),
                "right_correct": int(((rows[right].to_numpy() >= 0.5) == target).sum()),
            }
        )
    table = pd.DataFrame(readers)
    if set(table["goal"]) != {0, 1}:
        raise ValueError("Reader bootstrap requires both goals")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    groups = {label: table.loc[table["goal"] == label].reset_index(drop=True) for label in (0, 1)}
    for replicate in range(replicates):
        difference = 0.0
        for label in (0, 1):
            group = groups[label]
            selected = rng.integers(0, len(group), size=len(group))
            sampled = group.iloc[selected]
            left_recall = sampled["left_correct"].sum() / sampled["trials"].sum()
            right_recall = sampled["right_correct"].sum() / sampled["trials"].sum()
            difference += 0.5 * (left_recall - right_recall)
        draws[replicate] = difference
    point = _balanced_accuracy(frame["goal_index"].to_numpy(), frame[left].to_numpy()) - _balanced_accuracy(
        frame["goal_index"].to_numpy(), frame[right].to_numpy()
    )
    return {
        "left": left,
        "right": right,
        "point_difference": float(point),
        "replicates": replicates,
        "seed": seed,
        "cluster": "reader, sampled with replacement within each goal",
        "ci_95_percentile": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_mean": float(draws.mean()),
        "probability_difference_le_zero": float(np.mean(draws <= 0)),
    }


def _crossed_bootstrap(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    reader_table = frame[["reader_id", "goal_index"]].drop_duplicates()
    if reader_table["reader_id"].duplicated().any():
        raise ValueError("Crossed bootstrap requires a constant goal per reader")
    reader_groups = {
        label: sorted(reader_table.loc[reader_table["goal_index"] == label, "reader_id"].astype(str))
        for label in (0, 1)
    }
    articles = sorted(frame["article_id"].astype(str).unique())
    reader_index = {value: index for index, value in enumerate(sorted(frame["reader_id"].astype(str).unique()))}
    article_index = {value: index for index, value in enumerate(articles)}
    row_readers = frame["reader_id"].astype(str).map(reader_index).to_numpy()
    row_articles = frame["article_id"].astype(str).map(article_index).to_numpy()
    target = frame["goal_index"].to_numpy(np.int64)
    left_correct = (frame[left].to_numpy() >= 0.5) == target
    right_correct = (frame[right].to_numpy() >= 0.5) == target
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        reader_weights = np.zeros(len(reader_index), dtype=np.int64)
        for label in (0, 1):
            group = reader_groups[label]
            sampled = rng.integers(0, len(group), size=len(group))
            counts = np.bincount(sampled, minlength=len(group))
            for position, reader_id in enumerate(group):
                reader_weights[reader_index[reader_id]] = counts[position]
        article_draw = rng.integers(0, len(articles), size=len(articles))
        article_weights = np.bincount(article_draw, minlength=len(articles))
        weights = reader_weights[row_readers] * article_weights[row_articles]
        difference = 0.0
        for label in (0, 1):
            mask = target == label
            denominator = weights[mask].sum()
            if denominator == 0:
                raise RuntimeError("Degenerate crossed bootstrap draw")
            difference += 0.5 * (
                np.sum(weights[mask] * left_correct[mask]) / denominator
                - np.sum(weights[mask] * right_correct[mask]) / denominator
            )
        draws[replicate] = difference
    return {
        "left": left,
        "right": right,
        "replicates": replicates,
        "seed": seed,
        "cluster": "independent goal-stratified reader and article multiplicities",
        "ci_95_percentile": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_mean": float(draws.mean()),
        "probability_difference_le_zero": float(np.mean(draws <= 0)),
    }


def audit_target_free_outputs(config_path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = project_root()
    resolved_config = resolve(root, config_path)
    config = load_yaml(resolved_config)
    lock = validate_kd_outer_lock(resolved_config, root)
    dev_config = load_yaml(resolve(root, config["development_config"]))
    split = load_split(resolve(root, config["split_manifest"]))
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    state_path = resolve(root, config["outputs"]["output_root"]) / "fit_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("Outer fit state is absent")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("outer_labels_accessed") is not False or state.get("outer_scientific_metrics_computed") is not False:
        raise PermissionError("Outer fit-state boundary flags are invalid")
    if state.get("completed_families") != sorted(FAMILIES):
        raise ValueError("Outer fit state does not contain all frozen families")
    frames: dict[str, list[pd.DataFrame]] = {family: [] for family in FAMILIES}
    entries: list[dict[str, Any]] = []
    expected_all = set(store.metadata["trial_id"].astype(str))
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        _, target_ids = outer_context_positions(store.metadata, split, context)
        family_results = state["results"].get(context_id, {})
        if set(family_results) != set(FAMILIES):
            raise ValueError(f"Incomplete family set for {context_id}")
        for family in FAMILIES:
            result = family_results[family]
            checkpoint_path = Path(result["checkpoint_path"])
            prediction_path = Path(result["prediction_path"])
            result_path = Path(result["result_path"])
            if result.get("status") != "complete" or result.get("family") != family:
                raise ValueError(f"Invalid result status for {context_id}/{family}")
            if int(result.get("prediction_rows", -1)) != len(target_ids):
                raise ValueError(f"Prediction row count mismatch for {context_id}/{family}")
            if result.get("outer_labels_accessed") is not False or result.get("outer_scientific_metrics_computed") is not False:
                raise PermissionError(f"Boundary flag failure for {context_id}/{family}")
            if sha256(checkpoint_path) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {context_id}/{family}")
            if sha256(prediction_path) != result["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch for {context_id}/{family}")
            if sha256(result_path) != result["result_sha256"]:
                raise ValueError(f"Result hash mismatch for {context_id}/{family}")
            run = result["run_config"]
            required = {
                "context_id": context_id,
                "model_family": family,
                "seed": int(config["seed"]),
                "lock_sha256": lock["lock_sha256"],
                "confirmatory_config_sha256": sha256(resolved_config),
                "outer_labels_accessed": False,
                "outer_scientific_metrics_computed": False,
            }
            for key, value in required.items():
                if run.get(key) != value:
                    raise ValueError(f"Run provenance mismatch for {context_id}/{family}: {key}")
            frame = pd.read_parquet(prediction_path)
            if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(target_ids):
                raise ValueError(f"Prediction membership mismatch for {context_id}/{family}")
            if set(frame["model_family"].astype(str)) != {family}:
                raise ValueError(f"Prediction family mismatch for {context_id}/{family}")
            expected_metadata = store.metadata.loc[frame["trial_id"].astype(str)]
            if not np.array_equal(
                frame["reader_id"].astype(str).to_numpy(),
                expected_metadata["reader_id"].astype(str).to_numpy(),
            ) or not np.array_equal(
                frame["article_id"].astype(str).to_numpy(),
                expected_metadata["article_id"].astype(str).to_numpy(),
            ):
                raise ValueError(f"Prediction metadata mismatch for {context_id}/{family}")
            prediction_provenance = {
                "context_id": context_id,
                "model_family": family,
                "seed": str(config["seed"]),
                "lock_sha256": lock["lock_sha256"],
                "run_identity_sha256": run["run_identity_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
            for key, value in prediction_provenance.items():
                if set(frame[key].astype(str)) != {str(value)}:
                    raise ValueError(f"Prediction provenance mismatch for {context_id}/{family}: {key}")
            if frame["outer_labels_accessed"].astype(bool).any() or frame["outer_scientific_metrics_computed"].astype(bool).any():
                raise PermissionError(f"Prediction boundary flag failure for {context_id}/{family}")
            if not np.isfinite(frame[["logit", "probability"]].to_numpy()).all():
                raise ValueError(f"Non-finite prediction for {context_id}/{family}")
            if not frame["probability"].between(0.0, 1.0, inclusive="both").all():
                raise ValueError(f"Out-of-range probability for {context_id}/{family}")
            preprocessor = result["preprocessor"]
            if (
                preprocessor.get("validation_or_outer_statistics_used") is not False
                or int(preprocessor.get("training_trial_count", -1)) != int(run["training_trials"])
            ):
                raise PermissionError(f"Preprocessor scope mismatch for {context_id}/{family}")
            frames[family].append(frame[["trial_id", "reader_id", "article_id", "probability"]].copy())
            entries.append(
                {"context_id": context_id, "family": family, "prediction_rows": len(frame),
                 "checkpoint_sha256": result["checkpoint_sha256"],
                 "prediction_sha256": result["prediction_sha256"],
                 "result_sha256": result["result_sha256"]}
            )
        b1_result = family_results["b1"]
        d_result = family_results["d"]
        if (
            b1_result["parameter_counts"] != d_result["parameter_counts"]
            or b1_result["run_config"]["architecture"]
            != d_result["run_config"]["architecture"]
        ):
            raise ValueError(f"B1/D deployment capacity mismatch for {context_id}")
    combined: dict[str, pd.DataFrame] = {}
    for family in FAMILIES:
        frame = pd.concat(frames[family], ignore_index=True)
        if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != expected_all:
            raise ValueError(f"Outer partition is not exact for {family}")
        combined[family] = frame
    audit = {
        "schema": "goalexit-outer-output-audit-v1", "status": "PASS",
        "lock_sha256": lock["lock_sha256"], "contexts": len(split["contexts"]),
        "families": list(FAMILIES), "expected_results": len(split["contexts"]) * len(FAMILIES),
        "validated_results": len(entries), "trials_per_family": len(expected_all),
        "prediction_partition_exact": True, "hashes_verified": True,
        "b1_d_deployment_capacity_identical": True,
        "outer_labels_accessed_before_complete_set_audit": False,
        "outer_scientific_metrics_computed_before_complete_set_audit": False,
        "entries": entries,
    }
    return audit, combined


def analyze(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    audit, family_frames = audit_target_free_outputs(config_path)

    # This is the sole outer-label materialization point, reached only after the complete-set audit.
    dev_config = load_yaml(resolve(root, config["development_config"]))
    labels = pd.read_parquet(
        resolve(root, dev_config["outputs"]["trial_table"]), columns=["trial_id", "goal_index"]
    )
    if labels["trial_id"].duplicated().any():
        raise ValueError("Duplicate outer label row")
    base = family_frames["b0"].rename(columns={"probability": "b0"})
    for family in FAMILIES[1:]:
        incoming = family_frames[family][["trial_id", "probability"]].rename(columns={"probability": family})
        base = base.merge(incoming, on="trial_id", how="inner", validate="one_to_one")
    frame = base.merge(labels, on="trial_id", how="inner", validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError("Outer label join is not exact")
    target = frame["goal_index"].to_numpy(np.int64)
    articles = frame["article_id"].astype(str).to_numpy()
    pooled = {
        family: evaluate_binary(target, frame[family].to_numpy(), articles).to_dict()
        for family in FAMILIES
    }
    for family in FAMILIES:
        prediction = frame[family].to_numpy() >= 0.5
        pooled[family]["recall_goal_0"] = float(np.mean(prediction[target == 0] == 0))
        pooled[family]["recall_goal_1"] = float(np.mean(prediction[target == 1] == 1))
    context_rows: list[dict[str, Any]] = []
    split = load_split(resolve(root, config["split_manifest"]))
    trial_to_context: dict[str, str] = {}
    dev_store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    for context in split["contexts"]:
        _, ids = outer_context_positions(dev_store.metadata, split, context)
        trial_to_context.update({trial_id: str(context["context_id"]) for trial_id in ids})
    frame["context_id"] = frame["trial_id"].map(trial_to_context)
    for context_id, rows in frame.groupby("context_id", sort=True):
        row: dict[str, Any] = {"context_id": str(context_id), "trials": len(rows)}
        local_target = rows["goal_index"].to_numpy(np.int64)
        for family in FAMILIES:
            row[family] = _balanced_accuracy(local_target, rows[family].to_numpy())
        context_rows.append(row)
    contrasts = {}
    for name, left, right in (
        ("d_minus_b1", "d", "b1"), ("d_minus_b0", "d", "b0"),
        ("teacher_minus_d", "teacher", "d"), ("teacher_minus_b1", "teacher", "b1"),
        ("t_text_minus_b1", "t_text", "b1"),
    ):
        point = pooled[left]["balanced_accuracy"] - pooled[right]["balanced_accuracy"]
        differences = np.asarray([row[left] - row[right] for row in context_rows])
        contrasts[name] = {
            "pooled_difference": float(point), "context_mean_difference": float(differences.mean()),
            "context_median_difference": float(np.median(differences)),
            "positive_contexts": int((differences > 0).sum()), "contexts": len(differences),
        }
    context_summary = {
        family: {
            "mean_balanced_accuracy": float(np.mean([row[family] for row in context_rows])),
            "median_balanced_accuracy": float(np.median([row[family] for row in context_rows])),
        }
        for family in FAMILIES
    }
    bootstrap = _reader_cluster_bootstrap(
        frame, "d", "b1", replicates=int(config["bootstrap"]["reader_cluster_replicates"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    crossed = _crossed_bootstrap(
        frame, "d", "b1", replicates=int(config["bootstrap"]["reader_article_replicates"]),
        seed=int(config["bootstrap"]["seed"]) + 1,
    )
    reader_scores = frame.assign(
        d_correct=(frame["d"] >= 0.5) == frame["goal_index"],
        b1_correct=(frame["b1"] >= 0.5) == frame["goal_index"],
    ).groupby(["reader_id", "goal_index"])[["d_correct", "b1_correct"]].mean().reset_index()
    reader_macro = {
        model: float(reader_scores.groupby("goal_index")[f"{model}_correct"].mean().mean())
        for model in ("d", "b1")
    }
    cpu = json.loads(resolve(root, config["cpu_benchmark"]).read_text(encoding="utf-8"))
    if cpu.get("status") != "complete" or cpu.get("outer_rows_touched") is not False:
        raise ValueError("Frozen CPU benchmark audit is invalid")
    d_resources = next(row for row in cpu["measurements"] if row["family"] == "d")
    resource_evidence = {
        "source_sha256": sha256(resolve(root, config["cpu_benchmark"])),
        "interpretation": cpu["interpretation"],
        "cpu_model": cpu["cpu_model"],
        "torch_threads": cpu["torch_threads"],
        "parameter_count": d_resources["parameter_count"],
        "serialized_state_bytes": d_resources["serialized_state_bytes"],
        "end_to_end_full": d_resources["end_to_end_trial_full"],
        "per_new_fixation_update": d_resources["per_new_fixation_update"],
    }
    viability_config = config["viability"]
    teacher_gain = contrasts["teacher_minus_b1"]["pooled_difference"]
    text_gain = contrasts["t_text_minus_b1"]["pooled_difference"]
    conditions = {
        "d_minus_b1_point_positive": contrasts["d_minus_b1"]["pooled_difference"] > 0,
        "d_minus_b1_reader_ci_lower_positive": bootstrap["ci_95_percentile"][0]
        > float(viability_config["primary_reader_cluster_ci_lower_must_exceed"]),
        "d_minus_b0_nonnegative": contrasts["d_minus_b0"]["pooled_difference"]
        >= float(viability_config["d_minus_b0_must_be_at_least"]),
        "d_within_0p010_of_teacher": -contrasts["teacher_minus_d"]["pooled_difference"]
        >= float(viability_config["d_minus_teacher_must_be_at_least"]),
        "text_gain_at_most_half_teacher_gain": text_gain
        <= float(viability_config["text_gain_fraction_of_teacher_gain_maximum"]) * teacher_gain,
        "complete_set_audit_pass": audit["status"] == "PASS",
    }
    verdict = "PASS" if all(conditions.values()) else "FAIL"
    output = {
        "schema": "goalexit-kd-confirmatory-outer-v1", "verdict": verdict,
        "interpretation": "positive KD manuscript authorized" if verdict == "PASS" else "positive KD manuscript stopped",
        "pooled_metrics": pooled, "contrasts": contrasts, "context_metrics": context_rows,
        "context_summary": context_summary, "resource_evidence": resource_evidence,
        "primary_reader_cluster_bootstrap": bootstrap,
        "reader_article_crossed_bootstrap_sensitivity": crossed,
        "reader_macro_accuracy": reader_macro,
        "viability_conditions": conditions, "outer_trials": len(frame),
        "outer_readers": int(frame["reader_id"].nunique()),
        "outer_articles": int(frame["article_id"].nunique()),
        "audit_sha256_pending_write": True,
    }
    audit_path = resolve(root, config["outputs"]["audit"])
    metrics_path = resolve(root, config["outputs"]["metrics"])
    save_json(audit_path, audit)
    output["audit_sha256"] = sha256(audit_path)
    output.pop("audit_sha256_pending_write")
    save_json(metrics_path, output)
    _write_report(resolve(root, config["outputs"]["report"]), output)
    return output


def _write_report(path: Path, result: dict[str, Any]) -> None:
    pooled = result["pooled_metrics"]
    contrasts = result["contrasts"]
    bootstrap = result["primary_reader_cluster_bootstrap"]
    crossed = result["reader_article_crossed_bootstrap_sensitivity"]
    lines = [
        "# GoalExit KD Confirmatory Report", "", f"**Verdict: {result['verdict']}**", "",
        "The positive KD manuscript is authorized." if result["verdict"] == "PASS" else "The preregistered viability rule stops the positive KD manuscript.",
        "", "## Pooled outer performance", "",
        "| Model | Balanced accuracy | AUROC | Macro F1 | NLL |", "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        row = pooled[family]
        lines.append(f"| {family} | {row['balanced_accuracy']:.4f} | {row['auroc']:.4f} | {row['macro_f1']:.4f} | {row['nll']:.4f} |")
    lines.extend([
        "", "## Frozen contrasts", "",
        f"- D − B1: {contrasts['d_minus_b1']['pooled_difference']:+.4f}",
        f"- D − B0: {contrasts['d_minus_b0']['pooled_difference']:+.4f}",
        f"- Teacher − D: {contrasts['teacher_minus_d']['pooled_difference']:+.4f}",
        f"- T-text − B1: {contrasts['t_text_minus_b1']['pooled_difference']:+.4f}",
        "", "## Primary uncertainty", "",
        f"Reader-cluster 95% CI for D − B1: [{bootstrap['ci_95_percentile'][0]:+.4f}, {bootstrap['ci_95_percentile'][1]:+.4f}].",
        f"Crossed reader/article sensitivity CI: [{crossed['ci_95_percentile'][0]:+.4f}, {crossed['ci_95_percentile'][1]:+.4f}].",
        "", "## Viability checks", "",
    ])
    for name, passed in result["viability_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot analysis of sealed GoalExit outer predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/confirmatory_kd.yaml"))
    args = parser.parse_args()
    result = analyze(args.config)
    print(json.dumps({"verdict": result["verdict"], "viability_conditions": result["viability_conditions"]}, indent=2))


if __name__ == "__main__":
    main()
