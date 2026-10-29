from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from goalexit.audit.phase0 import reader_parameter_scan
from goalexit.data.dataset import (
    GoalExitStore,
    context_positions,
    load_allowed_targets,
    load_split,
)
from goalexit.eval.metrics import aggregate_metrics, evaluate_binary
from goalexit.eval.outer_guard import OuterEvaluationSealedError, require_outer_authorization
from goalexit.train.run_development import (
    _candidate_grid,
    _candidate_id,
    _paths,
    _select_candidate,
    evaluate_gate_a0,
    evaluate_gate_d,
    evaluate_gate_t,
)
from goalexit.utils import (
    canonical_json_sha256,
    git_commit,
    load_yaml,
    project_root,
    resolve,
    sha256,
)


FAMILIES = ("b0", "b1", "teacher", "t_text", "d", "a")
FIXED_PREFIX_FAMILIES = {"b0", "b1", "a"}
PREFIXES = ("5", "10", "20", "40", "80", "FULL")
METRIC_NAMES = (
    "balanced_accuracy",
    "auroc",
    "macro_f1",
    "brier",
    "nll",
    "ece",
    "macro_by_article_balanced_accuracy",
)
BOOTSTRAP_SEED = 20271015
FORBIDDEN_PARAMETER_TOKENS = (
    "reader",
    "participant",
    "identity",
    "article_id",
    "paragraph_id",
    "question",
)
PREDICTION_COLUMNS = {
    "trial_id",
    "prefix",
    "observed_fixations",
    "logit",
    "probability_h",
    "target_goal_index",
    "reader_id",
    "article_id",
    "full_fixation_count",
    "git_commit",
    "config_sha256",
    "canonical_data_sha256",
    "split_manifest_sha256",
    "context_id",
    "inner_holdout",
    "model_family",
    "candidate_id",
    "seed",
    "checkpoint_sha256",
    "outer_test_evaluated",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _assert_close(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(f"Mapping mismatch: {label}")
        for key in expected:
            _assert_close(observed[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(observed) != len(expected):
            raise ValueError(f"Sequence mismatch: {label}")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _assert_close(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        if not math.isclose(float(observed), float(expected), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Numeric mismatch: {label}: {observed} != {expected}")
        return
    if observed != expected:
        raise ValueError(f"Value mismatch: {label}: {observed!r} != {expected!r}")


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _hash_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _candidate_settings(
    config: dict[str, Any], state: dict[str, Any], context_id: str, family: str
) -> dict[str, tuple[float, float | None]]:
    inherited = state["selections"][context_id]["d"] if family == "a" else None
    return {
        _candidate_id(family, learning_rate, alpha): (learning_rate, alpha)
        for learning_rate, alpha in _candidate_grid(config, family, inherited)
    }


def _add_cluster_rows(
    accumulator: dict[str, dict[str, int]], frame: pd.DataFrame
) -> None:
    local = frame[["reader_id", "target_goal_index", "probability_h"]].copy()
    local["correct"] = (
        local["probability_h"].ge(0.5).astype(np.int8)
        == local["target_goal_index"].astype(np.int8)
    ).astype(np.int64)
    grouped = local.groupby("reader_id", sort=False).agg(
        goal=("target_goal_index", "first"),
        goal_values=("target_goal_index", "nunique"),
        correct=("correct", "sum"),
        total=("correct", "size"),
    )
    if grouped["goal_values"].ne(1).any():
        raise ValueError("Reader goal is not constant in selected predictions")
    for reader_id, row in grouped.iterrows():
        key = str(reader_id)
        goal = int(row["goal"])
        current = accumulator.setdefault(key, {"goal": goal, "correct": 0, "total": 0})
        if current["goal"] != goal:
            raise ValueError(f"Reader goal changed across evaluations: {key}")
        current["correct"] += int(row["correct"])
        current["total"] += int(row["total"])


def _bootstrap_draws(
    reader_goals: dict[str, int], replicates: int, seed: int
) -> tuple[dict[int, list[str]], dict[int, np.ndarray]]:
    readers = {
        goal: sorted(reader for reader, value in reader_goals.items() if value == goal)
        for goal in (0, 1)
    }
    if not readers[0] or not readers[1]:
        raise ValueError("Both reader-goal strata are required for bootstrap")
    rng = np.random.default_rng(seed)
    draws = {
        goal: rng.integers(0, len(readers[goal]), size=(replicates, len(readers[goal])))
        for goal in (0, 1)
    }
    return readers, draws


def _bootstrap_values(
    stats: dict[str, dict[str, int]],
    readers: dict[int, list[str]],
    draws: dict[int, np.ndarray],
) -> tuple[float, np.ndarray]:
    recalls: list[float] = []
    replicate_recalls: list[np.ndarray] = []
    for goal in (0, 1):
        correct = np.asarray([stats.get(reader, {}).get("correct", 0) for reader in readers[goal]])
        total = np.asarray([stats.get(reader, {}).get("total", 0) for reader in readers[goal]])
        if total.sum() == 0:
            raise ValueError(f"No predictions for reader-goal stratum {goal}")
        sampled_correct = correct[draws[goal]].sum(axis=1)
        sampled_total = total[draws[goal]].sum(axis=1)
        if (sampled_total == 0).any():
            raise ValueError(f"Empty bootstrap replicate for reader-goal stratum {goal}")
        recalls.append(float(correct.sum() / total.sum()))
        replicate_recalls.append(sampled_correct / sampled_total)
    return float(np.mean(recalls)), np.mean(np.stack(replicate_recalls), axis=0)


def reader_cluster_bootstrap(
    stats: dict[str, dict[str, int]],
    readers: dict[int, list[str]],
    draws: dict[int, np.ndarray],
    *,
    seed: int,
) -> dict[str, Any]:
    estimate, values = _bootstrap_values(stats, readers, draws)
    return {
        "pooled_balanced_accuracy": estimate,
        "ci95_percentile": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "replicates": int(len(values)),
        "seed": seed,
        "cluster_unit": "reader",
        "stratified_by_reader_goal": True,
    }


def reader_cluster_contrast(
    left: dict[str, dict[str, int]],
    right: dict[str, dict[str, int]],
    readers: dict[int, list[str]],
    draws: dict[int, np.ndarray],
    *,
    seed: int,
) -> dict[str, Any]:
    left_point, left_values = _bootstrap_values(left, readers, draws)
    right_point, right_values = _bootstrap_values(right, readers, draws)
    differences = left_values - right_values
    return {
        "pooled_balanced_accuracy_difference": left_point - right_point,
        "ci95_percentile": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "replicates": int(len(differences)),
        "seed": seed,
        "paired_reader_draws": True,
        "stratified_by_reader_goal": True,
    }


def _validate_prediction(
    frame: pd.DataFrame,
    result: dict[str, Any],
    validation_ids: list[str],
    validation_targets: pd.DataFrame,
    store: GoalExitStore,
    ece_bins: int,
    label: str,
) -> None:
    if set(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction columns differ from frozen schema: {label}")
    if len(frame) != int(result["prediction_rows"]):
        raise ValueError(f"Prediction row count mismatch: {label}")
    if frame[["trial_id", "prefix"]].duplicated().any():
        raise ValueError(f"Duplicate trial/prefix prediction: {label}")
    numeric = frame[["observed_fixations", "logit", "probability_h", "full_fixation_count"]]
    if not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise ValueError(f"Non-finite prediction: {label}")
    probability = 1.0 / (1.0 + np.exp(-np.clip(frame["logit"].to_numpy(np.float64), -50, 50)))
    if not np.allclose(probability, frame["probability_h"], rtol=1e-12, atol=1e-12):
        raise ValueError(f"Logit/probability mismatch: {label}")
    run = result["run_config"]
    provenance = {
        "git_commit": run["git_commit"],
        "config_sha256": run["config_sha256"],
        "canonical_data_sha256": run["canonical_data_sha256"],
        "split_manifest_sha256": run["split_manifest_sha256"],
        "context_id": run["context_id"],
        "inner_holdout": int(run["inner_holdout"]),
        "model_family": run["model_family"],
        "candidate_id": run["candidate_id"],
        "seed": int(run["seed"]),
        "checkpoint_sha256": result["checkpoint_sha256"],
    }
    for column, expected in provenance.items():
        if not frame[column].eq(expected).all():
            raise ValueError(f"Prediction provenance mismatch {column}: {label}")
    if not frame["outer_test_evaluated"].eq(False).all():
        raise PermissionError(f"Outer-evaluation flag found in prediction: {label}")
    allowed = set(validation_ids)
    if not set(frame["trial_id"].astype(str)).issubset(allowed):
        raise PermissionError(f"Prediction includes non-validation row: {label}")
    target = validation_targets.set_index("trial_id")["goal_index"].astype(int)
    metadata = store.metadata.loc[validation_ids]
    expected_prefixes = PREFIXES if result["family"] in FIXED_PREFIX_FAMILIES else ("FULL",)
    if set(frame["prefix"].astype(str)) != set(expected_prefixes):
        raise ValueError(f"Prediction prefix set mismatch: {label}")
    for prefix in expected_prefixes:
        subset = frame.loc[frame["prefix"].astype(str).eq(prefix)].copy()
        expected_ids = {
            trial_id
            for trial_id in validation_ids
            if prefix == "FULL"
            or int(metadata.loc[trial_id, "full_fixation_count"]) >= int(prefix)
        }
        ids = subset["trial_id"].astype(str)
        if set(ids) != expected_ids or len(ids) != len(expected_ids):
            raise ValueError(f"Prefix membership mismatch {prefix}: {label}")
        expected_target = target.loc[ids].to_numpy(np.int64)
        if not np.array_equal(subset["target_goal_index"].to_numpy(np.int64), expected_target):
            raise ValueError(f"Target/input mismatch {prefix}: {label}")
        expected_meta = metadata.loc[ids]
        if not np.array_equal(
            subset["reader_id"].astype(str).to_numpy(),
            expected_meta["reader_id"].astype(str).to_numpy(),
        ):
            raise ValueError(f"Reader metadata mismatch {prefix}: {label}")
        if not np.array_equal(
            subset["article_id"].astype(str).to_numpy(),
            expected_meta["article_id"].astype(str).to_numpy(),
        ):
            raise ValueError(f"Article metadata mismatch {prefix}: {label}")
        full_counts = expected_meta["full_fixation_count"].to_numpy(np.int64)
        if not np.array_equal(subset["full_fixation_count"].to_numpy(np.int64), full_counts):
            raise ValueError(f"Full-length audit field mismatch {prefix}: {label}")
        expected_observed = full_counts if prefix == "FULL" else np.full(len(subset), int(prefix))
        if not np.array_equal(
            subset["observed_fixations"].to_numpy(np.int64), expected_observed
        ):
            raise ValueError(f"Prefix causality mismatch {prefix}: {label}")
        recomputed = evaluate_binary(
            expected_target,
            subset["probability_h"].to_numpy(np.float64),
            subset["article_id"].astype(str).to_numpy(),
            ece_bins=ece_bins,
        ).to_dict()
        _assert_close(recomputed, result["metrics"][prefix], f"recomputed metrics {label} {prefix}")


def _validate_checkpoint(
    checkpoint_path: Path, result: dict[str, Any], label: str
) -> set[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "goalexit-checkpoint-v1":
        raise ValueError(f"Checkpoint schema mismatch: {label}")
    if checkpoint.get("family") != result["family"]:
        raise ValueError(f"Checkpoint family mismatch: {label}")
    _assert_close(checkpoint.get("run_config"), result["run_config"], f"checkpoint run config {label}")
    _assert_close(checkpoint.get("preprocessor"), result["preprocessor"], f"checkpoint preprocessor {label}")
    if int(checkpoint.get("selected_epoch")) != int(result["selected_epoch"]):
        raise ValueError(f"Checkpoint selected epoch mismatch: {label}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Missing checkpoint state: {label}")
    violations = {
        name
        for name in state
        if any(token in name.casefold() for token in FORBIDDEN_PARAMETER_TOKENS)
    }
    if violations:
        raise ValueError(f"Identity-bearing checkpoint parameter: {label}: {sorted(violations)}")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"Non-finite checkpoint state: {label}")
    state_count = int(sum(value.numel() for value in state.values()))
    if state_count != int(result["parameter_counts"]["total"]):
        raise ValueError(f"Checkpoint parameter count mismatch: {label}")
    return set(state)


def _validate_result(
    result: dict[str, Any],
    *,
    family: str,
    candidate_id: str,
    learning_rate: float,
    alpha: float | None,
    context: dict[str, Any],
    inner: int,
    training_ids: list[str],
    validation_ids: list[str],
    canonical_hash: str,
    split_hash: str,
    config_hash: str,
    development_seed: int,
    expected_paths: dict[str, Path],
    label: str,
) -> None:
    if result.get("schema") != "goalexit-candidate-result-v1" or result.get("status") != "complete":
        raise ValueError(f"Incomplete candidate result: {label}")
    if result.get("family") != family or result.get("outer_test_evaluated") is not False:
        raise PermissionError(f"Candidate family/outer flag mismatch: {label}")
    run = result.get("run_config", {})
    expected = {
        "schema": "goalexit-development-run-v1",
        "context_id": context["context_id"],
        "outer_reader_fold": int(context["outer_reader_fold"]),
        "outer_article_fold": int(context["outer_article_fold"]),
        "inner_holdout": inner,
        "model_family": family,
        "candidate_id": candidate_id,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "temperature": 2.0 if family in {"d", "a"} else None,
        "seed": development_seed,
        "config_sha256": config_hash,
        "canonical_data_sha256": canonical_hash,
        "split_manifest_sha256": split_hash,
        "training_trial_ids_sha256": canonical_json_sha256(sorted(training_ids)),
        "validation_trial_ids_sha256": canonical_json_sha256(sorted(validation_ids)),
        "outer_target_trial_ids_sha256": context["outer_target_trial_ids_sha256"],
        "training_trials": len(training_ids),
        "validation_trials": len(validation_ids),
        "target_access_scope": "inner training and paired inner validation only",
        "outer_test_evaluated": False,
    }
    for key, value in expected.items():
        _assert_close(run.get(key), value, f"run config {label}.{key}")
    if int(run["seed"]) in {37, 71}:
        raise PermissionError(f"Unauthorized stability seed in {label}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(run.get("git_commit", ""))):
        raise ValueError(f"Invalid code commit provenance: {label}")
    identity = dict(run)
    recorded_identity = identity.pop("run_identity_sha256", None)
    if recorded_identity != canonical_json_sha256(identity):
        raise ValueError(f"Run identity hash mismatch: {label}")
    if Path(result["checkpoint_path"]) != expected_paths["checkpoint"]:
        raise ValueError(f"Checkpoint path mismatch: {label}")
    if Path(result["prediction_path"]) != expected_paths["prediction"]:
        raise ValueError(f"Prediction path mismatch: {label}")
    if int(result["epochs_ran"]) != len(result["history"]):
        raise ValueError(f"Epoch history count mismatch: {label}")
    selected_epoch = int(result["selected_epoch"])
    selected_history = [row for row in result["history"] if int(row["epoch"]) == selected_epoch]
    if len(selected_history) != 1:
        raise ValueError(f"Selected epoch missing from history: {label}")
    for metric in METRIC_NAMES:
        _assert_close(
            selected_history[0][metric], result["metrics"]["FULL"][metric], f"selected epoch {label}.{metric}"
        )
    preprocessor = result.get("preprocessor", {})
    if (
        preprocessor.get("training_trial_count") != len(training_ids)
        or preprocessor.get("validation_or_outer_statistics_used") is not False
        or "training" not in str(preprocessor.get("fit_scope", "")).casefold()
    ):
        raise ValueError(f"Preprocessor fit-scope mismatch: {label}")


def _aggregate_selected(
    selected_metrics: dict[tuple[str, str], list[dict[str, Any]]],
    cluster_stats: dict[tuple[str, str], dict[str, dict[str, int]]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, list[str]], dict[int, np.ndarray]]:
    full_stats = cluster_stats[("b1", "FULL")]
    reader_goals = {reader: int(values["goal"]) for reader, values in full_stats.items()}
    readers, draws = _bootstrap_draws(
        reader_goals, int(config["bootstrap_replicates"]), BOOTSTRAP_SEED
    )
    families: dict[str, Any] = {}
    parameter_counts: dict[str, int] = {}
    for family in FAMILIES:
        prefix_names = PREFIXES if family in FIXED_PREFIX_FAMILIES else ("FULL",)
        prefix_output: dict[str, Any] = {}
        for prefix in prefix_names:
            rows = selected_metrics[(family, prefix)]
            if len(rows) != 75:
                raise ValueError(f"Selected metric coverage mismatch: {family} {prefix}")
            prefix_output[prefix] = {
                "metrics": {
                    metric: aggregate_metrics(rows, metric) for metric in METRIC_NAMES
                },
                "reader_cluster_bootstrap_balanced_accuracy": reader_cluster_bootstrap(
                    cluster_stats[(family, prefix)],
                    readers,
                    draws,
                    seed=BOOTSTRAP_SEED,
                ),
            }
        families[family] = {
            "selected_evaluations": 75,
            "prefixes": prefix_output,
        }
    return families, parameter_counts, readers, draws


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _render_report(metrics: dict[str, Any], audit_sha256: str, metrics_sha256: str) -> str:
    families = metrics["families"]
    full_lines = [
        "| Model | BA mean | Reader-cluster 95% CI (pooled BA) | AUROC | Macro-F1 | NLL | Brier | ECE | Macro-article BA | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "b0": "B0",
        "b1": "B1",
        "teacher": "T",
        "t_text": "T-text",
        "d": "D",
        "a": "A (FULL)",
    }
    for family in FAMILIES:
        row = families[family]["prefixes"]["FULL"]
        summaries = row["metrics"]
        ci = row["reader_cluster_bootstrap_balanced_accuracy"]["ci95_percentile"]
        full_lines.append(
            "| {label} | {ba} | [{lo}, {hi}] | {auroc} | {f1} | {nll} | {brier} | {ece} | {article} | {params:,} |".format(
                label=labels[family],
                ba=_fmt(summaries["balanced_accuracy"]["mean"]),
                lo=_fmt(ci[0]),
                hi=_fmt(ci[1]),
                auroc=_fmt(summaries["auroc"]["mean"]),
                f1=_fmt(summaries["macro_f1"]["mean"]),
                nll=_fmt(summaries["nll"]["mean"]),
                brier=_fmt(summaries["brier"]["mean"]),
                ece=_fmt(summaries["ece"]["mean"]),
                article=_fmt(summaries["macro_by_article_balanced_accuracy"]["mean"]),
                params=metrics["parameter_counts"][family],
            )
        )
    prefix_lines = [
        "| Model | Prefix | Mean BA | Median BA | Paired evaluations |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("b0", "b1", "a"):
        for prefix in PREFIXES:
            row = families[family]["prefixes"][prefix]["metrics"]
            prefix_lines.append(
                f"| {labels[family]} | {prefix} | {_fmt(row['balanced_accuracy']['mean'])} | "
                f"{_fmt(row['balanced_accuracy']['median'])} | {row['balanced_accuracy']['evaluations']} paired evaluations |"
            )
    gate_t = metrics["gates"]["T"]
    gate_d = metrics["gates"]["D"]
    gate_a0 = metrics["gates"]["A0"]
    cpu_lines = [
        "| Model | Params | State bytes | Update mean / median / P95 (ms) | FULL mean / median / P95 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics["cpu_benchmark"]["measurements"]:
        update = row["per_new_fixation_update"]
        full = row["end_to_end_trial_full"]
        cpu_lines.append(
            f"| {labels[row['family']]} | {row['parameter_count']:,} | {row['serialized_state_bytes']:,} | "
            f"{update['mean_ms']:.3f} / {update['median_ms']:.3f} / {update['p95_ms']:.3f} | "
            f"{full['mean_ms']:.3f} / {full['median_ms']:.3f} / {full['p95_ms']:.3f} |"
        )
    canonical = metrics["canonical_population"]
    counts = metrics["artifact_counts"]
    hashes = metrics["artifact_set_hashes"]
    return f"""# GoalExit Phase 0-2 development report

Status: **STOP pending researcher review**. This report contains development-only
evidence. The 25 outer reader/article intersections remain sealed and were not
evaluated.

## Population, split, and audit status

- Canonical population: **{canonical['trials']:,} trials**, **{canonical['readers']} readers**,
  **{canonical['articles']} articles**, **{canonical['fixations']:,} valid fixations**, and
  **{canonical['paragraph_versions']} paragraph variants**.
- GoalExit split seed: **{metrics['split']['seed']}**; 25 outer contexts and 75 paired
  inner-development evaluations. Canonical SHA-256:
  `{metrics['canonical_data_sha256']}`.
- Forbidden-field, metadata-only, reader-identity parameter, target/input,
  non-finite prediction, training-only preprocessing, prefix-causality, teacher-cache,
  exact-output-set, and outer-guard audits: **PASS**.
- Uncertainty uses {metrics['bootstrap']['replicates']:,} reader-cluster bootstrap
  replicates, stratified by reader goal, seed {metrics['bootstrap']['seed']}.

## Selected development performance

Each value is summarized over the same 75 paired inner validations. Bootstrap
intervals use pooled selected prediction rows while resampling whole readers;
the gate calculations remain the preregistered mean of 75 paired evaluation scores.

{chr(10).join(full_lines)}

## Fixed-prefix curves

No prefix was selected. All frozen reporting prefixes are shown. Trial counts at
short prefixes vary only because a trial must contain at least that many fixations.

{chr(10).join(prefix_lines)}

## Gate decisions

- **Gate T: {gate_t['status']}**. Mean T−B1 BA =
  `{gate_t['teacher_minus_b1']['mean_difference']:+.6f}`; T>B1 in
  `{gate_t['teacher_minus_b1']['positive_direction_count']}/75`; mean
  T-text−B1 BA = `{gate_t['t_text_minus_b1']['mean_difference']:+.6f}`.
- **Gate D: {gate_d['status']}**. Mean D−B1 BA =
  `{gate_d['d_minus_b1']['mean_difference']:+.6f}`; D>B1 in
  `{gate_d['d_minus_b1']['positive_direction_count']}/75`; D and B1 both use
  {metrics['parameter_counts']['d']:,} trainable parameters.
- **Gate A0: {gate_a0['status']}**. Mean A(FULL)−D BA =
  `{gate_a0['a_full_minus_d']['mean_difference']:+.6f}` versus the required
  `-0.005`; mean ECE difference = `{gate_a0['mean_ece_difference']:+.6f}`;
  parameter equality passed. The BA preservation condition failed.
- **Gate E: NOT RUN**. Gate A0 did not authorize adaptive trace generation,
  calibration, threshold evaluation, AP-1, or AP-2. No rescue or alternative
  stopping rule was attempted.

The development decision is therefore: teacher and privileged distillation are
supported by the frozen relative gates, but multi-prefix training damaged FULL
performance beyond the allowed tolerance. The anytime/adaptive direction stops.
These are development findings, not confirmatory outer-test claims.

## Sanity checks

- Reader-level label permutation: **{metrics['sanity']['reader_label_permutation']['status'].upper()}**;
  mean permuted BA `{metrics['sanity']['reader_label_permutation']['mean_permuted_balanced_accuracy']:.6f}`
  versus observed subset BA `{metrics['sanity']['reader_label_permutation']['mean_observed_selected_b1_balanced_accuracy']:.6f}`
  (criterion mean permuted BA <= 0.55).
- T-text shortcut control: acceptable; it did not reproduce the teacher gain.
- Metadata-only shortcut probe: **PASS**, mean BA
  `{metrics['sanity']['metadata_probe']['mean_balanced_accuracy']:.4f}`.
- Prefix causality and target/input membership: **PASS** for every saved official
  prediction row.
- Outer evaluation guard: **PASS**; final lock absent, no outer prediction or
  scientific metric created.

## Single-thread CPU deployment proxy

Environment: `{metrics['cpu_benchmark']['cpu_model']}`;
`{metrics['cpu_benchmark']['os']}`; PyTorch `{metrics['cpu_benchmark']['torch']}`.
One framework thread, {metrics['cpu_benchmark']['warmup_iterations']} warm-up and
{metrics['cpu_benchmark']['measured_iterations']} measured iterations; disk I/O and
parsing excluded. These are CPU proxy measurements, not embedded-device claims.

{chr(10).join(cpu_lines)}

AP-1/AP-2 latency is unavailable because Gate A0 failed and adaptive stopping was
not authorized.

## Artifact coverage and hashes

- Official candidate results: **{counts['candidate_results']}** metrics,
  **{counts['checkpoints']}** checkpoints, and **{counts['predictions']}** prediction
  files; **{counts['selected_evaluations']}** selected evaluations across six families.
- Teacher caches: **{counts['teacher_caches']}**; permutation evaluations:
  **{counts['permutation_evaluations']}**; residual partial states: **0**.
- Metric-set digest: `{hashes['metrics']}`.
- Checkpoint-set digest: `{hashes['checkpoints']}`.
- Prediction-set digest: `{hashes['predictions']}`.
- Teacher-cache-set digest: `{hashes['teacher_caches']}`.
- Development metrics SHA-256: `{metrics_sha256}`.
- Development output audit SHA-256: `{audit_sha256}`.

Machine-readable evidence is in
`artifacts/metrics/development_goalexit.json` and
`artifacts/manifests/development_output_audit.json`; supporting manifests cover
the split, teacher cache, CPU benchmark, permutation sanity, RoBERTa cache, and
outer guard. Deviations and infrastructure incidents are recorded in
`docs/deviations.md`.

## Required stop

Outer rows were not evaluated. No final lock was created, seeds 37/71 were not
run, and no unauthorized model or hyperparameter rescue was attempted. Phase 0-2
stops here pending researcher review.
"""


def run(config_path: Path) -> dict[str, Any]:
    root = project_root()
    commit = git_commit(root)
    if commit == "UNCOMMITTED":
        raise RuntimeError("Final evidence generation requires a clean frozen Git commit")
    config_path = resolve(root, config_path)
    config = load_yaml(config_path)
    config_hash = sha256(config_path)
    state_path = resolve(root, config["outputs"]["output_root"]) / "selection_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    canonical_path = resolve(root, config["outputs"]["canonical_manifest"])
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    split_path = resolve(root, config["outputs"]["split_manifest"])
    split = load_split(split_path)
    split_hash = sha256(split_path)
    roberta_path = resolve(root, config["outputs"]["roberta_manifest"])
    roberta = json.loads(roberta_path.read_text(encoding="utf-8"))
    forbidden = json.loads(
        resolve(root, config["outputs"]["forbidden_field_audit"]).read_text(encoding="utf-8")
    )
    metadata_probe_path = resolve(root, config["outputs"]["metadata_probe"])
    metadata_probe = json.loads(metadata_probe_path.read_text(encoding="utf-8"))
    permutation_path = resolve(root, config["outputs"]["permutation_sanity"])
    permutation = json.loads(permutation_path.read_text(encoding="utf-8"))
    cpu_path = resolve(root, config["outputs"]["cpu_benchmark"])
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    outer_guard_path = resolve(root, config["outputs"]["outer_guard_audit"])
    outer_guard = json.loads(outer_guard_path.read_text(encoding="utf-8"))
    targets = {
        "teacher_cache": resolve(root, config["outputs"]["teacher_cache_manifest"]),
        "metrics": resolve(root, config["outputs"]["development_metrics"]),
        "audit": resolve(root, config["outputs"]["development_output_audit"]),
        "report": root / "docs/03_phase0_2_development_report.md",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite terminal evidence: {existing}")
    if (
        state.get("outer_test_evaluated") is not False
        or state.get("gate_t", {}).get("status") != "PASS"
        or state.get("gate_d", {}).get("status") != "PASS"
        or state.get("gate_a0", {}).get("status") != "FAIL"
        or "adaptive" in state
        or "gate_e" in state
    ):
        raise ValueError("Selection state does not match the authorized T/D PASS, A0 FAIL path")
    if set(state.get("selections", {})) != {f"r{r}_a{a}" for r in range(5) for a in range(5)}:
        raise ValueError("Selection state does not cover exactly 25 contexts")
    if canonical.get("status") != "pass" or forbidden.get("status") != "pass":
        raise ValueError("Canonical or forbidden-field audit is not passing")
    if metadata_probe.get("status") != "pass" or permutation.get("status") != "pass":
        raise ValueError("Required shortcut/permutation sanity check failed")
    if cpu.get("status") != "complete" or cpu.get("outer_rows_touched") is not False:
        raise ValueError("CPU benchmark manifest is incomplete")
    if outer_guard.get("status") != "pass" or outer_guard.get("split_manifest_sha256") != split_hash:
        raise ValueError("Outer guard manifest mismatch")
    try:
        require_outer_authorization(root)
    except OuterEvaluationSealedError:
        pass
    else:
        raise PermissionError("Unexpected outer-evaluation authorization")
    for path in (root / "experiments/outer", root / "research_storage/predictions/outer"):
        if path.exists():
            raise PermissionError(f"Unexpected outer output location exists: {path}")

    store = GoalExitStore.load(
        resolve(root, config["outputs"]["trial_table"]),
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    context_map = {str(context["context_id"]): context for context in split["contexts"]}
    support: dict[tuple[str, int], dict[str, Any]] = {}
    for context_id, context in context_map.items():
        for inner in range(int(config["inner_folds"])):
            training, validation, outer = context_positions(store.metadata, split, context, inner)
            validation_targets = load_allowed_targets(
                resolve(root, config["outputs"]["trial_table"]), validation, outer
            )
            support[(context_id, inner)] = {
                "training": training,
                "validation": validation,
                "outer": outer,
                "targets": validation_targets,
            }

    expected_metrics: set[Path] = set()
    expected_checkpoints: set[Path] = set()
    expected_predictions: set[Path] = set()
    expected_selections: set[Path] = set()
    metric_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    selected_metrics: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    cluster_stats: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(dict)
    parameter_counts: dict[str, set[int]] = defaultdict(set)
    candidate_counts: dict[str, int] = defaultdict(int)
    model_commits: dict[str, set[str]] = defaultdict(set)
    checkpoint_parameter_names: dict[str, set[str]] = defaultdict(set)
    selected_checkpoint_hashes: dict[str, set[str]] = defaultdict(set)

    for context_id in sorted(context_map):
        context = context_map[context_id]
        for family in FAMILIES:
            if set(state["selections"][context_id]) != set(FAMILIES):
                raise ValueError(f"Unauthorized/missing family set in {context_id}")
            settings = _candidate_settings(config, state, context_id, family)
            candidate_results: dict[str, list[dict[str, Any]]] = {}
            for candidate_id, (learning_rate, alpha) in sorted(settings.items()):
                results: list[dict[str, Any]] = []
                for inner in range(int(config["inner_folds"])):
                    paths = _paths(root, config, context_id, family, candidate_id, inner)
                    metrics_path = paths["run"] / "metrics.json"
                    for path in (metrics_path, paths["checkpoint"], paths["prediction"]):
                        if not path.is_file():
                            raise FileNotFoundError(f"Missing official candidate artifact: {path}")
                    result = json.loads(metrics_path.read_text(encoding="utf-8"))
                    current = support[(context_id, inner)]
                    label = f"{context_id}/{family}/{candidate_id}/inner_{inner}"
                    _validate_result(
                        result,
                        family=family,
                        candidate_id=candidate_id,
                        learning_rate=learning_rate,
                        alpha=alpha,
                        context=context,
                        inner=inner,
                        training_ids=current["training"],
                        validation_ids=current["validation"],
                        canonical_hash=canonical["canonical_data_sha256"],
                        split_hash=split_hash,
                        config_hash=config_hash,
                        development_seed=int(config["development_seed"]),
                        expected_paths=paths,
                        label=label,
                    )
                    metric_record = _hash_record(root, metrics_path)
                    checkpoint_record = _hash_record(root, paths["checkpoint"])
                    prediction_record = _hash_record(root, paths["prediction"])
                    if checkpoint_record["sha256"] != result["checkpoint_sha256"]:
                        raise ValueError(f"Checkpoint hash mismatch: {label}")
                    if prediction_record["sha256"] != result["prediction_sha256"]:
                        raise ValueError(f"Prediction hash mismatch: {label}")
                    names = _validate_checkpoint(paths["checkpoint"], result, label)
                    checkpoint_parameter_names[family].update(names)
                    frame = pd.read_parquet(paths["prediction"])
                    _validate_prediction(
                        frame,
                        result,
                        current["validation"],
                        current["targets"],
                        store,
                        int(config["ece_bins"]),
                        label,
                    )
                    enriched = dict(result)
                    enriched["metrics_path"] = str(metrics_path)
                    enriched["metrics_sha256"] = metric_record["sha256"]
                    results.append(enriched)
                    expected_metrics.add(metrics_path.resolve())
                    expected_checkpoints.add(paths["checkpoint"].resolve())
                    expected_predictions.add(paths["prediction"].resolve())
                    metric_records.append(metric_record)
                    checkpoint_records.append(checkpoint_record)
                    prediction_records.append(prediction_record)
                    candidate_counts[family] += 1
                    parameter_counts[family].add(int(result["parameter_counts"]["trainable"]))
                    model_commits[family].add(str(result["run_config"]["git_commit"]))
                    if candidate_id == state["selections"][context_id][family]["selected_candidate_id"]:
                        selected_checkpoint_hashes[family].add(result["checkpoint_sha256"])
                        for prefix, values in result["metrics"].items():
                            selected_metrics[(family, prefix)].append(
                                {"context_id": context_id, "inner_holdout": inner, **values}
                            )
                            prefix_frame = frame.loc[frame["prefix"].astype(str).eq(prefix)]
                            _add_cluster_rows(cluster_stats[(family, prefix)], prefix_frame)
                candidate_results[candidate_id] = results
            recomputed = _select_candidate(family, candidate_results)
            recomputed["context_id"] = context_id
            selection_path = (
                resolve(root, config["outputs"]["output_root"])
                / context_id
                / family
                / "selection.json"
            )
            expected_selections.add(selection_path.resolve())
            disk_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            core = {key: value for key, value in recomputed.items() if key != "selected_results"}
            _assert_close(disk_selection, core, f"selection file {context_id}/{family}")
            recomputed["selection_path"] = str(selection_path)
            recomputed["selection_sha256"] = sha256(selection_path)
            _assert_close(
                state["selections"][context_id][family],
                recomputed,
                f"selection state {context_id}/{family}",
            )

    output_root = resolve(root, config["outputs"]["output_root"])
    actual_metrics = {
        path.resolve()
        for path in output_root.glob("r*_a*/*/*/inner_*/metrics.json")
    }
    actual_selections = {
        path.resolve() for path in output_root.glob("r*_a*/*/selection.json")
    }
    actual_checkpoints = {
        path.resolve()
        for path in resolve(root, config["outputs"]["checkpoint_root"]).rglob("*.pt")
    }
    actual_predictions = {
        path.resolve()
        for path in resolve(root, config["outputs"]["prediction_root"]).rglob("*.parquet")
    }
    if actual_metrics != expected_metrics:
        raise ValueError("Official metric file set is not exact")
    if actual_selections != expected_selections:
        raise ValueError("Official selection file set is not exact")
    if actual_checkpoints != expected_checkpoints:
        raise ValueError("Official checkpoint file set is not exact")
    if actual_predictions != expected_predictions:
        raise ValueError("Official prediction file set is not exact")
    partials = list(output_root.rglob("partial_training_state.pt"))
    temporary = list(output_root.rglob("*.tmp"))
    if partials or temporary:
        raise ValueError("Residual partial or temporary development state exists")
    if (output_root / "adaptive_stopping.json").exists() or list(
        resolve(root, config["outputs"]["prediction_root"]).rglob("*_trace.parquet")
    ):
        raise PermissionError("Adaptive artifacts exist despite Gate A0 failure")
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise ValueError("A family has inconsistent trainable parameter counts")

    recomputed_gates = {
        "T": evaluate_gate_t(state["selections"]),
        "D": evaluate_gate_d(state["selections"]),
        "A0": evaluate_gate_a0(state["selections"]),
    }
    _assert_close(recomputed_gates["T"], state["gate_t"], "Gate T")
    _assert_close(recomputed_gates["D"], state["gate_d"], "Gate D")
    _assert_close(recomputed_gates["A0"], state["gate_a0"], "Gate A0")
    gates = {
        **recomputed_gates,
        "E": {
            "gate": "E",
            "status": "NOT_RUN",
            "reason": "Gate A0 failed; adaptive stopping was not authorized",
        },
    }

    teacher_cache = state.get("teacher_cache_manifest", {})
    cache_entries = teacher_cache.get("entries", [])
    cache_keyed = {
        (str(entry.get("context_id")), int(entry.get("inner_holdout", -1))): entry
        for entry in cache_entries
    }
    if (
        teacher_cache.get("status") != "complete"
        or teacher_cache.get("entry_count") != 75
        or teacher_cache.get("outer_rows_touched") is not False
        or set(cache_keyed) != set(support)
    ):
        raise ValueError("Teacher-cache state manifest is incomplete")
    teacher_cache_records: list[dict[str, Any]] = []
    expected_cache_paths: set[Path] = set()
    for key, current in sorted(support.items()):
        context_id, inner = key
        entry = cache_keyed[key]
        selected_teacher = next(
            result
            for result in state["selections"][context_id]["teacher"]["selected_results"]
            if int(result["run_config"]["inner_holdout"]) == inner
        )
        cache_path = output_root / context_id / "teacher_cache" / f"inner_{inner}.parquet"
        expected = {
            "context_id": context_id,
            "inner_holdout": inner,
            "training_trials": len(current["training"]),
            "training_trial_ids_sha256": canonical_json_sha256(sorted(current["training"])),
            "source_teacher_checkpoint_path": selected_teacher["checkpoint_path"],
            "source_teacher_checkpoint_sha256": selected_teacher["checkpoint_sha256"],
            "source_teacher_run_identity_sha256": selected_teacher["run_config"]["run_identity_sha256"],
            "cache_path": str(cache_path),
            "outer_rows_touched": False,
        }
        for name, value in expected.items():
            _assert_close(entry.get(name), value, f"teacher cache {context_id}/{inner}.{name}")
        record = _hash_record(root, cache_path)
        if entry.get("cache_sha256") != record["sha256"]:
            raise ValueError(f"Teacher cache hash mismatch: {context_id}/{inner}")
        cache = pd.read_parquet(cache_path)
        if (
            set(cache.columns) != {"trial_id", "teacher_logit"}
            or cache["trial_id"].duplicated().any()
            or set(cache["trial_id"].astype(str)) != set(current["training"])
            or not np.isfinite(cache["teacher_logit"]).all()
            or set(cache["trial_id"].astype(str)) & current["outer"]
        ):
            raise ValueError(f"Teacher cache content mismatch: {context_id}/{inner}")
        teacher_cache_records.append(record)
        expected_cache_paths.add(cache_path.resolve())
    actual_cache_paths = {
        path.resolve() for path in output_root.glob("r*_a*/teacher_cache/inner_*.parquet")
    }
    if actual_cache_paths != expected_cache_paths:
        raise ValueError("Teacher cache file set is not exact")

    permutation_records: list[dict[str, Any]] = []
    expected_permutation_metrics: set[Path] = set()
    expected_permutation_checkpoints: set[Path] = set()
    expected_permutation_predictions: set[Path] = set()
    for position, row in enumerate(permutation["evaluations"]):
        context_id = str(row["context_id"])
        inner = int(row["inner_holdout"])
        candidate_id = f"reader_permutation_{context_id}_i{inner}"
        metric_path = output_root / "sanity" / candidate_id / "metrics.json"
        result = json.loads(metric_path.read_text(encoding="utf-8"))
        checkpoint_path = root / "research_storage/permutation" / candidate_id / "checkpoint.pt"
        prediction_path = root / "research_storage/permutation" / candidate_id / "predictions.parquet"
        if (
            result["run_config"].get("reader_level_permutation") is not True
            or int(result["run_config"]["seed"]) != int(config["permutation_sanity"]["seed"]) + position
            or result.get("outer_test_evaluated") is not False
            or result["checkpoint_sha256"] != row["checkpoint_sha256"]
            or result["prediction_sha256"] != row["prediction_sha256"]
        ):
            raise ValueError(f"Permutation provenance mismatch: {candidate_id}")
        _assert_close(
            result["metrics"]["FULL"]["balanced_accuracy"],
            row["permuted_balanced_accuracy"],
            f"permutation BA {candidate_id}",
        )
        for path, recorded in (
            (checkpoint_path, row["checkpoint_sha256"]),
            (prediction_path, row["prediction_sha256"]),
        ):
            if not path.is_file() or sha256(path) != recorded:
                raise ValueError(f"Permutation artifact hash mismatch: {path}")
        prediction = pd.read_parquet(prediction_path)
        if (
            not np.isfinite(prediction[["logit", "probability_h"]]).all().all()
            or not prediction["outer_test_evaluated"].eq(False).all()
            or prediction.groupby("reader_id")["target_goal_index"].nunique().gt(1).any()
        ):
            raise ValueError(f"Invalid permutation prediction: {candidate_id}")
        expected_permutation_metrics.add(metric_path.resolve())
        expected_permutation_checkpoints.add(checkpoint_path.resolve())
        expected_permutation_predictions.add(prediction_path.resolve())
        permutation_records.append(_hash_record(root, metric_path))
    if {
        path.resolve() for path in (output_root / "sanity").glob("*/metrics.json")
    } != expected_permutation_metrics:
        raise ValueError("Permutation metric file set is not exact")
    if {
        path.resolve() for path in (root / "research_storage/permutation").glob("*/checkpoint.pt")
    } != expected_permutation_checkpoints:
        raise ValueError("Permutation checkpoint file set is not exact")
    if {
        path.resolve() for path in (root / "research_storage/permutation").glob("*/predictions.parquet")
    } != expected_permutation_predictions:
        raise ValueError("Permutation prediction file set is not exact")

    cpu_families = {row["family"] for row in cpu["measurements"]}
    if (
        cpu_families != {"b1", "d", "a"}
        or int(cpu["torch_threads"]) != 1
        or int(cpu["warmup_iterations"]) != int(config["benchmark"]["warmup"])
        or int(cpu["measured_iterations"]) != int(config["benchmark"]["iterations"])
    ):
        raise ValueError("CPU benchmark protocol mismatch")
    cpu_state_records: list[dict[str, Any]] = []
    expected_cpu_states: set[Path] = set()
    for row in cpu["measurements"]:
        family = str(row["family"])
        checkpoint_path = Path(row["checkpoint_path"])
        state_path = Path(row["serialized_state_path"])
        if (
            row["checkpoint_sha256"] not in selected_checkpoint_hashes[family]
            or sha256(checkpoint_path) != row["checkpoint_sha256"]
            or sha256(state_path) != row["serialized_state_sha256"]
            or state_path.stat().st_size != int(row["serialized_state_bytes"])
            or int(row["parameter_count"]) != next(iter(parameter_counts[family]))
            or row["disk_io_and_parsing_timed"] is not False
            or row["end_to_end_trial_ap1"] is not None
            or row["end_to_end_trial_ap2"] is not None
        ):
            raise ValueError(f"CPU benchmark evidence mismatch: {family}")
        expected_cpu_states.add(state_path.resolve())
        cpu_state_records.append(_hash_record(root, state_path))
    if {
        path.resolve() for path in (root / "research_storage/cpu_benchmarks").glob("*/state_dict.pt")
    } != expected_cpu_states:
        raise ValueError("CPU state file set is not exact")

    parameter_scan = reader_parameter_scan(int(roberta["maximum_words"]))
    if parameter_scan["status"] != "pass":
        raise ValueError("Reader-identity parameter scan failed")
    for family, names in checkpoint_parameter_names.items():
        if any(
            token in name.casefold()
            for name in names
            for token in FORBIDDEN_PARAMETER_TOKENS
        ):
            raise ValueError(f"Saved checkpoint identity scan failed: {family}")

    family_metrics, _, readers, draws = _aggregate_selected(
        selected_metrics, cluster_stats, config
    )
    contrasts = {
        "teacher_minus_b1": {
            "gate_summary": gates["T"]["teacher_minus_b1"],
            "reader_cluster_bootstrap": reader_cluster_contrast(
                cluster_stats[("teacher", "FULL")],
                cluster_stats[("b1", "FULL")],
                readers,
                draws,
                seed=BOOTSTRAP_SEED,
            ),
        },
        "t_text_minus_b1": {
            "gate_summary": gates["T"]["t_text_minus_b1"],
            "reader_cluster_bootstrap": reader_cluster_contrast(
                cluster_stats[("t_text", "FULL")],
                cluster_stats[("b1", "FULL")],
                readers,
                draws,
                seed=BOOTSTRAP_SEED,
            ),
        },
        "d_minus_b1": {
            "gate_summary": gates["D"]["d_minus_b1"],
            "reader_cluster_bootstrap": reader_cluster_contrast(
                cluster_stats[("d", "FULL")],
                cluster_stats[("b1", "FULL")],
                readers,
                draws,
                seed=BOOTSTRAP_SEED,
            ),
        },
        "a_full_minus_d": {
            "gate_summary": gates["A0"]["a_full_minus_d"],
            "reader_cluster_bootstrap": reader_cluster_contrast(
                cluster_stats[("a", "FULL")],
                cluster_stats[("d", "FULL")],
                readers,
                draws,
                seed=BOOTSTRAP_SEED,
            ),
        },
    }

    metric_records.sort(key=lambda row: row["path"])
    checkpoint_records.sort(key=lambda row: row["path"])
    prediction_records.sort(key=lambda row: row["path"])
    teacher_cache_records.sort(key=lambda row: row["path"])
    artifact_set_hashes = {
        "metrics": canonical_json_sha256(metric_records),
        "checkpoints": canonical_json_sha256(checkpoint_records),
        "predictions": canonical_json_sha256(prediction_records),
        "teacher_caches": canonical_json_sha256(teacher_cache_records),
    }
    metrics_value = {
        "schema": "goalexit-development-metrics-v1",
        "status": "complete",
        "phase": "0-2 development only",
        "generated_by_git_commit": commit,
        "outer_test_evaluated": False,
        "canonical_data_sha256": canonical["canonical_data_sha256"],
        "config_sha256": config_hash,
        "split_manifest_sha256": split_hash,
        "canonical_population": canonical["observed"],
        "split": {
            "seed": int(split["seed"]),
            "contexts": 25,
            "paired_inner_evaluations": 75,
        },
        "bootstrap": {
            "replicates": int(config["bootstrap_replicates"]),
            "seed": BOOTSTRAP_SEED,
            "cluster_unit": "reader",
            "stratified_by_reader_goal": True,
        },
        "families": family_metrics,
        "parameter_counts": {
            family: next(iter(parameter_counts[family])) for family in FAMILIES
        },
        "candidate_evaluations": dict(sorted(candidate_counts.items())),
        "contrasts": contrasts,
        "gates": gates,
        "adaptive_stopping": {
            "status": "not_run",
            "reason": "Gate A0 failed; adaptive stopping was not authorized",
            "AP1": None,
            "AP2": None,
        },
        "sanity": {
            "reader_label_permutation": permutation,
            "metadata_probe": {
                "status": metadata_probe["status"],
                "mean_balanced_accuracy": metadata_probe["mean_balanced_accuracy"],
                "maximum_balanced_accuracy": metadata_probe["maximum_balanced_accuracy"],
            },
            "t_text_shortcut_control": {
                "status": "pass",
                "mean_t_text_minus_b1": gates["T"]["t_text_minus_b1"]["mean_difference"],
                "maximum_allowed_gain": 0.5
                * gates["T"]["teacher_minus_b1"]["mean_difference"],
            },
            "reader_identity_parameter_scan": parameter_scan,
        },
        "cpu_benchmark": cpu,
        "artifact_counts": {
            "candidate_results": len(metric_records),
            "checkpoints": len(checkpoint_records),
            "predictions": len(prediction_records),
            "selected_evaluations": 75 * len(FAMILIES),
            "selection_files": len(expected_selections),
            "teacher_caches": len(teacher_cache_records),
            "permutation_evaluations": len(permutation_records),
            "cpu_state_dicts": len(cpu_state_records),
            "partial_training_states": 0,
            "adaptive_traces": 0,
        },
        "artifact_set_hashes": artifact_set_hashes,
        "outer_guard": outer_guard,
        "decision": "STOP pending researcher review because Gate A0 failed",
    }
    metrics_bytes = _json_bytes(metrics_value)
    metrics_sha = _bytes_sha256(metrics_bytes)
    teacher_bytes = _json_bytes(teacher_cache)
    teacher_sha = _bytes_sha256(teacher_bytes)
    audit_value = {
        "schema": "goalexit-development-output-audit-v1",
        "status": "pass",
        "generated_by_git_commit": commit,
        "outer_test_evaluated": False,
        "authorized_path": "Gate T PASS -> Gate D PASS -> Gate A0 FAIL -> STOP before adaptive",
        "canonical_data_sha256": canonical["canonical_data_sha256"],
        "config_sha256": config_hash,
        "split_manifest_sha256": split_hash,
        "roberta_revision": roberta["resolved_revision"],
        "roberta_cache_sha256": roberta["array_sha256"],
        "development_model_commits": {
            family: sorted(model_commits[family]) for family in FAMILIES
        },
        "authorized_candidate_ids_by_context": {
            context_id: {
                family: sorted(_candidate_settings(config, state, context_id, family))
                for family in FAMILIES
            }
            for context_id in sorted(context_map)
        },
        "checks": {
            "exact_candidate_grid": "pass",
            "exact_file_sets": "pass",
            "all_artifact_hashes": "pass",
            "checkpoint_load_and_finite_state": "pass",
            "checkpoint_run_identity": "pass",
            "prediction_provenance": "pass",
            "prediction_metrics_recomputed": "pass",
            "target_input_membership": "pass",
            "nonfinite_predictions": "pass",
            "prefix_causality": "pass",
            "training_only_preprocessing": "pass",
            "reader_identity_parameters": "pass",
            "teacher_cache_provenance": "pass",
            "reader_level_permutation": "pass",
            "cpu_benchmark": "pass",
            "adaptive_artifacts_absent_after_a0_fail": "pass",
            "outer_evaluation_guard": "pass",
            "unauthorized_seeds_37_71_absent": "pass",
        },
        "counts": metrics_value["artifact_counts"],
        "artifact_set_hashes": artifact_set_hashes,
        "files": {
            "metrics": metric_records,
            "checkpoints": checkpoint_records,
            "predictions": prediction_records,
            "teacher_caches": teacher_cache_records,
            "permutation_metrics": sorted(permutation_records, key=lambda row: row["path"]),
            "cpu_state_dicts": sorted(cpu_state_records, key=lambda row: row["path"]),
        },
        "terminal_manifests": {
            "development_metrics": {
                "path": _relative(root, targets["metrics"]),
                "sha256": metrics_sha,
            },
            "teacher_cache": {
                "path": _relative(root, targets["teacher_cache"]),
                "sha256": teacher_sha,
            },
            "reader_label_permutation": _hash_record(root, permutation_path),
            "cpu_benchmark": _hash_record(root, cpu_path),
            "outer_guard": _hash_record(root, outer_guard_path),
            "split_manifest": _hash_record(root, split_path),
        },
        "final_lock_exists": False,
        "outer_scientific_predictions_created": False,
        "outer_scientific_metrics_computed": False,
    }
    audit_bytes = _json_bytes(audit_value)
    audit_sha = _bytes_sha256(audit_bytes)
    report = _render_report(metrics_value, audit_sha, metrics_sha).encode()
    _atomic_write(targets["teacher_cache"], teacher_bytes)
    _atomic_write(targets["metrics"], metrics_bytes)
    _atomic_write(targets["audit"], audit_bytes)
    _atomic_write(targets["report"], report)
    print(
        f"status=pass candidates={len(metric_records)} selected={75 * len(FAMILIES)} "
        f"gate_t={gates['T']['status']} gate_d={gates['D']['status']} "
        f"gate_a0={gates['A0']['status']} gate_e={gates['E']['status']}"
    )
    return {
        "status": "pass",
        "metrics_sha256": metrics_sha,
        "audit_sha256": audit_sha,
        "report_sha256": sha256(targets["report"]),
        "teacher_cache_manifest_sha256": teacher_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
