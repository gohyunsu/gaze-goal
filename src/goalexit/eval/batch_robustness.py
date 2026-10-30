from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


FAMILIES = ("b1", "d")


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def analyze(source_root: Path, output_path: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    config_path = source_root / "configs/confirmatory_kd.yaml"
    config = load_yaml(config_path)
    dev_config = load_yaml(resolve(source_root, config["development_config"]))
    split_path = resolve(source_root, config["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    trial_path = resolve(source_root, dev_config["outputs"]["trial_table"])
    labels = pd.read_parquet(
        trial_path,
        columns=["trial_id", "reader_id", "article_id", "goal_index"],
    )
    if labels["trial_id"].duplicated().any():
        raise ValueError("Canonical trial table contains duplicate trial IDs")

    prediction_root = resolve(source_root, config["outputs"]["prediction_root"])
    audit_path = resolve(source_root, config["outputs"]["audit"])
    output_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_prediction_hashes = {
        (str(row["context_id"]), str(row["family"])): str(
            row["prediction_sha256"]
        )
        for row in output_audit["entries"]
        if str(row["family"]) in FAMILIES
    }
    frame = labels.copy()
    expected_ids = set(labels["trial_id"].astype(str))
    for family in FAMILIES:
        pieces: list[pd.DataFrame] = []
        for context in split["contexts"]:
            context_id = str(context["context_id"])
            path = prediction_root / context_id / f"{family}.parquet"
            if sha256(path) != expected_prediction_hashes[(context_id, family)]:
                raise ValueError(f"Prediction hash mismatch: {context_id}/{family}")
            incoming = pd.read_parquet(path, columns=["trial_id", "probability"])
            if incoming["trial_id"].duplicated().any():
                raise ValueError(f"Duplicate predictions: {context_id}/{family}")
            pieces.append(incoming)
        incoming = pd.concat(pieces, ignore_index=True)
        if incoming["trial_id"].duplicated().any() or set(
            incoming["trial_id"].astype(str)
        ) != expected_ids:
            raise ValueError(f"Prediction partition is not exact for {family}")
        frame = frame.merge(
            incoming.rename(columns={"probability": family}),
            on="trial_id",
            validate="one_to_one",
        )

    batch = frame["article_id"].astype(str).str.split("_", n=1).str[0]
    if set(batch) != {"1", "2", "3"}:
        raise ValueError("Unexpected OneStop article-batch identifiers")
    frame["article_batch"] = batch

    per_batch: dict[str, Any] = {}
    for batch_id, rows in frame.groupby("article_batch", sort=True):
        target = rows["goal_index"].to_numpy(np.int64)
        b1 = _balanced_accuracy(target, rows["b1"].to_numpy(np.float64))
        distilled = _balanced_accuracy(target, rows["d"].to_numpy(np.float64))
        per_batch[str(batch_id)] = {
            "trials": int(len(rows)),
            "readers": int(rows["reader_id"].nunique()),
            "articles": int(rows["article_id"].nunique()),
            "gaze_gru_balanced_accuracy": b1,
            "distilled_gru_balanced_accuracy": distilled,
            "distilled_minus_gaze_gru": distilled - b1,
        }

    metrics_path = resolve(source_root, config["outputs"]["metrics"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pooled_target = frame["goal_index"].to_numpy(np.int64)
    pooled = {
        family: _balanced_accuracy(
            pooled_target, frame[family].to_numpy(np.float64)
        )
        for family in FAMILIES
    }
    for family in FAMILIES:
        expected = float(metrics["pooled_metrics"][family]["balanced_accuracy"])
        if not np.isclose(pooled[family], expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"Pooled metric mismatch for {family}")

    result = {
        "schema": "goalexit-outer-kd-batch-robustness-v1",
        "status": "complete",
        "post_hoc": True,
        "descriptive_only": True,
        "trial_table_sha256": sha256(trial_path),
        "split_manifest_sha256": sha256(split_path),
        "confirmatory_metrics_sha256": sha256(metrics_path),
        "outer_output_audit_sha256": sha256(audit_path),
        "prediction_set_verified_against_outer_output_audit": True,
        "pooled": {
            "gaze_gru_balanced_accuracy": pooled["b1"],
            "distilled_gru_balanced_accuracy": pooled["d"],
            "distilled_minus_gaze_gru": pooled["d"] - pooled["b1"],
        },
        "per_article_batch": per_batch,
        "all_batch_differences_positive": all(
            row["distilled_minus_gaze_gru"] > 0.0 for row in per_batch.values()
        ),
    }
    save_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=project_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/outer_kd_batch_robustness.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else project_root() / args.output
    result = analyze(args.source_root, output)
    print(
        "all_batch_differences_positive="
        f"{result['all_batch_differences_positive']}"
    )


if __name__ == "__main__":
    main()
