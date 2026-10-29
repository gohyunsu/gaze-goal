from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from goalexit.data.dataset import (
    GoalExitStore,
    context_positions,
    load_split,
    outer_context_positions,
)
from goalexit.eval.metrics import evaluate_binary
from goalexit.eval.outer_analysis import _crossed_bootstrap, _reader_cluster_bootstrap
from goalexit.features.schema import B0_FEATURES
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


BASELINES = ("dwell_time_lr", "gaze_summary_lr")


def _feature_matrix(
    store: GoalExitStore, ids: list[str], baseline: str
) -> np.ndarray:
    if baseline == "dwell_time_lr":
        # Total fixation dwell divided by paragraph words mirrors the simple
        # time-per-word baseline while using only canonical, retained events.
        values = [
            store.metadata.loc[key, "full_fixation_duration_ms"]
            / len(store.teacher_gaze[key])
            for key in ids
        ]
        return np.log1p(np.asarray(values, dtype=np.float64))[:, None]
    if baseline == "gaze_summary_lr":
        return np.stack([store.b0[key]["FULL"] for key in ids]).astype(np.float64)
    raise ValueError(f"Unknown baseline {baseline}")


def _fit(
    x: np.ndarray, y: np.ndarray, *, c_value: float, seed: int
) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    model.fit(x, y)
    return model


def _balanced_accuracy(target: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    return float(
        np.mean([np.mean(prediction[target == label] == label) for label in (0, 1)])
    )


def _load_primary_predictions(
    root: Path,
    split: dict[str, Any],
    store: GoalExitStore,
) -> dict[str, pd.DataFrame]:
    result: dict[str, list[pd.DataFrame]] = {"b0": [], "b1": [], "d": []}
    for context in split["contexts"]:
        context_id = str(context["context_id"])
        _, target_ids = outer_context_positions(store.metadata, split, context)
        for family in result:
            path = (
                root
                / "research_storage/predictions/outer_kd"
                / context_id
                / f"{family}.parquet"
            )
            frame = pd.read_parquet(path)
            probability_column = (
                "probability_h" if "probability_h" in frame.columns else "probability"
            )
            frame = frame.loc[
                frame.get("prefix", "FULL").eq("FULL")
                if "prefix" in frame.columns
                else np.ones(len(frame), dtype=bool),
                ["trial_id", probability_column],
            ].rename(columns={probability_column: family})
            if frame["trial_id"].duplicated().any() or set(frame["trial_id"].astype(str)) != set(
                target_ids
            ):
                raise ValueError(f"Primary prediction membership mismatch: {context_id}/{family}")
            result[family].append(frame)
    return {
        family: pd.concat(pieces, ignore_index=True)
        for family, pieces in result.items()
    }


def run(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    if not config.get("outer_results_known_when_designed"):
        raise ValueError("Controls must be marked post hoc")
    dev_config = load_yaml(resolve(root, config["development_config"]))
    split = load_split(resolve(root, config["split_manifest"]))
    trial_path = resolve(root, dev_config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    labels = pd.read_parquet(
        trial_path,
        columns=["trial_id", "goal_index", "reader_id", "article_id"],
    )
    target_map = labels.set_index("trial_id")["goal_index"].astype(int).to_dict()
    article_map = labels.set_index("trial_id")["article_id"].astype(str).to_dict()
    seed = int(config["random_seed"])
    output_root = resolve(root, config["outputs"]["predictions"])
    selections: list[dict[str, Any]] = []
    context_metrics: list[dict[str, Any]] = []
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in BASELINES}

    for context in split["contexts"]:
        context_id = str(context["context_id"])
        outer_training, outer_target = outer_context_positions(store.metadata, split, context)
        context_row: dict[str, Any] = {
            "context_id": context_id,
            "trials": len(outer_target),
        }
        for baseline in BASELINES:
            candidates: list[dict[str, Any]] = []
            for c_value in map(float, config["candidates"]):
                scores: list[float] = []
                nll_values: list[float] = []
                for inner in range(int(config["inner_folds"])):
                    training_ids, validation_ids, _ = context_positions(
                        store.metadata, split, context, inner
                    )
                    model = _fit(
                        _feature_matrix(store, training_ids, baseline),
                        np.asarray([target_map[key] for key in training_ids]),
                        c_value=c_value,
                        seed=seed,
                    )
                    probability = model.predict_proba(
                        _feature_matrix(store, validation_ids, baseline)
                    )[:, 1]
                    target = np.asarray([target_map[key] for key in validation_ids])
                    metrics = evaluate_binary(
                        target,
                        probability,
                        np.asarray([article_map[key] for key in validation_ids]),
                    )
                    scores.append(metrics.balanced_accuracy)
                    nll_values.append(metrics.nll)
                candidates.append(
                    {
                        "c": c_value,
                        "mean_inner_balanced_accuracy": float(np.mean(scores)),
                        "mean_inner_nll": float(np.mean(nll_values)),
                    }
                )
            selected = max(
                candidates,
                key=lambda row: (
                    row["mean_inner_balanced_accuracy"],
                    -row["mean_inner_nll"],
                    -row["c"],
                ),
            )
            model = _fit(
                _feature_matrix(store, outer_training, baseline),
                np.asarray([target_map[key] for key in outer_training]),
                c_value=float(selected["c"]),
                seed=seed,
            )
            probability = model.predict_proba(
                _feature_matrix(store, outer_target, baseline)
            )[:, 1]
            frame = pd.DataFrame(
                {"trial_id": outer_target, "probability": probability}
            )
            path = output_root / context_id / f"{baseline}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, compression="zstd")
            pieces[baseline].append(frame)
            local_target = np.asarray([target_map[key] for key in outer_target])
            context_row[baseline] = _balanced_accuracy(local_target, probability)
            selections.append(
                {
                    "context_id": context_id,
                    "baseline": baseline,
                    "selected_c": selected["c"],
                    "candidates": candidates,
                    "prediction_sha256": sha256(path),
                }
            )
        context_metrics.append(context_row)

    frame = labels.copy()
    for baseline, baseline_pieces in pieces.items():
        incoming = pd.concat(baseline_pieces, ignore_index=True).rename(
            columns={"probability": baseline}
        )
        if incoming["trial_id"].duplicated().any() or len(incoming) != len(labels):
            raise ValueError(f"Outer predictions are not an exact partition: {baseline}")
        frame = frame.merge(incoming, on="trial_id", validate="one_to_one")
    for family, incoming in _load_primary_predictions(root, split, store).items():
        frame = frame.merge(incoming, on="trial_id", validate="one_to_one")
    trial_to_context: dict[str, str] = {}
    for context in split["contexts"]:
        _, target_ids = outer_context_positions(store.metadata, split, context)
        trial_to_context.update(
            {trial_id: str(context["context_id"]) for trial_id in target_ids}
        )
    frame["context_id"] = frame["trial_id"].map(trial_to_context)
    if frame["context_id"].isna().any():
        raise ValueError("A trial is absent from the outer-context partition")
    target = frame["goal_index"].to_numpy(np.int64)
    articles = frame["article_id"].astype(str).to_numpy()
    pooled = {
        name: evaluate_binary(target, frame[name].to_numpy(), articles).to_dict()
        for name in (*BASELINES, "b0", "b1", "d")
    }
    contrasts: dict[str, Any] = {}
    pairs = (
        ("d_minus_dwell_time_lr", "d", "dwell_time_lr"),
        ("d_minus_gaze_summary_lr", "d", "gaze_summary_lr"),
        ("gaze_summary_lr_minus_b1", "gaze_summary_lr", "b1"),
    )
    for index, (name, left, right) in enumerate(pairs):
        context_differences = np.asarray(
            [
                _balanced_accuracy(
                    rows["goal_index"].to_numpy(np.int64),
                    rows[left].to_numpy(),
                )
                - _balanced_accuracy(
                    rows["goal_index"].to_numpy(np.int64),
                    rows[right].to_numpy(),
                )
                for _, rows in frame.groupby("context_id", sort=True)
            ]
        )
        contrasts[name] = {
            "left": left,
            "right": right,
            "pooled_difference": pooled[left]["balanced_accuracy"]
            - pooled[right]["balanced_accuracy"],
            "positive_contexts": int((context_differences > 0).sum()),
            "contexts": len(context_differences),
            "reader_cluster_bootstrap": _reader_cluster_bootstrap(
                frame,
                left,
                right,
                replicates=int(config["bootstrap"]["reader_cluster_replicates"]),
                seed=int(config["bootstrap"]["seed"]) + index,
            ),
            "reader_article_crossed_bootstrap": _crossed_bootstrap(
                frame,
                left,
                right,
                replicates=int(config["bootstrap"]["reader_article_replicates"]),
                seed=int(config["bootstrap"]["seed"]) + 100 + index,
            ),
        }

    result = {
        "schema": "goalexit-postfeedback-controls-v1",
        "status": "complete",
        "post_hoc": True,
        "feature_definitions": {
            "dwell_time_lr": "log1p(total retained fixation dwell / paragraph word count)",
            "gaze_summary_lr": B0_FEATURES,
        },
        "selection": "C selected by mean balanced accuracy over the frozen three inner reader/article holdouts; NLL tie-break",
        "selections": selections,
        "pooled": pooled,
        "context_metrics": context_metrics,
        "contrasts": contrasts,
    }
    save_json(resolve(root, config["outputs"]["metrics"]), result)
    report = _report(result)
    report_path = resolve(root, config["outputs"]["report"])
    report_path.write_text(report, encoding="utf-8")
    return result


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Post hoc simple-baseline results",
        "",
        "**Status:** complete; post hoc.",
        "",
        "Both baselines use the frozen reader/article partitions and nested inner selection.",
        "",
        "| Model | BA | AUROC | NLL |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "dwell_time_lr": "Dwell time/word LR",
        "gaze_summary_lr": "Gaze-summary LR",
        "b0": "Gaze-summary MLP",
        "b1": "Gaze GRU",
        "d": "Distilled GRU",
    }
    for key in ("dwell_time_lr", "gaze_summary_lr", "b0", "b1", "d"):
        row = result["pooled"][key]
        lines.append(
            f"| {labels[key]} | {row['balanced_accuracy']:.4f} | {row['auroc']:.4f} | {row['nll']:.4f} |"
        )
    lines.extend(["", "| Contrast | Delta BA | Reader 95% CI | Positive contexts |", "|---|---:|---:|---:|"])
    for row in result["contrasts"].values():
        low, high = row["reader_cluster_bootstrap"]["ci_95_percentile"]
        lines.append(
            f"| {labels[row['left']]} - {labels[row['right']]} | {row['pooled_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] | {row['positive_contexts']}/{row['contexts']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/postfeedback_controls.yaml")
    )
    args = parser.parse_args()
    result = run(args.config)
    print(
        " ".join(
            f"{key}_ba={result['pooled'][key]['balanced_accuracy']:.6f}"
            for key in BASELINES
        )
    )


if __name__ == "__main__":
    main()
