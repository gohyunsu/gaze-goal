from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from goalexit.data.cache_roberta import load_cache
from goalexit.data.dataset import GoalExitStore, context_positions, load_allowed_targets, load_split
from goalexit.train.engine import train_candidate
from goalexit.utils import load_yaml, project_root, resolve


def _balanced_subset(
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    articles: int,
    per_article_goal: int,
) -> list[str]:
    frame = targets.merge(
        metadata[["trial_id", "article_id"]], on="trial_id", validate="one_to_one"
    )
    coverage = frame.groupby("article_id")["goal_index"].nunique()
    selected_articles = sorted(coverage.loc[coverage.eq(2)].index.astype(str))[:articles]
    frame = frame.loc[frame["article_id"].astype(str).isin(selected_articles)]
    return (
        frame.groupby(["article_id", "goal_index"], group_keys=False)
        .head(per_article_goal)["trial_id"]
        .astype(str)
        .tolist()
    )


def main() -> None:
    root = project_root()
    config_path = root / "configs/development_goalexit.yaml"
    config = load_yaml(config_path)
    split = load_split(resolve(root, config["outputs"]["split_manifest"]))
    trial_path = resolve(root, config["outputs"]["trial_table"])
    store = GoalExitStore.load(
        trial_path,
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    context = split["contexts"][0]
    training, validation, outer = context_positions(store.metadata, split, context, 0)
    train_target_all = load_allowed_targets(trial_path, training, outer)
    validation_target_all = load_allowed_targets(trial_path, validation, outer)
    metadata = store.metadata.reset_index(drop=True)
    training = _balanced_subset(
        train_target_all, metadata, articles=8, per_article_goal=12
    )
    validation = _balanced_subset(
        validation_target_all, metadata, articles=6, per_article_goal=8
    )
    train_target = train_target_all.set_index("trial_id").loc[training].reset_index()
    validation_target = validation_target_all.set_index("trial_id").loc[validation].reset_index()
    validation_metadata = store.metadata.loc[validation].reset_index(drop=True)
    text_cache, text_index = load_cache(
        resolve(root, config["roberta"]["cache_array"]),
        resolve(root, config["roberta"]["cache_ids"]),
    )
    smoke_root = root / "research_storage/smoke"
    outcomes = {}
    teacher_logits = {key: float(index % 3 - 1) for index, key in enumerate(training)}
    for family in ("b0", "b1", "teacher", "t_text", "d", "a"):
        run = smoke_root / family
        result = train_candidate(
            family=family,
            training_ids=training,
            validation_ids=validation,
            training_targets=train_target,
            validation_targets=validation_target,
            validation_metadata=validation_metadata,
            store=store,
            learning_rate=3e-4,
            seed=13,
            weight_decay=1e-4,
            maximum_epochs=1,
            patience=1,
            minimum_delta=0.0,
            batch_size=64 if family not in {"teacher", "t_text"} else 16,
            gradient_clip=1.0,
            dropout=0.2 if family not in {"teacher", "t_text"} else 0.1,
            ece_bins=15,
            device=torch.device("cpu"),
            run_directory=run / "run",
            checkpoint_path=run / "checkpoint.pt",
            prediction_path=run / "predictions.parquet",
            run_config={
                "git_commit": "SMOKE",
                "config_sha256": "SMOKE",
                "canonical_data_sha256": "SMOKE",
                "split_manifest_sha256": "SMOKE",
                "context_id": "SMOKE",
                "inner_holdout": 0,
                "model_family": family,
                "candidate_id": f"smoke_{family}",
                "seed": 13,
                "learning_rate": 3e-4,
                "dropout": 0.2 if family not in {"teacher", "t_text"} else 0.1,
                "run_identity_sha256": f"SMOKE_{family}",
            },
            text_cache=text_cache if family in {"teacher", "t_text"} else None,
            text_index=text_index if family in {"teacher", "t_text"} else None,
            teacher_training_logits=teacher_logits if family in {"d", "a"} else None,
            alpha=0.5 if family in {"d", "a"} else None,
            temperature=2.0,
        )
        outcomes[family] = {
            "prediction_rows": result["prediction_rows"],
            "full_balanced_accuracy": result["metrics"]["FULL"]["balanced_accuracy"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        }
        print(f"smoke complete family={family}", flush=True)
    (smoke_root / "summary.json").write_text(
        json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
