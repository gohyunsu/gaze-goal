from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from goalexit.utils import canonical_json_sha256, load_yaml, project_root, resolve, save_json, sha256


ASSIGNMENT_COLUMNS = ["trial_id", "reader_id", "article_id", "goal_index"]


def _stratified_reader_map(
    readers: pd.DataFrame, rng: np.random.Generator, folds: int
) -> dict[str, int]:
    if readers["reader_id"].duplicated().any():
        raise ValueError("Reader assignment table is not unique")
    result: dict[str, int] = {}
    for goal in (0, 1):
        values = np.asarray(
            sorted(readers.loc[readers["goal_index"].eq(goal), "reader_id"].astype(str)),
            dtype=object,
        )
        rng.shuffle(values)
        result.update({str(value): int(index % folds) for index, value in enumerate(values)})
    if len(result) != len(readers):
        raise ValueError("Reader lacks a unique goal stratum")
    return result


def _group_map(values: pd.Series, rng: np.random.Generator, folds: int) -> dict[str, int]:
    groups = np.asarray(sorted(values.astype(str).unique()), dtype=object)
    rng.shuffle(groups)
    return {str(group): int(index % folds) for index, group in enumerate(groups)}


def _inner_seed(seed: int, outer_reader: int, outer_article: int) -> int:
    return int(np.random.SeedSequence([seed, outer_reader, outer_article]).generate_state(1)[0])


def construct_manifest(
    metadata: pd.DataFrame, *, seed: int, outer_folds: int = 5, inner_folds: int = 3
) -> dict[str, Any]:
    if list(metadata.columns) != ASSIGNMENT_COLUMNS:
        raise ValueError(f"Assignment frame must be exactly {ASSIGNMENT_COLUMNS}")
    if metadata["trial_id"].duplicated().any():
        raise ValueError("Trial IDs are not unique")
    reader_goal = metadata[["reader_id", "goal_index"]].drop_duplicates()
    if reader_goal.groupby("reader_id")["goal_index"].nunique().ne(1).any():
        raise ValueError("Reader goal is not constant")
    outer_rng = np.random.default_rng(seed)
    reader_map = _stratified_reader_map(reader_goal, outer_rng, outer_folds)
    article_map = _group_map(metadata["article_id"], outer_rng, outer_folds)
    outer_reader_assignment = metadata["reader_id"].astype(str).map(reader_map)
    outer_article_assignment = metadata["article_id"].astype(str).map(article_map)
    reader_balance = {
        str(fold): {
            str(goal): int(
                reader_goal.loc[
                    reader_goal["reader_id"].astype(str).map(reader_map).eq(fold)
                    & reader_goal["goal_index"].eq(goal)
                ].shape[0]
            )
            for goal in (0, 1)
        }
        for fold in range(outer_folds)
    }
    contexts: list[dict[str, Any]] = []
    for outer_reader in range(outer_folds):
        for outer_article in range(outer_folds):
            population_mask = outer_reader_assignment.ne(outer_reader) & outer_article_assignment.ne(outer_article)
            outer_mask = outer_reader_assignment.eq(outer_reader) & outer_article_assignment.eq(outer_article)
            population = metadata.loc[population_mask].copy()
            outer_target = metadata.loc[outer_mask].copy()
            context_seed = _inner_seed(seed, outer_reader, outer_article)
            inner_rng = np.random.default_rng(context_seed)
            inner_reader_map = _stratified_reader_map(
                population[["reader_id", "goal_index"]].drop_duplicates(),
                inner_rng,
                inner_folds,
            )
            inner_article_map = _group_map(population["article_id"], inner_rng, inner_folds)
            reader_inner = population["reader_id"].astype(str).map(inner_reader_map)
            article_inner = population["article_id"].astype(str).map(inner_article_map)
            holdouts: list[dict[str, Any]] = []
            for inner in range(inner_folds):
                training = population.loc[reader_inner.ne(inner) & article_inner.ne(inner)]
                validation = population.loc[reader_inner.eq(inner) & article_inner.eq(inner)]
                if training.empty or validation.empty:
                    raise ValueError("Empty inner split")
                if set(training["reader_id"]) & set(validation["reader_id"]):
                    raise ValueError("Inner reader leakage")
                if set(training["article_id"]) & set(validation["article_id"]):
                    raise ValueError("Inner article leakage")
                if set(training["trial_id"]) & set(outer_target["trial_id"]):
                    raise ValueError("Outer target in inner training")
                if set(validation["trial_id"]) & set(outer_target["trial_id"]):
                    raise ValueError("Outer target in inner validation")
                holdouts.append(
                    {
                        "inner_holdout": inner,
                        "training_trials": len(training),
                        "validation_trials": len(validation),
                        "training_readers": int(training["reader_id"].nunique()),
                        "validation_readers": int(validation["reader_id"].nunique()),
                        "training_articles": int(training["article_id"].nunique()),
                        "validation_articles": int(validation["article_id"].nunique()),
                        "training_trial_ids_sha256": canonical_json_sha256(sorted(training["trial_id"].astype(str))),
                        "validation_trial_ids_sha256": canonical_json_sha256(sorted(validation["trial_id"].astype(str))),
                    }
                )
            contexts.append(
                {
                    "context_id": f"r{outer_reader}_a{outer_article}",
                    "outer_reader_fold": outer_reader,
                    "outer_article_fold": outer_article,
                    "inner_seed": context_seed,
                    "population_training_trials": len(population),
                    "outer_target_trials_sealed": len(outer_target),
                    "outer_target_trial_ids_sha256": canonical_json_sha256(sorted(outer_target["trial_id"].astype(str))),
                    "inner_reader_map": inner_reader_map,
                    "inner_article_map": inner_article_map,
                    "inner_holdouts": holdouts,
                }
            )
    return {
        "schema": "goalexit-crossed-splits-v1",
        "status": "pass",
        "seed": seed,
        "algorithm": "goal-stratified reader folds and shuffled whole-article folds; context-seeded three-fold paired diagonal inner cells",
        "outer_reader_folds": outer_folds,
        "outer_article_folds": outer_folds,
        "outer_contexts": len(contexts),
        "inner_holdouts_per_context": inner_folds,
        "development_evaluations": len(contexts) * inner_folds,
        "outer_reader_map": reader_map,
        "outer_article_map": article_map,
        "outer_reader_goal_balance": reader_balance,
        "contexts": contexts,
        "outer_target_outcomes_materialized_in_manifest": False,
        "outer_scientific_metrics_computed": False,
        "outer_scientific_predictions_created": False,
        "assignment_fields": ASSIGNMENT_COLUMNS,
        "assertions": {
            "outer_reader_goal_stratified": True,
            "whole_article_grouping": True,
            "outer_target_is_reader_article_intersection": True,
            "population_excludes_held_readers_and_articles": True,
            "inner_reader_disjoint": True,
            "inner_article_disjoint": True,
            "outer_target_labels_not_summarized": True,
        },
    }


def _development_support_audit(
    metadata: pd.DataFrame, manifest: dict[str, Any], trial_path: Path
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    for context in manifest["contexts"]:
        ro = int(context["outer_reader_fold"])
        ao = int(context["outer_article_fold"])
        reader_outer = metadata["reader_id"].astype(str).map(manifest["outer_reader_map"])
        article_outer = metadata["article_id"].astype(str).map(manifest["outer_article_map"])
        population = metadata.loc[reader_outer.ne(ro) & article_outer.ne(ao)].copy()
        outer_ids = set(metadata.loc[reader_outer.eq(ro) & article_outer.eq(ao), "trial_id"].astype(str))
        population_ids = population["trial_id"].astype(str).tolist()
        if set(population_ids) & outer_ids:
            raise PermissionError("Development support intersects sealed outer target")
        targets = pq.read_table(
            trial_path,
            columns=["trial_id", "goal_index"],
            filters=[("trial_id", "in", sorted(population_ids))],
        ).to_pandas()
        population = population.drop(columns=["goal_index"]).merge(
            targets, on="trial_id", validate="one_to_one"
        )
        ri = population["reader_id"].astype(str).map(context["inner_reader_map"])
        ai = population["article_id"].astype(str).map(context["inner_article_map"])
        for inner in range(3):
            training = population.loc[ri.ne(inner) & ai.ne(inner)]
            validation = population.loc[ri.eq(inner) & ai.eq(inner)]
            train_counts = {str(goal): int(training["goal_index"].eq(goal).sum()) for goal in (0, 1)}
            validation_counts = {str(goal): int(validation["goal_index"].eq(goal).sum()) for goal in (0, 1)}
            if min(train_counts.values()) == 0 or min(validation_counts.values()) == 0:
                raise ValueError("Development split lacks a target class")
            evaluations.append(
                {
                    "context_id": context["context_id"],
                    "inner_holdout": inner,
                    "training_class_counts": train_counts,
                    "validation_class_counts": validation_counts,
                }
            )
    return {
        "target_access_scope": "development population and inner validation only",
        "outer_target_class_counts_computed": False,
        "all_development_splits_have_both_classes": True,
        "evaluations": evaluations,
    }


def build(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, config_path)
    config = load_yaml(config_path)
    trial_path = resolve(root, config["outputs"]["trial_table"])
    assignment = pd.read_parquet(trial_path, columns=ASSIGNMENT_COLUMNS)
    manifest = construct_manifest(
        assignment,
        seed=int(config["split_seed"]),
        outer_folds=int(config["outer_folds"]),
        inner_folds=int(config["inner_folds"]),
    )
    repeated = construct_manifest(
        assignment.copy(),
        seed=int(config["split_seed"]),
        outer_folds=int(config["outer_folds"]),
        inner_folds=int(config["inner_folds"]),
    )
    if canonical_json_sha256(manifest) != canonical_json_sha256(repeated):
        raise AssertionError("Split construction is not deterministic")
    manifest["development_support_audit"] = _development_support_audit(
        assignment, manifest, trial_path
    )
    canonical_path = resolve(root, config["outputs"]["canonical_manifest"])
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    manifest["canonical_data_sha256"] = canonical["canonical_data_sha256"]
    manifest["trial_table_sha256"] = sha256(trial_path)
    manifest["mapping_sha256"] = canonical_json_sha256(
        {
            "outer_reader_map": manifest["outer_reader_map"],
            "outer_article_map": manifest["outer_article_map"],
            "contexts": [
                {
                    "context_id": context["context_id"],
                    "inner_seed": context["inner_seed"],
                    "inner_reader_map": context["inner_reader_map"],
                    "inner_article_map": context["inner_article_map"],
                }
                for context in manifest["contexts"]
            ],
        }
    )
    output = resolve(root, config["outputs"]["split_manifest"])
    save_json(output, manifest)
    print(f"status=pass contexts={len(manifest['contexts'])} evaluations=75")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
