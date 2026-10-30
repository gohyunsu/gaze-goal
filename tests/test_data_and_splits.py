from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from goalexit.data.dataset import context_positions, load_allowed_targets, outer_context_positions
from goalexit.features.schema import audit_feature_names
from goalexit.splits.build_splits import ASSIGNMENT_COLUMNS, construct_manifest
from goalexit.utils import canonical_json_sha256, sha256


ROOT = Path(__file__).resolve().parents[1]
TRIAL_TABLE = ROOT / "data/processed/goalexit_trials.parquet"
requires_canonical_data = pytest.mark.skipif(
    not TRIAL_TABLE.is_file(),
    reason="requires locally reconstructed OneStop canonical tables",
)


def test_canonical_population_and_word_alignment_contract() -> None:
    manifest = json.loads((ROOT / "artifacts/manifests/canonical_goalexit.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["observed"] == {
        "articles": 30,
        "fixations": 2_042_764,
        "information_seeking_readers": 180,
        "information_seeking_trials": 9_720,
        "ordinary_readers": 180,
        "ordinary_trials": 9_718,
        "paragraph_versions": 339,
        "readers": 360,
        "trials": 19_438,
    }
    assert manifest["reader_goal_constancy"] is True
    assert manifest["articles_have_both_goals"] is True
    assert manifest["word_alignment_audit"]["teacher_word_alignment_complete"] is True
    assert audit_feature_names()["status"] == "pass"


def test_fold_distribution_audit_is_internally_consistent() -> None:
    audit = json.loads(
        (ROOT / "artifacts/manifests/fold_distribution_audit.json").read_text()
    )
    split = json.loads((ROOT / "artifacts/manifests/split_manifest.json").read_text())
    assert audit["status"] == "pass"
    assert audit["trial_table_sha256"] == split["trial_table_sha256"]
    assert audit["split_mapping_sha256"] == split["mapping_sha256"]
    assert all(audit["assertions"].values())

    for counts_by_batch in audit["reader_fold_batch_goal_counts"].values():
        assert sum(counts["H"] for counts in counts_by_batch.values()) == 36
        assert sum(counts["O"] for counts in counts_by_batch.values()) == 36

    partitions = list(audit["test_partition_trial_counts"].values())
    assert len(partitions) == 25
    assert sum(partition["H"] for partition in partitions) == 9_720
    assert sum(partition["O"] for partition in partitions) == 9_718
    assert sum(partition["total"] for partition in partitions) == 19_438


def test_batch_robustness_artifact_matches_confirmatory_sources() -> None:
    result = json.loads(
        (ROOT / "artifacts/metrics/outer_kd_batch_robustness.json").read_text()
    )
    assert result["status"] == "complete"
    assert result["post_hoc"] is True
    assert result["descriptive_only"] is True
    assert result["prediction_set_verified_against_outer_output_audit"] is True
    assert result["confirmatory_metrics_sha256"] == sha256(
        ROOT / "artifacts/metrics/outer_kd_confirmatory.json"
    )
    assert result["outer_output_audit_sha256"] == sha256(
        ROOT / "artifacts/manifests/outer_kd_output_audit.json"
    )

    batches = result["per_article_batch"]
    assert set(batches) == {"1", "2", "3"}
    assert sum(batch["trials"] for batch in batches.values()) == 19_438
    assert all(batch["readers"] == 120 for batch in batches.values())
    assert all(batch["articles"] == 10 for batch in batches.values())
    assert all(
        batch["distilled_minus_gaze_gru"] > 0.0 for batch in batches.values()
    )
    assert result["all_batch_differences_positive"] is True


@requires_canonical_data
def test_splits_are_deterministic_stratified_and_group_disjoint() -> None:
    trials = pd.read_parquet(TRIAL_TABLE, columns=ASSIGNMENT_COLUMNS)
    first = construct_manifest(trials, seed=20271002)
    second = construct_manifest(trials.copy(), seed=20271002)
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert first["outer_contexts"] == 25
    assert first["development_evaluations"] == 75
    assert all(value == {"0": 36, "1": 36} for value in first["outer_reader_goal_balance"].values())
    assert all(list(first["outer_article_map"].values()).count(fold) == 6 for fold in range(5))


@requires_canonical_data
def test_outer_targets_cannot_be_loaded_through_development_loader() -> None:
    split = json.loads((ROOT / "artifacts/manifests/split_manifest.json").read_text())
    trials = pd.read_parquet(TRIAL_TABLE)
    metadata = trials.set_index("trial_id", drop=False)
    context = split["contexts"][0]
    training, validation, outer = context_positions(metadata, split, context, 0)
    assert not set(training) & set(validation)
    assert not set(training) & outer and not set(validation) & outer
    with pytest.raises(PermissionError, match="sealed outer"):
        load_allowed_targets(
            TRIAL_TABLE, [next(iter(outer))], outer
        )


@requires_canonical_data
def test_outer_refit_positions_match_frozen_population_and_partition() -> None:
    split = json.loads((ROOT / "artifacts/manifests/split_manifest.json").read_text())
    trials = pd.read_parquet(
        TRIAL_TABLE,
        columns=["trial_id", "reader_id", "article_id"],
    )
    metadata = trials.set_index("trial_id", drop=False)
    all_targets: list[str] = []
    for context in split["contexts"]:
        training, target = outer_context_positions(metadata, split, context)
        assert not set(training) & set(target)
        assert len(training) == context["population_training_trials"]
        assert len(target) == context["outer_target_trials_sealed"]
        all_targets.extend(target)
    assert len(all_targets) == len(trials)
    assert set(all_targets) == set(trials["trial_id"].astype(str))
    assert len(set(all_targets)) == len(all_targets)
