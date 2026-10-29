from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from goalexit.data.preprocessing import NumericPreprocessor, fit_numeric_preprocessor
from goalexit.features.extract import all_b0_checkpoints, student_prefix
from goalexit.features.schema import (
    B0_FEATURES,
    STUDENT_BASE_FEATURES,
    STUDENT_CONTINUOUS,
    TEACHER_GAZE_CONTINUOUS,
    TEACHER_GAZE_FEATURES,
)
from goalexit.utils import canonical_json_sha256


TARGET_COLUMNS = ["trial_id", "goal_index"]
METADATA_COLUMNS = [
    "trial_id",
    "reader_id",
    "article_id",
    "paragraph_version_id",
    "difficulty",
    "full_fixation_count",
    "full_fixation_duration_ms",
]


@dataclass
class GoalExitStore:
    metadata: pd.DataFrame
    student_base: dict[str, np.ndarray]
    durations: dict[str, np.ndarray]
    b0: dict[str, dict[str, np.ndarray]]
    teacher_gaze: dict[str, np.ndarray]
    paragraph_ids: dict[str, str]

    @classmethod
    def load(cls, trial_path: Path, fixation_path: Path, word_path: Path) -> "GoalExitStore":
        metadata = pd.read_parquet(trial_path, columns=METADATA_COLUMNS)
        if metadata["trial_id"].duplicated().any():
            raise ValueError("Duplicate trial metadata")
        fix_columns = [
            "trial_id",
            "fixation_index",
            "duration_ms",
            *STUDENT_BASE_FEATURES,
        ]
        fixation = pd.read_parquet(fixation_path, columns=fix_columns)
        student_base: dict[str, np.ndarray] = {}
        durations: dict[str, np.ndarray] = {}
        b0: dict[str, dict[str, np.ndarray]] = {}
        for trial_id, group in fixation.groupby("trial_id", sort=False):
            group = group.sort_values("fixation_index", kind="stable")
            if not np.array_equal(group["fixation_index"].to_numpy(), np.arange(len(group))):
                raise ValueError(f"Noncontiguous fixation sequence {trial_id}")
            key = str(trial_id)
            student_base[key] = group[STUDENT_BASE_FEATURES].to_numpy(np.float32)
            durations[key] = group["duration_ms"].to_numpy(np.float32)
            b0[key] = all_b0_checkpoints(group)
        word_columns = ["trial_id", "token_index", *TEACHER_GAZE_FEATURES]
        word = pd.read_parquet(word_path, columns=word_columns)
        teacher_gaze: dict[str, np.ndarray] = {}
        for trial_id, group in word.groupby("trial_id", sort=False):
            group = group.sort_values("token_index", kind="stable")
            if not np.array_equal(group["token_index"].to_numpy(), np.arange(len(group))):
                raise ValueError(f"Noncontiguous teacher word sequence {trial_id}")
            teacher_gaze[str(trial_id)] = group[TEACHER_GAZE_FEATURES].to_numpy(np.float32)
        universe = set(metadata["trial_id"].astype(str))
        if set(student_base) != universe or set(teacher_gaze) != universe:
            raise ValueError("Canonical stores do not exactly cover trial universe")
        paragraph_ids = dict(
            zip(
                metadata["trial_id"].astype(str),
                metadata["paragraph_version_id"].astype(str),
            )
        )
        return cls(
            metadata=metadata.set_index("trial_id", drop=False),
            student_base=student_base,
            durations=durations,
            b0=b0,
            teacher_gaze=teacher_gaze,
            paragraph_ids=paragraph_ids,
        )

    def fit_preprocessor(self, model: str, training_ids: list[str]) -> NumericPreprocessor:
        if not training_ids:
            raise ValueError("No training trials")
        if model == "b0":
            values = np.stack([self.b0[key]["FULL"] for key in training_ids])
            return fit_numeric_preprocessor(
                values,
                B0_FEATURES,
                B0_FEATURES,
                training_trial_count=len(training_ids),
            )
        if model in {"b1", "d", "a", "d_gaze", "ls"}:
            values = np.concatenate([self.student_base[key] for key in training_ids])
            return fit_numeric_preprocessor(
                values,
                STUDENT_BASE_FEATURES,
                STUDENT_CONTINUOUS,
                training_trial_count=len(training_ids),
            )
        if model in {"teacher", "t_text", "gaze_teacher"}:
            values = np.concatenate([self.teacher_gaze[key] for key in training_ids])
            return fit_numeric_preprocessor(
                values,
                TEACHER_GAZE_FEATURES,
                TEACHER_GAZE_CONTINUOUS,
                training_trial_count=len(training_ids),
            )
        raise ValueError(f"Unauthorized model {model}")

    def transformed_student(
        self, ids: list[str], preprocessor: NumericPreprocessor
    ) -> dict[str, np.ndarray]:
        return {
            key: preprocessor.transform(self.student_base[key]) for key in ids
        }

    def student_at(
        self,
        trial_id: str,
        transformed_base: dict[str, np.ndarray],
        length: int,
    ) -> np.ndarray:
        return student_prefix(transformed_base[trial_id], length)

    def transformed_b0(
        self, ids: list[str], preprocessor: NumericPreprocessor
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            key: {
                checkpoint: preprocessor.transform(value[None, :])[0]
                for checkpoint, value in self.b0[key].items()
            }
            for key in ids
        }

    def transformed_teacher_gaze(
        self, ids: list[str], preprocessor: NumericPreprocessor
    ) -> dict[str, np.ndarray]:
        return {
            key: preprocessor.transform(self.teacher_gaze[key]) for key in ids
        }


def context_positions(
    metadata: pd.DataFrame,
    manifest: dict[str, Any],
    context: dict[str, Any],
    inner: int,
) -> tuple[list[str], list[str], set[str]]:
    reader_outer = metadata["reader_id"].astype(str).map(manifest["outer_reader_map"])
    article_outer = metadata["article_id"].astype(str).map(manifest["outer_article_map"])
    outer_reader = int(context["outer_reader_fold"])
    outer_article = int(context["outer_article_fold"])
    population = reader_outer.ne(outer_reader) & article_outer.ne(outer_article)
    outer = reader_outer.eq(outer_reader) & article_outer.eq(outer_article)
    reader_inner = metadata["reader_id"].astype(str).map(context["inner_reader_map"])
    article_inner = metadata["article_id"].astype(str).map(context["inner_article_map"])
    training_mask = population & reader_inner.ne(inner) & article_inner.ne(inner)
    validation_mask = population & reader_inner.eq(inner) & article_inner.eq(inner)
    training = metadata.loc[training_mask, "trial_id"].astype(str).tolist()
    validation = metadata.loc[validation_mask, "trial_id"].astype(str).tolist()
    outer_ids = set(metadata.loc[outer, "trial_id"].astype(str))
    if set(training) & set(validation) or set(training) & outer_ids or set(validation) & outer_ids:
        raise PermissionError("Development split intersects sealed outer target")
    expected = context["inner_holdouts"][inner]
    if canonical_json_sha256(sorted(training)) != expected["training_trial_ids_sha256"]:
        raise ValueError("Training membership differs from frozen split")
    if canonical_json_sha256(sorted(validation)) != expected["validation_trial_ids_sha256"]:
        raise ValueError("Validation membership differs from frozen split")
    return training, validation, outer_ids


def outer_context_positions(
    metadata: pd.DataFrame,
    manifest: dict[str, Any],
    context: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return population-fit and sealed target IDs without reading outcomes."""
    reader_outer = metadata["reader_id"].astype(str).map(manifest["outer_reader_map"])
    article_outer = metadata["article_id"].astype(str).map(manifest["outer_article_map"])
    outer_reader = int(context["outer_reader_fold"])
    outer_article = int(context["outer_article_fold"])
    training_mask = reader_outer.ne(outer_reader) & article_outer.ne(outer_article)
    target_mask = reader_outer.eq(outer_reader) & article_outer.eq(outer_article)
    training = metadata.loc[training_mask, "trial_id"].astype(str).tolist()
    target = metadata.loc[target_mask, "trial_id"].astype(str).tolist()
    if set(training) & set(target):
        raise PermissionError("Outer population fit intersects sealed target")
    if len(training) != int(context["population_training_trials"]):
        raise ValueError("Outer population-training count differs from frozen split")
    if len(target) != int(context["outer_target_trials_sealed"]):
        raise ValueError("Outer target count differs from frozen split")
    if canonical_json_sha256(sorted(target)) != context["outer_target_trial_ids_sha256"]:
        raise ValueError("Outer target membership differs from frozen split")
    return training, target


def load_allowed_targets(
    trial_path: Path, ordered_ids: list[str], prohibited_outer_ids: set[str]
) -> pd.DataFrame:
    requested = set(map(str, ordered_ids))
    if requested & prohibited_outer_ids:
        raise PermissionError("Attempted to materialize a sealed outer-target outcome")
    table = pq.read_table(
        trial_path,
        columns=TARGET_COLUMNS,
        filters=[("trial_id", "in", sorted(requested))],
    ).to_pandas()
    if set(table["trial_id"].astype(str)) != requested or table["trial_id"].duplicated().any():
        raise ValueError("Allowed target query does not exactly cover requested trials")
    return table.set_index("trial_id").loc[ordered_ids].reset_index()


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
