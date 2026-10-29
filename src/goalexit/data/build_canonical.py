from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch

from goalexit.features.schema import (
    B0_FEATURES,
    MODEL_INPUTS,
    STUDENT_BASE_FEATURES,
    STUDENT_CONTINUOUS,
    STUDENT_FEATURES,
    TEACHER_GAZE_CONTINUOUS,
    TEACHER_GAZE_FEATURES,
    audit_feature_names,
)
from goalexit.utils import (
    canonical_json_sha256,
    load_yaml,
    opaque_id,
    project_root,
    resolve,
    save_json,
    sha256,
)


TRIAL_COLUMNS = [
    "participant_id",
    "TRIAL_INDEX",
    "article_batch",
    "article_id",
    "difficulty_level",
    "paragraph_id",
    "practice_trial",
    "question_preview",
    "repeated_reading_trial",
]

IA_COLUMNS = TRIAL_COLUMNS + [
    "IA_ID",
    "IA_LABEL",
    "IA_LEFT",
    "IA_TOP",
    "IA_RIGHT",
    "IA_BOTTOM",
    "IA_FIXATION_COUNT",
    "IA_DWELL_TIME",
    "IA_FIRST_FIXATION_DURATION",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_REGRESSION_IN_COUNT",
    "IA_RUN_COUNT",
    "IA_SKIP",
]

FIXATION_COLUMNS = TRIAL_COLUMNS + [
    "CURRENT_FIX_INDEX",
    "CURRENT_FIX_START",
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "CURRENT_FIX_INTEREST_AREA_INDEX",
    "IP_START_TIME",
    "TRIAL_START_TIME",
]

FORBIDDEN_SOURCE_FIELDS = {
    "article_index",
    "trial_index",
    "onestopqa_question_id",
    "question",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "answers_order",
    "correct_answer_position",
    "selected_answer",
    "selected_answer_position",
    "is_correct",
    "ANSWER_RT",
    "QUESTION_RT",
    "PREVIEW_QUESTION_RT",
    "PARAGRAPH_RT",
}

SOURCES = (("ordinary", 0), ("hunting", 1))


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"Missing Boolean value in {values.name}")
        return values.astype(bool)
    normalized = values.astype(str).str.casefold().str.strip()
    if not set(normalized.unique()).issubset({"true", "false", "1", "0"}):
        raise ValueError(f"Unrecognized Boolean value in {values.name}")
    return normalized.isin({"true", "1"})


def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("TRIAL_INDEX", "article_batch", "article_id", "paragraph_id"):
        values = pd.to_numeric(result[column], errors="raise")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Non-integral identifier in {column}")
        result[column] = values.astype(np.int64)
    result["participant_id"] = result["participant_id"].astype(str)
    result["difficulty_level"] = result["difficulty_level"].astype(str)
    return result


def _zip_chunks(
    path: Path, member: str, columns: list[str], chunksize: int = 150_000
) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.endswith(member) and not name.startswith("__MACOSX/")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one member ending {member} in {path}: {matches}")
        with archive.open(matches[0]) as stream:
            yield from pd.read_csv(
                stream,
                usecols=columns,
                chunksize=chunksize,
                low_memory=False,
                na_values=["."],
            )


def _source_key(frame: pd.DataFrame) -> pd.Series:
    return frame["participant_id"].astype(str) + "|" + frame["TRIAL_INDEX"].astype(str)


def _regular_mask(frame: pd.DataFrame, goal: int) -> pd.Series:
    return (
        ~_bool_series(frame["practice_trial"])
        & ~_bool_series(frame["repeated_reading_trial"])
        & _bool_series(frame["question_preview"]).eq(bool(goal))
    )


def _scan_trials(
    config: dict[str, Any], sources: dict[str, Path]
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], dict[str, int]]:
    records: list[pd.DataFrame] = []
    maps: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for stem, goal in SOURCES:
        parts: list[pd.DataFrame] = []
        raw_rows = 0
        for chunk in _zip_chunks(
            sources[f"{stem}_ia"], config["source_members"][f"{stem}_ia"], TRIAL_COLUMNS
        ):
            raw_rows += len(chunk)
            parts.append(chunk.drop_duplicates())
        frame = _normalize_ids(pd.concat(parts, ignore_index=True).drop_duplicates())
        frame = frame.loc[_regular_mask(frame, goal)].copy()
        frame["source_key"] = _source_key(frame)
        if frame["source_key"].duplicated().any():
            raise ValueError(f"Duplicate regular trial key in {stem}")
        frame["trial_id"] = [
            "t_" + opaque_id(stem, reader, index)
            for reader, index in zip(frame["participant_id"], frame["TRIAL_INDEX"])
        ]
        frame["reader_id"] = frame["participant_id"].astype(str)
        frame["article_id_canonical"] = (
            frame["article_batch"].astype(str) + "_" + frame["article_id"].astype(str)
        )
        frame["paragraph_version_id"] = [
            "pv_" + opaque_id(batch, article, difficulty, paragraph)
            for batch, article, difficulty, paragraph in zip(
                frame["article_batch"],
                frame["article_id"],
                frame["difficulty_level"],
                frame["paragraph_id"],
            )
        ]
        frame["goal_index"] = goal
        frame["goal_class"] = "O" if goal == 0 else "H"
        frame["source_name"] = stem
        maps[stem] = dict(zip(frame["source_key"], frame["trial_id"]))
        records.append(
            frame[
                [
                    "trial_id",
                    "reader_id",
                    "article_id_canonical",
                    "paragraph_version_id",
                    "difficulty_level",
                    "goal_index",
                    "goal_class",
                    "source_name",
                    "source_key",
                ]
            ].rename(
                columns={
                    "article_id_canonical": "article_id",
                    "difficulty_level": "difficulty",
                }
            )
        )
        counts[f"{stem}_raw_ia_rows"] = raw_rows
        counts[f"{stem}_regular_trials"] = len(frame)
    trials = pd.concat(records, ignore_index=True)
    if trials["trial_id"].duplicated().any():
        raise ValueError("Opaque trial ID collision")
    reader_goal = trials[["reader_id", "goal_index"]].drop_duplicates()
    if reader_goal.groupby("reader_id")["goal_index"].nunique().ne(1).any():
        raise ValueError("Reading goal is not constant within reader")
    return trials, maps, counts


def _build_words(
    config: dict[str, Any],
    sources: dict[str, Path],
    trials: pd.DataFrame,
    source_maps: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    counts: Counter[str] = Counter()
    trial_lookup = trials.set_index("trial_id")
    skip_disagreements = 0
    for stem, goal in SOURCES:
        mapping = source_maps[stem]
        for chunk in _zip_chunks(
            sources[f"{stem}_ia"], config["source_members"][f"{stem}_ia"], IA_COLUMNS
        ):
            chunk = _normalize_ids(chunk)
            chunk = chunk.loc[_regular_mask(chunk, goal)].copy()
            chunk["source_key"] = _source_key(chunk)
            chunk = chunk.loc[chunk["source_key"].isin(mapping)].copy()
            if chunk.empty:
                continue
            chunk["trial_id"] = chunk["source_key"].map(mapping)
            numeric = [
                "IA_ID",
                "IA_LEFT",
                "IA_TOP",
                "IA_RIGHT",
                "IA_BOTTOM",
                "IA_FIXATION_COUNT",
                "IA_DWELL_TIME",
                "IA_FIRST_FIXATION_DURATION",
                "IA_FIRST_RUN_DWELL_TIME",
                "IA_REGRESSION_IN_COUNT",
                "IA_RUN_COUNT",
                "IA_SKIP",
            ]
            for column in numeric:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            if chunk[["IA_ID", "IA_LEFT", "IA_TOP", "IA_RIGHT", "IA_BOTTOM"]].isna().any().any():
                raise ValueError("A selected IA row lacks token ID or geometry")
            fixation_count = chunk["IA_FIXATION_COUNT"].fillna(0.0).clip(lower=0.0)
            dwell = chunk["IA_DWELL_TIME"].fillna(0.0).clip(lower=0.0)
            run_count = chunk["IA_RUN_COUNT"].fillna(0.0).clip(lower=0.0)
            derived_skip = fixation_count.eq(0)
            reported_skip = chunk["IA_SKIP"].fillna(derived_skip.astype(float)).astype(bool)
            skip_disagreements += int(derived_skip.ne(reported_skip).sum())
            selected = pd.DataFrame(
                {
                    "trial_id": chunk["trial_id"].astype(str),
                    "token_index": chunk["IA_ID"].astype(np.int32) - 1,
                    "word_text": chunk["IA_LABEL"].fillna("").astype(str),
                    "ia_left": chunk["IA_LEFT"].astype(np.float32),
                    "ia_top": chunk["IA_TOP"].astype(np.float32),
                    "ia_right": chunk["IA_RIGHT"].astype(np.float32),
                    "ia_bottom": chunk["IA_BOTTOM"].astype(np.float32),
                    "word_fixation_count": fixation_count.astype(np.float32),
                    "word_total_dwell_ms": dwell.astype(np.float32),
                    "word_mean_fixation_duration_ms": np.divide(
                        dwell,
                        fixation_count,
                        out=np.zeros(len(chunk), dtype=np.float64),
                        where=fixation_count.to_numpy() > 0,
                    ).astype(np.float32),
                    "word_first_fixation_duration_ms": chunk[
                        "IA_FIRST_FIXATION_DURATION"
                    ].fillna(0.0).clip(lower=0.0).astype(np.float32),
                    "word_first_pass_dwell_ms": chunk["IA_FIRST_RUN_DWELL_TIME"]
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .astype(np.float32),
                    "word_regression_in_count": chunk["IA_REGRESSION_IN_COUNT"]
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .astype(np.float32),
                    "word_revisit_count": (run_count - 1.0)
                    .clip(lower=0.0)
                    .astype(np.float32),
                    "word_skip_indicator": derived_skip.astype(np.float32),
                }
            )
            if selected["token_index"].lt(0).any():
                raise ValueError("Negative token index")
            parts.append(selected)
            counts[f"{stem}_selected_ia_rows"] += len(selected)
    words = pd.concat(parts, ignore_index=True)
    if words.duplicated(["trial_id", "token_index"]).any():
        raise ValueError("Duplicate word within a trial")
    words["paragraph_version_id"] = words["trial_id"].map(
        trial_lookup["paragraph_version_id"]
    )
    words["line_index"] = (
        words.groupby("trial_id")["ia_top"].rank(method="dense").astype(np.int16) - 1
    )
    contiguous = words.groupby("trial_id")["token_index"].apply(
        lambda value: np.array_equal(np.sort(value.to_numpy()), np.arange(len(value)))
    )
    if not contiguous.all():
        raise ValueError("Word indices are not contiguous")
    if set(words["trial_id"]) != set(trials["trial_id"]):
        raise ValueError("Word table does not cover every canonical trial")
    text_signatures = (
        words.sort_values(["trial_id", "token_index"])
        .groupby("trial_id", sort=False)
        .apply(
            lambda group: canonical_json_sha256(
                [
                    re.sub(r"\s+", " ", str(row.word_text).strip())
                    for row in group.itertuples(index=False)
                ]
            ),
            include_groups=False,
        )
        .rename("word_sequence_sha256")
    )
    layout_signatures = (
        words.sort_values(["trial_id", "token_index"])
        .groupby("trial_id", sort=False)
        .apply(
            lambda group: canonical_json_sha256(
                [
                    [
                        float(row.ia_left),
                        float(row.ia_top),
                        float(row.ia_right),
                        float(row.ia_bottom),
                    ]
                    for row in group.itertuples(index=False)
                ]
            ),
            include_groups=False,
        )
        .rename("layout_sha256")
    )
    signature_frame = (
        trials[["trial_id", "paragraph_version_id"]]
        .merge(text_signatures, left_on="trial_id", right_index=True, validate="one_to_one")
        .merge(layout_signatures, left_on="trial_id", right_index=True, validate="one_to_one")
    )
    source_version_count = int(signature_frame["paragraph_version_id"].nunique())
    word_variants_per_source = signature_frame.groupby("paragraph_version_id")[
        "word_sequence_sha256"
    ].nunique()
    signature_frame["canonical_paragraph_version_id"] = [
        "pv_" + opaque_id(source_version, word_signature)
        for source_version, word_signature in zip(
            signature_frame["paragraph_version_id"],
            signature_frame["word_sequence_sha256"],
        )
    ]
    canonical_map = dict(
        zip(
            signature_frame["trial_id"].astype(str),
            signature_frame["canonical_paragraph_version_id"].astype(str),
        )
    )
    trials["paragraph_source_id"] = trials["paragraph_version_id"].astype(str)
    trials["paragraph_version_id"] = trials["trial_id"].map(canonical_map)
    words["paragraph_version_id"] = words["trial_id"].map(canonical_map)
    signature_frame["paragraph_version_id"] = signature_frame[
        "canonical_paragraph_version_id"
    ]
    text_mismatch = signature_frame.groupby("paragraph_version_id")[
        "word_sequence_sha256"
    ].nunique()
    if text_mismatch.ne(1).any():
        raise AssertionError("Canonical paragraph word-sequence partition failed")
    layout_variants = signature_frame.groupby("paragraph_version_id")["layout_sha256"].nunique()
    representatives = (
        trials.sort_values("trial_id")
        .drop_duplicates("paragraph_version_id")
        [["paragraph_version_id", "trial_id"]]
    )
    paragraphs = representatives.merge(words, on=["trial_id", "paragraph_version_id"])
    paragraphs = paragraphs[
        [
            "paragraph_version_id",
            "token_index",
            "word_text",
            "ia_left",
            "ia_top",
            "ia_right",
            "ia_bottom",
            "line_index",
        ]
    ].sort_values(["paragraph_version_id", "token_index"])
    layout = words.groupby("trial_id", as_index=False).agg(
        paragraph_word_count=("token_index", "size"),
        paragraph_line_count=("line_index", "nunique"),
        text_left=("ia_left", "min"),
        text_top=("ia_top", "min"),
        text_right=("ia_right", "max"),
        text_bottom=("ia_bottom", "max"),
    )
    if (layout["text_right"] <= layout["text_left"]).any() or (
        layout["text_bottom"] <= layout["text_top"]
    ).any():
        raise ValueError("Invalid Paragraph bounding box")
    teacher_words = words[
        ["trial_id", "paragraph_version_id", "token_index", "line_index", *TEACHER_GAZE_FEATURES]
    ].copy()
    if not np.isfinite(teacher_words[TEACHER_GAZE_FEATURES].to_numpy()).all():
        raise ValueError("Teacher gaze summary contains non-finite values")
    audit = {
        **dict(counts),
        "canonical_trial_word_rows": len(teacher_words),
        "canonical_paragraph_word_rows": len(paragraphs),
        "paragraph_versions": int(paragraphs["paragraph_version_id"].nunique()),
        "source_paragraph_versions": source_version_count,
        "source_versions_with_multiple_ia_word_sequences": int(
            word_variants_per_source.gt(1).sum()
        ),
        "maximum_ia_word_sequence_variants_per_source_version": int(
            word_variants_per_source.max()
        ),
        "word_sequence_matches_across_trials": True,
        "paragraph_versions_with_multiple_rendered_layouts": int(layout_variants.gt(1).sum()),
        "maximum_rendered_layout_variants_per_paragraph": int(layout_variants.max()),
        "layout_policy": "retain each trial's rendered IA geometry for coordinate normalization",
        "teacher_word_alignment_complete": True,
        "reported_vs_derived_skip_disagreements": skip_disagreements,
        "maximum_paragraph_words": int(layout["paragraph_word_count"].max()),
    }
    return teacher_words, paragraphs, layout, audit


def _build_fixations(
    config: dict[str, Any],
    sources: dict[str, Path],
    trials: pd.DataFrame,
    source_maps: dict[str, dict[str, str]],
    words: pd.DataFrame,
    layout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    parts: list[pd.DataFrame] = []
    counts: Counter[str] = Counter()
    for stem, goal in SOURCES:
        mapping = source_maps[stem]
        for chunk in _zip_chunks(
            sources[f"{stem}_fixations"],
            config["source_members"][f"{stem}_fixations"],
            FIXATION_COLUMNS,
        ):
            counts[f"{stem}_raw_fixation_rows"] += len(chunk)
            chunk = _normalize_ids(chunk)
            chunk = chunk.loc[_regular_mask(chunk, goal)].copy()
            chunk["source_key"] = _source_key(chunk)
            chunk = chunk.loc[chunk["source_key"].isin(mapping)].copy()
            counts[f"{stem}_selected_source_fixations"] += len(chunk)
            if chunk.empty:
                continue
            chunk["trial_id"] = chunk["source_key"].map(mapping)
            numeric = [
                "CURRENT_FIX_INDEX",
                "CURRENT_FIX_START",
                "CURRENT_FIX_DURATION",
                "CURRENT_FIX_X",
                "CURRENT_FIX_Y",
                "CURRENT_FIX_INTEREST_AREA_INDEX",
                "IP_START_TIME",
                "TRIAL_START_TIME",
            ]
            for column in numeric:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            duration_valid = chunk["CURRENT_FIX_DURATION"].gt(0)
            coordinate_valid = np.isfinite(chunk["CURRENT_FIX_X"]) & np.isfinite(
                chunk["CURRENT_FIX_Y"]
            )
            scientific_valid = duration_valid & coordinate_valid
            chronology_valid = (
                np.isfinite(chunk["CURRENT_FIX_INDEX"])
                & np.isfinite(chunk["CURRENT_FIX_START"])
                & np.isfinite(chunk["IP_START_TIME"])
                & np.isfinite(chunk["TRIAL_START_TIME"])
            )
            counts["nonpositive_or_missing_duration_rows"] += int((~duration_valid).sum())
            counts["nonfinite_coordinate_rows"] += int((~coordinate_valid).sum())
            counts["invalid_fixation_rows_excluded"] += int((~scientific_valid).sum())
            counts["valid_rows_missing_chronology"] += int(
                (scientific_valid & ~chronology_valid).sum()
            )
            if (scientific_valid & ~chronology_valid).any():
                raise ValueError("A scientifically valid fixation lacks chronology")
            chunk = chunk.loc[scientific_valid].copy()
            chunk["source_fixation_index"] = chunk["CURRENT_FIX_INDEX"].astype(np.int32)
            chunk["onset_ms"] = chunk["CURRENT_FIX_START"] - (
                chunk["IP_START_TIME"] - chunk["TRIAL_START_TIME"]
            )
            chunk["token_index"] = chunk["CURRENT_FIX_INTEREST_AREA_INDEX"] - 1
            chunk.loc[~np.isfinite(chunk["token_index"]), "token_index"] = np.nan
            parts.append(
                chunk[
                    [
                        "trial_id",
                        "source_fixation_index",
                        "onset_ms",
                        "CURRENT_FIX_DURATION",
                        "CURRENT_FIX_X",
                        "CURRENT_FIX_Y",
                        "token_index",
                    ]
                ]
            )
    raw = pd.concat(parts, ignore_index=True).sort_values(
        ["trial_id", "source_fixation_index"], kind="stable"
    )
    if raw.duplicated(["trial_id", "source_fixation_index"]).any():
        raise ValueError("Duplicate fixation index within trial")
    if raw["onset_ms"].lt(0).any():
        raise ValueError("Fixation begins before Paragraph interest period")
    if not raw.groupby("trial_id")["source_fixation_index"].apply(
        lambda value: value.is_monotonic_increasing
    ).all():
        raise ValueError("Non-monotonic fixation chronology")
    raw["fixation_index"] = raw.groupby("trial_id").cumcount().astype(np.int32)
    raw["token_index"] = raw["token_index"].astype("Int32")
    line_support = words[["trial_id", "token_index", "line_index"]]
    joined = raw.merge(
        line_support, on=["trial_id", "token_index"], how="left", validate="many_to_one"
    ).merge(layout, on="trial_id", validate="many_to_one")
    width = joined["text_right"] - joined["text_left"]
    height = joined["text_bottom"] - joined["text_top"]
    group = joined.groupby("trial_id", sort=False)
    previous_x = group["CURRENT_FIX_X"].shift(1)
    previous_y = group["CURRENT_FIX_Y"].shift(1)
    previous_line = group["line_index"].shift(1)
    dx = ((joined["CURRENT_FIX_X"] - previous_x) / width).fillna(0.0)
    dy = ((joined["CURRENT_FIX_Y"] - previous_y) / height).fillna(0.0)
    amplitude = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    first = group.cumcount().eq(0)
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    sin_angle.loc[first] = 0.0
    cos_angle.loc[first] = 1.0
    both_lines = joined["line_index"].notna() & previous_line.notna()
    frame = pd.DataFrame(
        {
            "trial_id": joined["trial_id"].astype(str),
            "fixation_index": joined["fixation_index"].astype(np.int32),
            "source_fixation_index": joined["source_fixation_index"].astype(np.int32),
            "onset_ms": joined["onset_ms"].astype(np.float32),
            "duration_ms": joined["CURRENT_FIX_DURATION"].astype(np.float32),
            "token_index": joined["token_index"].astype("Int32"),
            "line_index": joined["line_index"].astype("Int16"),
            "x_norm": ((joined["CURRENT_FIX_X"] - joined["text_left"]) / width).astype(
                np.float32
            ),
            "y_norm": ((joined["CURRENT_FIX_Y"] - joined["text_top"]) / height).astype(
                np.float32
            ),
            "log_fixation_duration": np.log1p(joined["CURRENT_FIX_DURATION"]).astype(
                np.float32
            ),
            "dx_norm": dx.astype(np.float32),
            "dy_norm": dy.astype(np.float32),
            "log_saccade_amplitude": np.log1p(amplitude).astype(np.float32),
            "saccade_angle_sin": sin_angle.astype(np.float32),
            "saccade_angle_cos": cos_angle.astype(np.float32),
            "line_change": (
                both_lines & joined["line_index"].astype(float).ne(previous_line.astype(float))
            ).astype(np.float32),
        }
    )
    core = ["onset_ms", "duration_ms", *STUDENT_BASE_FEATURES]
    if not np.isfinite(frame[core].to_numpy(np.float64)).all():
        raise ValueError("Canonical fixation table contains non-finite core values")
    if set(frame["trial_id"]) != set(trials["trial_id"]):
        raise ValueError("At least one trial lacks valid Paragraph fixations")
    if not frame.groupby("trial_id")["fixation_index"].apply(
        lambda value: np.array_equal(value.to_numpy(), np.arange(len(value)))
    ).all():
        raise ValueError("Canonical fixation indices are not contiguous")
    counts["canonical_fixations"] = len(frame)
    counts["word_aligned_fixations"] = int(frame["token_index"].notna().sum())
    counts["out_of_text_fixations"] = int(frame["token_index"].isna().sum())
    return frame, dict(counts)


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        str(q): float(value)
        for q, value in values.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()
    }


def build(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config_path = resolve(root, config_path)
    config = load_yaml(config_path)
    sources = {key: resolve(root, value) for key, value in config["sources"].items()}
    source_manifest: dict[str, Any] = {}
    for key, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        if digest != config["source_hashes"][key]:
            raise ValueError(f"Source hash mismatch for {key}: {digest}")
        source_manifest[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "member": config["source_members"][key],
        }
    trials, source_maps, metadata_counts = _scan_trials(config, sources)
    words, paragraphs, layout, word_audit = _build_words(
        config, sources, trials, source_maps
    )
    fixations, fixation_counts = _build_fixations(
        config, sources, trials, source_maps, words, layout
    )
    sequence = fixations.groupby("trial_id", as_index=False).agg(
        full_fixation_count=("fixation_index", "size"),
        full_fixation_duration_ms=("duration_ms", "sum"),
    )
    trials = trials.merge(sequence, on="trial_id", validate="one_to_one")
    article_goal = trials.groupby(["article_id", "goal_index"]).size().unstack(fill_value=0)
    if set(article_goal.columns) != {0, 1} or (article_goal == 0).any().any():
        raise ValueError("An article is absent from one reading-goal condition")
    observed = {
        "trials": len(trials),
        "ordinary_trials": int(trials["goal_index"].eq(0).sum()),
        "information_seeking_trials": int(trials["goal_index"].eq(1).sum()),
        "readers": int(trials["reader_id"].nunique()),
        "ordinary_readers": int(
            trials.loc[trials["goal_index"].eq(0), "reader_id"].nunique()
        ),
        "information_seeking_readers": int(
            trials.loc[trials["goal_index"].eq(1), "reader_id"].nunique()
        ),
        "articles": int(trials["article_id"].nunique()),
        "paragraph_versions": int(trials["paragraph_version_id"].nunique()),
        "fixations": len(fixations),
    }
    expected = {
        "trials": 19_438,
        "ordinary_trials": 9_718,
        "information_seeking_trials": 9_720,
        "readers": 360,
        "ordinary_readers": 180,
        "information_seeking_readers": 180,
        "articles": 30,
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError(f"Canonical population differs from public expectation: {observed}")
    outputs = {
        key: resolve(root, config["outputs"][key])
        for key in (
            "trial_table",
            "fixation_table",
            "word_table",
            "paragraph_table",
            "canonical_manifest",
            "feature_schema",
            "forbidden_field_audit",
        )
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(trials, preserve_index=False), outputs["trial_table"], compression="zstd")
    pq.write_table(pa.Table.from_pandas(fixations, preserve_index=False), outputs["fixation_table"], compression="zstd")
    pq.write_table(pa.Table.from_pandas(words, preserve_index=False), outputs["word_table"], compression="zstd")
    pq.write_table(pa.Table.from_pandas(paragraphs, preserve_index=False), outputs["paragraph_table"], compression="zstd")
    output_hashes = {
        key: {"path": str(outputs[key]), "bytes": outputs[key].stat().st_size, "sha256": sha256(outputs[key])}
        for key in ("trial_table", "fixation_table", "word_table", "paragraph_table")
    }
    canonical_hash = canonical_json_sha256(
        {"sources": source_manifest, "outputs": output_hashes, "population": observed}
    )
    feature_audit = audit_feature_names()
    if feature_audit["status"] != "pass":
        raise ValueError(f"Forbidden model feature: {feature_audit}")
    sequence_distributions = {
        goal: _quantiles(group["full_fixation_count"])
        for goal, group in trials.groupby("goal_class")
    }
    reader_goal = trials[["reader_id", "goal_index"]].drop_duplicates()
    manifest = {
        "schema": "goalexit-canonical-v1",
        "status": "pass",
        "population_definition": "regular first-reading Paragraph trials only",
        "observed": observed,
        "public_expectation": expected,
        "public_expectation_matches": True,
        "metadata_counts": metadata_counts,
        "word_alignment_audit": word_audit,
        "fixation_counts": fixation_counts,
        "sequence_length_quantiles_by_goal": sequence_distributions,
        "overall_sequence_length_quantiles": _quantiles(trials["full_fixation_count"]),
        "reader_goal_constancy": bool(
            reader_goal.groupby("reader_id")["goal_index"].nunique().eq(1).all()
        ),
        "articles_have_both_goals": True,
        "article_goal_trial_count_minimum": int(article_goal.min().min()),
        "sources": source_manifest,
        "outputs": output_hashes,
        "canonical_data_sha256": canonical_hash,
        "requested_source_columns": {
            "trial": TRIAL_COLUMNS,
            "ia": IA_COLUMNS,
            "fixations": FIXATION_COLUMNS,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "assertions": {
            "paragraph_screen_only": True,
            "regular_first_reading_only": True,
            "reader_goal_constant": True,
            "all_articles_cover_both_goals": True,
            "all_trials_have_valid_fixations": True,
            "fixation_order_deterministic": True,
            "teacher_word_alignment_feasible": True,
            "outer_scientific_metrics_computed": False,
            "outer_scientific_predictions_created": False,
        },
    }
    save_json(outputs["canonical_manifest"], manifest)
    feature_manifest = {
        "schema": "goalexit-feature-schema-v1",
        "status": "frozen",
        "student_base_features": STUDENT_BASE_FEATURES,
        "student_dynamic_feature": "fixation_position = i / max(1, observed_prefix_length-1)",
        "student_features": STUDENT_FEATURES,
        "student_continuous_features": STUDENT_CONTINUOUS,
        "b0_features": B0_FEATURES,
        "teacher_gaze_features": TEACHER_GAZE_FEATURES,
        "teacher_gaze_continuous_features": TEACHER_GAZE_CONTINUOUS,
        "model_inputs": MODEL_INPUTS,
        "coordinate_normalization": "Paragraph IA bounding box available from rendered layout",
        "learned_scale_fit_scope": "current inner training population only",
        "feature_audit": feature_audit,
        "canonical_data_sha256": canonical_hash,
    }
    save_json(outputs["feature_schema"], feature_manifest)
    requested = set(TRIAL_COLUMNS + IA_COLUMNS + FIXATION_COLUMNS)
    forbidden_requested = sorted(FORBIDDEN_SOURCE_FIELDS & requested)
    forbidden_manifest = {
        "schema": "goalexit-forbidden-field-audit-v1",
        "status": "pass" if not forbidden_requested else "fail",
        "model_feature_audit": feature_audit,
        "forbidden_source_fields": sorted(FORBIDDEN_SOURCE_FIELDS),
        "forbidden_source_fields_requested": forbidden_requested,
        "source_condition_fields_used_only_for_population_and_target": [
            "question_preview",
            "repeated_reading_trial",
            "practice_trial",
        ],
        "reader_id_use": "split membership and reader-cluster audit only; never a model tensor",
        "article_id_use": "split membership and macro-by-article metric only; never a model tensor",
        "deployment_model_input_allowlists": {
            key: MODEL_INPUTS[key] for key in ("b0", "b1", "d", "a")
        },
        "teacher_input_allowlists": {
            key: MODEL_INPUTS[key] for key in ("teacher", "t_text")
        },
        "canonical_data_sha256": canonical_hash,
    }
    if forbidden_requested:
        raise ValueError(f"Forbidden source fields were requested: {forbidden_requested}")
    save_json(outputs["forbidden_field_audit"], forbidden_manifest)
    print(json.dumps(observed, sort_keys=True))
    print(f"canonical_data_sha256={canonical_hash}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
