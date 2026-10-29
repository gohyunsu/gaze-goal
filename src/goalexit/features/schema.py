from __future__ import annotations


PREFIXES: tuple[int | str, ...] = (5, 10, 20, 40, 80, "FULL")

STUDENT_BASE_FEATURES = [
    "x_norm",
    "y_norm",
    "log_fixation_duration",
    "dx_norm",
    "dy_norm",
    "log_saccade_amplitude",
    "saccade_angle_sin",
    "saccade_angle_cos",
    "line_change",
]
STUDENT_FEATURES = STUDENT_BASE_FEATURES + ["fixation_position"]
STUDENT_CONTINUOUS = [
    "x_norm",
    "y_norm",
    "log_fixation_duration",
    "dx_norm",
    "dy_norm",
    "log_saccade_amplitude",
]

B0_FEATURES = [
    "observed_fixation_count",
    "total_dwell_ms",
    "mean_fixation_duration_ms",
    "std_fixation_duration_ms",
    "median_fixation_duration_ms",
    "p25_fixation_duration_ms",
    "p75_fixation_duration_ms",
    "p90_fixation_duration_ms",
    "mean_saccade_amplitude",
    "std_saccade_amplitude",
    "median_saccade_amplitude",
    "p90_saccade_amplitude",
    "mean_absolute_dx_norm",
    "mean_absolute_dy_norm",
    "net_dx_norm",
    "net_dy_norm",
    "forward_movement_proportion",
    "backward_movement_proportion",
    "line_change_rate",
    "x_spatial_dispersion",
    "y_spatial_dispersion",
    "path_length_norm",
]

TEACHER_GAZE_FEATURES = [
    "word_fixation_count",
    "word_total_dwell_ms",
    "word_mean_fixation_duration_ms",
    "word_first_fixation_duration_ms",
    "word_first_pass_dwell_ms",
    "word_regression_in_count",
    "word_revisit_count",
    "word_skip_indicator",
]
TEACHER_GAZE_CONTINUOUS = TEACHER_GAZE_FEATURES[:-1]

MODEL_INPUTS = {
    "b0": B0_FEATURES,
    "b1": STUDENT_FEATURES,
    "d": STUDENT_FEATURES,
    "a": STUDENT_FEATURES,
    "teacher": ["frozen_roberta_word_embedding", *TEACHER_GAZE_FEATURES],
    "t_text": ["frozen_roberta_word_embedding"],
}

FORBIDDEN_INPUT_TOKENS = (
    "reader_id",
    "participant_id",
    "regime",
    "goal_index",
    "goal_label",
    "article_id",
    "paragraph_id",
    "question",
    "answer",
    "correct",
    "selected",
    "session",
    "trial_index",
    "article_index",
    "repeat",
    "timestamp",
    "full_fixation_count",
    "final_length",
)


def audit_feature_names() -> dict[str, object]:
    violations: list[dict[str, str]] = []
    for model, names in MODEL_INPUTS.items():
        if len(names) != len(set(names)):
            violations.append({"model": model, "feature": "<duplicate>"})
        for name in names:
            for token in FORBIDDEN_INPUT_TOKENS:
                if token in name.casefold():
                    violations.append({"model": model, "feature": name})
    expected = {
        "b0": B0_FEATURES,
        "b1": STUDENT_FEATURES,
        "d": STUDENT_FEATURES,
        "a": STUDENT_FEATURES,
        "teacher": ["frozen_roberta_word_embedding", *TEACHER_GAZE_FEATURES],
        "t_text": ["frozen_roberta_word_embedding"],
    }
    if MODEL_INPUTS != expected:
        violations.append({"model": "<all>", "feature": "<allowlist drift>"})
    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "models": MODEL_INPUTS,
        "forbidden_tokens": list(FORBIDDEN_INPUT_TOKENS),
    }
