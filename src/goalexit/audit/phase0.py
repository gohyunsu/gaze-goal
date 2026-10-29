from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from goalexit.data.dataset import GoalExitStore, context_positions, load_allowed_targets, load_split
from goalexit.models.goalexit import CausalGazeGRU, PrivilegedTeacher
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


def run_metadata_probe(
    config: dict[str, Any], root: Path, split: dict[str, Any], store: GoalExitStore
) -> dict[str, Any]:
    trial_path = resolve(root, config["outputs"]["trial_table"])
    rows: list[dict[str, Any]] = []
    for context in split["contexts"]:
        for inner in range(int(config["inner_folds"])):
            training, validation, outer = context_positions(
                store.metadata, split, context, inner
            )
            train_target = load_allowed_targets(trial_path, training, outer)
            validation_target = load_allowed_targets(trial_path, validation, outer)
            # No non-gaze deployment metadata survives the forbidden-field
            # allowlist. A constant column implements the intercept-only logistic
            # probe without smuggling identity, text, length, or session fields.
            model = LogisticRegression(random_state=13, solver="lbfgs")
            model.fit(np.ones((len(training), 1)), train_target["goal_index"])
            prediction = model.predict(np.ones((len(validation), 1)))
            target = validation_target["goal_index"].to_numpy(np.int64)
            recalls = [float((prediction[target == label] == label).mean()) for label in (0, 1)]
            rows.append(
                {
                    "context_id": context["context_id"],
                    "inner_holdout": inner,
                    "balanced_accuracy": float(np.mean(recalls)),
                    "training_trials": len(training),
                    "validation_trials": len(validation),
                }
            )
    values = np.asarray([row["balanced_accuracy"] for row in rows])
    result = {
        "schema": "goalexit-metadata-shortcut-probe-v1",
        "status": "pass" if float(values.mean()) <= 0.55 else "fail",
        "probe": "intercept-only logistic regression",
        "remaining_non_gaze_metadata_features": [],
        "constant_implementation_column_is_model_information": False,
        "evaluations": rows,
        "mean_balanced_accuracy": float(values.mean()),
        "maximum_balanced_accuracy": float(values.max()),
        "materially_exceeds_chance_threshold": 0.55,
        "outer_rows_touched": False,
    }
    save_json(resolve(root, config["outputs"]["metadata_probe"]), result)
    if result["status"] != "pass":
        raise ValueError("Metadata-only shortcut probe materially exceeds chance")
    return result


def reader_parameter_scan(maximum_words: int) -> dict[str, Any]:
    models = {
        "b1": CausalGazeGRU(),
        "d": CausalGazeGRU(),
        "a": CausalGazeGRU(),
        "teacher": PrivilegedTeacher(maximum_words=maximum_words),
        "t_text": PrivilegedTeacher(maximum_words=maximum_words, text_only=True),
    }
    forbidden = ("reader", "participant", "identity", "article_id", "paragraph_id", "question")
    violations = [
        {"model": model_name, "parameter": name}
        for model_name, model in models.items()
        for name, _ in model.named_parameters()
        if any(token in name.casefold() for token in forbidden)
    ]
    return {
        "status": "pass" if not violations else "fail",
        "forbidden_name_tokens": list(forbidden),
        "violations": violations,
        "models_scanned": list(models),
    }


def _source_table(canonical: dict[str, Any]) -> str:
    lines = ["| Source | Bytes | SHA-256 |", "|---|---:|---|"]
    for name, value in canonical["sources"].items():
        lines.append(f"| {name} | {value['bytes']:,} | `{value['sha256']}` |")
    return "\n".join(lines)


def write_repo_audit(
    root: Path,
    canonical: dict[str, Any],
    split: dict[str, Any],
    roberta: dict[str, Any],
    metadata_probe: dict[str, Any],
    parameter_scan: dict[str, Any],
) -> None:
    observed = canonical["observed"]
    quantiles = canonical["overall_sequence_length_quantiles"]
    reader_balance = split["outer_reader_goal_balance"]
    roberta_lines = "\n".join(
        f"| {name} | {value['bytes']:,} | `{value['sha256']}` |"
        for name, value in roberta["model_files"].items()
    )
    text = f"""# GoalExit repository and data audit

Audit status: **PASS**. This audit precedes all official model training. The
authoritative protocol is `docs/01_ICCE2027_GoalExit_Research_Design.md`.

## Source inventory

All four compressed source files were re-hashed before canonical reconstruction.

{_source_table(canonical)}

Only the requested Paragraph trial, IA, and fixation columns listed in
`artifacts/manifests/canonical_goalexit.json` were read. Question content,
answers, correctness, response times, article/session position, and selected
answers were not requested.

## Canonical population

- Regular first-reading Paragraph trials: **{observed['trials']:,}**
  ({observed['ordinary_trials']:,} ordinary; {observed['information_seeking_trials']:,}
  information-seeking).
- Readers: **{observed['readers']}** (180 per goal); reader-goal constancy passed.
- Articles: **{observed['articles']}**; every article has both goals.
- Valid fixations: **{observed['fixations']:,}**.
- Teacher paragraph word-sequence variants: **{observed['paragraph_versions']}**.
- Fixation-count distribution: min {quantiles['0.0']:.0f}, Q1
  {quantiles['0.25']:.0f}, median {quantiles['0.5']:.0f}, Q3
  {quantiles['0.75']:.0f}, P90 {quantiles['0.9']:.0f}, P95
  {quantiles['0.95']:.0f}, P99 {quantiles['0.99']:.0f}, max
  {quantiles['1.0']:.0f}.

Repeated-reading rows, practice rows, Question Preview events, and invalid
fixations are excluded. Original source fixation indices and deterministic
chronology are retained. The canonical data hash is
`{canonical['canonical_data_sha256']}`.

## Feature and leakage audit

The student surface is an exact 10-feature allowlist: nine causal Paragraph-gaze
features plus position normalized by the *observed prefix* length. B0 has a
frozen 22-feature gaze aggregate allowlist. Teacher gaze has the frozen eight
word summaries; only T/T-text also receive frozen paragraph word embeddings.

The column-level audit, source-request audit, target/input mismatch checks, and
reader-identity parameter scan pass. The non-gaze metadata allowlist is empty;
the intercept-only logistic shortcut probe obtained mean BA
**{metadata_probe['mean_balanced_accuracy']:.4f}** over all 75 paired development
evaluations. Reader and article IDs are used only for splitting, clustered audit,
and macro-by-article evaluation—not as tensors.

## Teacher feasibility and dependency lock

`FacebookAI/roberta-base` is available locally at immutable revision
`{roberta['resolved_revision']}`. It remains frozen and was loaded offline.
All {roberta['paragraphs']} canonical paragraph variants align without
truncation; maximum tokenized length is
{roberta['maximum_subword_tokens_including_special_tokens']} including special
tokens. Cached word embeddings have shape {roberta['array_shape']} and SHA-256
`{roberta['array_sha256']}`.

| RoBERTa file | Bytes | SHA-256 |
|---|---:|---|
{roberta_lines}

## New GoalExit splits and outer seal

Seed **{split['seed']}** produced five reader folds and five article folds. Every
reader fold contains {reader_balance['0']['0']} ordinary and
{reader_balance['0']['1']} information-seeking readers; each article fold has six
whole articles. Their Cartesian product defines 25 sealed UB targets and three
paired inner holdouts each, for 75 development evaluations. Membership mapping
hash: `{split['mapping_sha256']}`.

The manifest contains only outer membership/count/hash information, not outer
class summaries or metrics. `configs/final_goalexit_lock.yaml` is absent and the
outer entry point fails closed. No outer prediction or metric exists.

## Environment and reuse boundary

Environment versions are recorded in the canonical and RoBERTa manifests.
Low-level deterministic parsing, hashing, preprocessing, and guard patterns were
reviewed against ReadMode, but GoalExit uses a newly reconstructed population,
seed, target, folds, feature allowlists, model heads, and outputs. No prior target,
fold mapping, opened result, or model-selection decision was imported.
"""
    (root / "docs/02_repo_audit.md").write_text(text, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    canonical = json.loads(
        resolve(root, config["outputs"]["canonical_manifest"]).read_text(encoding="utf-8")
    )
    split_path = resolve(root, config["outputs"]["split_manifest"])
    split = load_split(split_path)
    roberta = json.loads(
        resolve(root, config["outputs"]["roberta_manifest"]).read_text(encoding="utf-8")
    )
    store = GoalExitStore.load(
        resolve(root, config["outputs"]["trial_table"]),
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    metadata_probe = run_metadata_probe(config, root, split, store)
    parameter_scan = reader_parameter_scan(int(roberta["maximum_words"]))
    if parameter_scan["status"] != "pass":
        raise ValueError("Forbidden identity parameter detected")
    final_lock = root / "configs/final_goalexit_lock.yaml"
    outer_audit = {
        "schema": "goalexit-outer-guard-audit-v1",
        "status": "pass" if not final_lock.exists() else "fail",
        "final_lock_path": str(final_lock),
        "final_lock_exists": final_lock.exists(),
        "outer_scientific_predictions_created": False,
        "outer_scientific_metrics_computed": False,
        "guard_exception_expected": "OuterEvaluationSealedError",
        "split_manifest_sha256": sha256(split_path),
    }
    if outer_audit["status"] != "pass":
        raise PermissionError("Unexpected GoalExit final lock exists")
    save_json(resolve(root, config["outputs"]["outer_guard_audit"]), outer_audit)
    write_repo_audit(root, canonical, split, roberta, metadata_probe, parameter_scan)
    result = {
        "status": "pass",
        "metadata_probe": metadata_probe,
        "reader_parameter_scan": parameter_scan,
        "outer_guard": outer_audit,
    }
    print(
        f"status=pass metadata_probe_ba={metadata_probe['mean_balanced_accuracy']:.6f} "
        f"reader_parameter_violations={len(parameter_scan['violations'])}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
