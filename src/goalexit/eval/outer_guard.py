from __future__ import annotations

import subprocess
from pathlib import Path

from goalexit.utils import git_commit, load_yaml, project_root, resolve, sha256


class OuterEvaluationSealedError(PermissionError):
    """Raised before any outer scientific outcome can be accessed."""


def require_outer_authorization(root: Path | None = None) -> dict:
    root = project_root() if root is None else root
    lock = root / "configs" / "final_goalexit_lock.yaml"
    if not lock.is_file():
        raise OuterEvaluationSealedError(
            "GoalExit outer evaluation is sealed: configs/final_goalexit_lock.yaml is absent"
        )
    value = load_yaml(lock)
    if value.get("authorized") is not True or value.get("phase") != "final_outer":
        raise OuterEvaluationSealedError(
            "GoalExit outer evaluation is sealed: no explicit final_outer authorization"
        )
    return value


def validate_kd_outer_lock(
    config_path: Path = Path("configs/confirmatory_kd.yaml"),
    root: Path | None = None,
) -> dict:
    root = project_root() if root is None else root
    lock_path = root / "configs/final_goalexit_lock.yaml"
    lock = require_outer_authorization(root)
    config_path = resolve(root, config_path)
    config = load_yaml(config_path)
    if lock.get("schema") != "goalexit-kd-final-lock-v1":
        raise OuterEvaluationSealedError("GoalExit outer evaluation is sealed: wrong lock schema")
    expected = {
        "confirmatory_config_sha256": sha256(config_path),
        "protocol_sha256": sha256(resolve(root, config["protocol"])),
        "development_state_sha256": sha256(resolve(root, config["development_state"])),
        "development_metrics_sha256": sha256(resolve(root, config["development_metrics"])),
        "development_audit_sha256": sha256(resolve(root, config["development_audit"])),
        "cpu_benchmark_sha256": sha256(resolve(root, config["cpu_benchmark"])),
        "canonical_manifest_sha256": sha256(resolve(root, config["canonical_manifest"])),
        "split_manifest_sha256": sha256(resolve(root, config["split_manifest"])),
        "roberta_manifest_sha256": sha256(resolve(root, config["roberta_manifest"])),
        "authorized_families": list(config["families"]),
        "seed": int(config["seed"]),
        "epoch_rule": config["epoch_rule"],
        "primary_contrast": config["viability"]["primary_contrast"],
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise OuterEvaluationSealedError(
                f"GoalExit outer evaluation is sealed: lock mismatch for {key}"
            )
    implementation_commit = str(lock.get("implementation_commit", ""))
    if len(implementation_commit) != 40:
        raise OuterEvaluationSealedError(
            "GoalExit outer evaluation is sealed: implementation commit absent"
        )
    current_commit = git_commit(root)
    if current_commit == "UNCOMMITTED":
        raise OuterEvaluationSealedError(
            "GoalExit outer evaluation is sealed: source tree is not clean"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, current_commit],
        cwd=root,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise OuterEvaluationSealedError(
            "GoalExit outer evaluation is sealed: implementation commit is not an ancestor"
        )
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{implementation_commit}..{current_commit}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if changed != ["configs/final_goalexit_lock.yaml"]:
        raise OuterEvaluationSealedError(
            "GoalExit outer evaluation is sealed: post-freeze source changes detected"
        )
    output = dict(lock)
    output["lock_path"] = str(lock_path)
    output["lock_sha256"] = sha256(lock_path)
    output["execution_commit"] = current_commit
    return output
