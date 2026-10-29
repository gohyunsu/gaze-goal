from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from goalexit.data.dataset import GoalExitStore
from goalexit.features.extract import student_prefix
from goalexit.stopping.adaptive import stop_trace
from goalexit.train.engine import load_checkpoint
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.casefold().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _timing_summary(nanoseconds: list[int]) -> dict[str, float]:
    values = np.asarray(nanoseconds, dtype=np.float64) / 1_000_000.0
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "iterations": len(values),
    }


def _representative_result(state: dict[str, Any], family: str) -> dict[str, Any]:
    values = [
        result
        for families in state["selections"].values()
        if family in families
        for result in families[family]["selected_results"]
    ]
    if len(values) != 75:
        raise ValueError(f"Expected 75 selected {family} evaluations")
    mean_ba = float(np.mean([value["metrics"]["FULL"]["balanced_accuracy"] for value in values]))
    return min(
        values,
        key=lambda value: abs(value["metrics"]["FULL"]["balanced_accuracy"] - mean_ba),
    )


def _measure_model(
    family: str,
    result: dict[str, Any],
    store: GoalExitStore,
    config: dict[str, Any],
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device("cpu")
    model, preprocessor, checkpoint = load_checkpoint(Path(result["checkpoint_path"]), device)
    prediction = pd.read_parquet(result["prediction_path"])
    full = prediction.loc[prediction["prefix"].eq("FULL")].copy()
    median_length = float(full["full_fixation_count"].median())
    representative_id = str(
        full.iloc[(full["full_fixation_count"] - median_length).abs().argsort().iloc[0]]["trial_id"]
    )
    base = preprocessor.transform(store.student_base[representative_id])
    full_input = student_prefix(base, len(base))
    lengths = torch.tensor([len(base)], dtype=torch.long)
    full_tensor = torch.from_numpy(full_input[None, :])
    model.eval()
    torch.set_num_threads(int(config["benchmark"]["threads"]))
    warmup = int(config["benchmark"]["warmup"])
    iterations = int(config["benchmark"]["iterations"])
    with torch.inference_mode():
        for _ in range(warmup):
            model(full_tensor, lengths)
        full_times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            model(full_tensor, lengths)
            full_times.append(time.perf_counter_ns() - start)
        update_times = []
        for index in range(warmup + iterations):
            length = 1 + (index % len(base))
            values = torch.from_numpy(student_prefix(base, length)[None, :])
            observed = torch.tensor([length], dtype=torch.long)
            start = time.perf_counter_ns()
            model(values, observed)
            elapsed = time.perf_counter_ns() - start
            if index >= warmup:
                update_times.append(elapsed)
    adaptive_timings: dict[str, Any] = {"AP1": None, "AP2": None}
    if family == "a" and state.get("adaptive"):
        context_record = next(
            row
            for row in state["adaptive"]["context_records"]
            if row["context_id"] == checkpoint["run_config"]["context_id"]
        )
        trace_source = next(
            row
            for row in context_record["trace_sources"]
            if int(row["inner_holdout"]) == int(checkpoint["run_config"]["inner_holdout"])
        )
        trace = pd.read_parquet(trace_source["path"])
        for operating_point in ("AP1", "AP2"):
            selected = context_record[operating_point]
            if selected is None:
                continue
            stopped = stop_trace(
                trace,
                float(selected["threshold"]),
                float(context_record["temperature_calibration"]["temperature"]),
            ).set_index("trial_id")
            length = int(stopped.loc[representative_id, "observed_fixations"])
            tensor = torch.from_numpy(student_prefix(base, length)[None, :])
            observed = torch.tensor([length], dtype=torch.long)
            with torch.inference_mode():
                for _ in range(warmup):
                    model(tensor, observed)
                times = []
                for _ in range(iterations):
                    start = time.perf_counter_ns()
                    model(tensor, observed)
                    times.append(time.perf_counter_ns() - start)
            adaptive_timings[operating_point] = {
                **_timing_summary(times),
                "observed_fixations": length,
                "threshold": float(selected["threshold"]),
            }
    state_path = root / "research_storage/cpu_benchmarks" / family / "state_dict.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "family": family,
        "representative_selection": "selected evaluation nearest 75-evaluation mean FULL BA",
        "context_id": checkpoint["run_config"]["context_id"],
        "inner_holdout": checkpoint["run_config"]["inner_holdout"],
        "trial_id": representative_id,
        "input_fixations": len(base),
        "input_feature_dimension": int(full_input.shape[1]),
        "checkpoint_path": result["checkpoint_path"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "parameter_count": parameters,
        "serialized_state_path": str(state_path),
        "serialized_state_bytes": state_path.stat().st_size,
        "serialized_state_sha256": sha256(state_path),
        "per_new_fixation_update": {
            **_timing_summary(update_times),
            "semantics": "exact causal prefix re-evaluation because fixation_position is normalized by current observed prefix length",
        },
        "end_to_end_trial_full": _timing_summary(full_times),
        "end_to_end_trial_ap1": adaptive_timings["AP1"],
        "end_to_end_trial_ap2": adaptive_timings["AP2"],
        "peak_rss_kib_after_benchmark": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "disk_io_and_parsing_timed": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    state_path = resolve(root, config["outputs"]["output_root"]) / "selection_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    store = GoalExitStore.load(
        resolve(root, config["outputs"]["trial_table"]),
        resolve(root, config["outputs"]["fixation_table"]),
        resolve(root, config["outputs"]["word_table"]),
    )
    families = ["b1"]
    if state.get("gate_t", {}).get("status") == "PASS" and "gate_d" in state:
        families.append("d")
    if state.get("gate_d", {}).get("status") == "PASS" and "gate_a0" in state:
        families.append("a")
    measurements = [
        _measure_model(
            family,
            _representative_result(state, family),
            store,
            config,
            root,
            state,
        )
        for family in families
    ]
    manifest = {
        "schema": "goalexit-cpu-benchmark-v1",
        "status": "complete",
        "cpu_model": _cpu_model(),
        "os": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_threads": int(config["benchmark"]["threads"]),
        "warmup_iterations": int(config["benchmark"]["warmup"]),
        "measured_iterations": int(config["benchmark"]["iterations"]),
        "device": "CPU",
        "consumer_or_embedded_hardware_claimed": False,
        "interpretation": "single-thread CPU deployment proxy",
        "measurements": measurements,
        "not_authorized_or_unavailable": [
            family for family in ("d", "a") if family not in families
        ],
        "outer_rows_touched": False,
    }
    output = resolve(root, config["outputs"]["cpu_benchmark"])
    save_json(output, manifest)
    print(
        " ".join(
            f"{row['family']}_p95_ms={row['per_new_fixation_update']['p95_ms']:.6f}"
            for row in measurements
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/development_goalexit.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
