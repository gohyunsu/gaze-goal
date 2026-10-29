from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from goalexit.benchmarks.cpu import _cpu_model
from goalexit.data.dataset import GoalExitStore
from goalexit.features.extract import student_prefix
from goalexit.train.engine import load_checkpoint
from goalexit.utils import load_yaml, project_root, resolve, save_json, sha256


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def run(config_path: Path) -> dict[str, Any]:
    root = project_root()
    config = load_yaml(resolve(root, config_path))
    dev_config = load_yaml(resolve(root, config["development_config"]))
    benchmark = json.loads(
        resolve(root, config["latency"]["source_manifest"]).read_text(encoding="utf-8")
    )
    family = str(config["latency"]["family"])
    source = next(row for row in benchmark["measurements"] if row["family"] == family)
    checkpoint_path = Path(source["checkpoint_path"])
    if sha256(checkpoint_path) != source["checkpoint_sha256"]:
        raise ValueError("Latency checkpoint hash mismatch")
    model, preprocessor, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    store = GoalExitStore.load(
        resolve(root, dev_config["outputs"]["trial_table"]),
        resolve(root, dev_config["outputs"]["fixation_table"]),
        resolve(root, dev_config["outputs"]["word_table"]),
    )
    trial_ids = store.metadata["trial_id"].astype(str).tolist()
    transformed = store.transformed_student(trial_ids, preprocessor)
    rng = np.random.default_rng(int(config["latency"]["seed"]))
    order = rng.permutation(len(trial_ids))
    torch.set_num_threads(int(config["latency"]["threads"]))
    model.eval()
    median_id = min(
        trial_ids,
        key=lambda key: abs(
            len(transformed[key])
            - float(np.median([len(values) for values in transformed.values()]))
        ),
    )
    median_values = transformed[median_id]
    median_tensor = torch.from_numpy(
        student_prefix(median_values, len(median_values))[None, :]
    )
    median_length = torch.tensor([len(median_values)], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(int(config["latency"]["warmup"])):
            model(median_tensor, median_length)
        elapsed = np.empty(len(trial_ids), dtype=np.float64)
        lengths = np.empty(len(trial_ids), dtype=np.int32)
        for output_position, input_position in enumerate(order):
            trial_id = trial_ids[int(input_position)]
            values = transformed[trial_id]
            tensor = torch.from_numpy(student_prefix(values, len(values))[None, :])
            length = torch.tensor([len(values)], dtype=torch.long)
            start = time.perf_counter_ns()
            model(tensor, length)
            elapsed[output_position] = (time.perf_counter_ns() - start) / 1_000_000.0
            lengths[output_position] = len(values)
    quartiles = np.quantile(lengths, [0.25, 0.50, 0.75])
    bins = [
        ("shortest_quartile", lengths <= quartiles[0]),
        ("second_quartile", (lengths > quartiles[0]) & (lengths <= quartiles[1])),
        ("third_quartile", (lengths > quartiles[1]) & (lengths <= quartiles[2])),
        ("longest_quartile", lengths > quartiles[2]),
    ]
    result = {
        "schema": "goalexit-cpu-length-distribution-v1",
        "status": "complete",
        "family": family,
        "checkpoint_sha256": source["checkpoint_sha256"],
        "device": "CPU",
        "cpu_model": _cpu_model(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": int(config["latency"]["threads"]),
        "warmup_iterations": int(config["latency"]["warmup"]),
        "trials": len(trial_ids),
        "measurement_order_seed": int(config["latency"]["seed"]),
        "semantics": "one model-only single-sequence forward pass for every retained trial after warm-up; tensor construction, preprocessing, parsing, and I/O excluded",
        "input_fixations": _summary(lengths.astype(np.float64)),
        "latency_ms": _summary(elapsed),
        "length_quartiles": {
            name: {
                "trials": int(mask.sum()),
                "input_fixations": _summary(lengths[mask].astype(np.float64)),
                "latency_ms": _summary(elapsed[mask]),
            }
            for name, mask in bins
        },
        "consumer_or_embedded_hardware_claimed": False,
    }
    save_json(resolve(root, config["outputs"]["latency_manifest"]), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/postfeedback_controls.yaml")
    )
    args = parser.parse_args()
    result = run(args.config)
    print(
        f"trials={result['trials']} "
        f"median_ms={result['latency_ms']['median']:.6f} "
        f"p95_ms={result['latency_ms']['p95']:.6f}"
    )


if __name__ == "__main__":
    main()
