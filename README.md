<div align="center">

# Gaze-Only Reading-Goal Distillation

### Code and reproducibility materials for gaze-only reading-goal classification

Hyunsu Go · Sumin Lee · Jinyong Jun · Kyeonghun Kim · Hyuk-Jae Lee · Nam-Joon Kim

</div>

## Overview

This repository accompanies the ICCE 2027 paper *Learning a Gaze-Only
Reading-Goal Classifier from a Word-Aligned Teacher*. The study asks whether
word-aligned supervision can be confined to training while a GRU uses only
fixation sequences at inference.

The evaluation uses regular first-reading trials from OneStop Eye Movements and
holds out readers and articles simultaneously. The primary distilled GRU
improves balanced accuracy from 0.5866 to 0.6205 without changing the student's
43,797-parameter inference architecture. Post hoc controls test label
smoothing, teacher input modality, random-seed sensitivity, simple baselines,
and the role of trial-level target correspondence. Shuffling teacher targets
within each class reduces but does not eliminate the distillation gain.

The repository contains source code, frozen configurations, aggregate results,
audit manifests, tests, and the paper source. OneStop data, model checkpoints,
and trial-level predictions are not distributed here.

## Repository map

| Path | Contents |
|---|---|
| `src/goalexit/` | Data construction, models, training, evaluation, and audits |
| `configs/` | Frozen experiment and analysis configurations |
| `artifacts/metrics/` | Machine-readable aggregate results |
| `artifacts/manifests/` | Data, split, model, and provenance audits |
| `docs/` | Protocols, reports, and reproducibility notes |
| `tests/` | Unit and integrity tests |
| `paper/` | IEEE manuscript source and compiled PDF |

## Setup

Python 3.10 is the reference version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[teacher,dev]'
PYTHONPATH=src pytest -q
```

Without the source dataset, the test suite skips three checks that reconstruct
the frozen reader/article splits from canonical trial tables. They run
automatically after the tables are built locally.

Download the ordinary-reading and information-seeking Paragraph reports from
the [OneStop Eye Movements release](https://github.com/lacclab/OneStop-Eye-Movements).
Copy `configs/development_goalexit.yaml`, replace its four source paths with
the local archives, and keep the supplied source hashes unchanged.

## Reproduction

The canonical tables and reader/article partitions can be rebuilt with:

```bash
PYTHONPATH=src python -m goalexit.data.build_canonical --config configs/local.yaml
PYTHONPATH=src python -m goalexit.splits.build_splits --config configs/local.yaml
PYTHONPATH=src python -m goalexit.audit.phase0 --config configs/local.yaml
PYTHONPATH=src python -m goalexit.data.cache_roberta --config configs/local.yaml
```

Training the teachers is most practical on a GPU. The deployed GRU and the
reported model-only latency benchmark run on CPU. Exact split seeds,
model-selection rules, software environments, execution stages, and artifact
mappings are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Results and provenance

The paper's main numerical sources are:

- `artifacts/metrics/outer_kd_confirmatory.json`
- `artifacts/metrics/outer_kd_batch_robustness.json`
- `artifacts/metrics/postconfirmatory_seed_extension.json`
- `artifacts/metrics/postfeedback_controls.json`
- `artifacts/metrics/postfeedback_shuffled_kd.json`
- `artifacts/manifests/fold_distribution_audit.json`
- `artifacts/manifests/cpu_benchmark_manifest.json`
- `artifacts/manifests/cpu_length_distribution.json`

Configurations and reports state whether an analysis was confirmatory or post
hoc. Released manifests use repository-relative paths or explicit placeholders
for external assets. Stored hashes identify the corresponding artifacts.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Code is
released under the [MIT License](LICENSE). The OneStop dataset remains under
its original licenses and is not redistributed by this repository.
