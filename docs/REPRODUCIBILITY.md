# Reproducibility guide

The paper reports the design choices needed to interpret the experiment. Exact
seeds, software versions, artifact mappings, and hashes live here and in the
frozen configurations so they do not interrupt the four-page argument.

## Data

Use the Paragraph Interest Period reports from OneStop Ordinary Reading and
OneStop Information Seeking. The four expected archive members and their
SHA-256 hashes are listed in `configs/development_goalexit.yaml`. Source data
are not redistributed.

Canonical construction retains regular first-reading trials, removes invalid
fixations, and excludes questions, answers, correctness, and post-reading
events. Feature definitions are frozen in `src/goalexit/features/schema.py` and
implemented in `src/goalexit/data/build_canonical.py` and
`src/goalexit/features/extract.py`.

## Splits and selection

- Outer split seed: `20271002`
- Outer design: five goal-stratified reader folds by five article folds
- Inner design: three reader/article holdouts within each outer training set
- Primary training seed: `13`
- Post hoc sensitivity seeds: `13`, `29`, and `47`
- Primary metric: balanced accuracy
- Decision threshold: `0.5`, fixed before outer evaluation

The exact reader and article assignments are in
`artifacts/manifests/split_manifest.json`. Learning-rate candidates,
distillation weights, epoch rules, batch sizes, and optimizer settings are in
`configs/development_goalexit.yaml` and `configs/confirmatory_kd.yaml`.

## Software and hardware

Canonical construction recorded Python 3.10, NumPy 2.2.6, pandas 2.3.3,
PyArrow 25.0.1, scikit-learn 1.7.2, and PyTorch 2.7.0. The post hoc matched
comparisons ran with PyTorch 2.6.0, CUDA 11.8, and one NVIDIA L40S. CPU latency
was measured with one PyTorch thread on an Intel Core Ultra 7 155H. The
machine-readable manifests are authoritative for individual runs.

## Execution order

1. Build canonical data and frozen splits.
2. Cache frozen RoBERTa-base word representations.
3. Run development selection without opening outer targets.
4. Validate `configs/final_goalexit_lock.yaml`.
5. Run the one-shot outer evaluation and analysis.
6. Run separately marked post hoc controls.

Representative commands are in the root README. Protocols in `docs/04`,
`docs/09`, `docs/12`, and `docs/17` define the corresponding analyses.

## Result mapping

| Paper evidence | Artifact |
|---|---|
| Primary model comparison | `artifacts/metrics/outer_kd_confirmatory.json` |
| Descriptive consistency across article batches | `artifacts/metrics/outer_kd_batch_robustness.json` |
| Matched-seed ablations | `artifacts/metrics/postconfirmatory_seed_extension.json` |
| Post hoc logistic controls | `artifacts/metrics/postfeedback_controls.json` |
| Class-conditional target shuffling | `artifacts/metrics/postfeedback_shuffled_kd.json` |
| Original CPU benchmark | `artifacts/manifests/cpu_benchmark_manifest.json` |
| Length-distribution benchmark | `artifacts/manifests/cpu_length_distribution.json` |
| Figure 1 trace | `artifacts/manifests/figure1_scanpath_provenance.json` |

Released manifests use repository-relative paths or explicit placeholders for
external data and run archives. Stored SHA-256 hashes and the mappings above
identify the corresponding artifacts.
