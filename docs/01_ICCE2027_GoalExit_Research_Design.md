# GoalExit: Resource-Aware Reading-Goal Recognition from Eye Movements for Adaptive Consumer Interfaces

## ICCE 2027 Research Design and Development Protocol

**Status:** Authoritative design for GoalExit Phase 0-2 development
**Target venue:** IEEE ICCE 2027
**Project:** `icce2027/GoalExit`
**Primary objective:** Obtain strong reading-goal recognition performance while reducing deployment model cost and the amount of gaze that must be observed before a reliable decision.

---

## 1. Research question

GoalExit studies a deployment-oriented problem rather than introducing reading-goal decoding itself.

> Can a strong text+gaze teacher transfer reading-goal knowledge to a compact gaze-only student, and can the student make reliable trial-specific early decisions under unseen-reader and unseen-text generalization?

The target is binary reading goal:

- `O`: ordinary reading
- `H`: information seeking

The scientific contribution is the **accuracy-efficiency-acquisition trade-off**, not the existence of a reading-goal signal.

The intended pipeline is:

`strong text+gaze teacher -> compact gaze-only student via privileged distillation -> anytime student -> calibrated adaptive stopping`

This design deliberately does **not** revisit comprehension correctness, reader-history personalization, SpanGaze, or the previous 4-state ReadMode target.

---

## 2. Contribution boundary and novelty

### 2.1 What prior work already establishes

Prior reading-goal work has already shown that eye movements can distinguish ordinary reading from information seeking, including under generalization settings and at partial prefixes. Therefore GoalExit must not claim that reading-goal decoding or early reading-goal prediction is new.

### 2.2 GoalExit contribution

GoalExit is successful only if it contributes at least two of the following three deployment properties while preserving competitive recognition quality:

1. **Privileged multimodal distillation:** a training-time text+gaze teacher improves a gaze-only deployment model.
2. **Trial-adaptive acquisition:** different trials stop at different fixation counts rather than using one fixed prefix for all trials.
3. **Compact deployment efficiency:** the final student has small parameter/storage/CPU cost and is evaluated as an edge/consumer inference model.

The paper should be framed around **resource-aware online recognition for adaptive consumer reading interfaces**.

### 2.3 Prohibited novelty claims

Do not claim:

- first reading-goal classifier;
- first early reading-goal classifier;
- first gaze Transformer for reading goals;
- first use of OneStop for reading-goal decoding;
- that a CPU proxy is embedded-device latency unless actual embedded hardware is measured.

---

## 3. Data and canonical task

### 3.1 Development population

Use regular first-reading Paragraph trials from both OneStop reading regimes:

- ordinary reading regular trials;
- information-seeking regular trials.

Expected public-data scale is approximately 19,438 regular trials, 360 readers, and 30 articles. Phase 0 must reconstruct and report exact counts from the canonical source files. The report, not this expectation, is authoritative for actual counts.

Repeated-reading trials are excluded from the primary GoalExit population.

### 3.2 Unit of prediction

One prediction corresponds to one regular Paragraph-reading trial.

### 3.3 Target

`reading_goal = 0` for ordinary reading and `reading_goal = 1` for information seeking.

The label is an experimental condition, not a post-reading outcome.

### 3.4 Permitted deployment-time inputs

The final student may use only Paragraph-screen gaze information available up to the current fixation. Permitted raw/canonical fields include, when derivable without target leakage:

- normalized fixation x/y;
- fixation duration;
- delta x/y from the previous fixation;
- saccade amplitude;
- saccade direction represented by sine/cosine or equivalent continuous encoding;
- line-transition indicator derivable from the Paragraph screen;
- normalized within-trial fixation index;
- validity/missingness masks that do not encode the target.

If IA/word mapping is required for a feature, it may be used only when the same information is realistically available from the rendered Paragraph layout at deployment. The student must not use lexical identity, question content, or article identity.

### 3.5 Teacher-only privileged inputs

The teacher may additionally use:

- Paragraph text;
- word identity/subword embeddings;
- word-aligned aggregate gaze derived from the same Paragraph gaze stream.

The teacher must never use question text, answer options, correctness, selected answer, or any post-Paragraph information.

### 3.6 Forbidden inputs and shortcuts

All models are forbidden from using:

- reader ID or learned reader embedding;
- reading-regime flag except as the target label;
- article ID, paragraph ID, question ID, or file/path tokens that encode them;
- question text or answer options;
- Question Preview gaze or question-screen events;
- selected answer or correctness;
- trial/session position, block index, timestamp, repeat flag, or metadata that can reveal regime;
- prior trials from the same reader as identity/history features;
- any statistics computed using validation/test readers or articles.

Phase 0 must run a column-level and code-level leakage audit before training.

---

## 4. Generalization design

### 4.1 New split seed

Use a new GoalExit split seed:

`20271002`

Do not reuse ReadAdapt, EarlyComprehend, SpanGaze, or ReadMode fold assignments.

### 4.2 Outer structure

Construct five reader folds and five article folds. Their Cartesian product yields 25 unseen-reader + unseen-text (`UB`) outer contexts.

Reader folds must be stratified by reading goal because goal is constant at the participant level. Each fold should contain near-equal numbers of ordinary and information-seeking readers.

### 4.3 Development structure

For Phase 0-2, the 25 outer contexts remain sealed. For each outer context, construct three deterministic paired inner holdouts from the remaining readers and articles.

All model families for a given evaluation must use identical train/validation populations.

Expected development evaluations:

`25 outer contexts x 3 inner holdouts = 75 paired evaluations`

### 4.4 Primary metric

Primary scientific metric:

**Balanced Accuracy (BA)**

This is necessary because reading goal is participant-level and a within-reader AUROC is undefined for readers belonging to only one goal condition.

Secondary metrics:

- AUROC;
- macro-F1;
- Brier score;
- NLL;
- ECE;
- macro-by-article balanced accuracy.

### 4.5 Aggregation

For every model, report:

- mean metric across 75 paired evaluations;
- median;
- interquartile range;
- positive-direction count for paired contrasts;
- per-outer-context summary after averaging the three inner holdouts.

Do not treat 75 evaluations as independent binomial trials.

---

## 5. Canonical fixation representation

### 5.1 Event filtering

Use only valid Paragraph fixations after the same deterministic validity rules across both regimes. At minimum:

- positive finite duration;
- finite screen coordinates;
- deterministic chronological ordering;
- no question-screen or non-Paragraph events.

Retain the original source fixation index for auditing.

### 5.2 Student event vector

For fixation `i`, construct a compact continuous vector. The default design is:

1. `x_norm`
2. `y_norm`
3. `log1p(duration_ms)`
4. `dx_norm`
5. `dy_norm`
6. `log1p(saccade_amplitude)`
7. `sin(saccade_angle)`
8. `cos(saccade_angle)`
9. `line_change`
10. `fixation_position = i / max(1, current_prefix_length-1)`

All scale parameters must be fit on the training population only.

`fixation_position` may depend only on the observed prefix length, never the unknown final trial length.

### 5.3 Prefixes

Fixed reporting prefixes:

`K = {5, 10, 20, 40, 80, FULL}` fixations.

If a trial ends before a fixed checkpoint, that checkpoint is missing for that trial; do not pad with future-length information. FULL always uses the observed complete Paragraph fixation sequence.

Percentage-of-final-length prefixes are prohibited because final length is future information.

---

## 6. Models

## 6.1 B0 - GlobalStats-MLP

Purpose: inexpensive gaze-only baseline.

For every prefix, aggregate only the observed fixation events into a frozen feature vector including:

- fixation count;
- mean/SD/median fixation duration;
- total observed dwell time;
- mean/SD saccade amplitude;
- horizontal/vertical displacement summaries;
- forward/backward movement proportions;
- line-change rate;
- x/y dispersion;
- path length;
- duration and saccade quantiles.

Architecture:

`input -> 128 ReLU -> dropout 0.2 -> 64 ReLU -> binary logit`

The exact feature list must be frozen before any target-conditioned signal audit.

## 6.2 B1 - Undistilled Gaze-Only GRU

Purpose: compact sequence baseline and direct comparator for distillation.

Architecture:

- input LayerNorm;
- 2-layer causal GRU;
- hidden size 64;
- final observed hidden state;
- MLP head `64 -> 64 -> 1`;
- ReLU and dropout 0.2 in the head.

No bidirectional recurrence is allowed because deployment is online.

B1 is trained with ordinary binary cross-entropy on the reading-goal label.

## 6.3 T - Privileged Text+Gaze Teacher

Purpose: produce the strongest feasible development reference while using information available at training but intentionally absent from the final device model.

### Text encoder

Use `roberta-base` as the default frozen contextual encoder. Record the resolved model revision/commit and file hashes. If the model cannot be obtained reproducibly in the execution environment, stop and document the dependency failure rather than silently substituting another language model.

### Word representation

For each Paragraph word:

- average its RoBERTa subword hidden states;
- compute word-aligned gaze summaries from Paragraph fixations only;
- include a learned relative word-position embedding.

Teacher gaze summaries per word should include the frozen set:

- fixation count;
- total dwell;
- mean fixation duration;
- first fixation duration;
- first-pass dwell when derivable deterministically;
- regression-in count;
- revisit count;
- skip indicator.

### Fusion encoder

Default teacher head:

- text projection `768 -> 128`;
- gaze projection `8 -> 64`;
- concatenate with position embedding;
- project to `d_model=192`;
- prepend learned CLS token;
- 3-layer Transformer encoder;
- 6 attention heads;
- feed-forward size 384;
- dropout 0.1;
- CLS binary classifier.

RoBERTa remains frozen in Phase 0-2. Do not fine-tune it unless a later researcher-authorized addendum explicitly permits it.

## 6.4 T-text - Text-Only Shortcut Control

Use the exact teacher text branch and fusion capacity but zero/remove all gaze features.

This control should not meaningfully decode reading goal under strict reader/article generalization. A surprisingly strong T-text result triggers a shortcut audit before proceeding.

## 6.5 D - Full-Sequence Distilled Student

D uses exactly the B1 student architecture and deployment inputs.

Teacher T is training-only. For each training trial, cache the teacher's full-sequence logit/probability using a teacher fit that did not train on the validation population used to evaluate D.

Loss:

`L_D = (1-alpha) * BCE(y, p_S) + alpha * T_kd^2 * KL(p_T^T_kd || p_S^T_kd)`

Default distillation temperature:

`T_kd = 2.0`

Development candidates:

`alpha in {0.25, 0.50, 0.75}`

Do not add feature-level, attention-level, contrastive, or self-supervised losses during Phase 0-2.

## 6.6 A - Anytime Distilled Student

A uses the same architecture and parameter count as D, but it is trained to produce predictions at all fixed prefixes.

For each training trial, evaluate prefixes in `K={5,10,20,40,80,FULL}` that exist for that trial. At each prefix `k`:

`L_k = (1-alpha) * BCE(y, p_S,k) + alpha * T_kd^2 * KL(p_T,FULL^T_kd || p_S,k^T_kd)`

Use equal weight across available prefixes within a trial, then average across trials.

The teacher's full-sequence output is privileged training information. It is never available at deployment.

---

## 7. Training and model selection

### 7.1 Common principles

- All preprocessing is fit on training data only.
- Use paired splits across model families.
- Early stopping uses inner validation only.
- No model may inspect an outer-test row during Phase 0-2.
- Every candidate must record code commit, data hash, split hash, config hash, random seed, and selected epoch.

### 7.2 Seeds

Primary development seed:

`13`

Additional seed stability is not authorized in Phase 0-2. Seeds 37 and 71 are reserved for a later addendum if all development gates pass.

### 7.3 Candidate grids

Keep grids intentionally small.

B0:

- learning rate `{1e-3, 3e-4}`

B1:

- learning rate `{1e-3, 3e-4}`

Teacher T / T-text fusion head:

- learning rate `{3e-4, 1e-4}`

D:

- student learning rate `{1e-3, 3e-4}`
- `alpha in {0.25, 0.50, 0.75}`
- `T_kd = 2.0` fixed

A:

- freeze the selected D learning rate and alpha for the same outer context;
- do not launch a new hyperparameter grid for A.

Maximum epochs and patience should be fixed before candidate training. Recommended defaults:

- max epochs 30;
- patience 5;
- minimum delta 0.

If hardware requires deterministic batch-size adjustment, document it without changing scientific inputs or sequence truncation.

### 7.4 Selection metric

All model selection uses mean validation balanced accuracy. Tie-breakers, in order:

1. lower validation NLL;
2. smaller model / lower parameter count if still tied;
3. lower learning rate only under an exact remaining tie.

No prefix is selected as the primary result. Fixed-prefix curves are reported in full.

---

## 8. Distillation leakage controls

Teacher predictions used as soft targets must be generated without fitting the teacher on the validation examples used to evaluate the student.

For each inner evaluation:

1. fit candidate T only on the inner training population;
2. generate teacher logits for that same inner training population for student training;
3. train D/A on the inner training population;
4. evaluate all models on the untouched paired inner validation population.

If teacher training logits are generated in-sample, this is allowed because they are training-only privileged targets. Validation labels must never enter teacher or student fitting.

Do not use a teacher trained on the full development pool to generate soft targets before the split is applied.

---

## 9. Adaptive stopping

Adaptive stopping is evaluated only after A is trained.

### 9.1 Online decision times

Starting after fixation 5, evaluate A every 5 observed fixations:

`5, 10, 15, 20, ... , FULL`

No final trial length is provided to the policy.

### 9.2 Calibration

Fit one temperature-scaling scalar on the inner validation training-selection procedure only. Calibration parameters are model/context-specific and must never use outer labels.

### 9.3 Stopping rule family

At observed fixation count `t`, define confidence:

`c_t = max(p_t, 1-p_t)`

A trial may stop when both conditions hold:

1. `c_t >= tau`
2. the predicted class has been identical at the current and previous decision time.

If no rule fires, stop at FULL.

Candidate confidence thresholds are frozen:

`tau in {0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}`

### 9.4 Operating points

Choose two operating points using inner validation only.

**AP-1: accuracy-preserving**

Among policies with:

`BA_exit >= BA_A_FULL - 0.010`

choose the policy with the smallest mean observed fixation fraction. Ties: lower median fraction, then higher BA, then larger tau.

**AP-2: aggressive**

Among policies with:

`BA_exit >= BA_A_FULL - 0.020`

choose the policy with the smallest mean observed fixation fraction using the same tie-breakers.

If no policy satisfies an operating-point constraint, report that operating point as unavailable.

### 9.5 Acquisition metrics

Report:

- mean observed fixations;
- median observed fixations;
- P90 observed fixations;
- mean observed fraction of the full trial;
- median observed fraction;
- mean acquisition reduction;
- median acquisition reduction;
- BA vs observed-fixation Pareto curve.

Observed fraction is computed only for evaluation/reporting after the stopping decision; final length is not an input to the policy.

---

## 10. Deployment efficiency

For B1, D, and A report:

- trainable parameter count;
- serialized model size;
- input feature dimension;
- single-thread CPU mean/median/P95 inference latency per newly observed fixation;
- single-thread CPU mean/median/P95 end-to-end trial inference latency under FULL and AP-1/AP-2;
- peak resident memory when measurable reproducibly.

CPU benchmark requirements:

- record CPU model, OS, Python/PyTorch versions;
- set framework thread count to 1;
- warm up before timing;
- benchmark fixed saved checkpoints and saved evaluation inputs;
- do not include disk I/O or dataset parsing in model latency.

If a Raspberry Pi, Android, Jetson, or other actual consumer/edge device is available later, that measurement may be added by addendum. It is not required for Phase 0-2.

---

## 11. Phase 0 - Audit and reconstruction

Before training, the implementation must produce a repository/data audit covering:

1. exact ordinary/information-seeking regular trial counts;
2. exact reader and article counts;
3. class balance overall and per proposed reader fold;
4. fixation counts and sequence-length distribution;
5. Paragraph-only event reconstruction;
6. forbidden-field audit;
7. reader-goal constancy audit;
8. article coverage in both reading goals;
9. student feature computability;
10. teacher word-alignment feasibility;
11. availability and reproducibility of `roberta-base`;
12. split hashes for the new GoalExit folds;
13. confirmation that all 25 outer contexts are sealed.

Also perform a **text-only shortcut audit** before interpreting teacher performance.

If any target can be inferred deterministically from an input metadata field, stop and fix the canonical schema before training.

---

## 12. Phase 1 - Strong recognition and distillation

Run B0, B1, T, T-text, and D on all 75 paired inner development evaluations.

### Gate T - Teacher value

Teacher development is considered technically worthwhile for distillation when all are true:

1. `mean BA(T - B1) >= +0.010`;
2. `T > B1` in at least `45/75` paired evaluations;
3. T-text does not reproduce the teacher gain: `mean BA(T-text - B1) <= 0.5 * mean BA(T - B1)` whenever the teacher gain is positive;
4. no shortcut/leakage audit failure.

If Gate T fails, stop before D/A expansion and report that privileged text+gaze teaching did not establish sufficient incremental value.

### Gate D - Distillation value

If Gate T passes, D is supported when all are true:

1. `mean BA(D - B1) >= +0.005`;
2. `D > B1` in at least `45/75` paired evaluations;
3. D uses exactly the same deployment inputs and architecture capacity as B1;
4. D parameter count is no larger than B1 except negligible scalar calibration state.

If Gate D fails, do not claim privileged distillation improves the deployment model.

---

## 13. Phase 2 - Anytime model and adaptive stopping

Only run Phase 2 if Gate T and Gate D pass.

Train A using the selected D settings, with no new hyperparameter grid.

### Gate A0 - Anytime training preservation

Require:

1. `mean BA(A_FULL - D) >= -0.005`;
2. A parameter count equals D parameter count;
3. no systematic calibration degradation larger than 0.02 absolute ECE without explanation.

If A0 fails, stop: multi-prefix training damaged the deployment model too much.

### Gate E - Adaptive acquisition value

Gate E passes when AP-1 exists and:

1. `BA_AP1 >= BA_A_FULL - 0.010` by construction;
2. mean fixation acquisition reduction is at least 30%;
3. median fixation acquisition reduction is at least 25%;
4. at least 20 of 25 outer-context development summaries show positive mean fixation reduction at their paired AP-1 policy;
5. no subgroup defined by reading goal loses more than 3.0 pp BA relative to A_FULL when aggregated across development predictions.

AP-2 is descriptive and is not required to pass Gate E.

### Development decision

- Gate T FAIL -> stop GoalExit teacher/distillation direction.
- Gate T PASS, Gate D FAIL -> teacher useful but distillation unsupported; stop before final claims.
- T/D PASS, A0 FAIL -> stop anytime direction.
- T/D/A0 PASS, Gate E FAIL -> report strong compact recognition but no sufficient adaptive-acquisition contribution; do not open outer.
- All PASS -> prepare Phase 3 addendum for seed stability, final lock, and one sealed outer evaluation. Phase 0-2 execution still stops here.

---

## 14. Sanity checks and negative controls

Mandatory checks:

### 14.1 Reader-level label permutation

Because reading goal is participant-level, permute goal labels at the reader level, not trial level. Re-run a reduced B1 pipeline on a fixed subset of development splits. Performance should approach chance.

### 14.2 Text-only control

T-text must not become a strong reading-goal classifier under strict UB splitting. If it does, audit article/version/session shortcuts.

### 14.3 Regime-metadata probe

Train a simple logistic probe on all non-gaze metadata that remains in the canonical model table after forbidden fields are removed. If it materially exceeds chance, identify and remove the shortcut before official training.

### 14.4 Prefix causality

For every saved prefix prediction, verify that no fixation after the prefix cutoff contributes to any student input or preprocessing statistic.

### 14.5 Outer guard

The evaluation entry point must refuse to evaluate an outer context unless a later researcher-approved final-lock file exists with matching hashes.

---

## 15. Required artifacts

The Phase 0-2 run must create at least:

- `docs/02_repo_audit.md`
- `docs/03_phase0_2_development_report.md`
- `docs/deviations.md`
- `artifacts/metrics/development_goalexit.json`
- `artifacts/manifests/development_output_audit.json`
- `artifacts/manifests/split_manifest.json`
- `artifacts/manifests/teacher_cache_manifest.json`
- `artifacts/manifests/cpu_benchmark_manifest.json`
- `artifacts/manifests/outer_guard_audit.json`

Per-run records must include:

- model family;
- split/context/inner-holdout IDs;
- seed;
- candidate hyperparameters;
- selected epoch;
- train/validation counts;
- class balance;
- data/split/config/code hashes;
- whether any outer row was touched;
- checkpoint hash;
- prediction hash.

Teacher soft-target caches require explicit provenance back to the exact teacher checkpoint and training population.

---

## 16. Deviations policy

Scientific ambiguities must be appended to `docs/deviations.md` before the affected official result is generated whenever possible.

Permitted implementation deviations include:

- memory-safe batching without sequence truncation;
- deterministic numerical-stability fixes;
- field-name/schema mapping supported by source evidence;
- unavailable optional device benchmarking.

The following require researcher authorization and must not be silently changed:

- task population;
- target definition;
- split seed or fold structure;
- student deployment inputs;
- teacher privileged inputs;
- model architecture sizes;
- candidate grids;
- distillation temperature/grid;
- prefix set;
- stopping-rule family;
- gate criteria;
- outer-test access.

Technical bugs may be fixed without rerunning unaffected scientific outputs only when the bug demonstrably did not change those outputs; add regression tests and document the evidence.

---

## 17. Phase 0-2 stop rule

The Phase 0-2 execution must stop after producing the development evidence package.

It is explicitly **not authorized** to:

- run seeds 37/71;
- tune teacher/student architecture after seeing Gate results;
- fine-tune RoBERTa;
- add another pretrained text model;
- add self-supervised gaze pretraining;
- add Transformer students;
- change fixed prefixes;
- invent new stopping policies after inspecting results;
- create a final lock;
- evaluate any outer row;
- write final-paper claims as if development results were confirmatory.

The next action after Phase 0-2 is a researcher review.

---

## 18. Intended ICCE 2027 paper story if development succeeds

Working title:

**GoalExit: Resource-Aware Reading-Goal Recognition from Eye Movements for Adaptive Consumer Interfaces**

Planned contribution statements:

1. We formulate reading-goal recognition as a resource-aware online inference problem for adaptive consumer reading interfaces.
2. We transfer training-time text-conditioned gaze knowledge from a privileged teacher to a compact gaze-only deployment model.
3. We characterize a trial-adaptive accuracy-vs-gaze-acquisition trade-off under strict unseen-reader and unseen-text generalization, together with CPU deployment cost.

The expected headline table contains:

- Teacher BA and deployment cost;
- B1 undistilled student BA;
- D distilled student BA;
- A FULL BA;
- AP-1/AP-2 BA;
- mean/median fixation reduction;
- parameter count/model size;
- CPU latency.

A successful paper must not depend on a single favorable fixed prefix.

---

## 19. Researcher decision checklist after Phase 0-2

Before any final evaluation, verify all of the following:

- [ ] canonical trial/reader/article counts audited;
- [ ] forbidden inputs absent;
- [ ] new GoalExit split hashes fixed;
- [ ] 25 outer contexts untouched;
- [ ] T-text shortcut control acceptable;
- [ ] Gate T passed;
- [ ] Gate D passed;
- [ ] Gate A0 passed;
- [ ] Gate E passed;
- [ ] reader-level permutation sanity check passed;
- [ ] all 75 paired development evaluations complete;
- [ ] teacher-cache provenance complete;
- [ ] CPU benchmark reproducible;
- [ ] no unauthorized model family or hyperparameter search performed;
- [ ] deviations log reviewed;
- [ ] Phase 0-2 report explicitly stops before outer evaluation.

Only after this checklist is satisfied should a separate Phase 3/final-confirmation addendum be written.
