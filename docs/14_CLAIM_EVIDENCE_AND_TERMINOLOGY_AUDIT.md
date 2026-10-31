# Claim, evidence, and terminology audit

This audit maps the manuscript's substantive statements to either a primary
source or a reproducible project artifact. It also records the terminology
used consistently in the paper.

## Claim-to-evidence map

| Manuscript statement | Evidence | Scope retained in the paper |
|---|---|---|
| Reading goals are associated with differences in fixation behavior and scanpaths. | Rayner (1998); Shubi and Berzak (2023) | Motivation, not a causal claim from the present experiment |
| OneStop contains ordinary-reading, information-seeking, and repeated-reading regimes. | Berzak et al. (2025), *Scientific Data* | Dataset description |
| Prior reading-goal work combines fixation-level gaze with contextual language representations. | Shubi et al. (2025), ACL, New Item & Participant condition | Prior-work context for the inference inputs |
| The retained data contain 19,438 trials from 360 readers and 30 articles. | `artifacts/manifests/canonical_goalexit.json` | Reported once in the data section |
| Student inputs contain ten fixation-level features and exclude reader, article, question, answer, and session identifiers. | `artifacts/manifests/feature_schema.json`; `artifacts/manifests/forbidden_field_audit.json` | Exact input definition and leakage boundary |
| Every test trial has a reader and article absent from its training set. | `artifacts/manifests/split_manifest.json`; `artifacts/manifests/outer_guard_audit.json` | Simultaneous reader and article holdout |
| Every reader fold contains both goals from all three article batches, and all test partitions contain both goals. | `artifacts/manifests/fold_distribution_audit.json` | Batch composition and test-partition balance |
| Text--gaze KD improves balanced accuracy over the architecture-matched gaze GRU by 0.0339. | `artifacts/metrics/outer_kd_confirmatory.json`; `artifacts/manifests/outer_kd_output_audit.json` | Primary out-of-fold comparison |
| The primary KD difference is positive within each of the three article batches. | `artifacts/metrics/outer_kd_batch_robustness.json` | Post hoc descriptive consistency check; no batch-specific inferential claim |
| Reader-clustered and crossed reader/article intervals for the primary contrast exclude zero. | `artifacts/metrics/outer_kd_confirmatory.json` | Sampling uncertainty under the two specified resampling schemes |
| Label smoothing does not reproduce the KD improvement. | `artifacts/metrics/postconfirmatory_seed_extension.json`; seeds 13, 29, and 47 | Post hoc control, not part of the primary analysis |
| Original text--gaze KD exceeds class-shuffled KD by 0.0190 BA; reader and crossed intervals exclude zero. The shuffled control remains 0.0185 above the GRU. | `artifacts/metrics/postfeedback_shuffled_kd.json`; `docs/18_POSTFEEDBACK_SHUFFLED_KD_REPORT.md` | Trial-level correspondence contributes, but class-conditional soft targets or their optimization effect also retain a gain |
| A word-aligned gaze teacher improves the student across all three tested seeds. | `artifacts/metrics/postconfirmatory_seed_extension.json` | Supports teacher distillation without contextual text |
| Contextual text has no demonstrated incremental benefit in the direct teacher or student contrasts. | `artifacts/metrics/postconfirmatory_seed_extension.json`; `artifacts/manifests/postconfirmatory_seed_extension_audit.json` | Absence of demonstrated incremental benefit, not proof of equivalence |
| The gaze-summary logistic control reaches 0.6066 BA under the frozen outer splits and nested selection. | `artifacts/metrics/postfeedback_controls.json` | Post hoc stronger baseline; the crossed interval for Distilled GRU minus this control includes zero |
| Distilled and undistilled GRUs have identical inputs, parameters, and inference operations. | model configuration, result manifests, and parameter-count audit | Distillation changes training only |
| The distilled GRU occupies 176 KiB; model-only latency has a 5.08 ms median and 16.57 ms P95 across the 19,438 retained sequences. | `artifacts/manifests/cpu_length_distribution.json` | Single-thread forward-pass latency; no preprocessing, eye-tracking, end-to-end, or embedded-device claim |
| Figure 1's scanpath schematic reflects observed line progression, a return sweep, and relative fixation duration. | `artifacts/manifests/figure1_scanpath_provenance.json` | Five-marker schematic; three size classes follow observed durations while positions remain qualitative |

## Terminology decisions

| Term used | Reason |
|---|---|
| **reading goal** | Matches the ACL 2025 task and distinguishes the intended purpose from a generic classification label |
| **ordinary reading for comprehension** / **information seeking** | Matches the OneStop and ACL task labels |
| **fixation**, **saccade**, **regression**, **scanpath** | Standard eye-movement terminology; `event` is avoided when a fixation is meant |
| **fixation sequence** | The student's chronological input representation |
| **fixation-to-word alignment** | The inference-time operation required to associate fixations with paragraph words |
| **word-level gaze summaries** | The teacher input produced after alignment and aggregation; not used as a synonym for the alignment operation |
| **word-aligned text--gaze teacher** / **word-aligned gaze teacher** | Distinguishes teacher modalities while making their shared word-level aggregation explicit |
| **Gaze-summary MLP** | Identifies a baseline built from trial-level aggregate gaze features rather than a fixation sequence |
| **Dwell-time LR** / **Gaze-summary LR** | Identifies the two post hoc logistic controls by their input features |
| **Distilled GRU** / **Gaze GRU** | Distinguishes training objective while preserving the fact that the inference architectures are identical |
| **primary analysis** / **post hoc ablation** | States when the analyses were designed without repeatedly defending the protocol |
| **random seed** | Standard term; `optimization seed` is avoided |
| **reader-clustered bootstrap** / **crossed reader/article bootstrap** | Names the resampled units and dependence structure directly |
| **model-only latency** | Restricts the timing claim to the measured forward pass |

## Post hoc evidence status

The seed-balanced extension audit is `PASS`. It validates 150/150 new result
artifacts, revalidates 300 retained base artifacts, verifies checkpoint,
prediction, and result hashes, and confirms exact 19,438-trial prediction
partitions. All four student conditions and both teacher modalities use seeds
13, 29, and 47 with matched student capacity and KD settings within seed.

## Statements deliberately not made

- The paper does not claim state-of-the-art reading-goal accuracy.
- The paper does not claim that contextual text causes the distillation gain.
- The paper does not claim equivalence between text--gaze and word-aligned gaze
  teachers; the direct contrasts are interpreted as no demonstrated
  incremental benefit.
- The paper does not claim partial-scanpath, end-to-end, embedded-device, or
  autonomous-interface performance.
- The paper does not treat the number of reader--article partitions as a
  contribution.

## Reference quality

The bibliography uses the official ACL and EMNLP proceedings for the task and
related prediction work, the peer-reviewed *Scientific Data* descriptor for
OneStop, the original papers for knowledge distillation, privileged
information, label smoothing, AdamW, balanced accuracy, calibration, GRUs, and
Transformers, and the original RoBERTa technical report. All citation keys in
the manuscript are defined and used. A final source-level cross-check confirmed
the author lists, titles, venues, years, page ranges, and available DOIs for all
14 entries. It also confirmed that the cited sources support the associated
dataset, prior-work, and method statements without extending their scope.
