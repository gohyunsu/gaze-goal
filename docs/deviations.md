# GoalExit Phase 0-2 deviations and implementation resolutions

## Pre-training resolutions

### Trial-specific Paragraph layout geometry

The first fail-closed canonical reconstruction found that the raw IA geometry is
not byte-identical for every reader viewing the same paragraph version. The
scientific text sequence is audited separately and must be identical. Coordinate
normalization and line-change derivation retain each trial's actual rendered IA
geometry, which is the Paragraph layout information available to a deployed
interface. Layout-variant counts are retained in the canonical audit. No target,
identity, question, answer, session, or post-Paragraph field is added.

The diagnostic also found layout-induced IA tokenization variants such as
`credit-card` versus `credit-` + `card`. The canonical teacher paragraph key is
therefore the source article/difficulty/paragraph key plus the exact IA word
sequence hash. This preserves one-to-one RoBERTa-word/gaze alignment without
changing article-level splitting or adding a model input. Counts of source keys
with multiple IA word sequences are reported in the canonical manifest.

### Teacher relative-position embedding

The design fixes a learned relative word-position embedding but does not state
its width. Before official training, it is frozen at 32 dimensions. The specified
128-dimensional text projection and 64-dimensional gaze projection are
concatenated with this embedding and projected to the required `d_model=192`.

### GRU inter-layer dropout

The design specifies dropout 0.2 in the `64 -> 64 -> 1` head but does not specify
recurrent inter-layer dropout. It is frozen at zero; the required 0.2 dropout is
applied only in the head. No architecture result was inspected before this
resolution.

### Frozen RoBERTa cache precision

The frozen `roberta-base` last hidden states are averaged over subwords in
float32 and stored as float16 to reduce cache storage. They are converted back to
float32 before the trainable teacher fusion encoder. RoBERTa is never fine-tuned.
The unused RoBERTa pooler is not instantiated; word features use only the frozen
last hidden state.

### Exact-tie fallback for distillation alpha

The prescribed selection tie-breakers end with lower learning rate but do not
define an ordering if two D candidates remain exactly tied while differing only
in alpha. Before official training, a final deterministic fallback of lower
alpha is frozen. It applies only after identical mean BA, mean NLL, parameter
count, and learning rate.

### Reduced reader-permutation sanity subset

Before official training, the reduced fixed subset is frozen to inner holdout 0
of contexts `r0_a0`, `r1_a1`, `r2_a2`, `r3_a3`, and `r4_a4`. Labels are shuffled
as a reader-level mapping with base seed 2027100213. The sanity criterion is mean
permuted BA at most 0.55; it is not a scientific model-selection gate.

## Pre-run technical fixes

The first official command stopped before training because the clean-tree guard
tested the `rev-parse` output variable instead of the `git status` output. No
official checkpoint, prediction, or metric was created. The variable reference
was corrected, a regression test was added, and the implementation was committed
again before any official run.

The next execution was externally terminated with signal 143 before any teacher
candidate completed. The 24 B0 and 13 B1 partial runs from commit `a07f616` were
moved intact to `research_storage/quarantine/pre_cpu_rebalance_a07f616`; none is
used as official evidence. To make individual candidates finish within the
execution-session lifetime, CPU scheduling was changed from four workers with
three threads each to two workers with ten threads each. Scientific batch size,
optimizer, architecture, candidate grid, epochs, patience, seed, and data were
not changed. All official candidates are regenerated under the new commit.

The two-worker execution then exposed a different session-lifetime risk: two
long B1 candidates produced no completed candidate artifact for more than 24
minutes because progress was recorded only after an entire candidate finished.
The run was stopped intentionally. Its 12 complete B0 and 3 complete B1
candidate triples (checkpoint, prediction, and metric) were preserved under
`research_storage/quarantine/pre_epoch_resume_60a382d` and are not used as
official evidence. Candidate training now atomically saves model, optimizer,
best-model, early-stopping history, and Python/NumPy/Torch RNG state after every
completed epoch. A resumed process is bound to the exact frozen run identity and
continues at the next epoch; completed final artifacts remove the transient
state. This is an execution-resilience change only. The scientific data,
features, architecture, optimization settings, candidate grids, seeds, and
gates are unchanged, and all official candidates are regenerated under the
post-fix commit.

With epoch resume in place, the two-worker schedule was benchmarked against the
earlier four-worker trace using identical completed B1 metrics. The candidate
histories and FULL balanced accuracies were exactly reproduced, but the
two-worker/ten-thread schedule completed only 15 candidates in 45.6 minutes;
the four-worker/three-thread trace completed 37 in 26.8 minutes. Small GRU
operations suffered substantial intra-op thread overhead. The slower run's 12
B0 results, 3 B1 results, and one in-progress epoch state were preserved under
`research_storage/quarantine/pre_thread_reversion_8070161` and are excluded from
official evidence. CPU scheduling was restored to four workers with three
threads each while retaining exact epoch resume. This changes execution
throughput only; all scientific settings remain unchanged, and official
candidates are regenerated under the new frozen commit.

## Official initial-stage execution incidents

The repository is stored on a case-insensitive Windows filesystem but is also
addressable through case-distinct Linux path spellings. One early launch used a
lowercase path identity while the frozen run configuration records the official
uppercase path. Outputs from that launch were excluded from official evidence.
All subsequent official launches used
`repository root`,
and final artifact validation requires the frozen run identity rather than
accepting a path-only match.

The initial B0/B1/T/T-text stage was interrupted by five host reboots after
authoritative completed-result counts of 457, 460, 567, 598, and 599 out of 600.
The observed PID 1 start times were 2026-10-06 21:59:56,
2026-10-06 23:38:21, 2026-10-07 18:56:41, 2026-10-08 09:22:51, and
2026-10-10 13:41:39 Asia/Seoul. Each restart first verified that no prior
training process remained, that the official worktree was clean at commit
`5c2f73d62906ca81758ac1d2cc0543c8c0ea8183`, and that the saved epoch state
matched the frozen run identity. Training then resumed at the next incomplete
epoch. The completed stage contains exactly 150 results for each of B0, B1,
Teacher, and T-text, with no remaining partial state.

Two additional forward wall-clock discontinuities were observed on 2026-10-07
(approximately 3 hours 40 minutes and 15 minutes 16 seconds). The same PIDs
remained alive and monotonic CPU time continued increasing, so these were
treated as environment-clock discontinuities rather than scientific changes.
Artifact hashes and run identities, not wall-clock duration, are used for the
evidence audit.

## Pre-D teacher-cache manifest ordering fix

After Gate T passed, inspection found that the original D entry point wrote the
tracked `artifacts/manifests/teacher_cache_manifest.json` before starting D
workers. That made the worktree dirty, so every worker's clean-commit provenance
guard would fail before producing a D result. No D checkpoint, prediction, or
metric had been generated. Teacher-cache construction was changed to persist a
small resumable progress manifest only under the ignored development output
root, validate the exact training membership, finite logits, cache hash,
selected-teacher checkpoint hash, and teacher run identity for all 75 entries,
and place the completed manifest in the ignored selection state before D
training. The tracked teacher-cache manifest is deferred to terminal evidence
finalization. This fixes execution/provenance ordering only and does not change
the population, inputs, architecture, candidate grid, seed, loss, or gate.

The first D launch generated and validated all 75 teacher caches, then created
four `fork`-based workers after the parent process had used PyTorch's
multithreaded teacher inference runtime. All four workers became permanently
blocked in `futex_wait_queue` with identical, non-increasing CPU times. No D
metric, checkpoint, prediction, or partial epoch state existed. The hung process
group was terminated, and the development process pool was changed to the
explicit `spawn` start method. A regression test verifies that `_run_stage`
passes a spawn context to the executor. The already validated teacher caches
remain unchanged and are reused by hash. This is a process-isolation fix only;
no scientific configuration or result was changed.

## Official D-stage execution incidents

The official spawned-worker D run was interrupted by a host reboot after 259
of 450 candidate results had completed. At recovery, PID 1 had started at
2026-10-11 12:13:42 Asia/Seoul, no process from the prior launch remained, and
the temporary launch log had been cleared with the host's temporary storage.
The exact on-disk state contained 259 complete D metric triples and three
in-progress epoch states; the fourth worker had completed its active candidate
immediately before the interruption. Before resuming, the official worktree was
verified clean and the saved states remained bound to the frozen run identity
and scientific commit `5a8a5e3955f2f98f949706ef4eef541779e520f4`.
Training resumes from the next incomplete epoch and revalidates the existing 75
teacher caches by hash. This is an environmental interruption only: the data,
splits, features, architecture, candidate grid, seeds, losses, and gate rules
are unchanged.

## Official A-stage execution incidents

The first detached A launch from the restricted execution environment exited
before producing a metric or partial epoch state. A subsequent monitored launch
using the official uppercase repository path reached its first checkpoint save
after approximately 24 minutes, where the restricted filesystem rejected the
write as read-only; it also produced no metric or partial state. The run was
restarted with explicit workspace write authority and reached 38 of 75 complete
results plus four resumable epoch states before its external session received
signal 143. A durable authorized launch then validated and resumed those exact
states and completed all 75 evaluations. The scientific configuration and run
identities remained those of commit
`5a8a5e3955f2f98f949706ef4eef541779e520f4` throughout.

## Permutation-sanity serialization fix

The first reader-level permutation command stopped before model training because
the shuffled reader-label mapping retained NumPy `int64` values that canonical
JSON cannot serialize. No checkpoint, prediction, metric, or partial epoch state
was created. The values are now normalized to built-in Python integers before
hashing; a regression test covers serialization. During diagnosis, a detached
launch became visible only after a delayed process start and briefly overlapped
a monitored launch. Both were stopped before either wrote a checkpoint or
metric. The fixed five-evaluation sanity run was then executed once from clean
commit `47c198421e7b20ea4c227c54b3679ef5c26eecac`.

## Reader-cluster bootstrap seed

The protocol fixes 2,000 reader-cluster bootstrap replicates but does not state
the random seed. Before terminal aggregation, seed `20271015` was fixed. Readers
are sampled with replacement within their reader-constant goal strata, and the
same draws are used for paired model contrasts. This affects uncertainty
estimation only and does not alter training, selection, or any gate.
