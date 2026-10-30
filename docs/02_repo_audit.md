# GoalExit repository and data audit

Audit status: **PASS**. This audit precedes all official model training. The
authoritative protocol is `docs/01_ICCE2027_GoalExit_Research_Design.md`.

## Source inventory

All four compressed source files were re-hashed before canonical reconstruction.

| Source | Bytes | SHA-256 |
|---|---:|---|
| hunting_fixations | 245,787,943 | `49fc77d9b8ac38e0665df611e22430b625077c53e3bf2080679faf28715eef88` |
| hunting_ia | 165,756,770 | `05e9ab76a9ac47269cfb1386614f47b582da520b4953be2452fa81bf707e0e30` |
| ordinary_fixations | 288,349,184 | `1de8fb62e10f69b417aaeff7ddfc380261713f7d4e927e025eb781d051b1a248` |
| ordinary_ia | 177,291,322 | `8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9` |

Only the requested Paragraph trial, IA, and fixation columns listed in
`artifacts/manifests/canonical_goalexit.json` were read. Question content,
answers, correctness, response times, article/session position, and selected
answers were not requested.

## Canonical population

- Regular first-reading Paragraph trials: **19,438**
  (9,718 ordinary; 9,720
  information-seeking).
- Readers: **360** (180 per goal); reader-goal constancy passed.
- Articles: **30**; every article has both goals.
- Valid fixations: **2,042,764**.
- Teacher paragraph word-sequence variants: **339**.
- Fixation-count distribution: min 9, Q1
  69, median 96, Q3
  131, P90 170, P95
  199, P99 266, max
  814.

Repeated-reading rows, practice rows, Question Preview events, and invalid
fixations are excluded. Original source fixation indices and deterministic
chronology are retained. The canonical data hash is
`9c3560ee8cf8e5eadd1ca22d9704544e5403ac9885080a8a284681f78032c75d`.

## Feature and leakage audit

The student surface is an exact 10-feature allowlist: nine causal Paragraph-gaze
features plus position normalized by the *observed prefix* length. B0 has a
frozen 22-feature gaze aggregate allowlist. Teacher gaze has the frozen eight
word summaries; only T/T-text also receive frozen paragraph word embeddings.

The column-level audit, source-request audit, target/input mismatch checks, and
reader-identity parameter scan pass. The non-gaze metadata allowlist is empty;
the intercept-only logistic shortcut probe obtained mean BA
**0.5000** over all 75 paired development
evaluations. Reader and article IDs are used only for splitting, clustered audit,
and macro-by-article evaluation, not as tensors.

## Teacher feasibility and dependency lock

`FacebookAI/roberta-base` is available locally at immutable revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. It remains frozen and was loaded offline.
All 339 canonical paragraph variants align without
truncation; maximum tokenized length is
223 including special
tokens. Cached word embeddings have shape [339, 165, 768] and SHA-256
`29c4de97edc9866314374ab0647f950d0161a7106fdca5fb39c819a9ea4ed47a`.

| RoBERTa file | Bytes | SHA-256 |
|---|---:|---|
| config.json | 481 | `ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9` |
| merges.txt | 456,318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| model.safetensors | 498,818,054 | `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81` |
| tokenizer.json | 1,355,863 | `847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e` |
| tokenizer_config.json | 25 | `994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce` |
| vocab.json | 898,823 | `9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655` |

## New GoalExit splits and outer seal

Seed **20271002** produced five reader folds and five article folds. Every
reader fold contains 36 ordinary and
36 information-seeking readers; each article fold has six
whole articles. Their Cartesian product defines 25 sealed UB targets and three
paired inner holdouts each, for 75 development evaluations. Membership mapping
hash: `a92dc35355642a1a6cd5c35f9750d5a20048c8417a2c0229d94585a00406844d`.

The manifest contains only outer membership/count/hash information, not outer
class summaries or metrics. `configs/final_goalexit_lock.yaml` is absent and the
outer entry point fails closed. No outer prediction or metric exists.

## Environment and reuse boundary

Environment versions are recorded in the canonical and RoBERTa manifests.
Low-level deterministic parsing, hashing, preprocessing, and guard patterns were
reviewed against ReadMode, but GoalExit uses a newly reconstructed population,
seed, target, folds, feature allowlists, model heads, and outputs. No prior target,
fold mapping, opened result, or model-selection decision was imported.
