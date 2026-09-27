# V7 final iteration plan: one stacker over V6's frozen survivors

Written 27 Sep 2026, about 17:15 IST, from the Claude session on Aditya's machine. Status: **plan only**.
No code, training, cloud job or portal action was started for it. Review: Astra. Training run:
Abhigyan. Final collector: Aditya. Official close 23:59 IST, internal acceptance 21:30 IST.

Steps are not owned. Any agent may do any step. Names below mark whose account, machine or portal
access a step needs, nothing more.

## 0. Decision in five lines

1. **V6 ens3 stays the fallback, untouched.** `AML_submission_v6.zip` (SHA-256 `7d66db41...`), audit
   0.97089 on 220,682 S1, 5.35 candidates per test S1, validator PASS. If it is not on the leaderboard
   yet, a human should upload its `matching_results.tsv` now to get a real public score.
2. **V7 = a third-stage stacker that scores exactly V6's survivors** (v5.1 top 12, p >= 0.01). It
   sees V6's probabilities plus the three kinds of evidence no current model has: sibling context,
   decoy clusters, and population-level competition for each target. The candidate file does not
   change (same 9,266,800 pairs, SHA `9c438703...`), so the "smaller candidate set" ranking criterion
   is unaffected.
3. Train on the 100k tune partition (out of sample for v5.1 and V6), 5-fold cross-fitted for
   every choice, with a pre-registered promotion gate on tune and one pre-registered look at the audit.
4. Test inference runs as 88 shards split across three >= 64 GB machines (Aditya, Abhigyan, Akash).
   Aditya collects, applies global target ownership and set selection, writes and validates both TSVs.
5. V7 replaces V6 only if every gate in section 6 passes. Otherwise V6 ships, with no retuning.

## 1. Theory: where the last points are

The score has five layers. Each teammate's work attacks one of them.

| Layer | Question it answers | Best current state | Remaining loss (audit) |
| --- | --- | --- | --- |
| 1. Recall | Is the true target among the candidates? | v5.1 TF-IDF union, then v5.1 top 12 p >= 0.01: U 0.99408 at 4.91/S1 | 1 - U = 0.00592 |
| 2. Pair evidence | Does this record look like this S1? | V6 ens3 (3 survivor-trained LightGBMs, 92 features) | inside U - S = 0.02319 |
| 3. Sibling context | Do the other strong candidates of this S1 agree with it? | Aditya's stage-2 (measured, reproduced by Akash); decoy clusters in the v3 lineage | not in V6 |
| 4. Ownership | Does another S1 own this target? Every labelled target has exactly one S1 | only the test-time one-owner rule, which tune/audit cannot measure (they hold 4.5% and 10% of train S1) | not in V6 |
| 5. Set decision | How many to return for this S1 under F0.5? | global threshold 0.72 | small (+0.0006 measured for expected-F) |

V6 reached the ceiling of layers 1-2 at a small candidate set. The stage-2 residual analysis
(`coordination/evidence/stage2/claude-20260927-1130.md` section 5) shows what is left:

- Rejected true matches: 58% have a missing address. Among missing-address survivors only 4.8% are
  true, because the data holds many same-name records without an address and most belong to other S1.
- False positives and rejected true matches look alike: a renamed variant at the S1's exact
  address versus a different business registered at that address and owned by another S1.
- "The deciding fact is which S1 record the candidate resembles most, and nothing in stage 1 or
  stage 2 measures it."

That residual is layers 3 and 4. Three sessions, working apart, built features for exactly this:
Aditya's `v6_stage2` sibling features and `v6_reverse`, the v3 lineage's decoy-cluster group features,
and Aadish's population reverse index. **V7 is the first model that uses layers 1-5 together.**

Honest arithmetic: 0.98 needs +0.0091 over V6's audit, which means recovering 39% of the U - S gap.
The trusted evidence supports roughly +0.002 to +0.003 (sibling + expected-F). The rest depends on
layers 3-4 working as the untrusted claims suggest. Expected audit range: 0.973-0.977. 0.98 is
possible, not likely. The plan ships whichever wins the pre-registered gates.

## 2. What each branch contributes, judged objectively

Rule used here: a number counts as evidence only if it was produced by a run we can reproduce
from recorded SHAs, or was reproduced independently. Aadish's reported numbers are treated as
unverified claims. His ideas are judged on their merits and on his code.

| Branch / author | Idea | Evidence status | Use in V7 |
| --- | --- | --- | --- |
| `codex/v6-train-claude-20260927-1420` (Aditya, 60ec04d) | survivor-specialist LightGBM x3 on v5.1 survivors | tune 0.97061, **audit 0.97089** once, paired +0.00265 (se 0.00013) over v5.1 same cut; validator PASS | base scorer, fallback, p6 feature |
| `codex/stage2-claude-20260927-1130` (Aditya, db4056b) | sibling features (61), expected-F prefix selection | tune 2-fold OOF 0.96780 -> 0.96994 (+0.0021), expected-F 0.97052 (+0.0006) | feature family F1, selection policy |
| `codex/s2-model-claude-newacct-20260927-1155` (Akash, a35aead) | reproduction + `null_on_oob` fix + thread setting | reproduced every stage-2 number exactly | **cherry-pick the fix** |
| `akash/v3-pipeline` (Akash) | transliteration skeleton folds, address-only keys, one-owner assignment | v2 val 0.9119; v3 lineage below | skeleton view in F3 keys (already in `text_norm_v5`) |
| `codex/stage2-plan-review-20260927` (Abhigyan) | zero/one/two-survivor safety, input gates, tie rules, candidate accounting | static review | release checks in section 8 |
| `results/v3-v6-v7-claude` (Aadish commit, 44cb55c) | IDF name/address coverage, typo-aware "true extra" tokens, house-number structure, decoy-cluster group support (`features3.py`), learned stage-1 shortlist | claim: 0.9737 on a 10k check half, 37 candidates/S1, U 0.9904. **Code review:** features are label-free, splits are disjoint. Flaw: early stopping and the threshold both use the other 10k half, so only the check half is honest. Never audited | feature family F2 (port NEW_FEATURES only) |
| `codex/v51-final-results-20260927` (Aadish, ca522df) | population reverse competition over all S1 via exact normalized keys, anchor similarities, directional token/number agreement, LGBM+XGB, expected-F with logit bias | claim: 0.97647 on 15k vs 0.96777 baseline. **Code review:** `full_reverse_*` features are sound (label-free, whole S1 population, same procedure possible on test). **Flaw:** the `rev_*` features (`prepare.py`) measure competition only among the 100k tune S1, i.e. 4.5% of the population, while on test all 1.73M S1 compete. They cannot transfer. Trained on 65k queries, 20+ variants selected on 10k | feature family F3 = **population keys only**; `rev_*` dropped |
| `results/v3-v6-v7-claude` v7 | digit-letter fold (0->o, 1->l, 5->s, 6->g, 8->b) in mixed tokens | claim: 10.5% of that lineage's misses | out of scope: it changes retrieval, which needs a full feature rebuild |
| same | per-country thresholds | +0.0004 on 10k, noise level | not adopted |

## 3. Step A0: adjudicate the claims on identical S1 (20 min, read-only, no cloud)

Both claimed check sets are subsets of V6's tune partition, so V6 can be scored on exactly the same
S1 from V6's saved `model/tune_scores.parquet` (three queues) and the supplied truth. The mean over an
identical S1 set is an exact comparison of means. No per-query file from the claimant is needed.

- v3-lineage check half: `perm = np.random.default_rng(42).permutation(2206821)`,
  `val = np.sort(perm[200000:220000])`, `p = np.random.default_rng(7).permutation(val)`,
  `check = np.sort(p[10000:])` (see `src/exp_v3.py` and `src/splits.py` on 44cb55c). First assert
  `check` is inside `plan.json["tune_rows"]` of `shared/er-v51-20260927`. If it is not, the two
  pipelines number S1 rows differently: map through `source1_entity_id` before comparing, and say so
  in the report. Report V6 ens3 (cut top 12,
  p >= 0.01, `unique_assign`, threshold 0.72) and v5.1 (threshold 0.785) on these S1 next to 0.9737.
- Aadish's 15k check: `queries = plan["tune_rows"]` in plan order,
  `perm = np.random.default_rng(27092026).permutation(100000)`, `check = queries[perm[85000:]]`
  (`prepare.py`). Report V6 ens3 and v5.1 on these S1 next to his 0.97647 / 0.96777. If our v5.1
  number differs from his 0.96777, his pipeline differs from ours and his gain figure is unusable.

Use: A0 only orders the engineering (F3 before F2 by default, F2 first if the v3 lineage beats V6
by >= 0.002 on identical S1). It is not a promotion gate. Record results in the evidence file.

## 4. V7 design

### 4.1 Frozen inputs (publish with hashes before any V7 work, then give scoped read access)

| Input | Location (Aditya's bucket) | Pin |
| --- | --- | --- |
| Code base | branch `codex/v6-train-claude-20260927-1420` | 60ec04d + cherry-pick a35aead |
| Stage-1 model, v5.1 | `shared/er-v51-20260927/model/model.txt` | `eec93e35...` |
| V6 members | `shared/er-v6-20260927`, `shared/er-v6s7-20260927`, `shared/er-v6l127-20260927` `model/model.txt` | `b7e7db65...`, `ad57c4f2...`, `c93f3b87...` |
| Candidate policy | v5.1 p, sort (p desc, target row asc), top 12, drop p < 0.01 | ensemble id `d0fa7781...`, V6 threshold 0.72 |
| Tune scores | each V6 queue `model/tune_scores.parquet` (top-15 tune survivors, v5.1 p and member p) | record SHA-256 |
| Test scores | `shared/er-v6ens3-20260927/score/` (88 shards) and each member queue `score/` | record per-shard SHA-256 |
| Audit features | `shared/er-v51-audit-20260927/feat/` + `plan.json` (`audit_sha`) | for the one audit run only |
| Data | supplied `train_*`/`test_*` TSVs | existing checksums |

Verify columns of the ens3 and member score shards before coding against them. If per-member test
p is not in the ens3 shards, read it from the member queues (they were scored at the looser
p >= 0.001 cut, a superset).

### 4.2 Rows

Every (S1, target) pair that V6's champion scores: v5.1 top 12 with p >= 0.01. About 491k tune rows
(4.91 per S1), 9,266,800 test rows (5.35 per S1). S1 with zero survivors (23,644 on test) have no
rows but must appear in outputs and count in every metric.

### 4.3 Feature families (all label-free; no raw IDs or row numbers as features)

- **F0, scores.** Mean V6 p (p6), member standard deviation, v5.1 p (p1), their logits, rank of p6
  within the S1, gap to the S1's best p6, the S1's best/second/third/sum p6, survivor count, counts
  of p6 >= 0.5 and >= 0.9.
- **F1, siblings.** `v6_stage2.features` exactly as tested (61 features: S1-candidate text
  similarities with the v5 normalizer, sibling agreement against the 5 highest-p other survivors),
  with `p = p6` so siblings are ranked by the better model. Requires the a35aead fix.
- **F2, decoy clusters and weighted tokens.** Port `NEW_FEATURES` of `features3.py` (44cb55c) onto
  `text_norm_v5` views: IDF-weighted name/address coverage, typo-aware "true extra/missing" tokens,
  number and house-number structure, group support (how many other survivors of the same S1 share
  this candidate's core name, house, address or its "true extra" token). Do not port `base.build`
  (it needs v3 blocking columns). IDF tables come from the target catalog of the same split
  (train catalog for tune and audit, test catalog for test), as the Sol plan already allows.
  Group features use the S1's survivors as context.
- **F3, population competition.** For each survivor target t of S1 q, find other S1 of the **same
  split's complete S1 file** (train 2,206,821 for tune and audit, test 1,732,544 for test) that
  share an exact normalized key with t. Views: core, sorted core tokens, concat, skeleton,
  addr_norm, house + addr_tok. Key = country | view | text, length >= 5, keys with more than 20
  S1 dropped. For those competitors compute name token-set, name ratio, address token-set, address
  ratio and joint sqrt(name ratio x address ratio) against t. Keep the top 4 per criterion, drop q,
  then per pair: best other-S1 value of each similarity, competitor count, and gaps = q's own
  similarity minus the best other (own similarity computed by the same scorer on the same views).
  - Use a version-independent key encoding (the key string, or blake2b of it). Do not rely on
    polars `hash` across machines.
  - Density differs: 0.214 S1 per target in train, 0.174 in test. Before training, compare tune vs
    test quantiles of every F3 column. If the count feature shifts materially (median ratio beyond
    1.3x), drop it and keep the max/gap features.
- **Dropped:** Aadish's within-sample `rev_*`, the digit-letter retrieval fix, anything needing a
  new retrieval pass.

### 4.4 Model and selection

- LightGBM binary, `learning_rate 0.05, num_leaves 63, min_data_in_leaf 40, feature_fraction 0.8,
  bagging_fraction 0.8, bagging_freq 1, lambda_l2 1`, early stopping 100 rounds on an inner 10% of
  the training folds (by S1). No class weights. Seeds fixed and recorded.
- 5-fold cross-fitting on the 100k tune S1, folds by a stable hash of the S1 row. OOF p7 for every row.
- Ablation ladder, one CV run each (a few minutes each): **L0** F0; **L1** F0+F1; **L2** F0+F1+F3;
  **L3** F0+F1+F2+F3. If F2 or F3 is not ready by its cutoff (section 7), skip that rung.
- Selection policies on OOF p7, after `unique_assign` (same code as V6):
  (a) global threshold, 0.02 grid then 0.0025 refinement;
  (b) expected-F0.5 prefix per S1 under an independence approximation, with calibration
  q = sigmoid((logit p7 + b) / T), b in {-0.6, -0.5, ..., +0.3}, T in {0.8, 1.0, 1.25}, empty set
  allowed (E[F] of the empty set = P(no positive)). Reuse `tools/v6_analysis/expected_f.py` with
  Abhigyan's fix (queries without survivors get an empty prediction). Adopt (b) only if it beats (a)
  by >= +0.0003 on OOF. All scores use complete truth, including unretrieved positives.
- Final model: refit on all 100k tune S1 with rounds = 1.1 x mean best iteration, same params. Save
  model, ordered feature list, config and policy JSON with SHA-256. Reload and check predictions on
  1,000 rows match the in-memory model (abs diff < 1e-6).
- Why not train on fit survivors: v5.1 and V6 were trained on those rows, so p1/p6 are in sample
  there. Tune is the largest partition where both are honest inputs.

## 5. Execution workflow (adapted from the teammate's sketch)

```text
Aditya: V6 ens3 frozen and packaged (done, 475ca2b on codex/control)
   -> publish exact pins of section 4.1 (code SHA, candidate policy, model/config manifests,
      tune scores, 88 test score shards) as a manifest JSON with SHA-256 per object
   -> scoped, time-limited read access for Abhigyan and Akash to exactly those prefixes
   -> V6 fallback zip and its S3 prefixes stay read-only
Any agent: A0 adjudication (section 3), in parallel
Abhigyan (training run): branch codex/v7-stack-<session> from 60ec04d + a35aead
   -> F0-F3 code + unit tests (section 8) -> tune features -> ladder -> policy -> OOF gate
   -> final fit -> publish model/config/feature list/policy with SHA-256 -> read access for all three
Aditya: restart ONE >= 64 GB notebook (er-v5-r7i) for the audit gate, one run, pre-registered
   (may run on a second r-class notebook in parallel with test shards if time is short)
Each of Aditya / Abhigyan / Akash, one machine each (>= 64 GB preferred; Akash's 32 GB
ml.r5.xlarge is acceptable with a shorter list, see section 9):
   -> build the test S1 competition index and test IDF tables locally, record their SHA-256
   -> benchmark ONE assigned test shard (seconds, peak RSS)
   -> receive the explicit shard list from the collector (balanced by pair count; France shards
      are heaviest at 7.31 candidates/S1)
   -> process ONLY assigned shards -> write to own prefix shared/er-v7-20260927-<name>/score/
Aditya, FINAL COLLECTOR:
   -> 88/88 shards, none missing, none duplicated, same code/model/config/index/IDF hashes
   -> each shard's (s1, t) pair hash equals the V6 ens3 survivors of that shard
   -> global target ownership (unique_assign on p7 over all 88 shards)
   -> frozen selection policy -> candidate_pairs.tsv + matching_results.tsv
   -> official validator --check-ids + Astra's strict validator -> France guard
   -> AML_submission_v7.zip -> champion gate (section 6) -> portal upload by a human
```

Per-shard output: `v7score-<shard>.parquet` with `s1, t, target_id, p1, p6, p7` sorted by
`(s1, p7 desc, t)`, and `v7score-<shard>.json` with input shard SHA, row and S1 counts, sorted pair
hash, code SHA, model/config/feature-list/policy SHA, index and IDF SHA, package versions, seconds,
peak RSS and machine. The collector accepts exactly one output per shard. If a machine has not
finished by 20:40, the collector reassigns its unfinished shards explicitly and records it.

Pin one environment everywhere: the V6 worker venv (`requirements-v5.txt`, Python 3.12). Record
`pip freeze` in every manifest; the collector rejects mismatched LightGBM or polars versions.

## 6. Gates (pre-registered; write them into the evidence file before running)

| Gate | When | Pass rule | If it fails |
| --- | --- | --- | --- |
| G1 inputs | before coding against inputs | 88 test shards, 1,732,544 S1 covered, disjoint S1 ranges, finite p, survivors recomputed from v5.1 p equal the ens3 rows | fix inputs; no training |
| G2 OOF | after the ladder | best rung + policy beats V6 ens3 on the same 100k tune S1 (0.97061 at 0.72) by **>= +0.0015**, paired lower bound > 0, India and US both non-negative | stop, ship V6 |
| G3 audit, once | after freeze | on 220,682 audit S1 in one run computing V6 and V7 per-S1 F: paired mean gain **>= +0.0010**, mean - 1.645 se > 0, no India/US/singleton slice worse than -2 se | ship V6; no retuning |
| G4 release | after collection | section 8 checks all pass, candidate_pairs.tsv SHA equals V6's `9c438703...` | fix and rerun collection, else V6 |
| G5 France guard | after collection | France mean predicted per S1 in [3.1, 3.7] (V6 3.41, labelled truth 3.45), empty rate in [3%, 9%], 8+ matches <= 2% | ship V6 |

The audit was opened once for V6. G3 is a second, single, pre-registered binary decision between two
frozen systems, recorded as such in the documentation. Nothing may be selected on the audit.

## 7. Timeline and cutoffs (IST; assumes Astra approves by 17:45)

| Time | Work | Hard cutoff rule |
| --- | --- | --- |
| now-17:45 | Astra review. Aditya publishes pins and grants read access. Human uploads V6 to the portal if not done | |
| 17:45-18:05 | A0 adjudication | skip if not done by 18:15 |
| 17:45-19:05 | V7 code: F0+F1 (existing), F3, then F2; tests; tune features | F3 not built by 19:05: drop L2/L3 and continue with L1. F2 not built by 19:15: drop L3 |
| 19:05-19:35 | ladder + policy search + G2 | no G2 pass by 19:45: ship V6, stop |
| 19:35-19:50 | final fit, reload check, freeze and publish | |
| 19:50-20:20 | G3 audit run (Aditya's r-class notebook). In parallel: three machines build indexes and benchmark one shard | |
| 20:00-20:50 | 88 test shards across three machines | unfinished at 20:40: reassign. Not complete by 21:10: V6 stays champion for the 21:30 acceptance |
| 20:50-21:20 | collection, G4, G5, zip, docs | |
| 21:20-21:30 | champion decision recorded; human uploads the winner | |
| 21:30-23:00 | repair only. A V7 that already passed G2-G5 may still replace V6 until 23:00; no model changes | after 23:00 the package on record is final |

## 8. Tests and release checks (from Abhigyan's review, Astra's validator and V6's release)

Unit tests on synthetic fixtures before any real run:
1. S1 with 0, 1, 2 and 3 survivors: F1-F3 produce finite features, no out-of-bounds, no self-sibling,
   no self-competitor; zero-survivor S1 appear in outputs with empty lists and count in the metric.
2. Ties: equal p in permuted input order give identical survivors, features, ownership and output.
3. F3 never reads truth; a test asserts the function has no truth argument and that removing
   `train_ground_truth.tsv` from the data dir does not change F3 values.
4. Expected-F: both-empty scores 1, nonempty truth with empty prediction scores 0, the PDF example
   scores 10/14, and the policy handles all-empty batches.
5. Shard scoring is deterministic: same shard twice gives byte-identical parquet.
6. Collector rejects a missing shard, a duplicated shard, a hash mismatch and a pair-set mismatch.

Release checks: every test S1 exactly once in both files; matches subset of candidates; only existing
test S2/S3 IDs; no duplicate IDs in a row; no target matched to two S1; candidate file identical to
V6's; official validator `--check-ids` PASS on files extracted from the zip; strict validator
(`src/strict_validate.py` on `codex/submission-support-astra-20260927-0450`) PASS; license notes
updated (LightGBM MIT, no new model family; XGBoost only if used, Apache-2.0).

Documentation: the cascade is retrieval -> v5.1 stage-1 over the union -> survivors (the candidate
file) -> V6 ens3 + V7 stacker scoring exactly those survivors -> global ownership -> set selection.
State tune OOF, audit (with the second-look note), candidate counts (mean 5.35, p95) and France limits.

## 9. Compute, budget and data handling

- Training side is small: about 491k rows x ~180 features. Tune features plus the train S1
  competition index need roughly 8-12 GB RAM; any >= 16 GB machine works. Abhigyan's 4 GiB notebooks
  are not enough; a teammate machine or local workstation is.
- Audit gate: v5.1 re-scores the audit union (about 58M rows) because V6's audit run saved only a JSON
  summary. One r-class notebook, about 20-30 min.
- Test side: benchmark first. Planning estimate is 1.5-3 min per shard (F1 measured at 39 s per
  France shard on 4 threads in Akash's smoke run). Split the 88 shards by pair count and measured
  speed, not evenly. Aditya's list is smaller if the same notebook also runs G3; running G3 on a
  second r-class notebook in parallel is recommended. Akash's account allows only ml.r5.xlarge
  (32 GB, 4 vCPU): enough memory for per-shard work (2.3 GiB peak for F1 in the smoke run) at
  about half the speed, so it gets fewer shards. If the benchmark exceeds 2 min per shard, Aditya
  may add r-class workers for his list.
- Incremental cloud cost: under $10 at list prices. Record job ID, account alias, machine, cap and
  shutdown in `codex/control` before launch; stop instances when idle.
- Data, features, models and TSVs stay in private buckets. Git gets code, tests and aggregate evidence
  only. The origin repository was observed public.

## 10. Not doing today, and why

- Retrieval changes (digit-letter fold, v3 keys, larger K): need a full feature rebuild and retraining
  of v5.1 and V6 (3 h or more). Retrieval loss is 0.0059; the matching gap is 0.0232.
- Neural models: no time for license checks, training and routed inference.
- Training V7 on fit survivors: p1 and p6 are in sample there.
- Within-sample competition features and per-country thresholds: see section 2.
- A wider context set (v5.1 top 15) for group features: possible later ablation; survivors only today.

## 11. Questions for Astra's review

1. Is the second audit look (G3), pre-registered as one binary V7-vs-V6 decision, acceptable?
2. Any leakage path in F3 (population features over the train S1 file, which includes tune and audit
   S1 as unlabeled records)? Any difference in meaning between train and test beyond density?
3. Are the G2 and G3 margins (+0.0015 OOF, +0.0010 audit) right for a 5.35-candidate system with
   paired standard errors near 0.0002?
4. Is per-machine index building with a SHA check better than one published index, given
   cross-account access delays?
5. Anything in the timeline that will not fit, and which rung to drop first?
