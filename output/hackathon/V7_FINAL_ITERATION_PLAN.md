# V7 final implementation and training plan — revision 3 (final)

**Status: final specification for execution. Nothing in it has been implemented, trained or run.**
Revision 2 (Astra, main 2795605) was reviewed and amended by the Claude session on Aditya's machine,
finished at 17:35 IST. Time left then: 3 h 55 min to internal acceptance (21:30) and 6 h 24 min to the
official close (23:59). Recalculate at execution start.

This document authorizes no compute, spending, access grant or portal action. Account owners approve
those in chat. **No agent owns any work.** Any agent may pick up any work package through the normal
claim push on `codex/control`. The names below only say whose account, machine or portal login a step
needs. Background rules: [Sol plan](SOL_IMPLEMENTATION_PLAN.md). Sources: [branch map](../../coordination/BRANCH_CONSOLIDATION.md).

## 0. The decision

1. **V6 ens3 is the incumbent and the fallback,** byte for byte: audit macro F0.5 0.97089 on 220,682 S1,
   candidate oracle U 0.99408, 5.35 candidates per test S1, official validator PASS.
2. **V7 is a correction head that scores exactly V6's candidate set** C(q): v5.1 top 12 by p1, ties by
   target row, then p1 >= 0.01. No retrieval change and no pruning. `candidate_pairs.tsv` is V6's file,
   reused byte for byte after a pair-set proof.
3. **Evidence families:** F0 (V6 and v5.1 scores), F1 (sibling agreement, existing tested code) and F3
   (coherent whole-population rivals, new). F2 (decoy clusters) and expected-F selection are dropped
   from today's critical path.
4. **Protocol:** 5-fold cross-fitting on all 100,000 tune S1 for every choice. The deployed predictor is
   the mean of the 5 fold models. One pre-registered comparison on the audit (already opened once for
   V6) tests that exact predictor. No D/C split, no refit.
5. **Schedule:** three work packages run in parallel. Test and audit features are computed while the head
   trains, so scoring after the freeze takes minutes.
6. **France routing:** a label-free rule keeps V6's decisions for France if V7's France inputs or outputs
   leave the envelope of the labelled countries. France is separable (country-partitioned catalogs).

The target needs +0.00911 over V6, 39.3% of the remaining matching gap (0.99408 - 0.97089 = 0.02319).
Earlier sibling and expected-F gains were measured over v5.1, a weaker base, and do not add to V6.
There is no defensible forecast of the V7 audit score. Promotion and reaching 0.98 are separate claims.

## 1. What changed from revision 2, and why

| Revision 2 | Revision 3 | Reason |
| --- | --- | --- |
| 80k D / 20k C split, head trained on D only, gate G2b on C | 5-fold on all 100k; deploy the 5-fold mean; G3 tests the deployed predictor | G3 covers 220,682 S1 in one run on the exact deployed predictor, with about 3.3x smaller paired SE than C would give. C is also tune data, so it cannot remove the historical selection either. D/C costs 20% of the head's training data and adds a weak gate whose failure action equals G3's. One pre-registered binary decision on the audit adds optimism of about one paired SE (0.00013 for V6 vs v5.1 there), a tenth of the margin. Five folds (b607e93 used two) keep each fold model at 80% of the data, so their mean stays close to the OOF predictors that set the threshold. |
| Folds by components of shared survivor targets | Folds by a stable hash of the S1 entity ID; exact identity-fingerprint duplicates share a fold | Labels are single-owner: 7,638,365 links point to 7,638,365 distinct targets. Features carry no IDs. Two tune S1 sharing a candidate target open no label path across folds. Component building risks one giant component and costs time. |
| Expected-F prefix policy optional after helper fixes | Dropped today; one global threshold | Over v5.1 it gained +0.0006 but cost singletons 0.9741 -> 0.9514. France's singleton rate is unknown, and the helper still needs fixes. |
| Member spread optional | Dropped | The ens3 test shards carry p1 and p6 only. Spread would need member shards on every worker. |
| none | Monotone +1 constraints on p6 and v5.1 p features | With all other evidence fixed, a higher incumbent score can never lower V7's score. It blocks local inversions, helps transfer and costs nothing. |
| none | Real v5.1 p1 as a feature | In the scaffold, the F1 column named `p1` holds p6. Disagreement between v5.1 and V6 is signal. |
| Model, then test features, then prediction | Test and audit features cached right after the feature-code freeze | Features depend only on frozen inputs and frozen code. This takes about 45 minutes of test work off the critical path. |
| Audit survivors extracted after the freeze | Extracted now | Pure v5.1 + V6 scoring, label-free and model-independent. The audit is opened only when labels are joined, once, at G3. |
| New files `v7_stack`, `v7_features`, `v7_decoy_features` | Extend `stage2_release.py` (b607e93, on main) and add one module, `src/v7_population.py` | The smallest change surface. The existing gate, assign, predict and collect steps and `run_final` already pass synthetic end-to-end tests with the official validator. |
| France guard rolls back the whole system | France alone falls back to V6 decisions | A French anomaly should not discard a validated US/India gain. |
| Four slices, each rejected at mean < -0.001 or upper bound < 0 | India and US mean >= 0. Singletons and one-true-match mean >= -0.003. All slices reported | The metric is the overall mean. Singletons are 5.6% of S1, so a -0.001 singleton change moves the total by less than 0.0001. The strict rule would reject real gains. |

Kept from revision 2 as written: input pins, candidate accounting and byte reuse, the coherent-rival
idea, stable keys and censoring flags, the second-audit disclosure, pair-set and query-ledger digests,
collector corruption tests and license notes.

## 2. Inputs, pins and invariants

| Artifact | SHA-256 |
| --- | --- |
| V6 ZIP `AML_submission_v6.zip` | 7d66db4163f24c674282e1b94aa866c190f94b6532f7d1caec55ea88e98f0778 |
| V6 `matching_results.tsv` | 3f69fe5b983dc1407acfa3a45fb55dc59fce8a3d334f153961d0868521715f09 |
| V6 `candidate_pairs.tsv` | 9c438703e605ac9d5fc4c0931b1c2272743054e75aeeb019b95891a2c83a9872 |
| V6 ensemble identity | d0fa7781e62a1a4226d7b9e40fe2ad65529db2af5503766b2691ace0a186ff63 |
| v5.1 base model | eec93e350d320b46453d69f47010f07883c0b3e5eb6dd47d2dda0153f46cd982 |
| V6 member (seed 42) | b7e7db652d4ab6519a243fb590b9d02ddb20336c38df6cf2b5a3f6194a11523d |
| V6 member (seed 7) | ad57c4f219a7e470801e8ae822e6b4395848dc3d7f2eeb3921daa129057ef255 |
| V6 member (127 leaves) | c93f3b87a34e2f2e0594520becebb28374b4653b787bfdcc0ec694ca91b38271 |

Private inputs in Aditya's bucket: `shared/er-v6ens3-20260927/{plan.json, model/model_manifest.json,
score/*.parquet}` (88 shards: s1, t, target_id, p1, p), each member queue's `model/{model_manifest.json,
tune_scores.parquet}`, `shared/er-v51-20260927/{plan.json, model/}`, and the audit features
`shared/er-v51-audit-20260927/{plan.json, feat/}`. Supplied TSVs are on every machine. Record every
object's SHA-256 in the run manifest before use. Every V7 artifact (indexes, feature caches, release,
per-shard outputs, audit result, final files) goes under `shared/er-v7-20260927/` in the same bucket. It is
the only prefix the teammates' roles can write, and the bucket enforces owner ownership and SSE-S3.

Invariants: 1,732,544 test S1 (US 663,106, India 809,986, France 259,452); 9,266,800 test pairs;
23,644 test S1 with zero survivors; 100,000 tune S1; 220,682 audit S1. Missing S1 in a score shard are
legitimate only when the query ledger proves an empty survivor set. Keys (s1, t) are unique, t maps
one-to-one to target_id, train and test row maps are separate namespaces, and member scores are aligned
by key, never by physical row order.

C(q) is fixed by p1 before any V7 computation. Reranking by p6 or p7 never adds or removes a candidate.
Whole-population key hits in F3 are label-free context comparisons, not candidates; the head scores only
C(q). The documentation reports broad stage-one scoring, the survivor set and the reverse-context
comparisons separately, and keeps the organizer clarification provenance on candidate accounting.

## 3. Work packages and required code changes

Base: main at 2795605 or later. Each work package gets its own branch. Integration is one
fast-forward merge at the feature freeze (section 7).

**WP1, population rivals (F3).** New `code/business_entity_resolution/src/v7_population.py` and
`tests/test_v7_population.py`. Interface:

```python
build_index(s1_raw: pl.DataFrame) -> PopulationIndex        # label-free; one per split
index_digest(index) -> str                                   # SHA-256 of sorted (key, s1_row) stream
rival_features(pairs: pl.DataFrame,  # columns s1, t (rows of this split)
               s1_raw: pl.DataFrame, t_raw: pl.DataFrame,
               index: PopulationIndex, threads: int) -> pl.DataFrame  # s1, t + F3_COLUMNS, fixed order
F3_COLUMNS: list[str]
```

Freeze `build_index` first; its key specification in section 4 is final. As soon as its tests pass, build
the train index (2,206,821 S1) and the test index (1,732,544 S1) on idle machines, and publish both
privately with their logical digests.

**WP2, protocol and release hardening.** `tools/v6_analysis/stage2_release.py` and
`tests/test_v6_stage2.py`:

1. Folds: 5, fold = SHA-256 of the S1 entity ID mod 5. Exact identity-fingerprint duplicates
   (country, core, addr_norm) among tune S1 go to the lowest-row member's fold. The inner 10%
   early-stopping split uses a second stable hash. No Polars `hash` anywhere in the protocol.
2. `--families F0F1 | F0F1F3`. F0 adds the real v5.1 p1 as `v51_p` and `v51_logit`. The v6_stage2
   feature named `p1` (the stage-1 input, here p6) is renamed `p6` in the feature frame; values unchanged.
3. LightGBM: the scaffold recipe plus `deterministic=True`, `force_col_wise=True`, and
   `monotone_constraints` of +1 on `p6`, `p6_logit` (whatever the frame names them), `v51_p` and
   `v51_logit`, 0 elsewhere, with `monotone_constraints_method="intermediate"`. Build the constraint
   vector from the ordered feature names and store it in `release.json`.
4. Selection on OOF: s = sigmoid((1 - alpha) * logit(p6) + alpha * logit(p7)), alpha in {0.5, 1.0},
   logits clipped at 1e-6. Global threshold grid 0.02, then 0.0025 around the best. Ties go to the
   larger alpha, then the higher threshold. `unique_assign` before thresholding, exactly as V6.
5. Comparator: V6 at its manifest threshold 0.72 on the same S1 and survivor rows. It must reproduce
   0.97061 within 1e-4 (G0). A swept V6 threshold is reported only as a diagnostic.
6. Gate rule G2 from section 6 replaces "gain > 2 SE".
7. New `features` step: compute and cache per-shard features (all families available) for test, audit
   or tune. Each cache file carries the input shard SHA, feature-code SHA, index digest, family list and
   ordered column list. A cached shard is reused only if every pin matches. Write to a temp name, then
   rename.
8. `predict` reads the cache, or computes if absent, applies the frozen head (5-fold mean) and alpha,
   and writes `s1, t, target_id, p1, p6, p7, s` with a sidecar manifest (section 8). A predict JSON
   alone never counts as done: resume re-verifies every pin and the output hash.
9. `collect`, in section 8: pair-set and query-ledger digests, common pins, France routing, V6
   candidate bytes, `validator_exit == 0` required (never `None`), strict validator, label-free report.

**WP3, audit survivors and the G3 evaluator.** New `tools/v7_audit.py` and a fixture test.

- `survivors` (start now, label-free): for each audit feature shard, v5.1 p1, the fixed cut, the three
  V6 members on survivors and their mean. Write lean `s1, t, target_id, p1, p6` shards plus a query
  ledger covering all 220,682 audit S1 (zero-survivor S1 included), checked against the audit plan's
  `audit_sha`. Reuse `cut`, `load_models` and `stage1` from `src/v6_train.py`. It must never read
  ground truth. This streams about 58M union rows through v5.1 once, on one r-class worker.
- `gate` (once, after the freeze): read the lean survivors and the cached audit features, apply the
  frozen release, join labels, compute per-S1 F for V6 at 0.72 and for V7, and write one JSON. It
  refuses to run if a result already exists.

## 4. Features

All features come from supplied records and frozen base scores. Raw IDs, row numbers, fold IDs, label
counts and country lookup tables are never model inputs. Missing evidence is NaN (LightGBM missing
handling), never 0 and never perfect agreement.

**F0, scores:** p6 and its clipped logit; v5.1 p and its logit; rank of p6 within the S1 (ties by
target row); gap to the S1's best p6; the S1's best, second, third and summed p6; survivor count;
counts of p6 >= 0.5 and >= 0.9.

**F1, siblings:** the tested `v6_stage2.features` on C(q) with rank recomputed from p6 (the scaffold's
`survivors()` on p6 already does this), a35aead's one/two-survivor fix, at most the five highest-p6
*other* survivors, never self, and a typed zero-row schema.

**F3, coherent population rivals.** Every labelled target has exactly one owner. The deciding evidence
for the residual errors is whether some *other* S1 record resembles the target better than q does.

- Population: every S1 row of the split (train file for tune and audit pairs, test file for test pairs).
  Tune and audit S1 appear in the train index as unlabelled records. The module never opens truth.
- Normalizer: pinned `src/text_norm_v5.normalise`. Views: (1) core, (2) sorted core tokens,
  (3) concat, (4) skel, (5) addr_norm, (6) house + addr_tok, only when both are nonempty.
- Key: the string `country \x1f view \x1f text`, with text length >= 5. Empty text gives no key. No
  Python or Polars hashing as a cross-machine contract.
- Postings count over the whole population. A key with more than 20 S1 is an **overflow key:** no
  expansion, but `t_overflow` is set. Overflow means censored search, not "no rival".
- R(t) = deduplicated union of postings over t's non-overflow keys. Compute similarities once per
  (t, r) per shard: name token-set on core, name ratio on concat, address token-set and address ratio
  on addr_norm, all scaled to [0, 1]. Address scores are NaN when either address is empty.
  joint = sqrt(name_ratio x address_ratio), NaN without valid addresses.
- Per pair (q, t), over rivals R(t) \ {q}, ties to the lower S1 row:
  - Own similarities of q vs t with the identical scorers and views (5 columns).
  - **Joint rival J** = argmax joint: its five similarities, `rJ_present`, and gaps own - J for joint,
    name ratio and address ratio.
  - **Name witness N** = argmax name ratio: its name ratio, name token-set and address ratio,
    `rN_present`, own - N name gap, `rN_is_J`.
  - **Address witness A** = argmax address ratio: its address ratio, address token-set and name ratio,
    `rA_present`, own - A address gap, `rA_is_J`.
  - Flags: `t_overflow`, `t_n_keys` (0-6), `q_addr_missing`, `t_addr_missing`.
  - No raw competitor counts. Never combine rival A's name with rival B's address into one rival.
- Per-country diagnostics on every split (label-free): rates of `rJ_present` and `t_overflow`, and
  p50/p90 of `rJ_joint` and `gap_joint`. These feed the France routing rule.

**F2, decoy clusters: not in today's run.** Its specification stays in revision 2 section 4
(2795605). Start it only if every critical-path item is done before 19:00, which is not expected.

**Rungs:** B0 = V6 at 0.72. B1 = F0+F1. B2 = F0+F1+F3. Run B1, then B2. If B2 beats B1 by less
than +0.0003 on OOF, choose B1, the simpler model. No hyperparameter search.

## 5. Training and selection protocol

1. Rows: C(q) for the 100,000 tune S1, about 491k pairs. Labels are `y = 1` iff (q, t) is a labelled
   link. Metrics always use **complete truth**, including unretrieved positives, and every tune S1,
   including zero-survivor and singleton S1.
2. Base scores are honest here: v5.1 was fit on fit S1, V6 on fit survivors with early stopping on stop.
   The survivor cut, the ens3 choice and 0.72 were selected on this tune set. Cross-fitting cannot erase
   that, so G3 exists.
3. For each rung: 5-fold OOF p7. LightGBM binary, learning_rate 0.05, num_leaves 63, min_data_in_leaf
   40, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1, at most 4000 rounds,
   early stopping 150 on the inner split, seeds recorded. No class weights.
4. Choose the rung, alpha and threshold on OOF only (section 3, WP2 item 4). Then freeze.
5. Deployed predictor: the mean of the five fold models' probabilities, then alpha, then the threshold.
   No refit. Reload every fold model and check predictions on 1,000 rows (max abs diff <= 1e-6).
6. Uncertainty: per-S1 paired differences with standard errors. S1 are independent units here (single
   owners, fingerprint duplicates co-assigned). A 1,000-replicate bootstrap is optional reporting.
7. Record per rung: OOF F, gain vs B0 with SE, India, US, singletons, one-true-match, missing-address
   slices, before/after-ownership counts, feature time and peak RAM.

## 6. Gates (pre-registered; copy into the evidence file before any V7 number is seen)

| Gate | Pass rule | If it fails |
| --- | --- | --- |
| G0 inputs | All pins resolved; 88 shards; 9,266,800 test pairs; query ledger = 1,732,544 S1; V6 at 0.72 reproduces tune 0.97061 +- 1e-4 | Fix inputs; no training |
| G1 code | Section 8 blocking tests pass; feature schema versioned and frozen; label isolation proven | No experiments until fixed |
| G2 development | Chosen rung on 100k OOF beats B0 on the same S1 by **>= +0.0015**, gain - 1.645 SE > 0; India and US deltas >= 0 | V6 ships; stop |
| G3 audit, once | Deployed predictor on 220,682 audit S1: V6 at 0.72 reproduces 0.97089 +- 1e-4; gain **>= +0.0010**, gain - 1.645 SE > 0; India and US deltas >= 0; singleton and one-true-match deltas >= -0.003 | V6 ships; no retuning |
| G4 compute | Benchmarked US, India and France shards; projected finish fits section 7 with 30% margin | Drop B2 (use B1) or keep V6 |
| G5 release | Section 8 collector checks pass; V6 candidate bytes reused after pair-set equality; extracted archive validates | Fix mechanical defects only; otherwise V6 |
| G6 France | See the routing rule below | France keeps V6 decisions |

**The second audit look.** V6's audit result was observed. G3 is one disclosed, pre-registered binary
decision between two frozen systems, and it is selection on the audit. Report both systems and the
decision whatever the outcome. Its interval is not an untouched confirmation.

**France routing (label-free, fixed now).** Let `rJ_present` and `t_overflow` rates be measured per
country on test pairs, and let lo/hi be the lower/higher of US and India. France is out of envelope if
either rate exceeds max(1.5 x hi, hi + 0.05) or falls below min(lo / 1.5, lo - 0.05). V7's France output
must also have mean predicted matches per S1 in [3.1, 3.7] (V6: 3.41), empty rate in [3%, 9%] (V6:
4.9%) and 8+ matches <= 2% (V6: 0.99%). If either check fails, every France S1 row in
`matching_results.tsv` is V6's row, byte for byte. Before any mixed output, the collector proves that
no target appears in candidates of S1 from two countries, so per-country ownership is exact. France
truth is unknown. Passing the envelope does not show French accuracy. The rule applies to any country
without labels.

## 7. Execution schedule (IST, 27 September)

| Time | Track | Work | Needs |
| --- | --- | --- | --- |
| **Now** | Access | Bucket-policy statements `AMLV7Team*`: Abhigyan's and Akash's SageMaker roles may read the ens3, V6 member, audit-feature and `shared/er-v7-20260927/` prefixes, and write only under `shared/er-v7-20260927/`, until 05:30 IST (their v5.1 read grants already exist). Record job IDs, caps and shutdown on control | Aditya (account owner) |
| Now | Fallback | No upload now (Aditya's decision). `AML_submission_v6.zip` stays ready; SHA-256 re-verified at 17:40 IST. If V7 has not passed every gate by the cutoff, a human uploads V6 before the deadline | Portal operator |
| 17:50-18:20 | Early read | Run the **existing** scaffold `gate` as-is on real inputs (2-fold B1). A measurement only, never a release | Abhigyan's machine |
| 17:50-18:55 | WP1, WP2, WP3 | Three agents in parallel, one package each | Any agents |
| ~18:15 | WP3 run | Lean audit survivors extraction (about 58M rows through v5.1 once) | One of Aditya's r-class notebooks |
| ~18:30 | WP1 index | `build_index` frozen; build train and test indexes; publish with digests | Idle machines |
| **18:55-19:05** | Freeze | Merge WP1+WP2+WP3, blocking tests pass, **feature-code freeze** (commit SHA) | Any agent |
| 19:05-19:40 | Train | Tune features, B1 and B2 5-fold, alpha and threshold, G2, release frozen | Training machine |
| 19:05-19:45 | Test features | Benchmark US, India and France shards; assign by measured rate; cache features for all 88 shards | Aditya's workers + Abhigyan + Akash |
| 19:05-19:40 | Audit features | Cached F0/F1/F3 on lean audit survivors with the train index | Aditya's r-class worker |
| 19:40-19:55 | G3 | One run of `v7_audit.py gate` | Audit worker |
| 19:55-20:10 | Predict | 88 shards from cache on the machines that hold them | Same workers |
| 20:10-20:45 | Collect | Verification, France routing, global ownership, threshold, TSVs, both validators, archive, docs | Aditya (collector) |
| 20:45-21:30 | Buffer | Independent hash check; human upload; submission record | Portal operator |
| 21:30-23:00 | Repair only | Mechanical fixes to an already-frozen, validated system; no new models or thresholds | |

**Hard cutoffs.**
- 19:05: WP1 not merged means B1 only (F0+F1).
- 19:40: no release passing G2 means V6.
- 19:55: G3 not passed means V6.
- 20:15: not all 88 shards predicted means V6 stays the 21:30 incumbent.
- 21:15: no validated archive means V6.
- A V7 that has passed every gate may replace V6 until 23:00 if a mechanical delay pushed it past 21:30.

Worker assignment uses the scaffold's `assign --rates` on the measured shards. France shards are heaviest
(7.31 candidates per S1). A worker running the audit is not available for test work until it finishes.
Reassignment is explicit and versioned, and a retried shard's earlier assignment is fenced first.
The collector accepts exactly one complete output per shard.

## 8. Tests and release contract

**Blocking tests (G1), synthetic fixtures only:**

1. S1 with 0, 1, 2 and 3 survivors, plus a fully empty shard: fixed schema, no out-of-bounds, no
   self-sibling, no self-rival; zero-survivor S1 kept in the ledger and scored as empty.
2. Coherent rivals: a name-only rival and a different address-only rival never form one joint rival;
   missing address is NaN, not equality; an overflow key sets `t_overflow` and adds no rivals;
   duplicate hits through several views count once.
3. Label isolation: F3 and every cached feature are identical when the truth file is missing or
   replaced with invented labels.
4. Permuted input order and tied p1/p6/p7 give identical C(q), features, ownership and decisions.
5. Collector rejects: a missing shard, a duplicated shard, same row count with an altered target,
   mismatched target_id, NaN p7, mismatched code/model/feature/index pins, a stale assignment, and
   validator exit `None` or nonzero.
6. Metric fixtures: both empty = 1, missed nonempty = 0, singleton false positive = 0, PDF case 10/14.
7. Reload: every fold model reproduces in-memory predictions within 1e-6; alpha, threshold and the
   monotone vector are serialized and exercised.

Non-blocking, run if time allows: worker-parity golden records across machines, determinism of cached
shards, and a large synthetic collection for memory.

**Per-shard output:** `s1, t, target_id, p1, p6, p7, s`, sorted by (s1, t), plus a sidecar with input
shard SHA, query-ledger digest (nonempty and empty S1), logical pair digest, output SHA, code SHA and
executable file hashes, release SHA, index digest, assignment version, worker, runtime, peak RSS and
`pip freeze`. Pair digest: sort numeric (s1, t), serialize fixed-width little-endian uint32 pairs after
a header naming the dataset and row-map identities, then SHA-256. Verify target_id for every pair.

**Final assembly:**
1. Exactly the 88 expected shard IDs, with disjoint query assignments whose ledgers union to all
   1,732,544 test S1.
2. The union of scored pairs equals V6's pair set, by digest.
3. France routing (section 6).
4. Global ownership once over all test S1: highest s per target, ties to the lower S1 row. Then the
   frozen threshold. No forced matches and no cardinality caps.
5. `candidate_pairs.tsv` = V6's file, byte for byte. `matching_results.tsv`: every S1 once, France and
   empty lists included, existing test S2/S3 IDs only, no duplicates, matches within candidates, no
   target given to two S1.
6. Official validator with `--check-ids`: exit 0 required. `src/strict_validate.py`: pass required.
   Then package, extract to a fresh directory, validate the extracted files and compare SHA-256.
7. The archive holds reproducible code, `release.json`, fold models, index build commands, the
   documentation template and a short summary, with no temporary teammate URLs. V6 and every submitted
   version are preserved. A human uploads and records the actual scored receipt.

## 9. Rules, licenses and documentation

Only supplied records and labels are used. Test S1 records serve as label-free context, like the V6
ownership rule. No lookups, external corpora, hosted resolvers, geocoding or generated labels.

Inventory every model: v5.1, three V6 members and five V7 fold models, all LightGBM (MIT), with actual
tree and leaf counts. The combined size is far below 8B parameters, but state the counts rather than
asserting it. Unidecode 1.4.0 (GPL-2.0-or-later) remains a preprocessing dependency of the existing
normalizer. Disclose it as before. Changing it now would change every model input.

The methodology describes the full cascade: TF-IDF retrieval of about 265 candidates per S1, v5.1
scoring of the union, the fixed survivor cut (the candidate file), V6 ens3 plus the V7 head scoring
exactly those survivors, whole-population rival context, global ownership, the threshold and France
routing. Report tune OOF, the audit result with the second-look disclosure, candidate counts (mean
5.35, p95 by country) and France's limits. Keep data, scores, features, models and records private;
Git receives code, tests and aggregate evidence only.

## 10. Not today

Retrieval changes (digit-letter folding, v3 keys, larger K), F2, expected-F, neural models, member
spread, per-country thresholds and a new holdout partition. Each needs either a feature rebuild, more
time than remains or labels that do not exist. Retrieval loss is 0.00592; the matching gap is 0.02319.

No new V7 score, throughput, spend, artifact or submission is claimed by this document.
