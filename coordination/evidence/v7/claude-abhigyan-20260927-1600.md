# V7: a correction head over the V6 ens3 candidate set (plan revision 3)

## Update 20:55 IST: B4 is the V7 submission

B4 adds the F2 decoy-cluster features (revision 2 section 4; `src/v7_decoy.py`, `533bcae`) to B2. Rule pre-registered at
control `f63cea8` before any B4 number: B4 replaces B2 only if its OOF beats B2 by >= 0.0003, G2 passes, and a second,
disclosed audit look passes G3 with an audit score >= 0.97859.

| Rung | Tune OOF (100k S1, same folds) | Audit (220,682 S1) | Test `matching_results.tsv` |
| --- | ---: | ---: | --- |
| V6 ens3 at 0.72 | 0.97061 | 0.97089 | `3f69fe5b...` |
| B2 = F0+F1+F3 | 0.97769 (+0.00708, se 0.00021) | 0.97809 (+0.00720, se 0.00014) | `0504edf4...` |
| **B4 = F0+F1+F3+F2** | **0.98002 (+0.00941, se 0.00023)** | **0.98045 (+0.00957, se 0.00015)** | **`2bceeace7ca080f799922aecbed233a3a1a56f101ec8e90cb35e05813ed822bb`** |

B4: alpha 1.0, threshold 0.7375, 128 features, `release.json` SHA-256
`43987d03144b43c77d4893c673040fb2999ef1889681f354562272a355c77f90`; 5,751,635 matched pairs; both validators exit 0;
`candidate_pairs.tsv` is V6's file byte for byte; France routed to V6 by the same G6 rule. The audit has now been looked
at three times (V6, B2, B4), each as a pre-registered decision; treat 0.98045 as selected, not untouched.
Reproduce B4 with `gate --families F0F1F3F2` and the same steps below.


Session `claude-abhigyan-20260927-1600` (Abhigyan), 27 September 2026, 16:00-19:50 IST. Code on `main` from this
integration: WP2 `5ea334e`, WP3 `3bc79e5`, WP1 `09533fd`. Runs on one `ml.r7i.4xlarge` notebook in Abhigyan's account
(`er-v7-abhigyan-r7i4xl-20260927`). Private artifacts are in `shared/er-v7-20260927/` of Aditya's bucket. This file holds
aggregate numbers only.

## Short version

V7 rescores exactly V6's survivors (v5.1 top 12 by p1, ties by target row, then p1 >= 0.01), so `candidate_pairs.tsv` is
V6's file byte for byte. The chosen rung B2 adds population-rival evidence (F3) to V6's scores (F0) and sibling agreement
(F1). On the locked audit (220,682 S1, one pre-registered look after V6's), V7 scores **0.97809** against V6's **0.97089**.
France keeps V6's decisions by the pre-registered routing rule.

## Results

| Gate | Measurement | Result |
| --- | --- | --- |
| G0 inputs | V6 at its manifest threshold 0.72 on the 100k tune S1 | 0.97061 (expected 0.97061) |
| G2 B1 = F0+F1 | 5-fold OOF, alpha 1.0, threshold 0.7225 | 0.97215, +0.00155 (se 0.00018), pass |
| G2 B2 = F0+F1+F3 | 5-fold OOF, alpha 1.0, threshold 0.71 | **0.97769, +0.00708 (se 0.00021), pass**; beats B1 by >= 0.0003, chosen |
| G3 audit, once | V6 at 0.72 vs B2 on 220,682 audit S1 | V6 0.97089 (reproduced), **V7 0.97809, +0.00720 (se 0.00014), pass** |
| G5 release | official validator `--check-ids`, `src.strict_validate` | exit 0 and exit 0 |
| G6 France | France rJ_present 0.395 vs US/India 0.208-0.226 | out of envelope: France rows = V6 rows |

Tune and audit use the same folds and survivors as V6 (491,490 tune pairs, 4.91 per S1; 1,084,336 audit pairs, 3,945 audit
S1 without survivors). The audit survivors were extracted label-free with a query ledger of all 220,682 S1 before the
labels were joined once.

## Release

| Item | Value |
| --- | --- |
| `release.json` | SHA-256 `1f135bde2cf24327cd39d7e28fdc80f224bc423f852c94f14d1a01e61a83b7df` (5 fold models, 93 features, monotone +1 on p6/v5.1 p) |
| `matching_results.tsv` | SHA-256 `0504edf4098cc7ae174583b0b1585881a6d59c87d5815fe78b35896d68f3896b`, 5,804,378 matched pairs |
| `candidate_pairs.tsv` | SHA-256 `9c438703e605ac9d5fc4c0931b1c2272743054e75aeeb019b95891a2c83a9872` (V6 bytes), 9,266,800 pairs |
| test pair digest | `89e1aa978d44499feb52b7f2c0d5295e51fea39be4dab7b23f17ba9c3382bb73` |

Label-free test output per S1: India 3.28 predicted (6.3% empty), US 3.40 (5.8%), France 3.41 (4.9%, V6's rows).

## Reproduction (from `code/business_entity_resolution`, requirements-v5.txt)

Inputs are local copies of `shared/er-v6ens3-20260927/{plan.json, model/model_manifest.json, score/}`, the three member
`model/{model_manifest.json, tune_scores.parquet}` folders, the audit feature queue `shared/er-v51-audit-20260927/`, and the
supplied dataset. Pass `--data` as a clean path to `student_resource/dataset`.

```bash
python tools/v6_analysis/stage2_release.py gate --families F0F1F3 --ens3 E --members M --data D --out R
python tools/v7_audit.py survivors --bucket B --ens3 shared/er-v6ens3-20260927 --audit shared/er-v51-audit-20260927 \
    --out shared/er-v7-20260927/audit-survivors --data D --work W
python tools/v6_analysis/stage2_release.py assign --release R/release.json --scores S --out A0.json --benchmark abhigyan-r7i4xl=v6score-test-France-000
python tools/v6_analysis/stage2_release.py predict --release R/release.json --assignment A.json --worker abhigyan-r7i4xl \
    --scores S --data D --out O --cache C
python tools/v7_audit.py gate --survivors SURV --release R/release.json --data D --out G3.json --cache AC
python tools/v6_analysis/stage2_release.py collect --release R/release.json --assignment A.json --inputs O --data D --out F \
    --expect-full --v6-matching V6/matching_results.tsv --v6-candidates V6/candidate_pairs.tsv
```

Tests: `tests/test_v6_stage2.py` (end-to-end on synthetic data with the official validator via `ER_VALIDATOR`),
`tests/test_v7_population.py`, `tests/test_v7_audit.py`.

## Runtime

Tune gate B2 about 12 minutes (features 45 s, five folds). Audit survivor extraction 27 minutes on 8 threads (58M union
rows through v5.1). F3 test features about 10 minutes, prediction 2 minutes, collect 3 minutes. Peak RSS below 7 GiB.

## Caveats

- The audit had been opened once for V6. G3 is a second, pre-registered binary decision, not an untouched confirmation.
- The threshold and alpha were chosen on out-of-fold tune predictions; tune had also selected V6's cut and threshold.
- France has no labels. Routing it to V6 is a label-free safety rule, not evidence about French accuracy.
- Candidate accounting: v5.1 scores the full retrieved union (about 265 per S1); the survivor cut gives the candidate
  file; the head scores exactly those pairs. Population-rival lookups are label-free context, not extra candidates.
