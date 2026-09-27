# S2-MODEL bounded smoke run: results (Akash's account)

- **Session:** `claude-newacct-20260927-1155`. Claim recorded on `codex/control` at `941be5c`.
- **Code:** base `db4056b04647948c307a93ca988534a21d06f5ee` plus `smoke_patch.diff` (SHA-256 `3b1406d88837bfed050ab6a6a64fac48f7ca00880356bc053eb2cd6cfdd2b4a0`).
- **Run:** 2026-09-27, 09:25:44 to 09:29:21 UTC, on Studio `amlc-er` (`ml.r5.xlarge`, 4 threads). Exit code 0, and no limit was hit (limits: 240 min, 22 GiB, 3 GiB free).
- **Scope:** baseline reproduction, stage-two CV and one France shard only. No full inference, no extra workers.
- **Results location:** `s3://<akash-private-bucket>/shared/er-v6s2-smoke-akash-20260927/`
  - `prerun/`: code, scripts and manifest
  - `logs/`: every step log plus input and code hashes
  - `s2_cv_K10_f0.json`
  - `work/`: record-level outputs, with `outputs.sha256`
- **Verification:** every log and JSON file read back from S3 with a matching SHA-256. The `work/` files match on size.

## Input checks

- **Synthetic 0/1/2/3-survivor check:** passed with the patch applied.
- **Row order:** 200,000 target rows sampled from `tune_scores.parquet` and 200,000 from `score-test-France-000.parquet` all index the same local dataset rows (0 mismatches). The local dataset TSVs were checked earlier against the canonical sizes, row counts and ZIP CRC32s.
- **Input SHA-256:**

  | File | SHA-256 |
  |---|---|
  | `plan.json` | `82a533bb33a6c5af0a3e1803fb936cbd119698c75911b5a9a2c7b9a2578d2ff5` |
  | `model_manifest.json` | `d51501517fdc87bb5f98ca3cc64b868286cd77776d84fa51eea19aaef23a4813` |
  | `tune_scores.parquet` | `e9bf1a401d031d99689d267272ba199b34df1872fa78b4c8242f40662523219a` |
  | `score-test-France-000.parquet` | `928f6aa2a1f78a6d72c46fcc36c97b06a4be2948a4780bb74d82d43851ff2133` |

## 1. Baseline reproduction (`loss.py`)

The stage-two evidence document is the reference: `coordination/evidence/stage2/claude-20260927-1130.md` at `db4056b`.

| Metric (tune, 100,000 S1) | Reproduced | Evidence document |
|---|---:|---:|
| Macro F0.5 at threshold 0.785 | 0.96794 | 0.96794 |
| Without the one-owner rule | 0.96777 | 0.96777 |
| Oracle U | 0.99520 | 0.99520 |
| India F / U | 0.95875 / 0.98967 | 0.95875 / 0.98967 |
| US F / U | 0.97406 / 0.99888 | 0.97406 / 0.99888 |
| Top 12, p ≥ 0.001: U / candidates per S1 | 0.99481 / 6.67 | 0.99481 / 6.67 |

The counterfactuals and pair counts also match: false positives 3,026, rejected positives 17,177 S1, and 5 matches moved by the one-owner rule.

## 2. Stage-two CV (`stage2_features.py tune 10` and `stage2_cv.py K10_f0`)

This is 2-fold out-of-fold (OOF) cross-fitting on the tune partition, not an untouched holdout.

| Top 10, tune | Stage 1 | Stage 2, out of fold | Evidence document (stage 2) |
|---|---:|---:|---:|
| Threshold | 0.784 | 0.651 | 0.651 |
| Macro F0.5 | 0.96780 | **0.96994** | 0.96994 |
| Fold 0 / fold 1 | 0.96801 / 0.96759 | 0.96986 / 0.97003 | same |
| India / US | 0.95859 / 0.97394 | 0.96253 / 0.97489 | same |
| Truth count 0 / 1 / 2 | 0.9646 / 0.9068 / 0.9596 | 0.9713 / 0.8943 / 0.9599 | same |
| False positives / rejected true matches | 3,103 / 19,244 | 4,029 / 14,290 | same |
| Best iteration, fold 0 / fold 1 | — | 398 / 330 | 398 / 330 |

The stage-two gain of +0.21 reproduces exactly, even with 4 threads instead of 20. Matches with a single true link still get worse under stage two (0.9068 falls to 0.8943).

## 3. Peak memory and runtime per step

| Step | Wall time | Peak memory (process tree) |
|---|---:|---:|
| edge check | 0.5 s | <0.1 GiB |
| row-order check | 4.5 s | 2.35 GiB |
| `loss.py` | 17.1 s | 3.69 GiB |
| stage-two features, tune K10 (1,000,000 rows) | 84.7 s | 6.48 GiB |
| stage-two CV | 68.3 s | 1.51 GiB |
| stage-two features, test K12 p ≥ 0.001 (one shard) | 39.1 s | 2.27 GiB |
| **Whole run, including downloads** | **222.8 s** | **6.51 GiB** |

## 4. Single France test shard (`score-test-France-000`, 20,000 S1)

The shard **completed**: 2 feature files, 204,966 rows × 64 columns.

- No non-finite values and no duplicate (S1, target) pairs.
- 19,996 S1 have survivors, averaging 10.25 candidates per S1. The 95th percentile is 12, which is the cap.
- **4 S1 have zero survivors**: every candidate scored below 0.001. They are missing from the feature output.

This step built stage-two **features only**. It produced no stage-two model predictions, match lists or submission files, because no release fit-and-predict entry point exists yet.

## Still open (not covered by this smoke test)

1. **Zero-survivor handling in final evaluation and output.** S1 with no survivors (4 of 20,000 in this shard) must still appear in both TSVs with empty lists and count in the metric. `expected_f.py` still requires every query to have survivors. This remains outstanding.
2. **Release entry point.** No code yet fits stage two on the allowed tune cohort, saves the model and feature list, reloads it and predicts test survivors.
3. **Pin the patch.** The `null_on_oob` fix and `ER_V6_THREADS` exist only in `smoke_patch.diff` in S3. They need a commit on a published branch, and this session has no Git push access.
4. **Evidence publication.** The results aren't on `codex/control` or an evidence branch yet. Someone with push access needs to record them, or copy this file to `coordination/evidence/s2-model/`.
5. **Not validated:** full stage-two validation, full test inference, France behavior beyond one shard, and the locked audit (still unopened).
