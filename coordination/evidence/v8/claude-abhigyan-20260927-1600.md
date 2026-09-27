# V8: SC-Block contrastive similarity features on top of V7-B4, and the final submissions (27 September 2026)

Session `claude-abhigyan-20260927-1600` (Abhigyan). Code: `tools/v8/{prep,gpu,cpu}.py`, `tools/v7_thresholds.py`,
`stage2_release.py collect --france-routing never`. Aggregate numbers only; private artifacts are in
`shared/er-v8-20260927/` and `shared/er-v7-20260927/` of Abhigyan's bucket.

## Leaderboard (public subset, macro F0.5)

| File | SHA-256 | Public LB |
| --- | --- | ---: |
| V7-B4, France routed to V6 (G6 rule) | `2bceeace7ca080f799922aecbed233a3a1a56f101ec8e90cb35e05813ed822bb` | 0.968 |
| V7-B4 with V7's own France rows (`collect --france-routing never`) | `97ceedb9ce85c2d2f19fc7d5c697a03b95964b2e5535962151176c09aef1b6c5` | **0.972** |
| V8 = B5 (B4 + SC-Block cosine), own France rows | `63ea83262f19b6b81afb0048e8c444c6c6b8bd9a4b0f634fbccce89d6ad28628` | not recorded here |

Only France differs between the first two files (56,191 of 259,452 French rows), so the +0.004 is France alone,
about +0.027 on French S1: the pre-registered G6 routing rule was too cautious. France thresholds 0.80/0.85/0.90 were
built with `tools/v7_thresholds.py` (label-free: V7 France predicts 3.30 matches per S1 vs V6 3.41).

## V8 method (paper: Brinkmann, Shraga, Bizer, SC-Block, arXiv 2303.03132)

- Serialization `[COL] name [VAL] ... [COL] address [VAL] ...`; groups = each FIT S1 with its labelled S2/S3 matches
  (377,448 groups, 1.76M records; one owner per target, so groups are disjoint; tune and audit S1 never used).
- Encoder `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), mean pooling, L2 norm, supervised contrastive loss
  (two members per group, other groups in the batch as negatives), temperature 0.07, batch 256 groups, lr 5e-5, one epoch
  (1,474 steps, about 100 s on one A10G, `ml.g5.2xlarge`).
- Features per candidate pair: cosine, its rank and gap within the S1, the best other candidate's cosine.
- B5 = B4's 128 features + these 4, the same 5 SHA-256 folds and LightGBM recipe. B4's own out-of-fold predictions were
  rebuilt with each B4 fold model scoring only its held-out fold (0.98002 reproduced exactly).

| Tune (100k S1) | OOF macro F0.5 | Threshold |
| --- | ---: | ---: |
| B4 | 0.98002 | 0.7375 |
| B5 | 0.98043 (+0.00041, paired se 0.00010) | 0.7000 |

The pre-registered bar was +0.0010, so B5 failed its gate; the team lead chose to override it for one submission. The
test cosines come from the same encoder run as the tune cosines. The V8 file passed the official validator
(`--check-ids`); the strict validator was not run before the deadline. No audit look was taken for V8.

## Reproduce

```bash
python tools/v8/prep.py                                   # CPU: groups and pair texts
V8_S3=s3://BUCKET/shared/er-v8-20260927 python tools/v8/gpu.py   # GPU: train, tune and test cosines
python tools/v8/cpu.py gate && python tools/v8/cpu.py test       # CPU: B5 gate, then test file
```
