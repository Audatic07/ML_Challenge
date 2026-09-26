# Review and v4 training plan

Reviewed `aadish_v1` a1d42bd and `akash/v3-pipeline` d5f3775 on 27 Sep 2026.
User target: macro per-S1 F0.5 >= 0.99; not yet established.

## Evidence and strengths

Akash v2 has measured results on 100k development queries and all 10.32M train
targets: link recall .8817, candidate oracle .9483, model .9119. Address keys
improved v1 (.8652). V3's extra Indic phonetic folds address observed miss
patterns, but have no real-data score. Chunked features and complete output rows
are useful foundations.

Aadish has nested validation samples, a locked audit reserve, held-out target
exclusion, deterministic tie-breaking, null-safe keys, address token pairs, text
reranking and 30 tests. The strongest checked-in retrieval result is oracle
U=.968438 on 5k queries against the full target pool (top100), with link recall
.920705. This is a retrieval ceiling, not an achieved matcher score. Its sample
is different from Akash's 100k.

## Defects and limits

* Aadish deletes `excluded_targets` then uses `len(excluded_targets)` for model
  metadata, which raises after training. Fixed in this branch.
* Akash moves the validation interval when fit size changes, which can invalidate
  a model comparison or train on the old validation.
* Akash ownership `p == max(p)` assigns tied candidates to every tied query.
* Both branches discard an entire block when its posting count exceeds a cap.
  Increasing shared caps can crowd the top-K with weak competitors.
* Akash replaces raw address words with phonetic skeletons in v3. Raw and folded
  views should coexist because folds improve recall and introduce collisions.
* Old caches are keyed by filename, not code/data digest. Old prediction retains
  every scored pair in memory and cannot resume a completed individual shard.
* U below .99 cannot reach a .99 end-to-end score for that recorded candidate
  policy. Repeated business fingerprints across different S1 IDs remain unaudited.

## v4 implementation

Four separate target-wide sparse text channels: name character 3/4-grams,
address character 3/4-grams, auxiliary phonetic name word 1/2-grams and auxiliary
phonetic address word 1/2-grams. IDF is fit on the complete supplied target pool;
no labels enter retrieval. Each channel keeps a separate per-source quota. The
union is never trimmed afterward. Name does not gate address, and no true pair is
injected. Country is an open set.

Normalization, retrieval and feature shards carry code/data/config pins and
checksums for safe resumption. New features include extra-name-token penalties,
asymmetric token coverage, numeric-set overlap, folded edit signals and each
retrieval channel score. Per-query context is computed after all target shards
are reunited.

Seed 42 and the historical `perm[200000:300000]` development population stay
fixed. `perm[300000:310000]` is reserved for early stopping. Threshold tuning is
on a separate nested subset of historical validation. Every reserved target is
excluded from fit negatives. The last 10% remains unopened. LightGBM fits only
supplied labels, with hard negatives from the full searchable target pool.

The bounded pilot runs only after its spend cap and job ID are recorded. First
inspect candidate oracle U and country slices; for the 0.99 target aim for U>=.997.
If retrieval dominates, repair it before adding model capacity. If matching
dominates, grow the fit set and use fit-only hard negatives. No sample or training
score establishes the target.

Risks: character channels still need measured full-pool runtime and recall; hash
collisions and Indic phonetic folds can add false candidates; France transfer is
unlabelled; repeated exact fingerprints are not audited. Unidecode is GPL and
requires organizer interpretation for a release. LightGBM is MIT. No external
pretrained weights or business lookup data are used.

The v4 pilot is experimental and cannot replace the incumbent without paired
validation. Final test-set retrieval/inference and release validation are not yet
part of this pilot.
