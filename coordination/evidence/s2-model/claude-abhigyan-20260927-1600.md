# S2-MODEL on the frozen v6 ens3 survivors: release path (Abhigyan's session)

Session `claude-abhigyan-20260927-1600`, 27 September 2026, from 16:00 IST. Branch
`codex/s2-model-claude-abhigyan-20260927-1600`. S2-MODEL was handed over by `claude-newacct-20260927-1155`
(Akash); his evidence file is unchanged.

## Base

- Akash's smoke patch as published: `a35aead` on `codex/s2-model-claude-newacct-20260927-1155`, tree `14e5d7ad`.
  Its code files are identical to the tree Akash verified (`f32eb112027231260ce4718e0701a92307743b87`,
  patch SHA-256 `cdcbbde0098412e2eba7b34c0871e84fbf5e0ea3bac3ffa7df12510f2ce85295`). The two trees differ by
  one line in his evidence note only. The exact `f32eb112` tree was reproduced here from his patch file.
- Merged unchanged: Aditya's published V6 branch at `42714bb`, so `src/v6_train.py` (`cut`, `best_threshold`)
  is the exact code that defined ens3. No V6-TRAIN path is edited.

## What was added

`tools/v6_analysis/stage2_release.py`, one entry point with four steps:

| Step | What it does |
|---|---|
| `gate` | Reads the ens3 `plan.json` and `model/model_manifest.json`, checks each member's `model_sha256`, averages the members' `model/tune_scores.parquet` as `tools/v6_ensemble_eval.py` does, and cuts with the manifest's `cascade` (`v6_train.cut`: top n by v5.1 p1, ties by target row, then p1 >= floor). Reproduces ens3 at its manifest threshold (fails if not within 1e-4 of `--expect-f`, default 0.97061 from control). Builds Akash's `v6_stage2.features` with the ens3 mean p as stage-1 p, runs the `stage2_cv.py` 2-fold LightGBM recipe, saves both fold models and checks that each reloads to the same predictions. Both sides get `v6_train.best_threshold` (one owner per target, then the tune sweep) on the same tune S1 and survivor rows. Pass needs a paired per-S1 gain above `--z` (default 2) standard errors. Only a pass writes `release.json`. |
| `assign` | Lists the ens3 score shards (expects 88), puts each worker's benchmark shard first, then assigns the rest by longest shard first to the worker with the earliest projected finish from measured rows per second. Static; records the release SHA-256. |
| `predict` | Runs only the shards assigned to `--worker` (`abhigyan-r7i4xl`, `aditya-worker`, `akash-r5xl`), or an explicit `--shards` subset of them. Checks the release gate, model checksums, the ens3 cut and the feature order. Writes one parquet file (`s1, t, target_id, p1, p_ens, p`) and one JSON file (rows, checksums, runtime, peak RSS) per shard. Restartable. |
| `collect` | Requires every assigned shard exactly once across the worker folders, the same release and code, matching checksums and row counts, and disjoint S1 rows. Then runs `v5_dist.run_final`: one owner per target over all test S1, the release threshold, both TSVs with every test S1 once (empty lists where no survivors), and the official validator with `--check-ids`. Adds label-free per-country match counts. |

The release model is the mean of the two fold models (each trained on half the tune S1); its threshold
comes from the out-of-fold sweep. The audit partition is never read.

## Checks run (local Windows, Python 3.14, `requirements-v5.txt` in a git-ignored venv)

- `tests/test_v6_stage2.py`: 3 passed. The new end-to-end test uses synthetic data (via
  `test_v5.make_split`): gate, fold-model reload, benchmark assignment, two workers, reassignment by
  measured rate, collector with the official validator (exit 0), an S1 with no survivors (empty in both
  TSVs), matches within candidates, refusal of an unassigned worker and of a shard delivered twice.
- `tests/test_v6_train.py`: 2 passed after the merge.

## Not run yet

The tune gate on the real ens3 inputs, any test inference, any paid compute. No stage-2 result on ens3
exists yet; the gate decides whether one is produced.
