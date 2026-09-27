# Handoff: AML entity resolution, v5.1 cascade to your next iteration

From session `claude-20260927-0340` (Claude Opus 5.5), written 27 Sep 2026 about 10:15 IST.
Addressed to the Claude session that iterates next. Read this file completely, then
`AGENTS.md`, `coordination/PROTOCOL.md` and the live `codex/control` state before you act.

**Your goal:** raise the private-leaderboard macro F0.5 **and** keep a small candidate set per
Source 1 entity. The organizers rank a smaller `candidate_pairs.tsv` higher, alongside the
score. A valid package already exists, so your work is to improve on it. The ideas below come
from what I measured. Do not stop at them. Think about the data and the metric yourself, look
at the actual errors, and pursue better ideas when you find them. Measure every change on the
fixed tune partition before you promote it.

---

## 1. State in twelve lines

- Code: branch `codex/v5-dist-claude-20260927` (the release code SHA is `ad291a7`; the head also
  holds the release records and these tools). The pipeline is `code/business_entity_resolution/src/v5_dist.py`
  (about 1,000 lines) plus `src/v5_cascade.py`, `src/text_norm_v5.py`, `src/v5_vec.py`, `deploy/` and `tests/test_v5.py`.
- Released package: `AML_submission.zip` on the user's machine (`C:\Ml_challenge_submission\`), SHA-256
  `c00693f465fba557fa38ab46a226a58da4abddd6e954b44194025a59348dfe0d`, 331 MB. It passes the official
  validator (`--check-ids`). Release records are in `coordination/evidence/v5-dist/release/`.
- Released model: v5.1 LightGBM, 2,000 trees, 127 leaves, 92 features, SHA `eec93e35...`, trained
  on 400k fit queries (39.2M pairs), with the threshold selected on 100k tune queries.
- Released candidates: stage-1 keeps the **top 30 per S1** by `combo + combo_word` cosine
  (label-free), and the model scores only those 30. That gives 51,976,320 candidate pairs.
- **Tune macro F0.5: 0.9605** at 30 candidates/S1, threshold 0.72. With the full retrieved union
  (about 256/S1) the same model scores **0.9679** (US 0.9741, India 0.9587). The v5 baseline scored 0.9590.
- Retrieval ceiling (oracle U) on the full union is 0.9952 (US 0.9989, India 0.9897). **At top 30 it
  is only 0.9841.** That 1.1-point loss is now the largest single problem.
- Matching gap on the full union: U-S = 2.7 points. Both trained models stopped at the
  2,000-round cap without early stopping, so they were still improving.
- The audit partition (the last 10% of the seed-42 permutation) has **never been opened**. Keep it locked
  until a champion is chosen.
- All 8 SageMaker notebook instances are **stopped**. Estimated spend so far is about $23-25 of the
  user's $100 cap on this account (instance-hours times list price; Cost Explorer showed no data).
- Deadlines: official close 27 Sep 23:59 IST. `AGENTS.md` sets an internal acceptance deadline of
  21:30 IST. Keep the released zip as the fallback until a better one is validated.
- Humans do the portal work. Never upload to the portal. Leaderboard uploads are limited to 5 per day.
- The user must approve paid compute (starting instances) in chat. An automatic permission
  check blocks starting instances or rewriting shared queue files without that approval.

---

## 2. Rules you must keep

- Use only the provided data: no external lookups, geocoding or hosted resolvers. The model must be
  MIT or Apache-2.0 with at most 8B parameters. LightGBM is MIT and has about 506k parameters.
- Unidecode (GPL-2.0-or-later) is used for transliteration and is disclosed in the docs. Replacing
  it with `anyascii` (ISC) is optional, but it changes normalization, so it needs a full feature rebuild.
- `candidate_pairs.tsv` must be the exact set the final model scores, and matches must be a subset
  of it. Every test S1 (1,732,544) appears exactly once in both files, with no duplicate IDs and only
  existing S2/S3 IDs. A multi-stage design is fine: the last stage before the final model defines the
  candidate set. Do not trim a scored candidate file just to shrink it.
- Package layout (from README): `output/{matching_results.tsv,candidate_pairs.tsv}`,
  `code/business_entity_resolution/{src/,README.md,requirements.txt}` and a filled `Documentation_template.md`.
  `tools/make_zip.py` builds and re-validates the zip. Update the documentation numbers
  (see the current filled doc in `coordination/evidence/v5-dist/release/Documentation_template.md`).
- Coordination: claim bounded work on `codex/control` (see PROTOCOL). Use your own branch
  `codex/<task>-<session>` based on `codex/v5-dist-claude-20260927`. Record job reservations before
  paid runs. Keep raw data, predictions and models out of git.

---

## 3. What was measured (use these numbers, do not re-derive them)

Data: train S1 2,206,821 rows (US 1,323,633, India 883,188), 10,320,219 train targets, test S1
1,732,544 (US 663,106, India 809,986, France 259,452) and 9,969,589 test targets. There are 3.5 true
links per S1 on average, up to 11, and 5.6% of S1s are singletons.
Note: Codex once misread the S1 count as 226,881 from a screenshot. The real count is 2,206,821.

**Retrieval benchmark** (4,000 tune queries per country, full catalogs, k per source):

| policy | India U | US U | pairs/q (India) |
| --- | ---: | ---: | ---: |
| name/address char + fold, each @20 | 0.9705 | 0.9926 | 125 |
| + combo char @40 | 0.9789 | 0.9981 | 177 |
| name@20, addr@40, folds@10, combo@40, combo_word@20 | 0.9854 | 0.9986 | 194 |
| everything @40-60 | 0.9876 | 0.9989 | 309 |

Single channels (India U@20 per source): name_char 0.729, address_char 0.925, combo 0.962,
combo_word 0.976 (0.981 @40). Names alone retrieve poorly because many unrelated businesses
share names. The combined name+address views carry most of the recall.

**Stage-1 cut-offs on tune** (the same v5.1 model; threshold re-selected on a 0.01 grid):

| stage-1 rank | N=20 | N=30 | N=40 | N=60 | N=80 |
| --- | --- | --- | --- | --- | --- |
| combo + combo_word cosine: F0.5 / U | 0.9592 / 0.9822 | **0.9605 / 0.9841** | 0.9611 / 0.9851 | 0.9620 / 0.9862 | 0.9625 / 0.9871 |
| + name_char + address_char cosine | 0.9581 / 0.9805 | 0.9595 / 0.9825 | 0.9604 / 0.9837 | 0.9623 / 0.9865 | 0.9635 / 0.9883 |
| all six channel cosines | 0.9504 / 0.9722 | 0.9576 / 0.9804 | 0.9600 / 0.9834 | 0.9622 / 0.9864 | 0.9636 / 0.9885 |
| `h_rank` (token-set heuristic) | 0.9276 / 0.9509 | 0.9442 / 0.9678 | 0.9522 / 0.9762 | 0.9608 / 0.9854 | - |

`h_rank` is poor because `token_set_ratio` returns 100 for any token subset, which creates huge ties.
The full union (about 256/q) reaches F 0.9679 and U 0.9952. The gap between the top-30 cut and the
full union, about 1.1 points of U, is the main opportunity.

**Stage timings on 8 workers** (with blocked search):
- Feature build, 115 shards of 20k queries (train 510k plus test 1.73M): about **2 h wall-clock**
  for v5.1 with 6-7 free workers (six channels plus anchors). A shard takes about 3-7 min on the
  c-class workers and less on r-class. Train shards alone (27) took about 50 min. Rebuilding
  features is the expensive step, so avoid it unless features change.
- Catalog build per worker: 2-4 min (US train 6.19M targets is the largest).
- LightGBM, 2,000 rounds on 39.2M rows x 92 features, 8 threads: about 65 min, at 1.8-2.0 s per round.
- Tune scoring plus threshold grid: about 10 min. Test scoring (88 shards): about 10 min.
- Final assembly: about 10 min for the 444M-pair union, about 1 min for the top-30 set.
- Cascade: `ceval` about 6 min. Cascade scoring plus final: about 6 min.

**Memory:** a v5.1 US-train feature task peaks around 15 GB, so 16 GB workers are excluded
(`memory_scale 2.0`). Training 39M rows needs about 40 GB, and final assembly now streams. `min_mem_gb`
per task handles placement automatically.

---

## 4. SageMaker setup, step by step

### 4.1 Access
- Account 580857071542, region **ap-south-1**. Execution role
  `arn:aws:iam::580857071542:role/service-role/AmazonSageMakerAdminIAMExecutionRole_1`
  (SageMakerStudioAdminIAMPermissiveExecutionPolicy, which grants full S3/SageMaker).
- Bucket: `amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt` (private, SSL-only, SSE-S3).
  Always pass `--sse AES256` / `ServerSideEncryption="AES256"`.
- CLI login: `aws login --region ap-south-1`. The user signs in through their browser; you never
  handle passwords. The local boto3 cannot read login credentials without `botocore[crt]`, so export
  them into your Python process:
  `$c = aws configure export-credentials --format process | ConvertFrom-Json` then set
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`. They are short-lived, so
  re-export per command.

### 4.2 Quotas (checked; do not waste time on other routes)
- **SageMaker training jobs and processing jobs: 0**, except tiny t3 processing. A quota request would take hours.
- **Notebook instances: 8 in total**, one per type. This is the compute we use. Instances, all with 8 vCPU:
  `er-v5-r7i` (ml.r7i.2xlarge, 62 GB), `er-v5-r6i` (r6i, 62 GB), `er-v5-r6id` (r6id, 62 GB),
  `er-v5-m5` (m5, 31 GB, slower), and `er-v5-c7i`, `er-v5-c6i`, `er-v5-c5`, `er-v5-c5d` (15.3 GB each).
  They all exist, carry lifecycle config `er-v5-worker`, and are currently **stopped**.
- Studio JupyterLab: one app per type. The domain is a SageMaker **Unified Studio (DataZone) managed**
  domain, so do not modify it. EC2 allows 16 on-demand and 32 spot vCPUs but needs an IAM role;
  do not create IAM roles without the user.
- Cost is about $4.5/h with all 8 running. Stop them when idle.

### 4.3 How the cluster works
- Each notebook instance runs `deploy/onstart.sh` at start. It fetches
  `s3://BUCKET/shared/er-v5-20260927/code/bootstrap.sh` and runs it in the background.
- `bootstrap.sh` installs uv and a Python 3.12 venv from `requirements-v5.txt` (the venv and data persist on
  the instance volume), downloads and extracts the dataset zip
  (`shared/entity-v4-job-20260927/input/student_resource.zip`), downloads `code/code.tar.gz` from its
  **home prefix** (`shared/er-v5-20260927`), and loops
  `python -m src.v5_dist worker --stay --reload --bucket ... --prefix <home> ...`.
- **Queues** are S3 prefixes. Each has `plan.json` (immutable run settings) and `tasks/*.json`. Workers
  claim a task by creating `claims/<id>` with an S3 conditional write (`IfNoneMatch='*'`). They then write
  `done/<id>.json` or `failed/<id>.json`, the outputs (`feat/`, `model/`, `score/`, `final/`, `ceval/`)
  and `attempts/<id>--<worker>` markers after an interruption. A failed task is **not** retried.
  To retry, delete its claim and its failed marker.
- **Control files** live in the home prefix `shared/er-v5-20260927/control/`:
  - `queues` is a JSON list of prefixes in priority order, e.g. `["shared/er-XYZ"]`. `queues-<worker>`
    overrides it per worker. Workers read this before every task.
  - `stop` and `stop-<worker>` make the worker exit after its current task. With `stop-instances` also
    present, the bootstrap then stops the instance. **Delete the stop flags before you restart.**
  - `prefix-<worker>` is a bootstrap-time home-prefix override (rarely needed).
- **Hot reload:** upload a new `code.tar.gz` to `shared/er-v5-20260927/code/`. Every worker notices the
  ETag change within about 30 s between tasks, exits with code 75, and the bootstrap re-extracts and
  restarts it. No reboot is needed. Package it with
  `tar -czf code.tar.gz -C <pkgdir> src requirements-v5.txt tests`. Upload `bootstrap.sh` with LF line
  endings only.
- **Monitoring:** `tools/queue_status.py` (set `ER_PREFIX`) prints done/running/failed per task group
  and worker heartbeats (`logs/<worker>.json` in the home prefix, refreshed every 60 s). Worker logs
  are in `logs/<worker>.log`. Prefer an S3-polling background loop over repeated manual checks.
- **Start/stop:** `aws sagemaker start-notebook-instance --notebook-instance-name er-v5-r7i --region ap-south-1`
  (about 5 min to InService; the worker starts during boot). Ask the user first.

### 4.4 Launching a run
1. Change code, then run `python -m pytest tests/test_v5.py -q --basetemp <dir>`. The synthetic end-to-end
   tests cover plan, worker, train, score, final, bench, and cascade eval and score, and take about
   2 min. Commit and push.
2. Publish a queue from your machine with the dataset at `C:\Ml_challenge\student_resource\dataset`, e.g.
   `python -m src.v5_dist plan --data <dataset> --bucket <B> --prefix shared/er-<name> --norm v5 --anchors --channels-json deploy/channels_v51.json --memory-scale 2.0 --neg-keep 0.3 [--fit N --top-k K]`.
   This writes `plan.json` plus 205 tasks: 115 feat, train, 88 score and final.
3. Upload the code, write `control/queues`, and start the instances (with user approval).
4. The cascade on a finished run:
   `python -m src.v5_cascade eval --bucket <B> --source <run> --prefix <run>c --ns 20,30,40 --ranks= --scores=combo+combo_word`,
   then `python -m src.v5_cascade score ... --n 30 --score combo+combo_word --threshold <T>`.
   PowerShell drops empty arguments, so write `--ranks=` rather than `--ranks ""`.

### 4.5 Where the artifacts are
- `shared/er-v51-20260927/`: v5.1 plan, `feat/` (27 train shards plus 88 test shards with **all** about 256
  candidates per query and all 92 features; about 20 GB), `model/` (model.txt, manifest, tune_scores,
  tune_per_query), `score/` (full-union test scores) and `final/` (full-union TSVs).
- `shared/er-v51c-20260927/`: the released cascade. It holds `ceval/` results, `model/` (the same model,
  cascade settings and threshold 0.72 in the manifest), `score/` (top-30 survivor scores) and `final/`
  (the released TSVs plus report).
- `shared/er-v5-20260927/`: the v5 baseline run, **plus the home prefix for code and control**.
- `shared/er-v5-bench1/`: retrieval benchmark results (`bench/bench-India.json`, `bench-US.json`).

---

## 5. Speed playbook

1. **Reuse the v5.1 feature shards.** Anything that leaves normalization, retrieval and features unchanged
   (retraining, stage-1 rankers, thresholds, ensembling, stacking on existing columns) should read
   `shared/er-v51-20260927/feat/` directly. The cascade code already takes `source_prefix`. Add the same to
   `run_train` (about 5 lines: build a `Store` for `task["source_prefix"]` and download feats from it).
   That saves the roughly 2 h feature phase on every iteration.
2. **Train on survivors.** With a stage-1 filter applied inside `run_train`, fit rows drop from 39M to
   about 12M. A round costs about 0.6 s instead of 1.9 s, so you can afford 4-6k rounds or larger trees.
3. **Run variants in parallel.** There are three 62 GB r-class workers. Give each variant its own queue
   prefix (plan and train task) and list them all in `control/queues`, so three trainings run at once.
   The other five workers can build features for new data meanwhile.
4. **Blocked sparse search** (250k-target blocks) is already in, and 10x faster than one big matrix,
   because sparse_dot_topn's dense accumulator must fit in cache. Keep it.
5. **Faster LightGBM options** if needed: `max_bin 63-127`, `bagging_fraction 0.5` or
   `data_sample_strategy goss`, and `num_threads 8` (already). Load feature parquet columns selectively.
6. **Memory rules:** training is only safe on r-class workers (`min_mem_gb 60`). A catalog with six channels
   needs `memory_scale 2.0`. Final assembly must stream shards, which it does since commit `4579f2f`.
7. **Avoid pointless restarts:** use hot reload for code and `control/queues` for scheduling. Restart an
   instance only when it has hung (for example, a heartbeat older than 5 min with `mem_avail` near 0). Then write
   an `attempts/<task>--<worker>` marker and delete the stale claim so another worker picks up the task.

---

## 6. Improvement backlog (my ranking, with rationale; verify before you trust it)

**P1. Retrain for the cascade you actually submit.** The model was trained on about 256 candidates per
query but is applied to the top 30. Retrain on stage-1 survivors (top 30-60), with context features still
computed on the full union exactly as at test time. Also raise the round cap (the loss was still falling
at 2,000) and try `num_leaves` 255. Tune F is computed on survivors, and truth still includes every
positive so misses count. Expected gain: several tenths of a point at the same N. Cost: about 1 h, with no
feature rebuild.
Implementation: (a) add `source_prefix` to `run_train`; (b) add
`if plan.get("cascade"): frame = survivors(frame, plan["cascade"])` right after reading each shard;
(c) write `manifest["cascade"] = plan["cascade"]` so `run_cascade` can score test; (d) queue the train task,
88 cascade tasks with `source_prefix=shared/er-v51-20260927`, and final.

**P2. A better stage-1 ranker, to close the 1.1-point U gap at N=30.** The cosine sum leaves U at 0.9841,
against 0.9952 for the full union. Train a small learned stage-1 model (LightGBM, about 100-300 trees,
31 leaves) on fit rows. Use cheap label-free inputs such as all six channel cosines, their ranks and gaps,
`n_cand`, source, and a few cheap string similarities. Rank candidates by its score and keep the top N.
This is still a legitimate blocking stage before the final model. Measure U@N and F@N on tune.
If U@30 approaches 0.99, final F at 30 should approach the full-union 0.9679. Extend `stage_rank()`
with a `{"model": ...}` policy.

**P3. Adaptive per-query N, for smaller candidate sets.** Every S1 currently gets exactly 30 because every
query retrieves more than 30. Most S1s have 1-5 true links. Keep candidates whose stage-1 score is within a
margin of the query's best, or above a calibrated floor, capped at N_max. That can cut the mean count well
below 30 with little loss of U. Report the mean and p95 candidates per S1 in the docs; the organizers rank on
this.

**P4. Threshold refinement.** `ceval` uses a 0.01 grid from 0.60 to 0.95; the full-union run used a 0.0025 fine
grid, and the optimum moved (0.785 against 0.72). Refine the grid, and test per-country thresholds for US
and India (France uses the global one). Adopt a per-country threshold only with a clear paired gain.

**P5. Stacking / model-based anchors (for the U-S gap of about 2.7 points).** Get 2-fold out-of-fold stage-1
probabilities on fit. Add features from the other candidates' scores: max and sum of p over other candidates,
this candidate's p-rank, the gap to the top p, how many candidates score above 0.5, and similarity to the
highest-p candidate. Then train a stage-2 model. True variants of one business support each other. This is
the biggest modelling lever left, but it is the most work (2-3 h).

**P6. More training data.** Only 400k of the roughly 1.8M eligible train S1s were used. Add feature tasks for
another 400-800k fit rows. `plan()` rebuilds every task, so publish only the new rows as extra feat tasks and
extend the train inputs. At 20k rows per shard that is 20-40 shards, about 15-25 min on 8 workers. This pairs
well with P1, since survivor filtering keeps training fast.

**P7. Ensembling.** After P1, average 2-3 LightGBM models (different seeds or feature fractions) at the
probability level. It is cheap with parallel r-workers, and usually gains 0.1-0.3 points.

**P8. Error analysis before guessing.** Astra's `src/analyze_tune.py` (branch
`codex/submission-support-astra-20260927-0450`, with `coordination/evidence/error-analysis/`) splits the loss
into unretrieved, false-positive and retrieved-but-rejected, by country and truth count, with examples. Run
it on `shared/er-v51-20260927/model/tune_*.parquet`. Join the stage-1 ranks from the train feat shards to
analyse the cascade. Singleton F is 0.965, so false merges on no-match S1s are measurable. Look at them.

**P9. Normalization and features.** Possible directions: `anyascii` for license cleanliness; better Indic
transliteration and state-name handling (the Telugu and Devanagari state names left by unidecode); address
house-number and range logic (`18-22` against `18`); name tokens that are web domains or handles;
frequency-weighted token overlap (rare shared tokens are stronger evidence). Each change requires a full
feature rebuild (about 2 h plus training), so batch them into one rebuild and start it early if you want it.

**P10. France.** It has no labels. At minimum, compare its candidate and score distributions with US and India
to catch collapse, such as too many empty predictions. Do not tune on France.

Also consider, and think beyond this list: a query-level no-match gate for singletons, a pairwise or listwise
ranking objective (`lambdarank` grouped by S1) for stage-1, source-specific thresholds (S2 and S3 behave
differently in the noise profile), and adding the combined channels' *ranks* to stage-1. **Look at the errors
and invent better approaches. These notes are a starting point, not a ceiling.**

---

## 7. Suggested schedule (13 h from about 10:30 IST)

| IST | Step |
| --- | --- |
| 10:30-11:15 | Read, claim, get user approval for compute, implement P1 (source_prefix, survivor training), test locally |
| 11:15-12:30 | Run P1 variants in parallel on the r-workers (N=30 and N=40 survivors, 4k rounds, leaves 127 and 255). Meanwhile run P8 error analysis locally |
| 12:30-14:30 | P2 learned stage-1 and P3 adaptive N, evaluated on tune. Optionally start P6 extra feature shards on the c/m workers |
| 14:30-17:30 | Best combination, plus P5 or P7 if time allows. Choose the champion by tune F, candidates/S1 and paired deltas |
| 17:30-18:30 | Open the locked audit **once** for the champion only. Score test, assemble, validate |
| 18:30-20:30 | Package (`tools/make_zip.py`), update the docs and README numbers, independent validator checks, hand to the user |
| 20:30-21:30 | Buffer. The user uploads to the portal. Never replace the fallback without a validated better zip |

Stop rules: if a step shows no tune gain after two tries, move on. Do not open the audit for model selection.
Always keep the current released zip as the fallback.

---

## 8. Lessons and pitfalls from this session

- Read counts from files, not screenshots. The 226,881 misread would have wasted the whole run.
- Workers used `time.monotonic()` against a 0 initial value, so new instances idled for 5 min (fixed).
  Task lists now refresh every 60 s.
- The first full-union final assembly ran out of memory on a 62 GB worker. It now streams shard by shard. Keep
  heavy steps streaming.
- Stage-1 `h_rank` looked sensible but ties destroyed it. Always measure U@N before choosing a filter.
- Both trained models hit the round cap. Check `best_iteration` in `done/train.json`.
- The permission classifier blocks starting instances or writing shared control files unless the user has
  approved in chat. Ask first and explain the cost.
- In PowerShell, a command that combines `Remove-Item` with the repo path gets blocked, so split it.
  Empty-string arguments vanish, so use `--flag=`.
- boto3 needs exported credentials (section 4.1). The `aws` CLI works directly after `aws login`.
- The official validator only warns on matches outside the candidates. Check that yourself too; it
  holds by construction in this pipeline.

## 9. Tools in `code/business_entity_resolution/tools/`
- `queue_status.py`: per-queue progress, failures and worker heartbeats (`ER_PREFIX`).
- `release_claims.py <worker...>`: releases a stuck worker's unfinished claims in `ER_PREFIX`.
- `shard_oracle.py <prefix> <feat-train-...>`: oracle U, link recall and per-channel recall from train feature shards.
- `bench_plan_example.py`: how the retrieval benchmark queue was published.
- `make_zip.py <values.json> <zip>`: fills doc placeholders, runs the validator, builds a ZIP64 archive and
  re-validates the extracted TSVs (paths inside are for the user's machine; adjust them).
