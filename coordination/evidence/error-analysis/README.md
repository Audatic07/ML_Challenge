# Read-only v5 tune error analysis

Prepared by astra-20260927-0450 from Claude's reachable temporary
`scratchpad/analyze.py` and pinned v5 source
`87298e02d8d8fe5f4df8aad815b41758b4b7f976`.
Only the tune partition is selected. No pipeline, notebook, job or S3 mutation is
implemented by these tools.

## Current execution status

Real-data analysis: **not run** at preparation time. The local SDK reports no
configured credentials/profiles, the exposed browser has no authenticated AWS
tab, and Claude's local `scratchpad/analysis/` folder does not exist.
This does **not** establish whether the remote artifacts exist or whether the
training job has finished. No score or real error examples are claimed.

Expected availability supplied by the user: around 2026-09-26 23:50 UTC /
2026-09-27 05:20 IST. A named profile or private local copy can unblock the run;
do not change credentials, job configuration or queue state merely to run it.

## Reproduce when access and all three files are ready

Use a new private directory outside Git. `fetch_tune.py` performs one
HeadObject/GetObject snapshot with ETags/version IDs, byte counts and SHA-256.
It never writes to S3 and does not poll. Alternatively use existing immutable
local copies.

~~~powershell
python coordination/evidence/error-analysis/fetch_tune.py --out "$env:TEMP/astra-v5-tune-snapshot"
python code/business_entity_resolution/src/analyze_tune.py --model-dir "$env:TEMP/astra-v5-tune-snapshot" --train-dir C:/Ml_challenge/student_resource/dataset/train --out "$env:TEMP/astra-v5-tune-analysis" --examples 20
~~~

An existing AWS profile can be passed to the snapshot tool with `--profile NAME`.
If a pinned local `plan.json` is available, pass `--plan PATH` to the analyzer;
this verifies the plan hash and exact tune membership. Otherwise, the saved
per-query index and manifest tune count define the selected partition.

The analyzer uses Polars and retains candidate scores in memory, as does the
starting script. Check available RAM against the full score table and join/sort
overhead before a real run; the 4-GB TSV validator has a separate, disk-backed
streaming implementation. If this workstation lacks capacity, execute this
read-only analysis on an already authorized machine with sufficient memory.
No such remote execution or resource launch is part of this preparation.

## Definitions and safeguards

For each query, let g be truth count, r retrieved truths, a predicted true
positives and b false positives. Let F(g,a,b) be the exact competition per-query
score (including the singleton convention), U=F(g,r,0), S=F(g,a,b), and
A=F(g,a,0).

- Retrieval loss: `1-U`.
- False-positive loss: `A-S`.
- Missed-but-retrieved loss: `U-A`.

These sum exactly to `1-S`. The latter two are an explicitly ordered
counterfactual attribution (remove FPs first); F0.5 is nonlinear, so other orders
can assign the interaction differently. Report both link counts and score loss.
A singleton false merge is attributed entirely to FP loss.

Recompute one-owner-per-target selection from scores with the production
tie-break (lower S1 row on equal probability) and the saved threshold. The
`p` and `tp` columns in the pinned `tune_per_query.parquet` schema come from
oracle predictions, **not** final predictions; the analyzer deliberately ignores
them. Recomputed g, F, U and aggregate F/U must match saved artifacts or the run
fails. This catches mixed versions, wrong thresholds and row-map mistakes.

Outputs are `aggregate.json`, `detail.private.parquet`, and
`examples.private.json`. Aggregates cover country, exact truth count and their
cross-product, with both within-slice means and contributions to total macro
loss. Link counts distinguish threshold rejections and ownership losses.
Examples are deterministic, severity-ranked selections across error
mechanism/country/truth-count groups (up to 20 distinct queries).

Publish only aggregate findings and checksums to Git. Keep raw query IDs, names,
addresses, scores and private examples outside the public repository. A completed
real analysis must add measured findings and its snapshot/model/plan/code pins
before it is described as run.
