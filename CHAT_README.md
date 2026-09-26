# Chat log: building v1 of the entity-resolution pipeline

Session with Claude Code on 26-27 September 2026, in `C:\Projects\ML_Challenge`.
Each **Prompt** is the user's message; each **Reply** is Claude's answer, with the work it did summarised.

---

## 1

**Prompt:**
> read the claude.md and the complete folder and then help me improve the accuracy of the model till 0.99

**Reply:**

I read `CLAUDE.md`, `AGENTS.md`, `SOL_IMPLEMENTATION_PLAN.md`, `v1_implementation.md`, the coordination files, the problem statement PDF and the data profiles.

Findings about the workspace:
- The repo had plans and coordination documents but **no model code**. The v1 pipeline described in `v1_implementation.md` (val F0.5 0.8652, blocking recall 0.79) lived on a teammate's SageMaker, not here.
- Machine: 4-core i5, 16 GB RAM (about 8 GB usable), no GPU. `lightgbm`, `rapidfuzz`, `polars` and `unidecode` were missing, so I created `.venv` and installed them.
- I extracted the dataset to `student_resource/` (git-ignored) and loaded it into `data/raw.duckdb` for exploration.

Findings about the data:
- It is synthetic, with a noise generator: typos, case changes, legal suffixes dropped or swapped, word reordering, junk prefixes (`--`, `<<`, `Mr`), website-style names (`baymedicalcenter.com`), names fully in Hindi/Punjabi/Bengali/Telugu script, missing addresses, state names abbreviated or in native script, and address parts reordered or dropped.
- No target is linked to more than one S1, and no link crosses countries.
- **About 26% of S2/S3 records match no S1, and they are deliberate decoys**: near-copies of a real business with an extra name word (e.g. "Summit", "Downtown") and a slightly different house number, sometimes with transposed digits (4067 vs 4076). Singletons have decoys too.
- 178k S1 names repeat at different addresses, so a name alone cannot decide a match.

I started writing a new package (normalisation, exact scorer, split) when the user interrupted.

---

## 2

**Prompt:**
> now help me create my v1 version so that i can push my v1 into the github repo and WARNING!!!!! NEVER PUSH ANY CODE WITHOUT TELLING ME

**Reply:**

I saved the rule "never push without telling the user" to memory and moved my draft out of the way. Then I rebuilt the v1 design from `v1_implementation.md` in `code/business_entity_resolution/`:

| File | Role |
|---|---|
| `src/config.py` | paths, env knobs, the 7 key types with weights and caps, LightGBM parameters |
| `src/text_norm.py` | views: `name_norm`, `core`, `skel`, `concat`, `addr_norm`, `addr_tok`, `postcode`, `house` |
| `src/prepare.py` | TSV reading (explicit tab, no quoting), normalisation, parquet cache, ground truth |
| `src/blocking.py` | 7 capped blocking keys, candidates ranked by summed key weight, top-K |
| `src/features.py` | about 45 rapidfuzz pair features, key flags, per-entity group features |
| `src/metric.py` | exact macro F0.5, candidate oracle, `unique_assign`, `decide` |
| `src/train.py` | seed-42 split (200k train / 100k val), LightGBM, threshold grid 0.20-0.95 |
| `src/predict.py` | test inference and the two submission TSVs |
| `README.md`, `requirements.txt`, `tests/test_metric.py` | run instructions, pinned versions, 3 metric tests (pass) |

Results on the laptop:

| | Recorded v1 | This run |
|---|---|---|
| Blocking recall (val) | 0.7913 | 0.7921 |
| Val macro F0.5 | 0.8652 | **0.8709** |
| Threshold | 0.625 | 0.675 |
| Test matches | 4,629,486 | 4,619,125 |
| Entities with no match | 11.3% | 11.4% |
| Candidate pairs | 33.6M | 34.0M |

Training took 11.5 minutes and test inference 29 minutes. The organiser's validator passed with `--check-ids`.

Submission files: `output/submissions/v1/matching_results.tsv` and `candidate_pairs.tsv` (git-ignored).

I committed locally as `598433e` on branch `codex/v1-pipeline`. **Nothing was pushed.** Commands to push when the user chooses:

```bash
git push -u origin codex/v1-pipeline
```

Notes:
- With v1's candidates, even a perfect matcher scores only **0.896** on validation, so blocking must improve before 0.99 is possible.
- `Unidecode` is GPL-licensed. It is a preprocessing helper, not the model (LightGBM is MIT), but check it against the licence rule.

---

## 3

**Prompt:**
> if i want to verify the work with astra or if i want to ask it to increase the accuracy then which all files i should provide it with the prompt from this folder??

**Reply:**

Context: `AGENTS.md`, `v1_implementation.md`, the problem statement PDF and the guidelines PDF.

Code: everything in `code/business_entity_resolution/` (`src/*.py`, `README.md`, `requirements.txt`, `tests/`).

Measured results (outside git): `data/er_work/meta.json`, `train_v1.log`, `predict_v1.log`.

For accuracy work, also: `output/hackathon/SOL_IMPLEMENTATION_PLAN.md` and `labelled_noise_profile.json`.

Don't send: the dataset zip, `student_resource/dataset/`, `output/submissions/`, `.venv/`, `coordination/`.

Put these findings in the prompt, because no file records them yet: the 0.896 ceiling, the decoys, the Indic-script names, and France having no labels.

Also: `src/text_norm.py` changed on disk after commit `598433e`. Send the current file if the edit was intended; `git diff` shows the change.

---

## 4

**Prompt:**
> but it is easy too add the folder in the chat like i have added it for u na??

**Reply:**

Yes. Attaching the whole folder works if Astra reads files on demand and respects `.gitignore`. Three things can go wrong:

- **Size.** The zip, the dataset, `data/raw.duckdb`, the submission TSVs and `.venv` add up to several GB. If Astra complains, exclude `student_resource/`, `data/`, `output/submissions/`, `.venv/` and the zip.
- **Results outside git.** `data/er_work/meta.json` is git-ignored, so point Astra at it in the prompt.
- **Findings not in any file.** Paste the four findings from reply 3 into the prompt.

I offered to write those findings into `v1_implementation.md` so the folder alone would be enough.

---

## 5

**Prompt:**
> can u create a readme file of this complete chat and type a prompt in front of my messages and reply in front of ur messages

**Reply:**

Created this file, `CHAT_README.md`, at the repository root. It is not committed or pushed.
