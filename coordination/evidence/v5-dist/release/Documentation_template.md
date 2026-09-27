# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** AML  
**Team Members:** Aditya Shrivastav, Aadish Sarin, Akash PES, Abhigyan Dutta  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We retrieve candidates with six sparse TF-IDF channels over names, addresses and a
combined name+address view, searching the full target catalog of each country. A
label-free stage-1 filter keeps the top 30 candidates per query, and a LightGBM
classifier scores those survivors with 92 pair features. One owner per target and a
threshold tuned for macro F0.5 give the final matches. On a held-out tune set of
100,000 train queries this scores **0.9605** macro F0.5 with **30.0**
candidates per query.

---

## 2. Methodology

### 2.1 Problem Analysis

- Source 1 is clean. The same business shows up several times in S2 and S3 (3.5 true
  matches per S1 on average, up to 11), and 5.6% of S1 records have no match at all.
- Name noise: typos, reordered words, legal-suffix changes (Pvt/Private, Ltd/Limited),
  bracketed or dropped tokens, repeated tokens, digits inside words (`G0mez`, `A1l`),
  alias prefixes (`Rizasol D.B.A. Flint Battery`, `X aka Y`), web and social forms
  (`flintbattery.com`, `@ingaborgmoreno`) and names in Telugu or Devanagari script.
- Address noise: reordered components, Street/St and Road/Rd, full state names against
  codes, PMB, PO Box and unit insertions, dropped house numbers, `N/A` fields, city typos
  and missing addresses (4.4% of true targets).
- Names alone are weak evidence. Many unrelated businesses share names, so a name-only
  top-40 search found just 63% (India) and 74% (US) of the true links. Addresses were
  far more discriminative, and name and address together were best.
- Test data adds France, which has no training labels. Country is treated as an open
  set: every country gets its own catalog built the same way.

### 2.2 Solution Strategy

**Approach Type:** Multi-channel sparse blocking, then a label-free stage-1 filter,
then a gradient-boosted tree classifier, then ownership and a threshold.

**Core Innovation:** Separate retrieval quotas for name, address and combined
name+address channels, so a changed name can still be found through its address and
common names cannot crowd out the right record. Anchor-support features let a
candidate borrow evidence from the query's other strong candidates, because the
variants of one business agree with each other. The measured candidate oracle (the
best score any matcher could reach with a candidate set) guided every retrieval
decision.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Binary hashed n-grams (2^20 dimensions) with IDF from the
  searchable catalog. Features appearing in more than 3% of records (character) or 5%
  (word) are dropped. Each channel keeps its own top-k per source (S2, S3) within the
  query's country, found by sparse cosine search over 250,000-target blocks:

  | Channel | View | n-grams | k per source |
  | --- | --- | --- | ---: |
  | name_char | canonical name | character 3-4 | 20 |
  | address_char | canonical address | character 3-4 | 40 |
  | name_fold | phonetic name skeleton | word 1-2 | 10 |
  | address_fold | phonetic address tokens | word 1-2 | 10 |
  | combo | name + address | character 3-4 | 40 |
  | combo_word | name + address | word 1-2 | 60 |

  The union of these channels averages about 256 records per query. Pair features are
  computed for all of them. Stage 1 then keeps the top **30** per query by the sum of the
  two combined name+address cosine scores (`combo` + `combo_word`), a label-free
  retrieval score. Only these survivors go to the matching model. They are exactly the rows of `candidate_pairs.tsv`.
- **Candidate pairs generated:** 51,976,320 for the 1,732,544 test S1 records, an
  average of 30.0 per S1 (at most 30).
- **How we made sure true matches were not lost:** Retrieval was measured with the
  candidate oracle U (predict exactly the retrieved true links; singletons count as 1).
  On 4,000 held-out queries per country against the full training catalog:

  | Policy | India U | US U |
  | --- | ---: | ---: |
  | Four name/address channels at k=20 | 0.9705 | 0.9926 |
  | + combined name+address character channel | 0.9789 | 0.9981 |
  | + combined word channel and wider address quota | 0.9854 | 0.9986 |

  Across the full 100,000-query tune set the retrieved union reaches U = 0.9952 and
  link recall 0.987. After the stage-1 filter at N = 30, U = 0.9841 and link recall
  0.958.

---

## 4. Matching Model

**Features used (92):**
- Name features: ratio, partial ratio, token-sort and token-set similarity on the
  normalized, canonical, consonant-skeleton and concatenated names; Jaro-Winkler;
  exact agreement; lengths; token precision and recall; extra and missing tokens;
  phonetic-fold similarities.
- Address features: ratio, partial and token similarities on normalized and token
  views; postcode and house-number agreement; Jaccard, precision and recall of number
  sets; lengths; missing-address flags. Address similarities are set to -1 when an
  address is empty.
- Other: target source; the cosine score of every retrieval channel; the candidate's
  rank and its gap to the best candidate within the query for key similarities; number
  of candidates; anchor support, meaning the similarity to the query's three strongest
  other candidates, weighted by their strength.

**Model type:** LightGBM binary classifier (MIT license, no pretrained weights), 2,000
trees with 127 leaves each, learning rate 0.08, trained on 400,000 train queries (39.2M
pairs; easy negatives subsampled to 30% and reweighted). Early stopping was monitored on
10,000 separate queries.

**Threshold selection method:** Grid search on the 100,000-query tune set to maximize
exact macro F0.5, including singletons and queries with no candidates. Each S2/S3
record belongs to at most one S1: the highest probability wins, and the pair is kept
only if its probability is at least the threshold (0.72). An S1 can get zero,
one or many matches.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9605** on the 100,000-query tune set with the
  top-30 stage-1 filter. Without the filter the same model scores 0.9679 at about 256
  candidates per query (US 0.9741, India 0.9587). The locked 10% audit set was never opened. The earlier
  baseline (four channels, no combined channels or anchors) scored 0.9590.
- **Where the loss comes from:** With the full union, 0.5 points are lost to matches
  that were never retrieved and 2.7 points to wrong decisions on retrieved candidates.
  India is harder than the US (retrieval ceiling 0.990 against 0.999).
- **Common false positives (wrong merges):** Different businesses with near-identical
  names at the same or neighbouring address (branches, related companies), and generic
  names that also share a street. With a single true match, one false positive costs 44%
  of that query's score, which is why the tuned threshold is high.
- **Common false negatives (missed matches):** Variants with both fields degraded,
  for example a truncated name with no house number, a transliterated name with a
  partial address, or no address at all.

---

## 6. Conclusion

Combined name+address retrieval was the largest single gain: it raised the India
candidate ceiling from 0.970 to 0.990 and the overall tune F0.5 from 0.959 to 0.968. A
label-free stage-1 filter then cut the candidate set to at most 30 records per S1 for a
tune-set change of -0.0074 F0.5. Measuring the candidate ceiling first, before tuning
the model, showed where to spend the effort.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the full pipeline. `README.md` has the exact
commands and `requirements.txt` the pinned versions.

- `src/v5_dist.py`: task queue, per-country catalogs, retrieval, pair features, training,
  stage-1 filter, scoring and final assembly with the official validator.
  Entry points: `python -m src.v5_dist plan ...` and `python -m src.v5_dist worker ...`.
- `src/v5_cascade.py`: stage-1 tune evaluation (`eval`) and test scoring set-up (`score`).
- `src/text_norm_v5.py`, `src/text_norm.py`: normalization. `src/features.py`,
  `src/v4_features.py`: pair features. `src/v4_retrieval.py`, `src/v5_vec.py`: views and
  vectorization. `src/metric.py`: macro F0.5 and ownership rule. `src/strict_validate.py`:
  streaming output checks.
- `deploy/`: SageMaker notebook-instance bootstrap and the channel configuration used.
- `tests/`: end-to-end tests on synthetic data, including the cascade.

### B. Additional Results

| Tune set (100,000 queries) | Baseline v5 | v5.1, full union | v5.1 + stage-1 N=30 (submitted) |
| --- | ---: | ---: | ---: |
| Macro F0.5 | 0.9590 | 0.9679 | 0.9605 |
| Candidate oracle U | 0.9842 | 0.9952 | 0.9841 |
| Candidates per query | 128 | 256 | 30 |
| Threshold | 0.735 | 0.785 | 0.72 |

Stage-1 cut-off on the tune set (the same final model scores the survivors; the
threshold is re-selected for each cut-off):

| Stage-1 rank | Top N | Macro F0.5 | Oracle U |
| --- | ---: | ---: | ---: |
| combo + combo_word cosine | 20 | 0.9592 | 0.9822 |
| combo + combo_word cosine | 30 (submitted) | 0.9605 | 0.9841 |
| combo + combo_word cosine | 40 | 0.9611 | 0.9851 |
| combo + combo_word cosine | 60 | 0.9620 | 0.9862 |
| combo + combo_word cosine | 80 | 0.9625 | 0.9871 |
| all six channel cosines | 30 | 0.9576 | 0.9804 |
| name/address token-set heuristic (`h_rank`) | 30 | 0.9442 | 0.9678 |

Beyond about 30 candidates each extra candidate adds little. The heuristic rank does
worse because token-set scores reach 100 for any subset, so many candidates tie.

Test output: 1,624,730 of 1,732,544 S1 records have at least one match
(5,522,613 matched pairs). Model SHA-256:
`eec93e350d320b46453d69f47010f07883c0b3e5eb6dd47d2dda0153f46cd982`.

Licenses: LightGBM (MIT) is the model. The other dependencies are MIT, BSD or
Apache-2.0, except Unidecode (GPL-2.0-or-later), which is used only to transliterate
text during normalization. No external data, lookups or hosted services were used.
