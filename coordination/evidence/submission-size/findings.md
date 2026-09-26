# Submission size: source review and safe options

Session: astra-20260927-0450. Review date: 2026-09-27 IST.
Scope: research only; no v5, cloud, queue, candidate policy or portal changes.

**Finding:** No numerical upload limit for the submission ZIP or
`candidate_pairs.tsv` is stated in the supplied problem PDF, guidelines, README
or candidate-generation notice. The portal's actual limit and ZIP64 support remain
unverified. An absent documented limit is not evidence of unlimited acceptance.

## What the supplied sources say

| Source | Relevant content |
| --- | --- |
| Problem statement, pp. 3-5 and 6-7 | Live leaderboard takes matching_results.tsv. Final package is a single ZIP containing both TSVs, runnable code and the filled template. Candidates equal the exact final matching-model input after blocking/filtering. No numerical file-size limit. |
| student_resource/README.md, Output Format / Final Submission Package / Submission Requirements | Same two-file contract and one ZIP. No numerical size or upload limit. |
| Guidelines, pp. 1-2 | 1-2-page approach note, code, version history, five submissions/day and participant-login restrictions. No numerical upload limit. |
| final_submission_candidate_generation.pdf, p. 1 | Candidate generation affects final ranking; smaller candidate sets are favored alongside match quality. No numerical cap or stated size-versus-score tradeoff formula. |
| AWS instructions PDF, p. 5 | Mentions 5 GB of S3 free-tier storage. This is an AWS allowance, **not** a submission-file cap. |

The problem's pp. 3-5 and the notice were also visually checked. The complete
problem/guideline/notice text and README were read. A size-term scan of all PDFs
found only the AWS storage reference above.

## Implication and proposed option

With 1,732,544 queries, 130-210 candidates means about 225.2-363.8 million pairs.
If IDs average 12 bytes, commas plus the query ID/tab/newline give approximately
`Q * (14 + 13*N)` bytes: about 2.95-4.75 GB (decimal) before ZIP compression.
This is an estimate, not a measurement of a final v5 file.

1. Preserve the actual final model's candidate set and first measure the compressed
   **single ZIP**, its uncompressed members and transfer time. Use normal lossless
   ZIP compression while retaining `output/candidate_pairs.tsv` as a TSV member;
   do not substitute a gzip or Parquet member. Enable ZIP64 when a member/archive
   requires it, and have the portal operator check support and the actual limit.
   Python documents this option in its [ZIP library reference](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile).
   ZIP compression has no guaranteed ratio for these outputs.
2. If the measured package is too large, evaluate a **real two-stage cascade**:
   broad label-free retrieval -> cheap first-stage ranking/pruning -> final
   LightGBM matching on survivors. Choose N or an adaptive policy from full-catalog
   tune data, including singletons/no-candidate queries; measure survivor oracle U,
   link recall, exact end-to-end macro F0.5 and country/truth-count slices.
   A nominal N=40 would be about 0.93 GB of TSV under the same ID-length assumption;
   N=60 about 1.38 GB. Neither N is a validated recommendation.
3. Declare and export exactly the survivors **before** the final matcher scores
   them. Retrain/revalidate affected context features, negatives and threshold.
   Pin first-stage policy/model, survivor pair manifest and final scored-input
   manifest; the latter two must describe the same pairs. Include both stages in
   runnable code and documentation. Under the README's last-blocking-stage wording,
   this is a defensible compliant design, subject to organizer interpretation if
   a preliminary learned model is itself considered part of the final matcher.

Do not trim an already scored candidate file merely for packaging, export only
accepted matches as candidates, omit S1 rows, or claim compressed byte savings
improve candidate-count ranking. A composite matcher that makes final decisions
over a broader union must disclose that whole union. No cascade was implemented
or promoted by this task.

Next release action: the single portal operator should establish the real upload
limit/ZIP64 support without consuming a submission, then compare measured ZIP
bytes and measured quality before requesting any model-policy change.

## Source fingerprints (SHA-256)

- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`:
  `084168f9af69d3f8e7f1bf9d86f917bd2c736bc30278469cfb1916a217350dc4`
- `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`:
  `bdb8db7f20a495cf614699009750be3ab5bca8226e354e487f46af96aef84921`
- `4cd78da9-38ed-4832-a95e-e074a69251c7.pdf`:
  `bc0adf38188ce54a6c2e947aa5190d8616a524d40d34c3558978d987942f1e38`
- `output/hackathon/sources/final_submission_candidate_generation.pdf`:
  `a3377227db7825945accf122274c018102c181fdd99f508f1495c6834511dde2`
- `student_resource/README.md`:
  `be67cd7d9dbc48a58f93e1f61402360655debac64214115bf9d27d3301939285`
