# Submission log (Akash's runs)

All validation numbers use the same 100,000 train S1 entities (seed 42, perm[200000:300000]),
macro F0.5 with singletons and blocking misses counted.

| Version | Change | Blocking recall (val) | Ceiling U | Val F0.5 | Threshold | Leaderboard | Status |
|---|---|---|---|---|---|---|---|
| v1 | baseline, 7 keys, TOP_K 30 | 0.7913 | not measured | 0.8652 | 0.625 | fill in | validator PASS, uploaded |
| v2 | + 3 address-only keys (H, I, J), caps ~5x, TOP_K 60, 2000 rounds | 0.8817 (US 0.926, India 0.816) | 0.9483 | 0.9119 | 0.675 | fill in | test prediction running |
| v3 | transliteration folds, skeleton address words, keys K and L, v2 caps | not run yet | | | | | ship only if val > 0.9119 |

Rejected experiment: v3 with wider caps (A/C 2000, G/J 1500, D/E/H 1000). On the synthetic test set it
lowered recall 0.926 -> 0.901 and F0.5 0.960 -> 0.952 because weak candidates filled the top-K.
