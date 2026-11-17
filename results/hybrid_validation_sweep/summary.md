# Validation-only hybrid sweep

## Experiment

This is a validation-only diagnostic using the saved MF entity-level model B and saved genre checkpoint. Neither model was trained. The locked chronological split was not changed. The primary method separately standardizes each model's scores over each user's full eligible catalogue:

```text
mf_z(u,m)    = (mf_score(u,m) - mean_u(mf_score)) / max(std_u(mf_score), 1e-8)
genre_z(u,m) = (genre_score(u,m) - mean_u(genre_score)) / max(std_u(genre_score), 1e-8)
hybrid(u,m)  = alpha * mf_z(u,m) + (1-alpha) * genre_z(u,m)
```

The population standard-deviation floor is `1e-8`; it is used as a denominator floor, not as a zero-score fallback. This keeps scores finite and preserves each model's ordering at alpha endpoints. Users below the floor: MF 0, genre 0. All scores were checked finite.

The secondary rank-fusion comparison follows the prior audit's explicit definition: rank each model by descending score with ascending movie-ID ties, map zero-based position `j` to `1-j/(n-1)`, then combine with the same alpha weights. There are 6,039 validation users. Each has the same candidate set for both models: all catalogue movies except that user's training-rated movies. Candidate counts (min/median/mean/max): 2031/3807/3750.7/3878. Candidate construction and scores did not use validation/test rating values.

After rankings were fixed, validation ratings were used only as labels (relevant iff rating >= 4). Metrics use binary-gain NDCG@10, fixed Precision@10 denominator 10, and Recall@10 over distinct relevant movies. 263 users with no relevant validation movie were excluded from Recall/NDCG but included in Precision. No test rating values or metrics were accessed.

## Primary z-score sweep

| Alpha | Precision@10 | Recall@10 | NDCG@10 | Evaluated users | Zero-relevant excluded |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.007584 | 0.012127 | 0.011453 | 6039 | 263 |
| 0.1 | 0.015665 | 0.022978 | 0.024679 | 6039 | 263 |
| 0.2 | 0.018298 | 0.027000 | 0.028522 | 6039 | 263 |
| 0.3 | 0.020417 | 0.029546 | 0.031448 | 6039 | 263 |
| 0.4 | 0.022967 | 0.033089 | 0.035117 | 6039 | 263 |
| 0.5 | 0.025302 | 0.035366 | 0.038299 | 6039 | 263 |
| 0.6 | 0.027637 | 0.036805 | 0.041093 | 6039 | 263 |
| 0.7 | 0.030005 | 0.037408 | 0.042895 | 6039 | 263 |
| 0.8 | 0.031959 | 0.037647 | 0.044507 | 6039 | 263 |
| 0.9 | 0.032737 | 0.037039 | 0.044956 | 6039 | 263 |
| 1.0 | 0.034261 | 0.037678 | 0.045670 | 6039 | 263 |

Best z-score alpha by validation NDCG@10: **1.0** (NDCG@10 0.045670).

Standalone MF B NDCG@10: 0.045670. Standalone genre NDCG@10: 0.011453. Any z-score hybrid alpha strictly beats MF B: **False**. Any secondary rank-fusion interior alpha beats MF B: **False**.

Alpha 0 and alpha 1 reproduced standalone genre and MF B recommendation lists exactly for both methods; metrics match saved standalone validation reports. All 22 score/ranking configurations evaluated identical user candidate counts.

## Secondary rank-fusion sweep

See `validation_metrics.csv` and `per_alpha_results.json` for each alpha. This is diagnostic only; alpha selection remains on the specified primary z-score sweep.

| Alpha | Precision@10 | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0.0 | 0.007584 | 0.012127 | 0.011453 |
| 0.1 | 0.014853 | 0.023308 | 0.023026 |
| 0.2 | 0.015698 | 0.024911 | 0.024388 |
| 0.3 | 0.016162 | 0.025713 | 0.025007 |
| 0.4 | 0.016311 | 0.025076 | 0.025143 |
| 0.5 | 0.016940 | 0.025769 | 0.025728 |
| 0.6 | 0.017553 | 0.025825 | 0.026601 |
| 0.7 | 0.018099 | 0.025831 | 0.027033 |
| 0.8 | 0.019258 | 0.026666 | 0.028428 |
| 0.9 | 0.021129 | 0.027629 | 0.030592 |
| 1.0 | 0.034261 | 0.037678 | 0.045670 |

## Interpretation and limitations

The best alpha is validation-selected only and is not final until the validation experiment is audited. The validation sweep does not establish generalization; do not inspect test performance until the hybrid method/configuration is approved. Per-user z-scores discard absolute score levels and may be sensitive to score variance, although no user fell below the chosen floor in this run. Rank fusion discards score-gap information and gives positions to ties using the deterministic movie-ID rule. Small alpha steps are diagnostic rather than extensive tuning.

## Integrity

Split SHA-256 stayed `f6f5f45d8d4bb8cdebb5e1559537b4e8d32e1f455c650e0f7ad989e693093538`. MF B and genre checkpoint hashes were unchanged. Existing results remained unchanged. No model was trained, and no test rating values or metrics were read.
