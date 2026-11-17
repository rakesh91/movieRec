# Randomized-split comparison for the paper

This is a supplementary experiment; the original chronological split,
paper draft, checkpoints, and results were not changed.

- Split: random per-user row assignment, 80/10/10, seed 42.
- Split SHA-256: `0446afd053607c6c2a73eb9d37816bb7a500e3a94c4981745d6486cee993bbc9`.
- Unlike the main chronological analysis, this is a random holdout:
  training ratings can be later in time than held-out ratings. Treat it
  as an interpolation/held-out interaction comparison, not a future-use
  simulation.
- Models were selected using training/validation only. Test metrics
  were calculated only after model and hybrid selection; hybrid test
  evaluation was not performed.
- User-level histories and candidate catalogues differ from the
  chronological experiment; scores should be compared within each
  split rather than interpreted as a controlled temporal improvement.

## Validation full-catalogue ranking

| Model | P@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Bias | 0.02424 | 0.02840 | 0.02719 |
| MF A | 0.03733 | 0.04391 | 0.05022 |
| MF B | 0.05283 | 0.06003 | 0.07375 |
| Genre | 0.00929 | 0.01265 | 0.01292 |

## Test full-catalogue ranking (standalone models only)

| Model | P@10 | Recall@10 | NDCG@10 | Rating RMSE |
|---|---:|---:|---:|---:|
| Bias | 0.02533 | 0.02880 | 0.02756 | 0.90765 |
| MF A | 0.03886 | 0.04400 | 0.05096 | 0.85422 |
| MF B | 0.05568 | 0.05920 | 0.07444 | 0.87062 |
| Genre | 0.01036 | 0.01420 | 0.01446 | n/a |

## Validation-only hybrid

Best z-score alpha: 0.8; NDCG@10=0.07891.
Hybrid test metrics were not calculated. Refer to
`hybrid_validation_metrics.csv` for the alpha grid and
`validation_history_buckets.csv` for same-split history results.
