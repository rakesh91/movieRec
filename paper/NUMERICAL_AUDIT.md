# Numerical audit

All values below were checked against the saved outputs and implementation. No
model was retrained and no saved result or split assignment was changed.

## Dataset and split

| Quantity | Paper value | Verification source |
|---|---:|---|
| MovieLens 1M users, catalogue movies, ratings | 6,040; 3,883; 1,000,209 | `results/data_audit.json` |
| Rating range; missing ratings fields; duplicate user--movie pairs | 1--5; 0; 0 | `results/data_audit.json` |
| Matrix sparsity; median user history; 4- or 5-star rating share | 95.74%; 96; 57.5% | `results/data_audit.json`, `results/rating_distribution.csv` |
| Movie exposure statistic | Median 123.5, range 1--3,428, among 3,706 movies with ratings; 177 of the 3,883 catalogue movies have no ratings | `results/ratings_per_movie.csv` (3,706 rows), `results/data_audit.json` |
| Per-user chronological train/validation/test rows and users | 799,076/100,613/100,520; 6,040/6,039/6,040 | `results/metrics.json` |
| Per-user chronological locked split SHA-256 | `f6f5f45d8d4bb8cdebb5e1559537b4e8d32e1f455c650e0f7ad989e693093538` | `results/regularization_sensitivity/B_entity_level/config.json`, `results/hybrid_validation_sweep/reproducibility_checks.json` |
| Random per-user train/validation/test rows and users | 797,758/97,383/105,068; 6,040/6,040/6,040 | `results/randomized_split_seed42/experiment_summary.json` |
| Random per-user split SHA-256 (seed 42) | `0446afd053607c6c2a73eb9d37816bb7a500e3a94c4981745d6486cee993bbc9` | `results/randomized_split_seed42/experiment_summary.json`, `integrity_checks.json` |
| Training mean, per-user chronological / random per-user | 3.6170 / 3.5805 | `results/metrics.json`, `results/randomized_split_seed42/experiment_summary.json` |

The reported 123.5 is a genuine median, but the original summary computed it
only over movies with at least one rating. It is not a statistic across all
3,883 catalogue movies; unrated catalogue titles have zero exposure.

## Training selections and standalone ranking

Chronological MF configurations are recorded in the A/B `config.json` files
under `results/regularization_sensitivity/`; random-per-user configurations
and validation selections are in `results/randomized_split_seed42/experiment_summary.json`.
Ranking metrics are macro averages on the full eligible catalogue. Each cell
below is Precision@10 / Recall@10 / NDCG@10.

| Model | Per-user chronological test | Random per-user test | Verification source |
|---|---:|---:|---|
| Bias | 0.01417 / 0.01628 / 0.01587 | 0.02493 / 0.02839 / 0.02653 | `results/final_analysis/bias_ranking.json` |
| MF A | 0.02075 / 0.02674 / 0.02833 | 0.03904 / 0.04430 / 0.05089 | `results/final_analysis/mf_b_selection_resolution.json` |
| MF B | 0.02775 / 0.03446 / 0.03955 | 0.05118 / 0.05404 / 0.06812 | `results/final_analysis/mf_b_selection_resolution.json` |
| Genre | 0.00586 / 0.00988 / 0.00852 | 0.01036 / 0.01420 / 0.01446 | `results/genre_based/test_metrics.json`; `results/randomized_split_seed42/test_metrics.csv` |

| Fit or validation quantity | Value | Verification source |
|---|---:|---|
| Chronological MF A / MF B selected factors, regularization, epoch | 50, 0.05, 22 / 10, 0.10, 29 | `results/final_analysis/mf_b_selection_resolution.json`; `.../config.json` |
| Chronological MF A / MF B validation RMSE | 0.86547 / 0.86738 | Same saved configuration/grid outputs |
| Random-per-user MF A / MF B selected factors, regularization, epoch | 50, 0.05, 24 / 10, 0.10, 30 | `results/final_analysis/mf_b_selection_resolution.json` |
| Random-per-user MF A / MF B validation RMSE | 0.84933 / 0.85106 | Same resolution output |
| Chronological validation ranking (P@10 / Recall@10 / NDCG@10), MF A | 0.02191 / 0.02647 / 0.02940 | `results/final_analysis/mf_b_selection_resolution.json` |
| Chronological validation ranking (P@10 / Recall@10 / NDCG@10), MF B | 0.02986 / 0.03546 / 0.04187 | Same resolution output |
| Chronological validation ranking (P@10 / Recall@10 / NDCG@10), Genre | 0.00758 / 0.01213 / 0.01145 | `results/genre_based/validation_metrics.json` |
| Bias validation MAE; chronological MF B test RMSE | 0.7179; 0.88638 | `results/validation_tuning.csv`; `results/final_analysis/mf_b_selection_resolution.json` |

Bias full-catalogue validation ranking was not saved. Chronological validation
ranking covers 6,039 users with validation interactions; one of the dataset's
6,040 users has none. Of those 6,039, 263 have no relevant validation movie
(and are retained in Precision@10 but excluded from Recall@10/NDCG@10).
Chronological test has 272 zero-relevant users of 6,040. Random validation
and test have 193 and 76 zero-relevant users, respectively, out of 6,040.
Sources: `results/metrics.json`, `results/genre_based/validation_metrics.json`,
`results/randomized_split_seed42/experiment_summary.json`.

## Split sensitivity, subgroup, and fusion

| Quantity | Per-user chronological | Random per-user | Verification source |
|---|---:|---:|---|
| Test NDCG@10, Bias / MF A / MF B / Genre | 0.01664 / 0.02887 / 0.04318 / 0.00852 | 0.02756 / 0.05096 / 0.07444 / 0.01446 | Saved test metric files listed above |
| Random-vs-chronological test NDCG relative differences (Bias / MF A / MF B / Genre) | -- | +65.6% / +76.5% / +72.4% / +69.7% | Calculated from the unrounded saved test NDCGs |
| Mean train history; mean eligible candidates/user | 132.3; 3,750.7 | 132.1; 3,750.9 | `results/split_assignments.csv`; `results/randomized_split_seed42/split_assignments.csv` |
| Users with at least one later training timestamp than validation / test | 0 / 0 | 6,027 / 6,038 | Same saved split-assignment CSVs |
| Mean relevant held-out movies/user, validation / test | 8.86 / 8.57 | 9.30 / 10.03 | Same saved split-assignment CSVs; distinct held-out movie IDs rated at least 4 |
| Random validation history groups: users; MF B / Popular / Genre NDCG@10 (6--20, 21--50, 51+) | -- | 553: 0.02342 / 0.06640 / 0.01194; 1,674: 0.03888 / 0.05925 / 0.01080; 3,813: 0.08384 / 0.09294 / 0.01392 | `results/final_analysis/history_popularity_corrected.csv` |
| Chronological Genre validation NDCG@10 (6--20, 21--50, 51+) | 0.00937 / 0.00979 / 0.01240 | -- | `results/genre_based/history_bucket_metrics.csv` |
| Chronological MF B/Genre mean shared top-10, Jaccard, candidate-wise Spearman | 0.053; 0.00281; 0.12954 | -- | `results/final_analysis/ranking_overlap_summary.json` |

Random per-user validation z-score fusion NDCG@10 by MF B weight
alpha=0.0, 0.1, ..., 1.0 is
0.01292, 0.03887, 0.04805, 0.05573, 0.06253, 0.06788, 0.07223, 0.07728,
0.07999, 0.07937, 0.07662. Source:
`results/final_analysis/random_mf_b_ndcg_fusion_validation.csv`.
The best is alpha=0.8.

Per-user chronological validation z-score fusion NDCG@10 over the same
alpha grid is
0.01145, 0.02606, 0.03033, 0.03375, 0.03686, 0.03946, 0.04158, 0.04389,
0.04556, 0.04640, 0.04652. Source:
`results/final_analysis/chronological_mf_b_ndcg_fusion_validation.csv`. Both
fusion studies used validation only; the randomized integrity report
confirms that each frozen hybrid is evaluated once on test after validation-only
alpha selection.

The metric implementation in `src/evaluation.py` uses fixed-denominator
Precision@10, Recall@10 over distinct relevant held-out movies, and binary
gain DCG/IDCG with discounts `log2(rank + 1)`; zero-relevant users are omitted
from Recall/NDCG. Fusion normalization and rank-score mapping are in
`Scripts/run_hybrid_validation_sweep.py`. Equations in the paper match
these implementations.

## Unresolved evaluation gaps

No numerical discrepancy remains in the values reported in the paper.
Repeated temporal splits, paired confidence intervals, batch-size sensitivity,
hybrid test metrics, global-timepoint evaluation, online outcomes, and
user-centred outcomes were not produced by the project. They remain
limitations, not results to infer or fill in.
