# Results change summary

All values below were produced by `Scripts\final_analysis.py` with seed 42,
the locked chronological split, and the random per-user split generated with
seed 42. The bootstrap used 1,000 user resamples. Test data were not used for
model or alpha selection.

## New controlled comparison

For each matched lambda, factors and epoch were selected by validation RMSE.

| lambda | A factors | A validation RMSE | A validation NDCG@10 | B factors | B validation RMSE | B validation NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 10 | 0.87324 | 0.02722 | 10 | 0.87856 | 0.03589 |
| 0.05 | 50 | 0.86547 | 0.02928 | 10 | 0.87275 | 0.03733 |
| 0.10 | 50 | 0.88064 | 0.01281 | 10 | 0.86738 | 0.04136 |

A wins validation RMSE at 0.02 and 0.05; B wins at 0.10. B wins validation
NDCG at all three lambdas.

## Popularity baseline

Full-catalogue P@10 / Recall@10 / NDCG@10:

| split | validation | test |
|---|---|---|
| chronological | 0.03355 / 0.04187 / 0.04824 | 0.02975 / 0.04026 / 0.04362 |
| random per-user | 0.05111 / 0.07174 / 0.08174 | 0.05391 / 0.06690 / 0.08235 |

Bias baseline full-catalogue values are 0.01563 / 0.01701 / 0.01683
(validation) and 0.01417 / 0.01628 / 0.01587 (test) chronologically; the
random-split values are 0.02376 / 0.02785 / 0.02595 and
0.02493 / 0.02839 / 0.02653.

## Unclipped ranking and ties

Unclipped corrected MF B test NDCG@10 is 0.03955 chronological and 0.06812
random, versus clipped values 0.03959 and 0.06793. NDCG-selected MF B reaches
0.04354 chronological and 0.07635 random. On clipped corrected MF B test
rankings, candidate scores tied at 5.0 / top-10 slots tied at 5.0 were
37,094 / 13,016 chronologically and 32,640 / 13,237 randomly.

## Validation-selected hybrid test results

The chronological alpha is 1.0; the random per-user alpha is 0.8. Corrected
full-grid RMSE-selected MF B hybrid test NDCG@10 is 0.03955 chronologically
and 0.07097 randomly. NDCG-selected MF B gives 0.04354 and 0.07635, and its
random hybrid gives 0.08120.

Bootstrap 95% CIs for test NDCG@10:

| split | MF A | MF B | hybrid |
|---|---|---|---|
| random per-user | 0.04841--0.05345 | 0.07265--0.07908 | 0.07717--0.08386 |

Paired bootstrap CIs for MF B minus MF A and hybrid minus MF B:

| split | MF B - MF A | hybrid - MF B |
|---|---|---|
| chronological | 0.00910--0.01339 (RMSE), 0.00807--0.01285 (NDCG) | hybrid equals MF B |
| random per-user | 0.01503--0.01955 (RMSE), 0.01973--0.02366 (NDCG) | 0.00302--0.00673 |

## Final patch-pass outputs

## MF B selection resolution

The earlier Table 1/2 MF B checkpoint was produced by the legacy sensitivity
runner with 50 factors and learning-rate selection. It was not the minimum of
the later 3x3 factors/lambda grid. The corrected full-grid validation-RMSE
minima are 10 factors, lambda 0.10, epoch 29 (chronological; validation/test
RMSE 0.86738/0.88638) and epoch 30 (random; validation/test RMSE
0.85106/0.85503). Their test NDCG@10 values are 0.03955 and 0.06812.
The complete grid and dependent metrics are saved in
`results/final_analysis/mf_b_selection_resolution.json`.

The paper changed the chronological MF B RMSE-selected test NDCG from 0.04299
to 0.03955 and the random value from 0.07580 to 0.06812. Its validation/test
RMSE changed from 0.87895/0.89702 to 0.86738/0.88638 chronologically and is
0.85106/0.85503 randomly. The corrected RMSE-selected random hybrid test
NDCG is 0.07097; the NDCG-selected hybrid remains 0.08120.

NDCG-selected minus RMSE-selected test NDCG@10 paired 95% CIs were:

| split | MF A | MF B |
|---|---|---|
| chronological | 0.00328--0.00626 | 0.00185--0.00601 |
| random per-user | 0.00157--0.00621 | 0.00560--0.01102 |

NDCG selection therefore has a positive interval for both MF variants.
NDCG-selected MF B fusion selected alpha 1.0 chronologically (test NDCG
0.04354) and 0.8 randomly (test NDCG 0.08120). The corrected full-grid
RMSE-selected MF B fusion gives test NDCG 0.03955 chronologically and 0.07097
randomly. Its clipped/unclipped tie audit is in
`results/final_analysis/corrected_tie_audit.csv`; clipping changes NDCG by
0.00003 chronologically and -0.00019 randomly.
The history comparison is in
`results/final_analysis/history_popularity_corrected.csv`; popularity is
higher than corrected MF B in all random validation history groups, with the
smallest NDCG gap in the 51+ group.

The RMSE-selected checkpoint reconciliation is in
`results/final_analysis/mf_b_selection_resolution.json`: chronological MF B
validation/test RMSE is 0.86738/0.88638 and random is 0.85106/0.85503.
The matched-lambda table's 0.86738 is the selected full-grid result at fixed
lambda 0.10.

Corrected paired intervals for the NDCG-selected hybrid are 0.00000--0.00000
versus its chronological MF B base and 0.00302--0.00673 versus its random MF
B base. Popular minus the random NDCG-selected hybrid is -0.00303--0.00506,
so that comparison includes zero.

## Remaining verification

The paper PDF still needs a local LaTeX compilation check after the text
edits. The AI-use disclosure intentionally contains author placeholders rather
than guessed tool names.
