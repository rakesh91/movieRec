"""Regenerate history-group ranking rows for corrected MF B."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.mf_pipeline import _read_locked_split
from Scripts.final_analysis import MostPopularModel, _random_split, _ranking_rows
from Scripts.resolve_mf_b_selection import load_checkpoint
from src.entity_regularized_matrix_factorization import EntityRegularizedMatrixFactorization

OUTPUT = ROOT / "results" / "final_analysis" / "history_popularity_corrected.csv"


def main():
    ratings, movies, _ = load_dataset(ROOT / "data" / "ml-1m")
    splits = {
        "chronological": _read_locked_split(
            ROOT / "results" / "split_assignments.csv", ratings
        ),
        "random": _random_split(ratings),
    }
    rows = []
    for split_name, split in splits.items():
        mf_b = load_checkpoint(
            EntityRegularizedMatrixFactorization,
            ROOT / "results" / "final_analysis"
            / f"{split_name}_MF_B_rmse_grid_min_checkpoint.npz",
            10,
            0.1,
        )
        popular = MostPopularModel().fit(split.loc[split["partition"] == "train"])
        history = split.loc[split["partition"] == "train"].groupby("user_id").size()
        buckets = pd.cut(
            history,
            bins=[0, 20, 50, np.inf],
            labels=["6-20", "21-50", "51+"],
            include_lowest=True,
        )
        for model_name, model in (("MF B", mf_b), ("Popular", popular)):
            frame, _ = _ranking_rows(
                model,
                movies,
                split,
                "validation",
                clip=False,
                score_kind="popularity" if model_name == "Popular" else "mf",
            )
            frame["history_bucket"] = frame["user_id"].map(buckets)
            for bucket, group in frame.groupby("history_bucket", observed=True):
                rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "history_bucket": str(bucket),
                        "users": int(len(group)),
                        "precision_at_10": float(group["precision_at_10"].mean()),
                        "recall_at_10": float(group["recall_at_10"].dropna().mean()),
                        "ndcg_at_10": float(group["ndcg_at_10"].dropna().mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
