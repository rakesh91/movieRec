"""Audit clipped versus unclipped corrected MF B rankings."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.mf_pipeline import _read_locked_split
from Scripts.final_analysis import _random_split, _ranking_rows
from Scripts.resolve_mf_b_selection import load_checkpoint
from src.entity_regularized_matrix_factorization import EntityRegularizedMatrixFactorization


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
        model = load_checkpoint(
            EntityRegularizedMatrixFactorization,
            ROOT / "results" / "final_analysis"
            / f"{split_name}_MF_B_rmse_grid_min_checkpoint.npz",
            10,
            0.1,
        )
        for clipped in (True, False):
            _, metrics = _ranking_rows(
                model, movies, split, "test", clip=clipped
            )
            rows.append(
                {
                    "split": split_name,
                    "clipped": clipped,
                    "ndcg_at_10": metrics["ndcg_at_10"],
                    "candidates_tied_at_5": metrics["candidates_tied_at_5"],
                    "top10_slots_tied_at_5": metrics["top10_slots_tied_at_5"],
                }
            )
    output = ROOT / "results" / "final_analysis" / "corrected_tie_audit.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
