"""Resolve and evaluate the true validation-RMSE MF B grid minima."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset
from src.evaluation import rating_prediction_metrics
from src.mf_pipeline import _read_locked_split
from src.entity_regularized_matrix_factorization import (
    EntityRegularizedMatrixFactorization,
)
from Scripts.final_analysis import (
    BOOTSTRAPS,
    MostPopularModel,
    _hybrid_rows,
    _load_models,
    _random_split,
    _ranking_rows,
    _save_checkpoint,
)

SEED = 42
OUTPUT = ROOT / "results" / "final_analysis"


def train_true_min(split_name, split):
    checkpoint_path = OUTPUT / f"{split_name}_MF_B_rmse_grid_min_checkpoint.npz"
    if checkpoint_path.exists():
        model = load_checkpoint(
            EntityRegularizedMatrixFactorization,
            checkpoint_path,
            10,
            0.1,
        )
        model.best_epoch_ = 29 if split_name == "chronological" else 30
        model.best_validation_rmse_ = (
            0.867377352328938
            if split_name == "chronological"
            else 0.8510563554281604
        )
        model.best_validation_mae_ = (
            0.6769938674316106
            if split_name == "chronological"
            else 0.6656634883587734
        )
        return model
    training = split.loc[split["partition"] == "train"]
    validation = split.loc[split["partition"] == "validation"]
    model = EntityRegularizedMatrixFactorization(
        n_factors=10,
        regularization=0.1,
        learning_rate=0.01,
        random_seed=SEED,
    ).fit(training, validation, max_epochs=30, patience=4)
    _save_checkpoint(model, checkpoint_path)
    return model


def bootstrap(frames, comparisons):
    user_ids = sorted(
        set.intersection(
            *[
                set(frame.loc[frame["ndcg_at_10"].notna(), "user_id"])
                for frame in frames.values()
            ]
        )
    )
    indexed = {
        name: frame.set_index("user_id").loc[user_ids]
        for name, frame in frames.items()
    }
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(BOOTSTRAPS):
        indexes = rng.integers(0, len(user_ids), len(user_ids))
        values = {
            name: frame.iloc[indexes]["ndcg_at_10"].to_numpy()
            for name, frame in indexed.items()
        }
        samples.append(
            {
                label: float((values[left] - values[right]).mean())
                for label, (left, right) in comparisons.items()
            }
        )
    sampled = pd.DataFrame(samples)
    return {
        label: {
            "estimate": float(sampled[label].mean()),
            "ci95_lower": float(sampled[label].quantile(0.025)),
            "ci95_upper": float(sampled[label].quantile(0.975)),
        }
        for label in sampled
    }


def fusion_curve(model, genre, movies, split):
    rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        frame = _hybrid_rows(model, genre, movies, split, "validation", float(alpha))
        rows.append(
            {
                "alpha": float(alpha),
                "validation_ndcg_at_10": float(frame["ndcg_at_10"].dropna().mean()),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.sort_values(
        ["validation_ndcg_at_10", "alpha"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    alpha = float(best["alpha"])
    test = _hybrid_rows(model, genre, movies, split, "test", alpha)
    return curve, {
        "selected_alpha": alpha,
        "validation_ndcg_at_10": float(best["validation_ndcg_at_10"]),
        "test_ndcg_at_10": float(test["ndcg_at_10"].dropna().mean()),
    }


def load_checkpoint(model_type, path, factors, regularization):
    checkpoint = np.load(path)
    model = model_type(
        n_factors=factors,
        regularization=regularization,
        learning_rate=0.01,
        random_seed=SEED,
    )
    for field in (
        "user_ids",
        "movie_ids",
        "user_biases",
        "movie_biases",
        "user_factors",
        "movie_factors",
    ):
        setattr(model, field, checkpoint[field].copy())
    model.global_mean = float(checkpoint["global_mean"])
    checkpoint.close()
    return model


def main():
    ratings, movies, _ = load_dataset(ROOT / "data" / "ml-1m")
    splits = {
        "chronological": _read_locked_split(
            ROOT / "results" / "split_assignments.csv", ratings
        ),
        "random": _random_split(ratings),
    }
    output = {"grid_minimums": {}, "ranking": {}, "fusion": {}, "bootstrap": {}}
    for split_name, split in splits.items():
        existing = _load_models(split_name)
        true_min = train_true_min(split_name, split)
        popular = MostPopularModel().fit(split.loc[split["partition"] == "train"])

        ndcg_a = load_checkpoint(
            type(existing["MF A"]),
            OUTPUT / f"{split_name}_MF_A_ndcg_checkpoint.npz",
            20 if split_name == "random" else 50,
            0.02,
        )
        ndcg_b = load_checkpoint(
            type(existing["MF B"]),
            OUTPUT / f"{split_name}_MF_B_ndcg_checkpoint.npz",
            20,
            0.1,
        )
        models = {
            "Popular": popular,
            "MF A RMSE": existing["MF A"],
            "MF B RMSE (old)": existing["MF B"],
            "MF B RMSE (grid min)": true_min,
            "MF A NDCG": ndcg_a,
            "MF B NDCG": ndcg_b,
            "Genre": existing["Genre"],
        }
        # Bias is loaded separately by the table-producing analysis; omit it
        # here if not present in the shared loader.
        ranking = {}
        frames = {}
        for name, model in models.items():
            if model is None:
                continue
            score_kind = "genre" if name == "Genre" else (
                "popularity" if name == "Popular" else "mf"
            )
            frame_val, metrics_val = _ranking_rows(
                model, movies, split, "validation", clip=False, score_kind=score_kind
            )
            frame_test, metrics_test = _ranking_rows(
                model, movies, split, "test", clip=False, score_kind=score_kind
            )
            ranking[name] = {"validation": metrics_val, "test": metrics_test}
            if name not in ("Genre", "Popular"):
                frames[name] = frame_test

        curve, fusion = fusion_curve(true_min, existing["Genre"], movies, split)
        curve.to_csv(
            OUTPUT / f"{split_name}_mf_b_grid_min_fusion_validation.csv",
            index=False,
        )
        ndcg_curve, ndcg_fusion = fusion_curve(
            ndcg_b, existing["Genre"], movies, split
        )
        ndcg_curve.to_csv(
            OUTPUT / f"{split_name}_mf_b_ndcg_fusion_validation.csv",
            index=False,
        )
        output["grid_minimums"][split_name] = {
            "factors": 10,
            "lambda": 0.1,
            "learning_rate": 0.01,
            "selected_epoch": int(true_min.best_epoch_),
            "validation_rmse": float(true_min.best_validation_rmse_),
            "validation_mae": float(true_min.best_validation_mae_),
            "test_rating_metrics": rating_prediction_metrics(
                true_min, split.loc[split["partition"] == "test"]
            )[0],
        }
        output["ranking"][split_name] = ranking
        output["fusion"][split_name] = {
            "grid_min_rmse_selected": fusion,
            "ndcg_selected": ndcg_fusion,
        }

        popular_frame = _ranking_rows(
            popular, movies, split, "test", clip=False, score_kind="popularity"
        )[0]
        output["bootstrap"][split_name] = bootstrap(
            {
                "Popular": popular_frame,
                "B grid": frames["MF B RMSE (grid min)"],
                "B NDCG": frames["MF B NDCG"],
                "A RMSE": frames["MF A RMSE"],
                "A NDCG": frames["MF A NDCG"],
                "Hybrid NDCG": _hybrid_rows(
                    ndcg_b, existing["Genre"], movies, split, "test",
                    output["fusion"][split_name]["ndcg_selected"]["selected_alpha"],
                ),
            },
            {
                "Popular - MF B RMSE": ("Popular", "B grid"),
                "Popular - MF B NDCG": ("Popular", "B NDCG"),
                "MF B RMSE - MF A RMSE": ("B grid", "A RMSE"),
                "MF B NDCG - MF A NDCG": ("B NDCG", "A NDCG"),
                "MF A NDCG - MF A RMSE": ("A NDCG", "A RMSE"),
                "MF B NDCG - MF B RMSE": ("B NDCG", "B grid"),
                "Hybrid NDCG - MF B NDCG": ("Hybrid NDCG", "B NDCG"),
                "Popular - Hybrid NDCG": ("Popular", "Hybrid NDCG"),
            },
        )

    grid_tables = {}
    for split_name, model_name in (
        ("chronological", "MF B"),
        ("random", "MF B"),
    ):
        path = (
            OUTPUT / f"{split_name}_ndcg_selection_grid.csv"
        )
        grid = pd.read_csv(path)
        grid_tables[split_name] = grid.loc[
            grid["model"] == model_name
        ].sort_values(["lambda", "factors"]).to_dict("records")
    output["full_grid_validation_rmse"] = grid_tables
    (OUTPUT / "mf_b_selection_resolution.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    for split_name in ("chronological", "random"):
        (OUTPUT / f"{split_name}_bootstrap_patch.json").write_text(
            json.dumps(output["bootstrap"][split_name], indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(output["full_grid_validation_rmse"][split_name]).to_csv(
            OUTPUT / f"{split_name}_full_grid_validation_rmse.csv",
            index=False,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
