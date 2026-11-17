"""Produce the final-pass analyses without retraining models."""

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
from Scripts.final_analysis import (
    BOOTSTRAPS,
    MostPopularModel,
    _hybrid_rows,
    _load_models,
    _random_split,
    _ranking_rows,
)

SEED = 42
OUTPUT = ROOT / "results" / "final_analysis"


def paired_ci(frames, comparisons):
    common = set.intersection(
        *[
            set(frame.loc[frame["ndcg_at_10"].notna(), "user_id"])
            for frame in frames.values()
        ]
    )
    user_ids = np.asarray(sorted(common), dtype=np.int64)
    indexed = {
        name: frame.set_index("user_id").loc[user_ids]
        for name, frame in frames.items()
    }
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(BOOTSTRAPS):
        indices = rng.integers(0, len(user_ids), len(user_ids))
        values = {
            name: frame.iloc[indices]["ndcg_at_10"].to_numpy()
            for name, frame in indexed.items()
        }
        samples.append(
            {
                label: float((values[left] - values[right]).mean())
                for label, (left, right) in comparisons.items()
            }
        )
    sample_frame = pd.DataFrame(samples)
    return {
        label: {
            "estimate": float(sample_frame[label].mean()),
            "ci95_lower": float(sample_frame[label].quantile(0.025)),
            "ci95_upper": float(sample_frame[label].quantile(0.975)),
        }
        for label in sample_frame
    }


def load_ndcg_models(split_name, rmse_models):
    configs = pd.read_csv(
        OUTPUT / f"{split_name}_ndcg_selected_configs.csv"
    ).set_index("model")
    models = {}
    for name in ("MF A", "MF B"):
        checkpoint = np.load(
            OUTPUT / f"{split_name}_{name.replace(' ', '_')}_ndcg_checkpoint.npz"
        )
        model = rmse_models[name].__class__(
            n_factors=int(configs.loc[name, "factors"]),
            regularization=float(configs.loc[name, "lambda"]),
            learning_rate=0.01,
            random_seed=SEED,
            batch_size=8192,
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
        models[name] = model
        checkpoint.close()
    return models, configs


def select_alpha(model, genre, movies, split):
    rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        frame = _hybrid_rows(model, genre, movies, split, "validation", float(alpha))
        rows.append(
            {
                "alpha": float(alpha),
                "validation_ndcg_at_10": float(frame["ndcg_at_10"].dropna().mean()),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["validation_ndcg_at_10", "alpha"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return float(best["alpha"]), table


def history_rows(model, popular, movies, split, split_name):
    history = split.loc[split["partition"] == "train"].groupby("user_id").size()
    buckets = pd.cut(
        history,
        bins=[0, 20, 50, np.inf],
        labels=["6-20", "21-50", "51+"],
        include_lowest=True,
    )
    rows = []
    for model_name, current in (("MF B", model), ("Popular", popular)):
        per_user, _ = _ranking_rows(current, movies, split, "validation", clip=False)
        per_user["history_bucket"] = per_user["user_id"].map(buckets)
        for bucket, frame in per_user.groupby("history_bucket", observed=True):
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "history_bucket": str(bucket),
                    "users": int(len(frame)),
                    "precision_at_10": float(frame["precision_at_10"].mean()),
                    "recall_at_10": float(frame["recall_at_10"].dropna().mean()),
                    "ndcg_at_10": float(frame["ndcg_at_10"].dropna().mean()),
                }
            )
    return rows


def main():
    ratings, movies, _ = load_dataset(ROOT / "data" / "ml-1m")
    splits = {
        "chronological": _read_locked_split(
            ROOT / "results" / "split_assignments.csv", ratings
        ),
        "random": _random_split(ratings),
    }
    selection_effect = {}
    fusion = {}
    reconciliation = {}
    history = []

    for split_name, split in splits.items():
        rmse_models = _load_models(split_name)
        ndcg_models, configs = load_ndcg_models(split_name, rmse_models)
        rmse_frames = {
            name: _ranking_rows(
                rmse_models[name], movies, split, "test", clip=False
            )[0]
            for name in ("MF A", "MF B")
        }
        ndcg_frames = {
            name: _ranking_rows(
                ndcg_models[name], movies, split, "test", clip=False
            )[0]
            for name in ("MF A", "MF B")
        }
        selection_effect[split_name] = paired_ci(
            {
                "MF A RMSE": rmse_frames["MF A"],
                "MF A NDCG": ndcg_frames["MF A"],
                "MF B RMSE": rmse_frames["MF B"],
                "MF B NDCG": ndcg_frames["MF B"],
            },
            {
                "MF A NDCG - RMSE": ("MF A NDCG", "MF A RMSE"),
                "MF B NDCG - RMSE": ("MF B NDCG", "MF B RMSE"),
            },
        )

        alpha, curve = select_alpha(
            ndcg_models["MF B"], rmse_models["Genre"], movies, split
        )
        curve.to_csv(
            OUTPUT / f"{split_name}_ndcg_mf_b_fusion_validation.csv",
            index=False,
        )
        test_hybrid = _hybrid_rows(
            ndcg_models["MF B"], rmse_models["Genre"], movies, split, "test", alpha
        )
        fusion[split_name] = {
            "selected_alpha": alpha,
            "validation_ndcg_at_10": float(
                curve.loc[curve["alpha"] == alpha, "validation_ndcg_at_10"].iloc[0]
            ),
            "test_ndcg_at_10": float(test_hybrid["ndcg_at_10"].dropna().mean()),
        }

        config_root = (
            ROOT / "results" / "regularization_sensitivity"
            if split_name == "chronological"
            else ROOT / "results" / "randomized_split_seed42" / "matrix_factorization"
        )
        reconciliation[split_name] = {
            "rmse_selected": {
                name: {
                    "validation_rmse": float(
                        json.loads(
                            (config_root / (
                                "A_interaction_weighted"
                                if name == "MF A"
                                else "B_entity_level"
                            ) / "config.json").read_text()
                        )["validation_rmse"]
                    ),
                    "test_rmse": float(
                        rating_prediction_metrics(
                            rmse_models[name],
                            split.loc[split["partition"] == "test"],
                        )[0]["rmse"]
                    ),
                }
                for name in ("MF A", "MF B")
            },
            "ndcg_selected_configs": configs.reset_index().to_dict("records"),
            "matched_lambda_table": {
                "protocol": "chronological",
                "selection_rule": "validation RMSE",
            },
        }
        popular = MostPopularModel().fit(split.loc[split["partition"] == "train"])
        history.extend(
            history_rows(rmse_models["MF B"], popular, movies, split, split_name)
        )

    (OUTPUT / "selection_effect_ci.json").write_text(
        json.dumps(selection_effect, indent=2), encoding="utf-8"
    )
    (OUTPUT / "ndcg_selected_fusion.json").write_text(
        json.dumps(fusion, indent=2), encoding="utf-8"
    )
    (OUTPUT / "rmse_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(
        OUTPUT / "history_popularity_comparison.csv", index=False
    )
    print(json.dumps({"selection_effect": selection_effect, "fusion": fusion}, indent=2))


if __name__ == "__main__":
    main()
