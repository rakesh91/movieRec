import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baseline_model import UserMovieBiasModel
from src.evaluation import ranking_metrics, rating_prediction_metrics
from src.matrix_factorization import RegularizedMatrixFactorization
from src.pipeline import _tune_model
from src.recommender import recommend_for_user
from src.user_eligibility import build_user_eligibility


LATENT_FACTORS = (10, 20, 50)
REGULARIZATION_VALUES = (0.02, 0.05, 0.1)
LEARNING_RATE = 0.01
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4
RANDOM_SEED = 42


def _read_locked_split(split_path, ratings):
    split = pd.read_csv(split_path)
    required = {
        "row_order",
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
        "partition",
    }
    if not required.issubset(split.columns):
        raise ValueError(f"Saved split is missing columns: {sorted(required - set(split))}")

    if not split["row_order"].is_unique or len(split) != len(ratings):
        raise ValueError("Saved split does not contain every source row exactly once.")
    source = ratings.set_index("row_order")[
        ["user_id", "movie_id", "rating", "timestamp"]
    ].sort_index()
    saved = split.set_index("row_order")[
        ["user_id", "movie_id", "rating", "timestamp"]
    ].sort_index()
    if not saved.astype(str).equals(source.astype(str)):
        raise ValueError("Saved split rows do not match the source ratings.")
    if set(split["partition"]) != {"train", "validation", "test"}:
        raise ValueError("Saved split must contain train, validation, and test labels.")
    return split


def _tune_matrix_factorization(training, validation):
    histories = []
    candidates = []
    for factors in LATENT_FACTORS:
        for regularization in REGULARIZATION_VALUES:
            model = RegularizedMatrixFactorization(
                n_factors=factors,
                regularization=regularization,
                learning_rate=LEARNING_RATE,
                random_seed=RANDOM_SEED,
            ).fit(
                training,
                validation,
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOPPING_PATIENCE,
            )
            for point in model.history_:
                histories.append(
                    {
                        "n_factors": factors,
                        "regularization": regularization,
                        "learning_rate": LEARNING_RATE,
                        **point,
                        "selected_epoch": point["epoch"] == model.best_epoch_,
                    }
                )
            candidates.append(
                {
                    "n_factors": factors,
                    "regularization": regularization,
                    "learning_rate": LEARNING_RATE,
                    "best_epoch": model.best_epoch_,
                    "epochs_trained": len(model.history_),
                    "validation_mae": model.best_validation_mae_,
                    "validation_rmse": model.best_validation_rmse_,
                    "model": model,
                }
            )

    tuning = pd.DataFrame(
        [
            {key: value for key, value in candidate.items() if key != "model"}
            for candidate in candidates
        ]
    ).sort_values(
        ["validation_rmse", "n_factors", "regularization"],
        kind="mergesort",
    ).reset_index(drop=True)
    best_config = tuning.iloc[0]
    best_model = next(
        candidate["model"]
        for candidate in candidates
        if candidate["n_factors"] == int(best_config["n_factors"])
        and candidate["regularization"] == float(best_config["regularization"])
    )
    return best_model, tuning, pd.DataFrame(histories)


def _save_validation_curves(history, figures_dir):
    for metric, label in (
        ("validation_rmse", "Validation RMSE"),
        ("validation_mae", "Validation MAE"),
    ):
        figure, axis = plt.subplots(figsize=(10, 6))
        for (factors, regularization), rows in history.groupby(
            ["n_factors", "regularization"], sort=True
        ):
            axis.plot(
                rows["epoch"],
                rows[metric],
                label=f"factors={factors}, reg={regularization:g}",
            )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.set_title(f"Matrix factorization {label.lower()} by epoch")
        axis.legend(fontsize="small", ncol=2)
        figure.tight_layout()
        figure.savefig(
            figures_dir / f"matrix_factorization_{metric}_curves.png",
            dpi=150,
        )
        plt.close(figure)


def _recommendation_overlap(model, movies, training, eligibility, results_dir):
    eligible_ids = eligibility.loc[
        eligibility["can_receive_top_10_recommendation"], "user_id"
    ].astype(int)
    selected_users = eligible_ids.head(5).tolist()
    if len(selected_users) < 5:
        raise ValueError(
            f"Need five eligible users for the overlap check; found {len(selected_users)}."
        )

    lists = {}
    recommendation_rows = []
    for user_id in selected_users:
        recommendations = recommend_for_user(
            model, user_id, movies, training, n=10
        )
        if len(recommendations) != 10:
            raise ValueError(
                f"User {user_id} received {len(recommendations)} rather than 10 recommendations."
            )
        lists[user_id] = set(recommendations["movie_id"].astype(int))
        for rank, row in enumerate(recommendations.itertuples(), start=1):
            recommendation_rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "movie_id": int(row.movie_id),
                    "title": row.title,
                    "genres": row.genres,
                    "predicted_rating": float(row.predicted_rating),
                }
            )

    pair_rows = []
    for first_user, second_user in combinations(selected_users, 2):
        shared = len(lists[first_user] & lists[second_user])
        union = len(lists[first_user] | lists[second_user])
        pair_rows.append(
            {
                "user_id_1": first_user,
                "user_id_2": second_user,
                "shared_recommendations": shared,
                "overlap_fraction_of_10": shared / 10.0,
                "jaccard_overlap": shared / union if union else 0.0,
            }
        )

    recommendations_frame = pd.DataFrame(recommendation_rows)
    pairs_frame = pd.DataFrame(pair_rows)
    recommendations_frame.to_csv(
        results_dir / "top_10_recommendations.csv", index=False
    )
    pairs_frame.to_csv(results_dir / "recommendation_pairwise_overlap.csv", index=False)
    return {
        "users": selected_users,
        "average_pairwise_shared_movies": float(
            pairs_frame["shared_recommendations"].mean()
        ),
        "average_pairwise_overlap_fraction": float(
            pairs_frame["overlap_fraction_of_10"].mean()
        ),
        "average_pairwise_jaccard": float(pairs_frame["jaccard_overlap"].mean()),
        "pairwise_results": pair_rows,
        "recommendations": recommendation_rows,
    }


def run_matrix_factorization(data_dir, split_path, results_dir):
    from src.data_loader import load_dataset

    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, users = load_dataset(data_dir)
    split = _read_locked_split(split_path, ratings)
    training = split.loc[split["partition"] == "train"].copy()
    validation = split.loc[split["partition"] == "validation"].copy()
    test = split.loc[split["partition"] == "test"].copy()
    if training.empty or validation.empty or test.empty:
        raise ValueError("The locked split must have nonempty train, validation, and test data.")

    model, tuning, history = _tune_matrix_factorization(training, validation)
    tuning.to_csv(results_dir / "validation_tuning.csv", index=False)
    history.to_csv(results_dir / "training_history.csv", index=False)
    history.to_csv(results_dir / "validation_curves.csv", index=False)
    _save_validation_curves(history, figures_dir)

    test_rating_metrics, per_user_metrics = rating_prediction_metrics(model, test)
    per_user_metrics.to_csv(results_dir / "per_user_test_metrics.csv", index=False)
    test_ranking_metrics = ranking_metrics(model, movies, split, k=10)

    bias_model, bias_tuning = _tune_model(training, validation)
    bias_tuning.to_csv(results_dir / "bias_baseline_validation_tuning.csv", index=False)
    bias_rating_metrics, _ = rating_prediction_metrics(bias_model, test)
    bias_ranking_metrics = ranking_metrics(bias_model, movies, split, k=10)
    bias_config = {
        "name": "regularized user-movie bias",
        "global_mean_training_only": bias_model.global_mean,
        "lambda_user": bias_model.lambda_user,
        "lambda_movie": bias_model.lambda_movie,
        "iterations": bias_model.iterations,
    }

    eligibility = build_user_eligibility(users, movies, split)
    eligibility.to_csv(results_dir / "user_eligibility.csv", index=False)
    overlap = _recommendation_overlap(
        model, movies, training, eligibility, results_dir
    )

    split_summary = {
        partition: {
            "ratings": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
        }
        for partition, frame in (
            ("train", training),
            ("validation", validation),
            ("test", test),
        )
    }
    selected_config = {
        "name": "regularized biased matrix factorization",
        "n_factors": model.n_factors,
        "regularization": model.regularization,
        "learning_rate": model.learning_rate,
        "random_seed": model.random_seed,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "epochs_trained": len(model.history_),
        "selected_epoch": model.best_epoch_,
        "global_mean_training_only": model.global_mean,
        "validation_rmse": model.best_validation_rmse_,
        "validation_mae": model.best_validation_mae_,
        "unknown_user": "zero user bias and zero user factors",
        "unknown_movie": "zero movie bias and zero movie factors",
    }
    comparison = {
        "matrix_factorization": {
            "validation_mae": model.best_validation_mae_,
            "validation_rmse": model.best_validation_rmse_,
            "test_rating_prediction": test_rating_metrics,
            "test_ranking": test_ranking_metrics,
        },
        "bias_only_baseline": {
            "selected_config": bias_config,
            "test_rating_prediction": bias_rating_metrics,
            "test_ranking": bias_ranking_metrics,
        },
    }
    comparison_rows = [
        {
            "model": "matrix_factorization",
            "validation_mae": model.best_validation_mae_,
            "validation_rmse": model.best_validation_rmse_,
            "test_mae": test_rating_metrics["mae"],
            "test_rmse": test_rating_metrics["rmse"],
            "restricted_precision_at_10": test_ranking_metrics[
                "restricted_test_candidate_ranking"
            ]["precision_at_10"],
            "restricted_recall_at_10": test_ranking_metrics[
                "restricted_test_candidate_ranking"
            ]["recall_at_10"],
            "restricted_ndcg_at_10": test_ranking_metrics[
                "restricted_test_candidate_ranking"
            ]["ndcg_at_10"],
            "catalogue_precision_at_10": test_ranking_metrics[
                "full_eligible_catalogue_ranking"
            ]["precision_at_10"],
            "catalogue_recall_at_10": test_ranking_metrics[
                "full_eligible_catalogue_ranking"
            ]["recall_at_10"],
            "catalogue_ndcg_at_10": test_ranking_metrics[
                "full_eligible_catalogue_ranking"
            ]["ndcg_at_10"],
        },
        {
            "model": "bias_only_baseline",
            "validation_mae": float(
                bias_tuning.sort_values("validation_mae").iloc[0]["validation_mae"]
            ),
            "validation_rmse": None,
            "test_mae": bias_rating_metrics["mae"],
            "test_rmse": bias_rating_metrics["rmse"],
            "restricted_precision_at_10": bias_ranking_metrics[
                "restricted_test_candidate_ranking"
            ]["precision_at_10"],
            "restricted_recall_at_10": bias_ranking_metrics[
                "restricted_test_candidate_ranking"
            ]["recall_at_10"],
            "restricted_ndcg_at_10": bias_ranking_metrics[
                "restricted_test_candidate_ranking"
            ]["ndcg_at_10"],
            "catalogue_precision_at_10": bias_ranking_metrics[
                "full_eligible_catalogue_ranking"
            ]["precision_at_10"],
            "catalogue_recall_at_10": bias_ranking_metrics[
                "full_eligible_catalogue_ranking"
            ]["recall_at_10"],
            "catalogue_ndcg_at_10": bias_ranking_metrics[
                "full_eligible_catalogue_ranking"
            ]["ndcg_at_10"],
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(
        results_dir / "baseline_comparison.csv", index=False
    )

    metrics = {
        "dataset": "MovieLens 1M",
        "split_source": str(Path(split_path)),
        "split_counts": split_summary,
        "matrix_factorization": selected_config,
        "matrix_factorization_test_rating_prediction": test_rating_metrics,
        "matrix_factorization_test_ranking": test_ranking_metrics,
        "bias_only_baseline_comparison": comparison["bias_only_baseline"],
        "recommendation_personalization": {
            key: value
            for key, value in overlap.items()
            if key not in {"pairwise_results", "recommendations"}
        },
        "hyperparameter_grid": {
            "latent_factors": list(LATENT_FACTORS),
            "regularization": list(REGULARIZATION_VALUES),
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection_metric": "validation RMSE",
        },
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return {
        "model": model,
        "tuning": tuning,
        "history": history,
        "test_rating_metrics": test_rating_metrics,
        "test_ranking_metrics": test_ranking_metrics,
        "bias_config": bias_config,
        "bias_rating_metrics": bias_rating_metrics,
        "bias_ranking_metrics": bias_ranking_metrics,
        "overlap": overlap,
        "metrics": metrics,
        "results_dir": results_dir,
    }
