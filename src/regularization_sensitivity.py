import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_model import UserMovieBiasModel
from src.entity_regularized_matrix_factorization import (
    EntityRegularizedMatrixFactorization,
)
from src.evaluation import ranking_metrics, rating_prediction_metrics
from src.mf_pipeline import _read_locked_split
from src.matrix_factorization import RegularizedMatrixFactorization
from src.pipeline import _tune_model
from src.recommender import recommend_for_user
from src.user_eligibility import build_user_eligibility


LATENT_FACTORS = 50
RANDOM_SEED = 42
REGULARIZATION_VALUES = (0.02, 0.05, 0.1)
LEARNING_RATE_VALUES = (0.0025, 0.005, 0.01)
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4


def _tune_formulation(model_type, training, validation):
    tuning_rows = []
    history_rows = []
    models = {}
    for regularization in REGULARIZATION_VALUES:
        for learning_rate in LEARNING_RATE_VALUES:
            model = model_type(
                n_factors=LATENT_FACTORS,
                regularization=regularization,
                learning_rate=learning_rate,
                random_seed=RANDOM_SEED,
            ).fit(
                training,
                validation,
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOPPING_PATIENCE,
            )
            key = (regularization, learning_rate)
            models[key] = model
            tuning_rows.append(
                {
                    "n_factors": LATENT_FACTORS,
                    "regularization": regularization,
                    "learning_rate": learning_rate,
                    "best_epoch": model.best_epoch_,
                    "epochs_trained": len(model.history_),
                    "validation_mae": model.best_validation_mae_,
                    "validation_rmse": model.best_validation_rmse_,
                }
            )
            history_rows.extend(
                {
                    "n_factors": LATENT_FACTORS,
                    "regularization": regularization,
                    "learning_rate": learning_rate,
                    **point,
                    "selected_epoch": point["epoch"] == model.best_epoch_,
                }
                for point in model.history_
            )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["validation_rmse", "regularization", "learning_rate"],
        kind="mergesort",
    ).reset_index(drop=True)
    best = tuning.iloc[0]
    best_model = models[
        (float(best["regularization"]), float(best["learning_rate"]))
    ]
    return best_model, tuning, pd.DataFrame(history_rows)


def _save_checkpoint(model, directory):
    np.savez_compressed(
        directory / "model_checkpoint.npz",
        user_ids=model.user_ids,
        movie_ids=model.movie_ids,
        user_biases=model.user_biases,
        movie_biases=model.movie_biases,
        user_factors=model.user_factors,
        movie_factors=model.movie_factors,
        global_mean=np.asarray(model.global_mean),
    )
    return {
        "checkpoint": "model_checkpoint.npz",
        "global_mean_training_only": model.global_mean,
        "n_factors": model.n_factors,
        "regularization": model.regularization,
        "learning_rate": model.learning_rate,
        "random_seed": model.random_seed,
        "batch_size": model.batch_size,
        "selected_epoch": model.best_epoch_,
        "epochs_trained": len(model.history_),
        "validation_mae": model.best_validation_mae_,
        "validation_rmse": model.best_validation_rmse_,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }


def _recommendations_for_users(
    model, movies, training, user_ids, output_path
):
    rows = []
    recommendation_sets = {}
    for user_id in user_ids:
        recs = recommend_for_user(model, user_id, movies, training, n=10)
        if len(recs) != 10:
            raise ValueError(f"User {user_id} did not receive exactly 10 recommendations.")
        recommendation_sets[int(user_id)] = set(recs["movie_id"].astype(int))
        for rank, row in enumerate(recs.itertuples(), start=1):
            rows.append(
                {
                    "user_id": int(user_id),
                    "rank": rank,
                    "movie_id": int(row.movie_id),
                    "title": row.title,
                    "genres": row.genres,
                    "predicted_rating": float(row.predicted_rating),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return recommendation_sets


def _overlap_rows(recommendation_sets):
    rows = []
    user_ids = list(recommendation_sets)
    for index, first_user in enumerate(user_ids):
        for second_user in user_ids[index + 1 :]:
            first = recommendation_sets[first_user]
            second = recommendation_sets[second_user]
            shared = len(first & second)
            union = len(first | second)
            rows.append(
                {
                    "user_id_1": first_user,
                    "user_id_2": second_user,
                    "shared_recommendations": shared,
                    "overlap_fraction_of_10": shared / 10.0,
                    "jaccard_overlap": shared / union if union else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_model(model, movies, split):
    training = split.loc[split["partition"] == "train"]
    test = split.loc[split["partition"] == "test"]
    rating_metrics, _ = rating_prediction_metrics(model, test)
    rankings = ranking_metrics(model, movies, split, k=10)
    return rating_metrics, rankings


def run_sensitivity_experiment(data_dir, split_path, results_dir):
    from src.data_loader import load_dataset

    results_dir = Path(results_dir)
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"Sensitivity output directory is not empty: {results_dir}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, users = load_dataset(data_dir)
    split = _read_locked_split(split_path, ratings)
    training = split.loc[split["partition"] == "train"].copy()
    validation = split.loc[split["partition"] == "validation"].copy()
    test = split.loc[split["partition"] == "test"].copy()
    if training.empty or validation.empty or test.empty:
        raise ValueError("The locked split must have nonempty train/validation/test data.")

    split_hash = hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
    formulations = {
        "A_interaction_weighted": (
            RegularizedMatrixFactorization,
            "The existing MF Version 1 objective and update implementation.",
            (
                "J_A(B)=1/2*sum_(u,i in B)e_ui^2 + "
                "lambda/2*sum_(u,i in B)(b_u^2+b_i^2+||p_u||^2+||q_i||^2)"
            ),
        ),
        "B_entity_level": (
            EntityRegularizedMatrixFactorization,
            "Each distinct user's and movie's bias/factor parameters are penalized once per batch.",
            (
                "J_B(B)=1/2*sum_(u,i in B)e_ui^2 + lambda/2*("
                "sum_(u in U_B)(b_u^2+||p_u||^2) + "
                "sum_(i in I_B)(b_i^2+||q_i||^2))"
            ),
        ),
    }
    results = {}
    model_objects = {}
    for name, (
        model_type,
        objective_description,
        objective_equation,
    ) in formulations.items():
        directory = results_dir / name
        directory.mkdir(parents=True)
        model, tuning, history = _tune_formulation(
            model_type, training, validation
        )
        tuning.to_csv(directory / "validation_tuning.csv", index=False)
        history.to_csv(directory / "training_history.csv", index=False)
        selected_history = history.loc[
            (history["regularization"] == model.regularization)
            & (history["learning_rate"] == model.learning_rate)
        ].copy()
        selected_history.to_csv(
            directory / "selected_training_history.csv", index=False
        )
        config = _save_checkpoint(model, directory)
        config.update(
            {
                "formulation": name,
                "objective": objective_description,
                "objective_equation": objective_equation,
                "factor_update_equation": (
                    "p_u <- p_u + eta*(sum_(i in B_u)e_ui*q_i - "
                    "lambda*penalty_multiplier_u*p_u); "
                    "q_i <- q_i + eta*(sum_(u in B_i)e_ui*p_u - "
                    "lambda*penalty_multiplier_i*q_i)"
                ),
                "bias_update_equation": (
                    "b_u <- b_u + eta*(sum_(i in B_u)e_ui - "
                    "lambda*penalty_multiplier_u*b_u); "
                    "b_i <- b_i + eta*(sum_(u in B_i)e_ui - "
                    "lambda*penalty_multiplier_i*b_i)"
                ),
                "penalty_multiplier": (
                    "number of this entity's interactions in batch"
                    if name == "A_interaction_weighted"
                    else "1 for each distinct entity present in batch"
                ),
                "objective_normalization": (
                    "one half times summed squared error over the current mini-batch; "
                    "one half times lambda times the formulation-specific parameter "
                    "penalty; gradients and learning-rate scale use summed error."
                ),
                "validation_selection_metric": "validation RMSE",
                "validation_interactions": int(len(validation)),
                "test_interactions_used_for_selection": 0,
                "locked_split_sha256": split_hash,
            }
        )
        with (directory / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
        results[name] = {
            "config": config,
            "tuning": tuning,
            "history": history,
        }
        model_objects[name] = model

    test_metrics = {}
    for name, model in model_objects.items():
        rating_metrics, rankings = _evaluate_model(model, movies, split)
        test_metrics[name] = {
            "rating_prediction": rating_metrics,
            "ranking": rankings,
        }
        directory = results_dir / name
        with (directory / "test_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(test_metrics[name], file, indent=2)

    bias_model, bias_tuning = _tune_model(training, validation)
    bias_tuning.to_csv(results_dir / "bias_only_validation_tuning.csv", index=False)
    bias_metrics, bias_rankings = _evaluate_model(bias_model, movies, split)
    with (results_dir / "bias_only_reference.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "test_rating_prediction": bias_metrics,
                "test_ranking": bias_rankings,
                "selected_lambda_user": bias_model.lambda_user,
                "selected_lambda_movie": bias_model.lambda_movie,
                "locked_split_sha256": split_hash,
            },
            file,
            indent=2,
        )

    eligibility = build_user_eligibility(users, movies, split)
    eligible_users = eligibility.loc[
        eligibility["can_receive_top_10_recommendation"], "user_id"
    ].astype(int).head(20).tolist()
    if len(eligible_users) != 20:
        raise ValueError("Could not find 20 eligible users for overlap comparison.")
    overlap_results = {}
    recommendation_sets = {}
    for name, model in model_objects.items():
        recommendation_sets[name] = _recommendations_for_users(
            model,
            movies,
            training,
            eligible_users,
            results_dir / name / "top_10_recommendations.csv",
        )

    for name, sets in recommendation_sets.items():
        pairs = _overlap_rows(sets)
        pairs.to_csv(
            results_dir / name / "pairwise_recommendation_overlap.csv",
            index=False,
        )
        overlap_results[name] = {
            "users": eligible_users,
            "pair_count": int(len(pairs)),
            "average_shared_movies": float(
                pairs["shared_recommendations"].mean()
            ),
            "average_overlap_fraction_of_10": float(
                pairs["overlap_fraction_of_10"].mean()
            ),
            "average_jaccard": float(pairs["jaccard_overlap"].mean()),
        }

    comparison = {
        name: {
            "selected_configuration": results[name]["config"],
            "validation": {
                "mae": results[name]["config"]["validation_mae"],
                "rmse": results[name]["config"]["validation_rmse"],
            },
            "test_rating_prediction": test_metrics[name]["rating_prediction"],
            "test_ranking": test_metrics[name]["ranking"],
            "recommendation_overlap": overlap_results[name],
        }
        for name in formulations
    }
    report = {
        "dataset": "MovieLens 1M",
        "split_path": str(Path(split_path)),
        "locked_split_sha256": split_hash,
        "split_counts": {
            part: int((split["partition"] == part).sum())
            for part in ("train", "validation", "test")
        },
        "fixed_latent_factors": LATENT_FACTORS,
        "fixed_random_seed": RANDOM_SEED,
        "hyperparameter_grid": {
            "regularization": list(REGULARIZATION_VALUES),
            "learning_rate": list(LEARNING_RATE_VALUES),
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection_metric": "validation RMSE",
        },
        "test_data_used_for_tuning": False,
        "objective_equations": {
            "A_interaction_weighted": (
                "J_A(B)=1/2*sum_(u,i in B)e_ui^2 + "
                "lambda/2*sum_(u,i in B)(b_u^2+b_i^2+||p_u||^2+||q_i||^2)"
            ),
            "B_entity_level": (
                "J_B(B)=1/2*sum_(u,i in B)e_ui^2 + lambda/2*("
                "sum_(u in U_B)(b_u^2+||p_u||^2) + "
                "sum_(i in I_B)(b_i^2+||q_i||^2))"
            ),
        },
        "formulations": comparison,
        "bias_only_reference": {
            "test_rating_prediction": bias_metrics,
            "test_ranking": bias_rankings,
        },
    }
    with (results_dir / "comparison.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    pd.DataFrame(
        [
            {
                "formulation": name,
                "validation_mae": results[name]["config"]["validation_mae"],
                "validation_rmse": results[name]["config"]["validation_rmse"],
                "test_mae": test_metrics[name]["rating_prediction"]["mae"],
                "test_rmse": test_metrics[name]["rating_prediction"]["rmse"],
                "restricted_precision_at_10": test_metrics[name]["ranking"][
                    "restricted_test_candidate_ranking"
                ]["precision_at_10"],
                "restricted_recall_at_10": test_metrics[name]["ranking"][
                    "restricted_test_candidate_ranking"
                ]["recall_at_10"],
                "restricted_ndcg_at_10": test_metrics[name]["ranking"][
                    "restricted_test_candidate_ranking"
                ]["ndcg_at_10"],
                "catalogue_precision_at_10": test_metrics[name]["ranking"][
                    "full_eligible_catalogue_ranking"
                ]["precision_at_10"],
                "catalogue_recall_at_10": test_metrics[name]["ranking"][
                    "full_eligible_catalogue_ranking"
                ]["recall_at_10"],
                "catalogue_ndcg_at_10": test_metrics[name]["ranking"][
                    "full_eligible_catalogue_ranking"
                ]["ndcg_at_10"],
                "average_pairwise_overlap": overlap_results[name][
                    "average_overlap_fraction_of_10"
                ],
                "average_pairwise_jaccard": overlap_results[name][
                    "average_jaccard"
                ],
            }
            for name in formulations
        ]
    ).to_csv(results_dir / "comparison.csv", index=False)
    return report
