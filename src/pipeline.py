import json
from pathlib import Path

import pandas as pd

from src.baseline_model import UserMovieBiasModel
from src.data_audit import run_data_audit
from src.data_loader import load_dataset
from src.data_split import chronological_split
from src.evaluation import ranking_metrics, rating_prediction_metrics
from src.user_eligibility import build_user_eligibility


REGULARIZATION_VALUES = (5.0, 10.0, 20.0)
MODEL_ITERATIONS = 15


def _timestamp_tie_diagnostics(split):
    crosses_partition = (
        split.groupby(["user_id", "timestamp"], sort=False)["partition"]
        .transform("nunique")
        .gt(1)
    )
    affected = split.loc[crosses_partition]
    return {
        "users_affected": int(affected["user_id"].nunique()),
        "rating_records_affected": int(len(affected)),
        "timestamp_tie_groups_crossing_partitions": int(
            affected.groupby(["user_id", "timestamp"], sort=False).ngroups
        ),
    }


def _partition_coverage(split):
    user_count = int(split["user_id"].nunique())
    counts = split.groupby("partition")["user_id"].nunique()
    return {
        "users_per_partition": {
            partition: int(counts.get(partition, 0))
            for partition in ("train", "validation", "test")
        },
        "users_without_validation": int(
            user_count - counts.get("validation", 0)
        ),
        "users_without_test": int(user_count - counts.get("test", 0)),
    }


def _tune_model(training_ratings, validation_ratings):
    candidates = []
    if validation_ratings.empty:
        selected = UserMovieBiasModel(
            lambda_user=10.0,
            lambda_movie=10.0,
            iterations=MODEL_ITERATIONS,
        ).fit(training_ratings)
        return selected, pd.DataFrame(
            [{"lambda_user": 10.0, "lambda_movie": 10.0, "validation_mae": None}]
        )

    for lambda_user in REGULARIZATION_VALUES:
        for lambda_movie in REGULARIZATION_VALUES:
            model = UserMovieBiasModel(
                lambda_user=lambda_user,
                lambda_movie=lambda_movie,
                iterations=MODEL_ITERATIONS,
            ).fit(training_ratings)
            predictions = model.predict_many(
                validation_ratings["user_id"].to_numpy(),
                validation_ratings["movie_id"].to_numpy(),
            )
            mae = float(
                (
                    abs(
                        predictions
                        - validation_ratings["rating"].to_numpy(dtype=float)
                    )
                ).mean()
            )
            candidates.append(
                {
                    "lambda_user": lambda_user,
                    "lambda_movie": lambda_movie,
                    "validation_mae": mae,
                }
            )

    scores = pd.DataFrame(candidates).sort_values(
        ["validation_mae", "lambda_user", "lambda_movie"],
        kind="mergesort",
    ).reset_index(drop=True)
    best = scores.iloc[0]
    model = UserMovieBiasModel(
        lambda_user=best["lambda_user"],
        lambda_movie=best["lambda_movie"],
        iterations=MODEL_ITERATIONS,
    ).fit(training_ratings)
    return model, scores


def prepare_experiment(data_dir, results_dir):
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ratings, movies, users = load_dataset(data_dir)
    audit = run_data_audit(ratings, movies, users, results_dir)
    split = chronological_split(ratings)
    split.to_csv(results_dir / "split_assignments.csv", index=False)
    training = split.loc[split["partition"] == "train"].copy()
    validation = split.loc[split["partition"] == "validation"].copy()
    test = split.loc[split["partition"] == "test"].copy()
    model, validation_scores = _tune_model(training, validation)
    validation_scores.to_csv(results_dir / "validation_tuning.csv", index=False)

    rating_metrics, per_user_metrics = rating_prediction_metrics(model, test)
    per_user_metrics.to_csv(results_dir / "per_user_test_metrics.csv", index=False)
    rankings = ranking_metrics(model, movies, split, k=10)
    eligibility = build_user_eligibility(users, movies, split)
    eligibility.to_csv(results_dir / "user_eligibility.csv", index=False)

    partition_summary = {}
    for name, frame in (
        ("train", training),
        ("validation", validation),
        ("test", test),
    ):
        partition_summary[name] = {
            "ratings": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
        }
    timestamp_ties = _timestamp_tie_diagnostics(split)
    metrics = {
        "dataset": "MovieLens 1M",
        "data_audit": {
            "users": audit["users"],
            "movies": audit["movies"],
            "ratings": audit["ratings"],
        },
        "split": partition_summary,
        "split_user_coverage": _partition_coverage(split),
        "timestamp_ties_split_across_partitions": timestamp_ties,
        "model": {
            "name": "regularized user-movie bias",
            "global_mean_training_only": model.global_mean,
            "lambda_user": model.lambda_user,
            "lambda_movie": model.lambda_movie,
            "iterations": model.iterations,
            "selection_metric": "validation MAE",
            "validation_interactions": int(len(validation)),
        },
        "rating_prediction_test": rating_metrics,
        "ranking_test": rankings,
        "eligibility": {
            "users_total": int(len(eligibility)),
            "rating_prediction": int(
                eligibility["eligible_for_rating_prediction"].sum()
            ),
            "ranking": int(eligibility["eligible_for_ranking_evaluation"].sum()),
            "full_catalogue_recommendation": int(
                eligibility["can_receive_top_10_recommendation"].sum()
            ),
            "sample_rating_prediction_user_ids": eligibility.loc[
                eligibility["eligible_for_rating_prediction"], "user_id"
            ]
            .head(20)
            .astype(int)
            .tolist(),
            "sample_full_catalogue_user_ids": eligibility.loc[
                eligibility["can_receive_top_10_recommendation"], "user_id"
            ]
            .head(20)
            .astype(int)
            .tolist(),
        },
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return {
        "ratings": ratings,
        "movies": movies,
        "users": users,
        "audit": audit,
        "split": split,
        "training": training,
        "validation": validation,
        "test": test,
        "model": model,
        "validation_scores": validation_scores,
        "rating_metrics": rating_metrics,
        "per_user_metrics": per_user_metrics,
        "rankings": rankings,
        "eligibility": eligibility,
        "metrics": metrics,
        "results_dir": results_dir,
    }


def display_experiment_summary(experiment):
    metrics = experiment["metrics"]
    eligibility = experiment["eligibility"]
    print("\nMovieLens 1M baseline is ready.")
    print(
        "Dataset: {users:,} users, {movies:,} movies, {ratings:,} ratings".format(
            **metrics["data_audit"]
        )
    )
    print("Partition counts:")
    for partition, summary in metrics["split"].items():
        print(
            f"  {partition}: {summary['ratings']:,} ratings, "
            f"{summary['users']:,} users"
        )
    print(
        "Eligibility: "
        f"{len(eligibility):,} total users; "
        f"{metrics['eligibility']['rating_prediction']:,} rating-test users; "
        f"{metrics['eligibility']['ranking']:,} ranking users; "
        f"{metrics['eligibility']['full_catalogue_recommendation']:,} users "
        "can receive top-10 recommendations."
    )
    print(
        "Sample eligible rating-test user IDs: "
        + ", ".join(map(str, metrics["eligibility"]["sample_rating_prediction_user_ids"]))
    )
    print(
        "Test rating prediction: "
        f"MAE {metrics['rating_prediction_test']['mae']:.4f}, "
        f"RMSE {metrics['rating_prediction_test']['rmse']:.4f}"
    )
    print(f"Reports and figures saved under: {experiment['results_dir']}")
