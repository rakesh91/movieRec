import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import rating_prediction_metrics
from src.pipeline import display_experiment_summary, prepare_experiment
from src.recommender import predict_test_ratings_for_user, recommend_for_user


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "ml-1m"
DEFAULT_RESULTS_DIR = ROOT / "results"


def _read_integer(prompt):
    value = input(prompt).strip()
    try:
        return int(value)
    except ValueError as error:
        raise ValueError("Please enter a whole-number ID.") from error


def _known_user(experiment, user_id, require_training=True):
    users = experiment["users"]
    if user_id not in set(users["user_id"].tolist()):
        raise ValueError(f"User ID {user_id} is not in the MovieLens users file.")
    if require_training and not (
        experiment["training"]["user_id"] == user_id
    ).any():
        raise ValueError(f"User {user_id} has no training ratings.")


def _show_recommendations(experiment, user_id):
    _known_user(experiment, user_id)
    print("Available recommendation method: regularized user-movie bias baseline.")
    recommendations = recommend_for_user(
        experiment["model"],
        user_id,
        experiment["movies"],
        experiment["training"],
        n=10,
    )
    if recommendations.empty:
        print("No unseen catalogue movies are available for this user.")
        return
    recommendations["in_training_history"] = recommendations[
        "in_training_history"
    ].map({True: "yes", False: "no"})
    print(
        recommendations[
            [
                "movie_id",
                "title",
                "genres",
                "predicted_rating",
                "in_training_history",
            ]
        ]
        .rename(
            columns={
                "movie_id": "Movie ID",
                "title": "Title",
                "genres": "Genres",
                "predicted_rating": "Predicted",
                "in_training_history": "In training history",
            }
        )
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


def _show_prediction(experiment, user_id, movie_id):
    _known_user(experiment, user_id, require_training=False)
    movie = experiment["movies"].loc[experiment["movies"]["movie_id"] == movie_id]
    if movie.empty:
        raise ValueError(f"Movie ID {movie_id} is not in the MovieLens catalogue.")
    predicted = experiment["model"].predict(user_id, movie_id)
    print(f"Predicted rating: {predicted:.3f} / 5")
    heldout = experiment["test"].loc[
        (experiment["test"]["user_id"] == user_id)
        & (experiment["test"]["movie_id"] == movie_id)
    ]
    if not heldout.empty:
        print(
            "Held-out test rating (evaluation reference only): "
            f"{heldout['rating'].iloc[0]}"
        )


def _show_user_test_evaluation(experiment, user_id):
    _known_user(experiment, user_id, require_training=False)
    predictions = predict_test_ratings_for_user(
        experiment["model"], user_id, experiment["test"]
    )
    if predictions.empty:
        print(f"User {user_id} has no held-out test ratings.")
        return
    print(
        predictions[["movie_id", "rating", "predicted_rating"]]
        .rename(
            columns={
                "movie_id": "Movie ID",
                "rating": "Actual test rating",
                "predicted_rating": "Predicted rating",
            }
        )
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    metrics, _ = rating_prediction_metrics(experiment["model"], predictions)
    print(
        f"User {user_id}: {metrics['test_interactions']} test interactions, "
        f"MAE {metrics['mae']:.4f}, RMSE {metrics['rmse']:.4f}"
    )


def _show_eligible_users(experiment):
    eligibility = experiment["eligibility"]
    print(
        "Criteria: rating prediction requires at least one test rating; ranking "
        "requires at least one relevant (rating >= 4) test movie; top-10 "
        "recommendation requires training history and at least 10 unseen "
        "catalogue movies."
    )
    for column, title in (
        ("eligible_for_rating_prediction", "Rating-prediction evaluation users"),
        ("eligible_for_ranking_evaluation", "Ranking evaluation users"),
        ("can_receive_top_10_recommendation", "Full-catalogue recommendation users"),
    ):
        ids = eligibility.loc[eligibility[column], "user_id"].astype(int).tolist()
        print(f"\n{title} ({len(ids)}):")
        print(", ".join(map(str, ids)) if ids else "(none)")


def _show_overall_evaluation(experiment):
    metrics = experiment["metrics"]
    print("Rating prediction on held-out test interactions:")
    print(f"  Interactions: {metrics['rating_prediction_test']['test_interactions']:,}")
    print(f"  Users: {metrics['rating_prediction_test']['users_evaluated']:,}")
    print(f"  MAE: {metrics['rating_prediction_test']['mae']:.4f}")
    print(f"  RMSE: {metrics['rating_prediction_test']['rmse']:.4f}")
    for experiment_name, values in metrics["ranking_test"].items():
        print(f"\n{experiment_name.replace('_', ' ').title()} (k=10):")
        print(
            f"  Precision@10: {values['precision_at_10']:.4f} "
            f"(users: {values['users_for_precision']:,})"
        )
        print(f"  Precision definition: {values['precision_at_10_definition']}")
        if experiment_name == "restricted_test_candidate_ranking":
            print(
                "  Users with fewer than 10 test candidates: "
                f"{values['users_with_fewer_than_10_candidates']:,}"
            )
        if values["recall_at_10"] is None:
            print("  Recall@10: undefined (no users with relevant test items)")
            print("  NDCG@10: undefined (no users with relevant test items)")
        else:
            print(
                f"  Recall@10: {values['recall_at_10']:.4f} "
                f"(users: {values['users_for_recall_and_ndcg']:,})"
            )
            print(f"  NDCG@10: {values['ndcg_at_10']:.4f}")
        print(
            "  Users excluded from Recall/NDCG for zero relevant test items: "
            f"{values['users_excluded_no_relevant_test_items']:,}"
        )
    print(
        "\nCatalogue results are offline estimates: unobserved movies have "
        "unknown relevance, not confirmed negative feedback."
    )
    tie_metrics = metrics["timestamp_ties_split_across_partitions"]
    print(
        "\nTimestamp ties split across partitions: "
        f"{tie_metrics['users_affected']:,} users, "
        f"{tie_metrics['rating_records_affected']:,} rating records "
        f"({tie_metrics['timestamp_tie_groups_crossing_partitions']:,} "
        "same-user timestamp groups). The tie-aware split keeps each "
        "same-user timestamp group in one partition."
    )


def _menu(experiment):
    while True:
        print(
            "\nMOVIE RECOMMENDATION SYSTEM\n\n"
            "1. Recommend movies for an existing MovieLens user\n"
            "2. Show rating predictions for a user\n"
            "3. Evaluate a user on held-out test ratings\n"
            "4. Show eligible users for testing\n"
            "5. Run overall evaluation\n"
            "6. Exit"
        )
        choice = input("Choose an option: ").strip()
        try:
            if choice == "1":
                _show_recommendations(
                    experiment, _read_integer("Enter MovieLens user ID: ")
                )
            elif choice == "2":
                user_id = _read_integer("Enter MovieLens user ID: ")
                movie_id = _read_integer("Enter MovieLens movie ID: ")
                _show_prediction(experiment, user_id, movie_id)
            elif choice == "3":
                _show_user_test_evaluation(
                    experiment, _read_integer("Enter MovieLens user ID: ")
                )
            elif choice == "4":
                _show_eligible_users(experiment)
            elif choice == "5":
                _show_overall_evaluation(experiment)
            elif choice == "6":
                print("Goodbye.")
                return
            else:
                print("Choose a menu option from 1 to 6.")
        except ValueError as error:
            print(f"Input error: {error}")


def _run_batch(experiment):
    display_experiment_summary(experiment)
    _show_overall_evaluation(experiment)
    eligible = experiment["eligibility"].loc[
        experiment["eligibility"]["can_receive_top_10_recommendation"], "user_id"
    ]
    sample_ids = experiment["metrics"]["eligibility"][
        "sample_full_catalogue_user_ids"
    ]
    print(
        "\nSample of at least 20 eligible user IDs: "
        + ", ".join(map(str, sample_ids))
    )
    if not eligible.empty:
        demonstration_user = int(eligible.iloc[0])
        print(
            f"\nDemonstration recommendation for eligible user "
            f"{demonstration_user}:"
        )
        _show_recommendations(experiment, demonstration_user)


def main():
    parser = argparse.ArgumentParser(description="MovieLens 1M recommendation baseline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="run the complete baseline and print metrics without opening the menu",
    )
    args = parser.parse_args()
    try:
        experiment = prepare_experiment(args.data_dir, args.results_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    display_experiment_summary(experiment)
    if args.batch:
        _run_batch(experiment)
        return
    _menu(experiment)


if __name__ == "__main__":
    main()
