import argparse
from pathlib import Path

from src.mf_pipeline import run_matrix_factorization


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate MovieLens biased matrix factorization"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "ml-1m")
    parser.add_argument(
        "--split-path",
        type=Path,
        default=ROOT / "results" / "split_assignments.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "matrix_factorization",
    )
    args = parser.parse_args()

    experiment = run_matrix_factorization(
        args.data_dir,
        args.split_path,
        args.results_dir,
    )
    metrics = experiment["metrics"]
    model = metrics["matrix_factorization"]
    print("\nRegularized matrix factorization complete.")
    print(
        "Selected config: "
        f"factors={model['n_factors']}, regularization={model['regularization']}, "
        f"learning_rate={model['learning_rate']}, epoch={model['selected_epoch']}"
    )
    print(
        f"Validation: MAE {model['validation_mae']:.4f}, "
        f"RMSE {model['validation_rmse']:.4f}"
    )
    rating = metrics["matrix_factorization_test_rating_prediction"]
    print(
        f"Test: MAE {rating['mae']:.4f}, RMSE {rating['rmse']:.4f} "
        f"({rating['test_interactions']:,} ratings)"
    )
    for label, ranking in (
        ("Restricted test-candidate", metrics["matrix_factorization_test_ranking"]["restricted_test_candidate_ranking"]),
        ("Full catalogue", metrics["matrix_factorization_test_ranking"]["full_eligible_catalogue_ranking"]),
    ):
        print(
            f"{label} ranking: P@10 {ranking['precision_at_10']:.4f}, "
            f"R@10 {ranking['recall_at_10']:.4f}, "
            f"NDCG@10 {ranking['ndcg_at_10']:.4f}"
        )

    baseline = metrics["bias_only_baseline_comparison"]
    baseline_rating = baseline["test_rating_prediction"]
    print(
        "Bias-only baseline test: "
        f"MAE {baseline_rating['mae']:.4f}, RMSE {baseline_rating['rmse']:.4f}"
    )
    for label, ranking in (
        (
            "Bias-only restricted",
            baseline["test_ranking"]["restricted_test_candidate_ranking"],
        ),
        (
            "Bias-only full catalogue",
            baseline["test_ranking"]["full_eligible_catalogue_ranking"],
        ),
    ):
        print(
            f"{label}: P@10 {ranking['precision_at_10']:.4f}, "
            f"R@10 {ranking['recall_at_10']:.4f}, "
            f"NDCG@10 {ranking['ndcg_at_10']:.4f}"
        )
    overlap = metrics["recommendation_personalization"]
    print(
        "Recommendation overlap for users "
        f"{overlap['users']}: average shared "
        f"{overlap['average_pairwise_shared_movies']:.2f}/10, "
        f"average Jaccard {overlap['average_pairwise_jaccard']:.4f}"
    )
    print(f"Outputs saved under: {experiment['results_dir']}")


if __name__ == "__main__":
    main()
