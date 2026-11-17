import argparse
from pathlib import Path

from src.regularization_sensitivity import run_sensitivity_experiment


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Compare MF interaction-weighted and entity-level regularization"
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
        default=ROOT / "results" / "regularization_sensitivity",
    )
    args = parser.parse_args()

    report = run_sensitivity_experiment(
        args.data_dir,
        args.split_path,
        args.results_dir,
    )
    print("Regularization sensitivity experiment complete.")
    print(
        f"Locked split SHA-256: {report['locked_split_sha256']}"
    )
    for name, result in report["formulations"].items():
        config = result["selected_configuration"]
        validation = result["validation"]
        test = result["test_rating_prediction"]
        restricted = result["test_ranking"]["restricted_test_candidate_ranking"]
        catalogue = result["test_ranking"]["full_eligible_catalogue_ranking"]
        overlap = result["recommendation_overlap"]
        print(
            f"\n{name}: lambda={config['regularization']}, "
            f"learning_rate={config['learning_rate']}, "
            f"factors={config['n_factors']}, epoch={config['selected_epoch']}"
        )
        print(
            f"  Validation MAE/RMSE: {validation['mae']:.4f}/"
            f"{validation['rmse']:.4f}"
        )
        print(f"  Test MAE/RMSE: {test['mae']:.4f}/{test['rmse']:.4f}")
        print(
            "  Restricted P/R/NDCG@10: "
            f"{restricted['precision_at_10']:.4f}/"
            f"{restricted['recall_at_10']:.4f}/"
            f"{restricted['ndcg_at_10']:.4f}"
        )
        print(
            "  Catalogue P/R/NDCG@10: "
            f"{catalogue['precision_at_10']:.4f}/"
            f"{catalogue['recall_at_10']:.4f}/"
            f"{catalogue['ndcg_at_10']:.4f}"
        )
        print(
            "  20-user overlap: "
            f"{overlap['average_shared_movies']:.2f}/10 shared; "
            f"Jaccard {overlap['average_jaccard']:.4f}"
        )
    print(f"\nFull report: {args.results_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
