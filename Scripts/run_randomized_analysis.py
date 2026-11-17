import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline_model import UserMovieBiasModel
from src.data_loader import load_dataset
from src.entity_regularized_matrix_factorization import (
    EntityRegularizedMatrixFactorization,
)
from src.evaluation import (
    _ranking_values,
    ranking_metrics,
    rating_prediction_metrics,
)
from src.genre_content import GenreContentModel
from src.matrix_factorization import RegularizedMatrixFactorization
from src.pipeline import _tune_model
from src.regularization_sensitivity import (
    _save_checkpoint,
    _tune_formulation,
)


DATA_DIR = ROOT / "data" / "ml-1m"
OUTPUT = ROOT / "results" / "randomized_split_seed42"
SEED = 42
ALPHAS = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))
K = 10
RELEVANCE_THRESHOLD = 4
STD_FLOOR = 1e-8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(directory, exclude=None):
    directory = Path(directory)
    exclude = Path(exclude).resolve() if exclude else None
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if exclude and (path.resolve() == exclude or exclude in path.resolve().parents):
            continue
        hashes[str(path.relative_to(directory))] = sha256(path)
    return hashes


def randomized_per_user_split(ratings):
    rng = np.random.default_rng(SEED)
    labels = np.full(len(ratings), "", dtype=object)
    ordered = ratings.sort_values("row_order", kind="mergesort")
    for _user_id, group in ordered.groupby("user_id", sort=True):
        positions = group.index.to_numpy()
        positions = rng.permutation(positions)
        count = len(positions)
        train_count = int(math.floor(0.8 * count))
        validation_count = int(math.floor(0.1 * count))
        labels[positions[:train_count]] = "train"
        labels[positions[train_count : train_count + validation_count]] = (
            "validation"
        )
        labels[positions[train_count + validation_count :]] = "test"
    split = ratings.copy()
    split["partition"] = labels
    if not (split["partition"] != "").all():
        raise ValueError("Random split left one or more source rows unassigned.")
    if set(split["partition"]) != {"train", "validation", "test"}:
        raise ValueError("Random split did not populate all three partitions.")
    return split.sort_values("row_order", kind="mergesort").reset_index(drop=True)


def _per_user_validation_metrics(rankings, target):
    relevant = (
        target.loc[target["rating"] >= RELEVANCE_THRESHOLD]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    rows = []
    for user_id, ranked_ids in rankings.items():
        precision, recall, ndcg = _ranking_values(
            ranked_ids, relevant.get(user_id, set()), K
        )
        rows.append(
            {
                "user_id": int(user_id),
                "precision_at_10": precision,
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "precision_at_10": float(frame["precision_at_10"].mean()),
        "recall_at_10": float(frame["recall_at_10"].dropna().mean()),
        "ndcg_at_10": float(frame["ndcg_at_10"].dropna().mean()),
        "users_evaluated": int(len(frame)),
        "users_excluded_zero_relevant": int(frame["recall_at_10"].isna().sum()),
    }, frame


def _rank_model(model, movies, training, target, score_kind="rating"):
    catalogue = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    rated = training.groupby("user_id")["movie_id"].apply(set).to_dict()
    rankings = {}
    candidate_counts = {}
    for user_id in sorted(target["user_id"].unique()):
        candidates = np.asarray(
            [movie_id for movie_id in catalogue if movie_id not in rated.get(user_id, set())],
            dtype=np.int64,
        )
        if len(candidates) < K:
            continue
        user_ids = np.full(len(candidates), int(user_id), dtype=np.int64)
        if score_kind == "genre":
            scores = model.predict_similarity(user_ids, candidates)
        else:
            scores = model.predict_many(user_ids, candidates)
        if not np.isfinite(scores).all():
            raise ValueError(f"{score_kind} scores are not finite for user {user_id}.")
        order = np.lexsort((candidates, -scores))
        rankings[int(user_id)] = candidates[order]
        candidate_counts[int(user_id)] = len(candidates)
    metrics, per_user = _per_user_validation_metrics(rankings, target)
    metrics["candidate_count_summary"] = {
        "minimum": int(min(candidate_counts.values())),
        "median": float(np.median(list(candidate_counts.values()))),
        "mean": float(np.mean(list(candidate_counts.values()))),
        "maximum": int(max(candidate_counts.values())),
    }
    return rankings, metrics, per_user


def _load_mf_checkpoint(directory, model_type):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = model_type(
        n_factors=config["n_factors"],
        regularization=config["regularization"],
        learning_rate=config["learning_rate"],
        random_seed=config["random_seed"],
        batch_size=config["batch_size"],
    )
    with np.load(directory / "model_checkpoint.npz", allow_pickle=False) as data:
        for name in (
            "user_ids",
            "movie_ids",
            "user_biases",
            "movie_biases",
            "user_factors",
            "movie_factors",
        ):
            setattr(model, name, data[name].copy())
        model.global_mean = float(data["global_mean"])
    return model, config


def _validation_hybrid(mf_model, genre_model, movies, training, validation):
    mf_rankings = {}
    genre_rankings = {}
    hybrid_results = {alpha: {} for alpha in ALPHAS}
    catalogue = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    rated = training.groupby("user_id")["movie_id"].apply(set).to_dict()
    candidate_counts = {}

    for user_id in sorted(validation["user_id"].unique()):
        candidates = np.asarray(
            [movie_id for movie_id in catalogue if movie_id not in rated.get(user_id, set())],
            dtype=np.int64,
        )
        if len(candidates) < K:
            continue
        users = np.full(len(candidates), int(user_id), dtype=np.int64)
        mf_scores = mf_model.predict_many(users, candidates)
        genre_scores = genre_model.predict_similarity(users, candidates)
        if not np.isfinite(mf_scores).all() or not np.isfinite(genre_scores).all():
            raise ValueError("Non-finite component scores found in hybrid evaluation.")
        mf_std = float(np.std(mf_scores))
        genre_std = float(np.std(genre_scores))
        mf_z = (mf_scores - float(np.mean(mf_scores))) / max(mf_std, STD_FLOOR)
        genre_z = (genre_scores - float(np.mean(genre_scores))) / max(
            genre_std, STD_FLOOR
        )
        mf_rankings[int(user_id)] = candidates[np.lexsort((candidates, -mf_scores))]
        genre_rankings[int(user_id)] = candidates[
            np.lexsort((candidates, -genre_scores))
        ]
        for alpha in ALPHAS:
            fused = alpha * mf_z + (1.0 - alpha) * genre_z
            order = np.lexsort((candidates, -fused))
            hybrid_results[alpha][int(user_id)] = candidates[order]
        candidate_counts[int(user_id)] = len(candidates)

    users_with_labels = set(validation["user_id"].unique())
    if set(mf_rankings) != users_with_labels:
        raise ValueError("Hybrid ranking user set differs from validation user set.")
    mf_metrics, mf_per_user = _per_user_validation_metrics(mf_rankings, validation)
    genre_metrics, genre_per_user = _per_user_validation_metrics(
        genre_rankings, validation
    )
    rows = []
    for alpha, rankings in hybrid_results.items():
        metrics, _ = _per_user_validation_metrics(rankings, validation)
        rows.append({"alpha": alpha, **metrics})
    return {
        "mf": mf_metrics,
        "genre": genre_metrics,
        "mf_per_user": mf_per_user,
        "genre_per_user": genre_per_user,
        "hybrid": rows,
        "candidate_summary": {
            "minimum": int(min(candidate_counts.values())),
            "median": float(np.median(list(candidate_counts.values()))),
            "mean": float(np.mean(list(candidate_counts.values()))),
            "maximum": int(max(candidate_counts.values())),
        },
        "users": int(len(candidate_counts)),
    }


def main():
    if OUTPUT.exists() and any(
        path.name not in {"run_randomized_analysis.py", "__pycache__"}
        for path in OUTPUT.iterdir()
    ):
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing_results_hashes = tree_hashes(ROOT / "results", exclude=OUTPUT)
    ratings, movies, users = load_dataset(DATA_DIR)

    split = randomized_per_user_split(ratings)
    split_path = OUTPUT / "split_assignments.csv"
    split.to_csv(split_path, index=False)
    split_hash = sha256(split_path)
    training = split.loc[split["partition"] == "train"].copy()
    validation = split.loc[split["partition"] == "validation"].copy()
    test = split.loc[split["partition"] == "test"].copy()
    split_counts = {
        name: {
            "ratings": int((split["partition"] == name).sum()),
            "users": int(split.loc[split["partition"] == name, "user_id"].nunique()),
        }
        for name in ("train", "validation", "test")
    }

    # Tune baseline only on training/validation, then freeze it before test access.
    baseline, baseline_tuning = _tune_model(training, validation)
    baseline_dir = OUTPUT / "bias_baseline"
    baseline_dir.mkdir()
    baseline_tuning.to_csv(baseline_dir / "validation_tuning.csv", index=False)
    baseline_val_rankings, baseline_val_ranking, _ = _rank_model(
        baseline, movies, training, validation
    )
    baseline_val_rating = rating_prediction_metrics(baseline, validation)[0]

    sensitivity_dir = OUTPUT / "matrix_factorization"
    sensitivity_dir.mkdir()
    formulations = {
        "A_interaction_weighted": RegularizedMatrixFactorization,
        "B_entity_level": EntityRegularizedMatrixFactorization,
    }
    mf_models = {}
    mf_configs = {}
    for name, model_type in formulations.items():
        model, tuning, history = _tune_formulation(model_type, training, validation)
        directory = sensitivity_dir / name
        directory.mkdir()
        tuning.to_csv(directory / "validation_tuning.csv", index=False)
        history.to_csv(directory / "training_history.csv", index=False)
        selected_history = history.loc[
            (history["regularization"] == model.regularization)
            & (history["learning_rate"] == model.learning_rate)
        ]
        selected_history.to_csv(
            directory / "selected_training_history.csv", index=False
        )
        config = _save_checkpoint(model, directory)
        config.update(
            {
                "formulation": name,
                "selection_metric": "validation RMSE",
                "test_used_for_selection": False,
                "randomized_split_sha256": split_hash,
            }
        )
        (directory / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        mf_models[name] = model
        mf_configs[name] = config
    mf_a_dir = sensitivity_dir / "A_interaction_weighted"
    mf_b_dir = sensitivity_dir / "B_entity_level"
    mf_a, mf_b = (
        mf_models["A_interaction_weighted"],
        mf_models["B_entity_level"],
    )
    mf_a_config, mf_b_config = (
        mf_configs["A_interaction_weighted"],
        mf_configs["B_entity_level"],
    )

    genre_model = GenreContentModel().fit(training, movies)
    genre_dir = OUTPUT / "genre_content"
    genre_dir.mkdir()
    np.savez_compressed(
        genre_dir / "model_checkpoint.npz",
        movie_ids=genre_model.movie_ids,
        genre_names=genre_model.genre_names,
        movie_genre_matrix=genre_model.movie_genre_matrix,
        user_ids=genre_model.user_ids,
        user_profiles=genre_model.user_profiles,
        user_history_counts=genre_model.user_history_counts,
        profile_norms=genre_model.profile_norms,
        global_mean=np.asarray(genre_model.global_mean),
    )
    genre_val_rankings, genre_val_ranking, genre_val_per_user = _rank_model(
        genre_model, movies, training, validation, score_kind="genre"
    )
    genre_val_per_user.to_csv(
        genre_dir / "validation_per_user_metrics.csv", index=False
    )

    mf_a_val_rankings, mf_a_val_ranking, mf_a_val_per_user = _rank_model(
        mf_a, movies, training, validation
    )
    mf_b_val_rankings, mf_b_val_ranking, mf_b_val_per_user = _rank_model(
        mf_b, movies, training, validation
    )
    mf_a_val_per_user.to_csv(mf_a_dir / "validation_per_user_ranking.csv", index=False)
    mf_b_val_per_user.to_csv(mf_b_dir / "validation_per_user_ranking.csv", index=False)

    hybrid = _validation_hybrid(mf_b, genre_model, movies, training, validation)
    pd.DataFrame(hybrid["hybrid"]).to_csv(
        OUTPUT / "hybrid_validation_metrics.csv", index=False
    )

    # All choices are frozen before evaluating any test labels.
    baseline_test_rating = rating_prediction_metrics(baseline, test)[0]
    baseline_test_ranking = ranking_metrics(baseline, movies, split, k=K)[
        "full_eligible_catalogue_ranking"
    ]
    mf_a_test_rating, _ = rating_prediction_metrics(mf_a, test)
    mf_a_test_ranking = ranking_metrics(mf_a, movies, split, k=K)[
        "full_eligible_catalogue_ranking"
    ]
    mf_b_test_rating, _ = rating_prediction_metrics(mf_b, test)
    mf_b_test_ranking = ranking_metrics(mf_b, movies, split, k=K)[
        "full_eligible_catalogue_ranking"
    ]
    genre_test_rankings, genre_test_ranking, genre_test_per_user = _rank_model(
        genre_model, movies, training, test, score_kind="genre"
    )
    genre_test_rating = {
        "note": "Genre model has no rating-prediction score; ranking-only method."
    }
    genre_test_per_user.to_csv(
        genre_dir / "test_per_user_metrics.csv", index=False
    )
    genre_test = {
        "rating_prediction": genre_test_rating,
        "full_eligible_catalogue_ranking": genre_test_ranking,
    }
    (genre_dir / "test_metrics.json").write_text(
        json.dumps(genre_test, indent=2), encoding="utf-8"
    )

    # History-bucket breakdown for both MF B and genre on validation.
    history_counts = training.groupby("user_id").size().to_dict()
    relevant_users = set(
        validation.loc[validation["rating"] >= RELEVANCE_THRESHOLD, "user_id"]
    )
    bucket_rows = []
    for model_name, per_user in (
        ("MF B", mf_b_val_per_user),
        ("Genre", genre_val_per_user),
    ):
        frame = per_user.copy()
        frame["history_length"] = frame["user_id"].map(history_counts)
        frame["bucket"] = pd.cut(
            frame["history_length"],
            bins=[0, 5, 20, 50, np.inf],
            labels=["1-5", "6-20", "21-50", "51+"],
        )
        for bucket, group in frame.groupby("bucket", observed=False):
            if group.empty:
                continue
            recall_users = group.loc[group["recall_at_10"].notna()]
            ndcg_users = group.loc[group["ndcg_at_10"].notna()]
            bucket_rows.append(
                {
                    "model": model_name,
                    "history_bucket": str(bucket),
                    "users": int(len(group)),
                    "users_with_relevant_validation": int(
                        group["user_id"].isin(relevant_users).sum()
                    ),
                    "precision_at_10": float(group["precision_at_10"].mean()),
                    "recall_at_10": (
                        float(recall_users["recall_at_10"].mean())
                        if len(recall_users)
                        else None
                    ),
                    "ndcg_at_10": (
                        float(ndcg_users["ndcg_at_10"].mean())
                        if len(ndcg_users)
                        else None
                    ),
                }
            )
    pd.DataFrame(bucket_rows).to_csv(
        OUTPUT / "validation_history_buckets.csv", index=False
    )

    validation_rows = [
        {"model": "Bias", **baseline_val_ranking},
        {"model": "MF A", **mf_a_val_ranking},
        {"model": "MF B", **mf_b_val_ranking},
        {"model": "Genre", **genre_val_ranking},
    ]
    test_rows = [
        {
            "model": "Bias",
            **baseline_test_ranking,
            "rating_rmse": baseline_test_rating["rmse"],
        },
        {
            "model": "MF A",
            **mf_a_test_ranking,
            "rating_rmse": mf_a_test_rating["rmse"],
        },
        {
            "model": "MF B",
            **mf_b_test_ranking,
            "rating_rmse": mf_b_test_rating["rmse"],
        },
        {
            "model": "Genre",
            **genre_test_ranking,
            "rating_rmse": None,
        },
    ]
    pd.DataFrame(validation_rows).to_csv(
        OUTPUT / "validation_metrics.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(OUTPUT / "test_metrics.csv", index=False)

    best_hybrid = max(hybrid["hybrid"], key=lambda row: row["ndcg_at_10"])
    config = {
        "experiment": "random per-user 80/10/10 MovieLens split",
        "split_method": (
            "Within each user, shuffle rating-row positions using NumPy "
            "default_rng(seed=42); assign floor(0.8*n) to train, "
            "floor(0.1*n) to validation, remainder to test. Timestamps are "
            "not used to order or assign rows."
        ),
        "seed": SEED,
        "split_sha256": split_hash,
        "split_counts": split_counts,
        "source_row_count": len(ratings),
        "user_count": int(ratings["user_id"].nunique()),
        "movies_count": int(movies["movie_id"].nunique()),
        "per_user_min_partition_counts": {
            part: int(split.groupby("user_id")["partition"].apply(lambda x: (x == part).sum()).min())
            for part in ("train", "validation", "test")
        },
        "candidate_definition": "full movie catalogue excluding that user's training-rated movies",
        "relevance_threshold": "rating >= 4",
        "precision_denominator": 10,
        "zero_relevant_users": "included in Precision; excluded from Recall and NDCG",
        "tie_break": "descending score, then ascending movie_id",
        "test_used_for_tuning_or_selection": False,
        "chronological_order_used_for_split": False,
        "known_interpretation_change": (
            "Random per-user holdout evaluates held-out preference prediction, "
            "not future recommendation; training may contain ratings later "
            "than validation/test timestamps."
        ),
        "selected_mf_b": mf_b_config,
        "selected_mf_a": mf_a_config,
        "hybrid": {
            "formula": "alpha * per-user MF B z-score + (1-alpha) * per-user genre z-score",
            "alpha_grid": list(ALPHAS),
            "population_std_floor": STD_FLOOR,
            "validation_best_alpha": best_hybrid["alpha"],
            "validation_best_ndcg_at_10": best_hybrid["ndcg_at_10"],
            "test_evaluation_performed_for_hybrid": False,
        },
        "validation_full_catalogue_metrics": validation_rows,
        "test_full_catalogue_metrics": test_rows,
        "baseline_validation_rating_prediction": baseline_val_rating,
        "baseline_test_rating_prediction": baseline_test_rating,
        "existing_results_unchanged": None,
    }
    config_path = OUTPUT / "experiment_summary.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    paper_comparison = [
        "# Randomized-split comparison for the paper",
        "",
        "This is a supplementary experiment; the original chronological split,",
        "paper draft, checkpoints, and results were not changed.",
        "",
        f"- Split: random per-user row assignment, 80/10/10, seed {SEED}.",
        f"- Split SHA-256: `{split_hash}`.",
        "- Unlike the main chronological analysis, this is a random holdout:",
        "  training ratings can be later in time than held-out ratings. Treat it",
        "  as an interpolation/held-out interaction comparison, not a future-use",
        "  simulation.",
        "- Models were selected using training/validation only. Test metrics",
        "  were calculated only after model and hybrid selection; hybrid test",
        "  evaluation was not performed.",
        "- User-level histories and candidate catalogues differ from the",
        "  chronological experiment; scores should be compared within each",
        "  split rather than interpreted as a controlled temporal improvement.",
        "",
        "## Validation full-catalogue ranking",
        "",
        "| Model | P@10 | Recall@10 | NDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in validation_rows:
        paper_comparison.append(
            f"| {row['model']} | {row['precision_at_10']:.5f} | "
            f"{row['recall_at_10']:.5f} | {row['ndcg_at_10']:.5f} |"
        )
    paper_comparison += [
        "",
        "## Test full-catalogue ranking (standalone models only)",
        "",
        "| Model | P@10 | Recall@10 | NDCG@10 | Rating RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        rmse = "n/a" if row["rating_rmse"] is None else f"{row['rating_rmse']:.5f}"
        paper_comparison.append(
            f"| {row['model']} | {row['precision_at_10']:.5f} | "
            f"{row['recall_at_10']:.5f} | {row['ndcg_at_10']:.5f} | {rmse} |"
        )
    paper_comparison += [
        "",
        "## Validation-only hybrid",
        "",
        f"Best z-score alpha: {best_hybrid['alpha']:.1f}; "
        f"NDCG@10={best_hybrid['ndcg_at_10']:.5f}.",
        "Hybrid test metrics were not calculated. Refer to",
        "`hybrid_validation_metrics.csv` for the alpha grid and",
        "`validation_history_buckets.csv` for same-split history results.",
    ]
    (OUTPUT / "paper_comparison.md").write_text(
        "\n".join(paper_comparison) + "\n", encoding="utf-8"
    )

    after_hashes = tree_hashes(ROOT / "results", exclude=OUTPUT)
    integrity = {
        "randomized_split_sha256": split_hash,
        "split_assignments_rows": int(len(split)),
        "split_partition_counts": {
            name: int((split["partition"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "split_user_partition_minimums": config["per_user_min_partition_counts"],
        "chronological_order_used_for_split": False,
        "existing_results_before_match_after": existing_results_hashes
        == after_hashes,
        "changed_existing_results": sorted(
            key
            for key in set(existing_results_hashes) | set(after_hashes)
            if existing_results_hashes.get(key) != after_hashes.get(key)
        ),
        "mf_checkpoints_created_only_in_new_output": True,
        "genre_checkpoint_created_only_in_new_output": True,
        "hybrid_test_evaluated": False,
        "test_used_for_model_selection": False,
    }
    (OUTPUT / "integrity_checks.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    if not integrity["existing_results_before_match_after"]:
        raise RuntimeError("An existing results file changed during this run.")
    print(f"Randomized-split experiment complete: {OUTPUT}")
    print(f"Split SHA-256: {split_hash}")
    print(f"Best validation hybrid alpha: {best_hybrid['alpha']:.1f}")


if __name__ == "__main__":
    main()
