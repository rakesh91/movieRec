import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from src.evaluation import _ranking_values
from src.mf_pipeline import _read_locked_split


LOCKED_SPLIT_SHA256 = (
    "f6f5f45d8d4bb8cdebb5e1559537b4e8d32e1f455c650e0f7ad989e693093538"
)
RELEVANCE_THRESHOLD = 4
RANKING_CUTOFF = 10
HISTORY_BINS = (
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21-50", 21, 50),
    ("51+", 51, None),
)


def _genres_for_movie(value):
    if pd.isna(value) or not str(value).strip():
        return ()
    genres = str(value).split("|")
    return tuple(genre for genre in genres if genre and genre != "(no genres listed)")


class GenreContentModel:
    """Content recommender using signed, training-only genre preference profiles."""

    def fit(self, training_ratings, movies):
        required_ratings = {"user_id", "movie_id", "rating"}
        required_movies = {"movie_id", "title", "genres"}
        if not required_ratings.issubset(training_ratings.columns):
            raise ValueError("Training ratings must include user_id, movie_id, and rating.")
        if not required_movies.issubset(movies.columns):
            raise ValueError("Movies must include movie_id, title, and genres.")
        if training_ratings.empty:
            raise ValueError("Cannot fit genre profiles without training ratings.")
        if movies["movie_id"].duplicated().any():
            raise ValueError("Movie catalogue contains duplicate movie IDs.")

        movie_table = movies.sort_values("movie_id", kind="mergesort").reset_index(drop=True)
        genre_names = sorted(
            {
                genre
                for genres in movie_table["genres"]
                for genre in _genres_for_movie(genres)
            }
        )
        genre_index = {genre: index for index, genre in enumerate(genre_names)}
        genre_matrix = np.zeros(
            (len(movie_table), len(genre_names)), dtype=np.float64
        )
        for movie_position, genres in enumerate(movie_table["genres"]):
            for genre in _genres_for_movie(genres):
                genre_matrix[movie_position, genre_index[genre]] = 1.0

        self.movie_ids = movie_table["movie_id"].to_numpy(dtype=np.int64)
        self.genre_names = np.asarray(genre_names, dtype=str)
        self.movie_genre_matrix = genre_matrix
        self.movie_genre_norms = np.linalg.norm(genre_matrix, axis=1)
        self.global_mean = float(training_ratings["rating"].mean())
        self.user_ids = np.sort(training_ratings["user_id"].unique()).astype(np.int64)
        self.user_history_counts = (
            training_ratings.groupby("user_id").size().reindex(self.user_ids).to_numpy()
        ).astype(np.int64)

        user_positions = np.searchsorted(
            self.user_ids, training_ratings["user_id"].to_numpy(dtype=np.int64)
        )
        movie_positions = np.searchsorted(
            self.movie_ids, training_ratings["movie_id"].to_numpy(dtype=np.int64)
        )
        if (
            np.any(user_positions >= len(self.user_ids))
            or np.any(movie_positions >= len(self.movie_ids))
            or not np.array_equal(self.user_ids[user_positions], training_ratings["user_id"])
            or not np.array_equal(self.movie_ids[movie_positions], training_ratings["movie_id"])
        ):
            raise ValueError("A training user or movie is missing from its catalogue.")

        centered_ratings = (
            training_ratings["rating"].to_numpy(dtype=np.float64) - self.global_mean
        )
        weighted_interactions = coo_matrix(
            (centered_ratings, (user_positions, movie_positions)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        ).tocsr()
        raw_profiles = np.asarray(weighted_interactions @ genre_matrix)
        self.profile_norms = np.linalg.norm(raw_profiles, axis=1)
        self.user_profiles = np.zeros_like(raw_profiles)
        nonzero_profiles = self.profile_norms > 0.0
        self.user_profiles[nonzero_profiles] = (
            raw_profiles[nonzero_profiles]
            / self.profile_norms[nonzero_profiles, np.newaxis]
        )
        self.zero_profile_users = self.user_ids[~nonzero_profiles]
        return self

    def predict_similarity(self, user_ids, movie_ids):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        movie_ids = np.asarray(movie_ids, dtype=np.int64)
        if user_ids.shape != movie_ids.shape:
            raise ValueError("user_ids and movie_ids must have matching shapes.")

        similarities = np.zeros(user_ids.shape, dtype=np.float64)
        if not len(self.user_ids) or not len(self.movie_ids):
            return similarities

        user_positions = np.searchsorted(self.user_ids, user_ids)
        movie_positions = np.searchsorted(self.movie_ids, movie_ids)
        known_users = user_positions < len(self.user_ids)
        known_movies = movie_positions < len(self.movie_ids)
        known_users &= self.user_ids[np.minimum(user_positions, len(self.user_ids) - 1)] == user_ids
        known_movies &= self.movie_ids[np.minimum(movie_positions, len(self.movie_ids) - 1)] == movie_ids
        valid = known_users & known_movies
        if not valid.any():
            return similarities

        profiles = self.user_profiles[user_positions[valid]]
        vectors = self.movie_genre_matrix[movie_positions[valid]]
        denominators = self.movie_genre_norms[movie_positions[valid]]
        numerators = np.einsum("ij,ij->i", profiles, vectors)
        nonzero = denominators > 0.0
        valid_similarities = np.zeros(len(denominators), dtype=np.float64)
        valid_similarities[nonzero] = numerators[nonzero] / denominators[nonzero]
        similarities[valid] = valid_similarities
        return similarities

    def predict_similarity_for_user(self, user_id, movie_ids):
        movie_ids = np.asarray(movie_ids, dtype=np.int64)
        return self.predict_similarity(
            np.full(movie_ids.shape, user_id, dtype=np.int64), movie_ids
        )


def _training_items_by_user(training):
    return training.groupby("user_id")["movie_id"].apply(set).to_dict()


def _rank_candidates(model, user_id, movies, training_items, k=RANKING_CUTOFF):
    rated_ids = training_items.get(int(user_id), set())
    candidates = movies.loc[~movies["movie_id"].isin(rated_ids)].copy()
    if candidates.empty:
        candidates["content_similarity"] = pd.Series(dtype=np.float64)
        return candidates
    candidate_ids = candidates["movie_id"].to_numpy(dtype=np.int64)
    scores = model.predict_similarity_for_user(user_id, candidate_ids)
    order = np.lexsort((candidate_ids, -scores))[:k]
    ranked = candidates.iloc[order].copy()
    ranked["content_similarity"] = scores[order]
    return ranked.reset_index(drop=True)


def _evaluate_partition(
    model, movies, training, target, partition_name, k=RANKING_CUTOFF
):
    training_items = _training_items_by_user(training)
    training_history_counts = training.groupby("user_id").size().to_dict()
    relevant_by_user = (
        target.loc[target["rating"] >= RELEVANCE_THRESHOLD]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    target_by_user = target.groupby("user_id", sort=True)
    metric_rows = []
    recommendation_rows = []
    insufficient_candidates = 0

    for user_id, user_target in target_by_user:
        ranked = _rank_candidates(model, user_id, movies, training_items, k)
        ranked_ids = ranked["movie_id"].astype(int).tolist()
        if len(ranked_ids) < k:
            insufficient_candidates += 1
        precision, recall, ndcg = _ranking_values(
            ranked_ids, relevant_by_user.get(user_id, set()), k
        )
        candidate_count = len(movies) - len(training_items.get(int(user_id), set()))
        metric_rows.append(
            {
                "user_id": int(user_id),
                "training_history_length": int(
                    training_history_counts.get(int(user_id), 0)
                ),
                "target_interactions": int(len(user_target)),
                "relevant_target_movies": int(
                    len(relevant_by_user.get(user_id, set()))
                ),
                "candidate_movies": int(candidate_count),
                "precision_at_10": precision,
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            }
        )
        for rank, row in enumerate(ranked.itertuples(), start=1):
            if int(row.movie_id) in training_items.get(int(user_id), set()):
                raise AssertionError("A training-rated movie entered recommendations.")
            recommendation_rows.append(
                {
                    "user_id": int(user_id),
                    "rank": rank,
                    "movie_id": int(row.movie_id),
                    "title": row.title,
                    "genres": row.genres,
                    "content_similarity": float(row.content_similarity),
                }
            )

    per_user = pd.DataFrame(metric_rows)
    precision_values = per_user["precision_at_10"].to_numpy(dtype=np.float64)
    recall_values = per_user["recall_at_10"].dropna().to_numpy(dtype=np.float64)
    ndcg_values = per_user["ndcg_at_10"].dropna().to_numpy(dtype=np.float64)
    candidate_counts = per_user["candidate_movies"]
    metrics = {
        "partition": partition_name,
        "k": k,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "precision_definition": "relevant recommendations in top 10 divided by 10",
        "candidate_definition": (
            "all catalogue movies excluding that user's training-rated movies"
        ),
        "users_for_precision": int(len(per_user)),
        "users_for_recall_and_ndcg": int(len(recall_values)),
        "users_excluded_no_relevant_items": int(per_user["recall_at_10"].isna().sum()),
        "users_with_fewer_than_10_candidates": int(insufficient_candidates),
        "candidate_movies_per_user": {
            "minimum": int(candidate_counts.min()) if len(candidate_counts) else 0,
            "median": float(candidate_counts.median()) if len(candidate_counts) else 0.0,
            "mean": float(candidate_counts.mean()) if len(candidate_counts) else 0.0,
            "maximum": int(candidate_counts.max()) if len(candidate_counts) else 0,
        },
        "precision_at_10": float(precision_values.mean()) if len(precision_values) else None,
        "recall_at_10": float(recall_values.mean()) if len(recall_values) else None,
        "ndcg_at_10": float(ndcg_values.mean()) if len(ndcg_values) else None,
        "target_interactions": int(len(target)),
    }
    return metrics, per_user, pd.DataFrame(recommendation_rows)


def _history_bucket_metrics(per_user, partition_name):
    rows = []
    for label, minimum, maximum in HISTORY_BINS:
        mask = per_user["training_history_length"] >= minimum
        if maximum is not None:
            mask &= per_user["training_history_length"] <= maximum
        group = per_user.loc[mask]
        rows.append(
            {
                "partition": partition_name,
                "history_length_bucket": label,
                "users": int(len(group)),
                "median_training_history": (
                    float(group["training_history_length"].median())
                    if len(group)
                    else None
                ),
                "users_with_relevant_items": int(group["recall_at_10"].notna().sum()),
                "precision_at_10": (
                    float(group["precision_at_10"].mean()) if len(group) else None
                ),
                "recall_at_10": (
                    float(group["recall_at_10"].mean())
                    if group["recall_at_10"].notna().any()
                    else None
                ),
                "ndcg_at_10": (
                    float(group["ndcg_at_10"].mean())
                    if group["ndcg_at_10"].notna().any()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _profile_frame(model):
    profiles = pd.DataFrame(model.user_profiles, columns=model.genre_names)
    profiles.insert(0, "user_id", model.user_ids)
    profiles.insert(1, "training_history_length", model.user_history_counts)
    profiles.insert(2, "raw_profile_l2_norm", model.profile_norms)
    profiles.insert(3, "zero_profile", model.profile_norms == 0.0)
    return profiles


def _sample_user_analysis(model, movies, training, user_profiles, count=5):
    eligible = user_profiles.loc[
        user_profiles["training_history_length"].gt(0), "user_id"
    ].astype(int)
    if len(eligible) < count:
        raise ValueError(f"Need at least {count} users with training histories.")
    training_items = _training_items_by_user(training)
    samples = []
    for user_id in eligible.head(count):
        profile_position = int(np.searchsorted(model.user_ids, user_id))
        preferences = [
            {"genre": str(genre), "normalized_weight": float(weight)}
            for genre, weight in sorted(
                zip(model.genre_names, model.user_profiles[profile_position]),
                key=lambda pair: (-pair[1], pair[0]),
            )
            if weight > 0.0
        ][:5]
        dislikes = [
            {"genre": str(genre), "normalized_weight": float(weight)}
            for genre, weight in sorted(
                zip(model.genre_names, model.user_profiles[profile_position]),
                key=lambda pair: (pair[1], pair[0]),
            )
            if weight < 0.0
        ][:3]
        recommendations = _rank_candidates(
            model, user_id, movies, training_items, RANKING_CUTOFF
        )
        samples.append(
            {
                "user_id": int(user_id),
                "training_history_length": int(
                    model.user_history_counts[profile_position]
                ),
                "zero_profile": bool(model.profile_norms[profile_position] == 0.0),
                "top_positive_genres": preferences,
                "top_negative_genres": dislikes,
                "recommendations": [
                    {
                        "rank": rank,
                        "movie_id": int(row.movie_id),
                        "title": row.title,
                        "genres": row.genres,
                        "content_similarity": float(row.content_similarity),
                    }
                    for rank, row in enumerate(
                        recommendations.itertuples(), start=1
                    )
                ],
            }
        )
    return samples


def _save_plots(model, history_metrics, results_dir):
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    genre_counts = pd.Series(
        model.movie_genre_matrix.sum(axis=0), index=model.genre_names
    ).sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    genre_counts.plot(kind="barh", ax=axis, color="#4472C4")
    axis.set_xlabel("Movies tagged with genre")
    axis.set_ylabel("Genre")
    axis.set_title("MovieLens 1M catalogue genre coverage")
    figure.tight_layout()
    figure.savefig(figures_dir / "genre_catalogue_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for partition, rows in history_metrics.groupby("partition", sort=True):
        axis.plot(
            rows["history_length_bucket"],
            rows["ndcg_at_10"],
            marker="o",
            label=partition.capitalize(),
        )
    axis.set_xlabel("Training rating count")
    axis.set_ylabel("NDCG@10")
    axis.set_title("Full-catalogue NDCG@10 by training history length")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "ndcg_by_history_length.png", dpi=150)
    plt.close(figure)


def run_genre_experiment(data_dir, split_path, results_dir):
    from src.data_loader import load_dataset

    results_dir = Path(results_dir)
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"Genre recommender output directory is not empty: {results_dir}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, _users = load_dataset(data_dir)
    split = _read_locked_split(split_path, ratings)
    split_hash = hashlib.sha256(Path(split_path).read_bytes()).hexdigest()
    if split_hash != LOCKED_SPLIT_SHA256:
        raise ValueError("Split file does not match the locked MovieLens assignments.")

    training = split.loc[split["partition"] == "train"].copy()
    validation = split.loc[split["partition"] == "validation"].copy()
    if training.empty or validation.empty:
        raise ValueError("The locked split must have nonempty train and validation data.")

    model = GenreContentModel().fit(training, movies)
    profile_frame = _profile_frame(model)
    profile_frame.to_csv(results_dir / "user_profiles.csv", index=False)
    movies_export = pd.DataFrame(
        model.movie_genre_matrix.astype(np.int8), columns=model.genre_names
    )
    movies_export.insert(0, "movie_id", model.movie_ids)
    movies_export.to_csv(results_dir / "movie_genre_vectors.csv", index=False)
    np.savez_compressed(
        results_dir / "model_checkpoint.npz",
        movie_ids=model.movie_ids,
        genre_names=model.genre_names,
        movie_genre_matrix=model.movie_genre_matrix,
        user_ids=model.user_ids,
        user_profiles=model.user_profiles,
        user_history_counts=model.user_history_counts,
        profile_norms=model.profile_norms,
        global_mean=np.asarray(model.global_mean),
    )

    config = {
        "model": "genre-based content recommender",
        "genre_representation": "multi-hot binary movie genre vector",
        "user_profile": (
            "L2-normalize the sum over training ratings of "
            "(rating - training_global_mean) * movie_multi_hot_genres"
        ),
        "scoring": (
            "cosine(user_profile, movie_multi_hot_genres); zero for unknown "
            "users, unknown movies, zero profiles, or movies without genres"
        ),
        "training_global_mean": model.global_mean,
        "rating_data_used_to_build_profiles": "training partition only",
        "candidate_exclusion": "movies rated by that user in training",
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "ranking_cutoff": RANKING_CUTOFF,
        "locked_split_sha256": split_hash,
        "split_counts": {
            name: int((split["partition"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "validation_used_for_model_tuning": False,
        "test_used_for_model_design_or_tuning": False,
        "test_evaluated_after_validation_checks": True,
        "zero_profile_fallback": (
            "all similarities are zero; deterministic movie_id ascending tie-break"
        ),
        "unknown_id_handling": (
            "unknown user or movie receives similarity zero; recommendations "
            "for an unknown user use deterministic catalogue ordering"
        ),
        "genre_names": model.genre_names.tolist(),
        "genreless_movies": int(np.count_nonzero(model.movie_genre_norms == 0.0)),
        "zero_profile_users": int(len(model.zero_profile_users)),
    }
    with (results_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    validation_metrics, validation_per_user, validation_recommendations = (
        _evaluate_partition(
            model, movies, training, validation, "validation"
        )
    )
    validation_per_user.to_csv(
        results_dir / "validation_per_user_metrics.csv", index=False
    )
    validation_recommendations.to_csv(
        results_dir / "validation_recommendations.csv", index=False
    )
    with (results_dir / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(validation_metrics, file, indent=2)

    expected_validation_users = int(validation["user_id"].nunique())
    if validation_metrics["users_for_precision"] != expected_validation_users:
        raise AssertionError("Validation ranking did not evaluate every target user.")
    if validation_metrics["ndcg_at_10"] is None:
        raise AssertionError("Validation NDCG@10 is undefined; refusing test evaluation.")
    if not validation_recommendations.empty:
        training_pairs = set(
            zip(training["user_id"].astype(int), training["movie_id"].astype(int))
        )
        if any(
            (int(row.user_id), int(row.movie_id)) in training_pairs
            for row in validation_recommendations.itertuples()
        ):
            raise AssertionError("Validation recommendations contain training items.")

    test = split.loc[split["partition"] == "test"].copy()
    if test.empty:
        raise ValueError("The locked split has no test data to evaluate.")
    test_metrics, test_per_user, test_recommendations = _evaluate_partition(
        model, movies, training, test, "test"
    )
    test_per_user.to_csv(results_dir / "test_per_user_metrics.csv", index=False)
    test_recommendations.to_csv(
        results_dir / "test_recommendations.csv", index=False
    )
    with (results_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    history_metrics = pd.concat(
        [
            _history_bucket_metrics(validation_per_user, "validation"),
            _history_bucket_metrics(test_per_user, "test"),
        ],
        ignore_index=True,
    )
    history_metrics.to_csv(results_dir / "history_bucket_metrics.csv", index=False)
    _save_plots(model, history_metrics, results_dir)

    samples = _sample_user_analysis(model, movies, training, profile_frame)
    with (results_dir / "sample_user_analysis.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(samples, file, indent=2)
    pd.DataFrame(
        [
            {
                "user_id": sample["user_id"],
                "training_history_length": sample["training_history_length"],
                "rank": recommendation["rank"],
                "movie_id": recommendation["movie_id"],
                "title": recommendation["title"],
                "genres": recommendation["genres"],
                "content_similarity": recommendation["content_similarity"],
                "top_positive_genres": "|".join(
                    genre["genre"] for genre in sample["top_positive_genres"]
                ),
            }
            for sample in samples
            for recommendation in sample["recommendations"]
        ]
    ).to_csv(results_dir / "sample_user_recommendations.csv", index=False)

    genre_catalogue_counts = pd.DataFrame(
        {
            "genre": model.genre_names,
            "movie_count": model.movie_genre_matrix.sum(axis=0).astype(int),
        }
    )
    genre_catalogue_counts.to_csv(
        results_dir / "genre_catalogue_distribution.csv", index=False
    )

    sample_lists = {
        int(sample["user_id"]): {
            int(recommendation["movie_id"])
            for recommendation in sample["recommendations"]
        }
        for sample in samples
    }
    sample_pairwise = []
    sample_user_ids = list(sample_lists)
    for index, first_user in enumerate(sample_user_ids):
        for second_user in sample_user_ids[index + 1 :]:
            first = sample_lists[first_user]
            second = sample_lists[second_user]
            sample_pairwise.append(
                {
                    "user_id_1": first_user,
                    "user_id_2": second_user,
                    "shared_recommendations": len(first & second),
                    "jaccard": len(first & second) / len(first | second),
                }
            )
    pd.DataFrame(sample_pairwise).to_csv(
        results_dir / "sample_user_pairwise_overlap.csv", index=False
    )

    lowest_test_bucket = history_metrics.loc[
        history_metrics["partition"] == "test"
    ].dropna(subset=["ndcg_at_10"]).sort_values("ndcg_at_10")
    failure_analysis = {
        "zero_profile_users": int(len(model.zero_profile_users)),
        "genreless_catalogue_movies": int(
            np.count_nonzero(model.movie_genre_norms == 0.0)
        ),
        "test_history_bucket_with_lowest_ndcg": (
            str(lowest_test_bucket.iloc[0]["history_length_bucket"])
            if not lowest_test_bucket.empty
            else None
        ),
        "test_ndcg_by_history_bucket": {
            str(row.history_length_bucket): (
                float(row.ndcg_at_10) if pd.notna(row.ndcg_at_10) else None
            )
            for row in history_metrics.loc[
                history_metrics["partition"] == "test"
            ].itertuples()
        },
        "sample_users_top_10_overlap": {
            "user_count": len(sample_lists),
            "pair_count": len(sample_pairwise),
            "average_shared_movies": (
                float(np.mean([row["shared_recommendations"] for row in sample_pairwise]))
                if sample_pairwise
                else None
            ),
            "average_jaccard": (
                float(np.mean([row["jaccard"] for row in sample_pairwise]))
                if sample_pairwise
                else None
            ),
        },
        "known_limitations": [
            "Genre vectors cannot distinguish movies that share the same genre tags.",
            "Sparse training histories produce less reliable genre profiles.",
            "A zero profile or genreless movie has no content signal and is ranked by movie ID on ties.",
            "Validation/test movies without a rating of at least 4 are unknown relevance, not confirmed dislikes.",
        ],
    }
    with (results_dir / "failure_analysis.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(failure_analysis, file, indent=2)

    report = {
        "model": config,
        "validation": validation_metrics,
        "test": test_metrics,
        "history_bucket_metrics": history_metrics.to_dict(orient="records"),
        "sample_users": samples,
        "failure_analysis": failure_analysis,
        "results_dir": str(results_dir),
    }
    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate the genre-based MovieLens content recommender"
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "ml-1m")
    parser.add_argument(
        "--split-path",
        type=Path,
        default=root / "results" / "split_assignments.csv",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=root / "results" / "genre_based"
    )
    args = parser.parse_args()

    report = run_genre_experiment(args.data_dir, args.split_path, args.results_dir)
    print("Genre-based content recommender complete.")
    print(f"Locked split SHA-256: {report['model']['locked_split_sha256']}")
    for partition in ("validation", "test"):
        metrics = report[partition]
        print(
            f"{partition.title()} full-catalogue P/R/NDCG@10: "
            f"{metrics['precision_at_10']:.4f}/"
            f"{metrics['recall_at_10']:.4f}/"
            f"{metrics['ndcg_at_10']:.4f} "
            f"(users: {metrics['users_for_precision']:,})"
        )
    print(
        "Users with zero profiles: "
        f"{report['model']['zero_profile_users']:,}; "
        "genreless catalogue movies: "
        f"{report['model']['genreless_movies']:,}"
    )
    print(f"Full report: {args.results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
