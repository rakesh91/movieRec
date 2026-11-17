import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def _column_summary(frame):
    return {
        column: {
            "dtype": str(frame[column].dtype),
            "missing_values": int(frame[column].isna().sum()),
        }
        for column in frame.columns
    }


def run_data_audit(ratings, movies, users, results_dir):
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    user_ids = set(users["user_id"].tolist())
    movie_ids = set(movies["movie_id"].tolist())
    rated_user_ids = set(ratings["user_id"].tolist())
    rated_movie_ids = set(ratings["movie_id"].tolist())
    unique_users = np.union1d(users["user_id"].unique(), ratings["user_id"].unique())
    unique_movies = np.union1d(
        movies["movie_id"].unique(), ratings["movie_id"].unique()
    )
    user_index = np.searchsorted(unique_users, ratings["user_id"].to_numpy())
    movie_index = np.searchsorted(unique_movies, ratings["movie_id"].to_numpy())
    sparse_interactions = coo_matrix(
        (
            np.ones(len(ratings), dtype=np.int8),
            (user_index, movie_index),
        ),
        shape=(len(unique_users), len(unique_movies)),
    ).tocsr()

    rating_counts = ratings["rating"].value_counts().sort_index()
    ratings_per_user = ratings.groupby("user_id").size().rename("rating_count")
    ratings_per_movie = ratings.groupby("movie_id").size().rename("rating_count")
    genre_counts = (
        movies.assign(genre=movies["genres"].str.split("|", regex=False))
        .explode("genre")
        .groupby("genre", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    timestamp_min = int(ratings["timestamp"].min())
    timestamp_max = int(ratings["timestamp"].max())

    report = {
        "users": int(len(users)),
        "movies": int(len(movies)),
        "ratings": int(len(ratings)),
        "columns": {
            "ratings": _column_summary(ratings),
            "movies": _column_summary(movies),
            "users": _column_summary(users),
        },
        "duplicate_records": {
            "ratings_full_rows": int(
                ratings.drop(columns=["row_order"]).duplicated().sum()
            ),
            "user_movie_interactions": int(
                ratings.duplicated(["user_id", "movie_id"]).sum()
            ),
            "movie_ids": int(movies["movie_id"].duplicated().sum()),
            "user_ids": int(users["user_id"].duplicated().sum()),
        },
        "rating_range": {
            "minimum": float(ratings["rating"].min()),
            "maximum": float(ratings["rating"].max()),
        },
        "id_consistency": {
            "rating_users_missing_from_users_file": sorted(
                rated_user_ids - user_ids
            ),
            "users_without_ratings": sorted(user_ids - rated_user_ids),
            "rating_movies_missing_from_movies_file": sorted(
                rated_movie_ids - movie_ids
            ),
            "movies_without_ratings": sorted(movie_ids - rated_movie_ids),
        },
        "timestamp_range": {
            "minimum_epoch_seconds": timestamp_min,
            "maximum_epoch_seconds": timestamp_max,
            "minimum_utc": pd.to_datetime(timestamp_min, unit="s", utc=True).isoformat(),
            "maximum_utc": pd.to_datetime(timestamp_max, unit="s", utc=True).isoformat(),
        },
        "rating_distribution": {
            str(int(key)): int(value) for key, value in rating_counts.items()
        },
        "user_rating_count_summary": {
            "minimum": int(ratings_per_user.min()),
            "median": float(ratings_per_user.median()),
            "maximum": int(ratings_per_user.max()),
        },
        "movie_rating_count_summary": {
            "minimum": int(ratings_per_movie.min()),
            "median": float(ratings_per_movie.median()),
            "maximum": int(ratings_per_movie.max()),
        },
        "user_movie_matrix": {
            "shape": [int(len(unique_users)), int(len(unique_movies))],
            "observed_unique_interactions": int(sparse_interactions.nnz),
            "sparsity": float(
                1.0 - sparse_interactions.nnz / (len(unique_users) * len(unique_movies))
            ),
        },
    }

    with (results_dir / "data_audit.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    rating_counts.rename_axis("rating").rename("count").to_csv(
        results_dir / "rating_distribution.csv"
    )
    genre_counts.rename_axis("genre").rename("movie_count").to_csv(
        results_dir / "genre_distribution.csv"
    )
    ratings_per_user.rename_axis("user_id").to_csv(
        results_dir / "ratings_per_user.csv"
    )
    ratings_per_movie.rename_axis("movie_id").to_csv(
        results_dir / "ratings_per_movie.csv"
    )

    rating_counts.plot(kind="bar", color="#4472C4")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("MovieLens 1M rating distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "rating_distribution.png", dpi=150)
    plt.close()

    genre_counts.sort_values().plot(kind="barh", color="#70AD47")
    plt.xlabel("Number of movies")
    plt.ylabel("Genre")
    plt.title("MovieLens 1M genre distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "genre_distribution.png", dpi=150)
    plt.close()

    ratings_per_user.plot(kind="hist", bins=30, color="#ED7D31")
    plt.xlabel("Ratings per user")
    plt.ylabel("Number of users")
    plt.title("Ratings per user")
    plt.tight_layout()
    plt.savefig(figures_dir / "ratings_per_user.png", dpi=150)
    plt.close()

    ratings_per_movie.plot(kind="hist", bins=30, color="#A5A5A5")
    plt.xlabel("Ratings per movie")
    plt.ylabel("Number of movies")
    plt.title("Ratings per movie")
    plt.tight_layout()
    plt.savefig(figures_dir / "ratings_per_movie.png", dpi=150)
    plt.close()
    return report
