"""Build static, precomputed recommendation data for the GitHub Pages demo."""

import csv
import gzip
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
from src.genre_content import GenreContentModel
from src.matrix_factorization import RegularizedMatrixFactorization
from src.pipeline import _tune_model


SITE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ml-1m"
RANDOM_DIR = ROOT / "results" / "randomized_split_seed42"
CHRON_DIR = ROOT / "results"

MODEL_NAMES = (
    ("bias", "Regularized bias baseline", "Popularity + user/movie bias"),
    ("mf_a", "MF A", "Interaction-weighted regularization"),
    ("mf_b", "MF B", "Entity-level regularization"),
    ("genre", "Genre profile", "Mean-centered multi-hot cosine"),
    ("hybrid", "Validation-selected hybrid", "Per-user z-score fusion"),
)
K = 10
USERS_PER_CHUNK = 2000


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_split(path, ratings):
    split = pd.read_csv(path)
    if len(split) != len(ratings) or not split["row_order"].is_unique:
        raise ValueError(f"Split does not contain every source row once: {path}")
    expected = ratings.set_index("row_order")[
        ["user_id", "movie_id", "rating", "timestamp"]
    ].sort_index()
    actual = split.set_index("row_order")[
        ["user_id", "movie_id", "rating", "timestamp"]
    ].sort_index()
    if not expected.astype(str).equals(actual.astype(str)):
        raise ValueError(f"Split rows do not match MovieLens source: {path}")
    return split


def load_mf(directory, model_type):
    directory = Path(directory)
    config = load_json(directory / "config.json")
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
    return model


def load_genre(directory):
    model = GenreContentModel()
    with np.load(Path(directory) / "model_checkpoint.npz", allow_pickle=False) as data:
        for name in (
            "movie_ids",
            "genre_names",
            "movie_genre_matrix",
            "user_ids",
            "user_profiles",
            "user_history_counts",
            "profile_norms",
        ):
            setattr(model, name, data[name].copy())
        model.global_mean = float(data["global_mean"])
        model.movie_genre_norms = np.linalg.norm(model.movie_genre_matrix, axis=1)
    return model


def metric_row(precision, recall, ndcg):
    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
    }


def ranking_metrics(top_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    hits = [int(movie_id in relevant_ids) for movie_id in top_ids[:K]]
    hit_count = sum(hits)
    precision = hit_count / K
    if not relevant_ids:
        return metric_row(precision, None, None)
    recall = hit_count / len(relevant_ids)
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(hits))
    ideal_count = min(len(relevant_ids), K)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return metric_row(precision, recall, dcg / ideal)


def per_user_rankings(models, split, movies, target_partition, candidate_train):
    training_ids = (
        candidate_train.groupby("user_id")["movie_id"].apply(set).to_dict()
    )
    targets = split.loc[split["partition"] == target_partition]
    target_groups = {
        int(user_id): group for user_id, group in targets.groupby("user_id", sort=True)
    }
    catalog = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    result = {}

    for user_id, target in target_groups.items():
        candidates = np.asarray(
            [
                movie_id
                for movie_id in catalog
                if movie_id not in training_ids.get(user_id, set())
            ],
            dtype=np.int64,
        )
        if len(candidates) < K:
            raise ValueError(f"User {user_id} has fewer than {K} candidates.")
        users = np.full(len(candidates), user_id, dtype=np.int64)
        target_ratings = dict(
            zip(target["movie_id"].astype(int), target["rating"].astype(int))
        )
        relevant = {
            movie_id
            for movie_id, rating in target_ratings.items()
            if rating >= 4
        }
        rows = {}
        raw_scores = {}

        for name in ("bias", "mf_a", "mf_b", "genre"):
            model = models[name]
            if name == "genre":
                scores = model.predict_similarity(users, candidates)
            else:
                scores = model.predict_many(users, candidates)
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite {name} score for user {user_id}.")
            raw_scores[name] = scores
            order = np.lexsort((candidates, -scores))[:K]
            top = candidates[order]
            rows[name] = {
                "metrics": ranking_metrics(top, relevant),
                "recommendations": [
                    {
                        "movie_id": int(movie_id),
                        "score": float(scores[index]),
                        "heldout_rating": target_ratings.get(int(movie_id)),
                    }
                    for index, movie_id in zip(order, top)
                ],
            }

        if target_partition == "validation":
            mf_scores = raw_scores["mf_b"]
            genre_scores = raw_scores["genre"]
            mf_z = (mf_scores - mf_scores.mean()) / max(
                float(np.std(mf_scores)), 1e-8
            )
            genre_z = (genre_scores - genre_scores.mean()) / max(
                float(np.std(genre_scores)), 1e-8
            )
            alpha = models["hybrid_alpha"]
            fused = alpha * mf_z + (1.0 - alpha) * genre_z
            order = np.lexsort((candidates, -fused))[:K]
            top = candidates[order]
            rows["hybrid"] = {
                "metrics": ranking_metrics(top, relevant),
                "recommendations": [
                    {
                        "movie_id": int(movie_id),
                        "score": float(fused[index]),
                        "heldout_rating": target_ratings.get(int(movie_id)),
                    }
                    for index, movie_id in zip(order, top)
                ],
            }

        history_count = int((candidate_train["user_id"] == user_id).sum())
        positive_genres = []
        if user_id in set(models["genre"].user_ids.tolist()):
            position = int(np.searchsorted(models["genre"].user_ids, user_id))
            profile = models["genre"].user_profiles[position]
            order = np.argsort(-profile, kind="stable")[:5]
            positive_genres = [
                {
                    "genre": str(models["genre"].genre_names[index]),
                    "preference": float(profile[index]),
                }
                for index in order
                if profile[index] > 0
            ]

        result[str(user_id)] = {
            "history_count": history_count,
            "heldout_count": len(target),
            "relevant_count": len(relevant),
            "positive_genres": positive_genres,
            "models": rows,
        }
    return result


def global_metric_table(split_name):
    result = {}
    if split_name == "chronological":
        paths = {
            "mf_a": CHRON_DIR
            / "regularization_sensitivity"
            / "validation_ranking"
            / "A_interaction_weighted.json",
            "mf_b": CHRON_DIR
            / "regularization_sensitivity"
            / "validation_ranking"
            / "B_entity_level.json",
            "genre": CHRON_DIR / "genre_based" / "validation_metrics.json",
        }
        for key, path in paths.items():
            validation = load_json(path)
            result[key] = {"validation": metric_row(
                validation["precision_at_10"],
                validation["recall_at_10"],
                validation["ndcg_at_10"],
            )}
        baseline = load_json(CHRON_DIR / "metrics.json")
        result["bias"] = {
            "validation": None,
            "test": baseline["ranking_test"]["full_eligible_catalogue_ranking"],
        }
        for key, path in {
            "mf_a": CHRON_DIR
            / "regularization_sensitivity"
            / "A_interaction_weighted"
            / "test_metrics.json",
            "mf_b": CHRON_DIR
            / "regularization_sensitivity"
            / "B_entity_level"
            / "test_metrics.json",
        }.items():
            test = load_json(path)["ranking"]["full_eligible_catalogue_ranking"]
            result[key]["test"] = metric_row(
                test["precision_at_10"], test["recall_at_10"], test["ndcg_at_10"]
            )
        genre = load_json(CHRON_DIR / "genre_based" / "test_metrics.json")
        result["genre"]["test"] = metric_row(
            genre["precision_at_10"],
            genre["recall_at_10"],
            genre["ndcg_at_10"],
        )
        hybrid = []
        for row in csv.DictReader(
            (CHRON_DIR / "hybrid_validation_sweep" / "validation_metrics.csv").open(
                encoding="utf-8-sig", newline=""
            )
        ):
            if row["method"] == "per_user_z_score":
                hybrid.append(
                    {
                        "alpha": float(row["alpha"]),
                        **metric_row(
                            float(row["precision_at_10"]),
                            float(row["recall_at_10"]),
                            float(row["ndcg_at_10"]),
                        ),
                    }
                )
        best = max(hybrid, key=lambda row: row["ndcg_at_10"])
        result["hybrid"] = {"validation": best, "validation_sweep": hybrid, "test": None}
        return result

    metrics = load_json(RANDOM_DIR / "experiment_summary.json")
    validation = {
        row["model"].lower().replace(" ", "_"): metric_row(
            row["precision_at_10"], row["recall_at_10"], row["ndcg_at_10"]
        )
        for row in metrics["validation_full_catalogue_metrics"]
    }
    test = {
        row["model"].lower().replace(" ", "_"): metric_row(
            row["precision_at_10"], row["recall_at_10"], row["ndcg_at_10"]
        )
        for row in metrics["test_full_catalogue_metrics"]
    }
    for key in ("bias", "mf_a", "mf_b", "genre"):
        result[key] = {"validation": validation[key], "test": test[key]}
    hybrid = []
    for row in csv.DictReader(
        (RANDOM_DIR / "hybrid_validation_metrics.csv").open(
            encoding="utf-8-sig", newline=""
        )
    ):
        hybrid.append(
            {
                "alpha": float(row["alpha"]),
                **metric_row(
                    float(row["precision_at_10"]),
                    float(row["recall_at_10"]),
                    float(row["ndcg_at_10"]),
                ),
            }
        )
    result["hybrid"] = {
        "validation": max(hybrid, key=lambda row: row["ndcg_at_10"]),
        "validation_sweep": hybrid,
        "test": None,
    }
    return result


def load_history(split_name):
    if split_name == "chronological":
        path = CHRON_DIR / "genre_based" / "history_bucket_metrics.csv"
        rows = [
            row
            for row in csv.DictReader(path.open(encoding="utf-8-sig", newline=""))
            if row["partition"] == "validation"
        ]
        return [
            {
                "model": "genre",
                "bucket": row["history_length_bucket"],
                "users": int(row["users"]),
                "ndcg_at_10": float(row["ndcg_at_10"]),
            }
            for row in rows
        ]
    path = RANDOM_DIR / "validation_history_buckets.csv"
    return [
        {
            "model": row["model"].lower().replace(" ", "_"),
            "bucket": row["history_bucket"],
            "users": int(row["users"]),
            "ndcg_at_10": float(row["ndcg_at_10"]),
        }
        for row in csv.DictReader(path.open(encoding="utf-8-sig", newline=""))
    ]


def compact_user_record(record, model_order, genre_names):
    genre_index = {name: index for index, name in enumerate(genre_names)}
    preferences = [
        [genre_index[item["genre"]], int(round(item["preference"] * 1000))]
        for item in record["positive_genres"]
    ]
    recommendations = []
    for model_name in model_order:
        model_record = record["models"].get(model_name)
        if not model_record:
            recommendations.append([])
            continue
        recommendations.append(
            [
                [
                    int(item["movie_id"]),
                    int(round(item["score"] * 1000)),
                    int(item["heldout_rating"] or 0),
                ]
                for item in model_record["recommendations"]
            ]
        )
    return [
        int(record["history_count"]),
        int(record["heldout_count"]),
        int(record["relevant_count"]),
        preferences,
        recommendations,
    ]


def build():
    ratings, movies, _users = load_dataset(DATA_DIR)
    movie_lookup = {
        str(int(row.movie_id)): {
            "title": str(row.title),
            "genres": [] if pd.isna(row.genres) else str(row.genres).split("|"),
        }
        for row in movies.itertuples(index=False)
    }
    split_definitions = {
        "chronological": {
            "path": CHRON_DIR / "split_assignments.csv",
            "models_dir": CHRON_DIR / "regularization_sensitivity",
            "genre_dir": CHRON_DIR / "genre_based",
            "alpha": None,
            "description": (
                "Per-user chronological split with equal-timestamp groups kept "
                "together; intended to approximate future recommendations."
            ),
        },
        "randomized": {
            "path": RANDOM_DIR / "split_assignments.csv",
            "models_dir": RANDOM_DIR / "matrix_factorization",
            "genre_dir": RANDOM_DIR / "genre_content",
            "alpha": None,
            "description": (
                "Per-user random 80/10/10 row split, seed 42. Training can include "
                "ratings later than held-out ratings; this is held-out-interaction "
                "evaluation, not a future-time simulation."
            ),
        },
    }
    splits = {}
    chunk_files = []
    model_order = [name for name, _, _ in MODEL_NAMES]
    genre_names = sorted(
        {genre for movie in movie_lookup.values() for genre in movie["genres"]}
    )
    data_dir = SITE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_chunk in data_dir.glob("*.json.gz"):
        old_chunk.unlink()

    for split_name, definition in split_definitions.items():
        split = read_split(definition["path"], ratings)
        training = split.loc[split["partition"] == "train"].copy()
        models = {}
        if split_name == "chronological":
            baseline_config = load_json(CHRON_DIR / "metrics.json")["model"]
            models["bias"] = UserMovieBiasModel(
                lambda_user=baseline_config["lambda_user"],
                lambda_movie=baseline_config["lambda_movie"],
                iterations=baseline_config["iterations"],
            ).fit(training)
        else:
            models["bias"], _ = _tune_model(
                training, split.loc[split["partition"] == "validation"]
            )

        mf_dir = definition["models_dir"]
        models["mf_a"] = load_mf(
            mf_dir / "A_interaction_weighted", RegularizedMatrixFactorization
        )
        models["mf_b"] = load_mf(
            mf_dir / "B_entity_level", EntityRegularizedMatrixFactorization
        )
        models["genre"] = load_genre(definition["genre_dir"])
        global_metrics = global_metric_table(split_name)
        alpha = global_metrics["hybrid"]["validation"]["alpha"]
        models["hybrid_alpha"] = alpha

        targets = {}
        for partition in ("validation", "test"):
            if partition == "test" and split_name in ("chronological", "randomized"):
                # Test labels are included solely to display offline hit diagnostics
                # for standalone models; hybrid recommendations are never scored on test.
                pass
            targets[partition] = per_user_rankings(
                models,
                split,
                movies,
                partition,
                training,
            )
            if partition == "test":
                for user_data in targets[partition].values():
                    user_data["models"].pop("hybrid", None)

        history_counts = training.groupby("user_id").size()
        candidate_counts = {
            "min_training_history": int(history_counts.min()),
            "median_training_history": float(history_counts.median()),
            "mean_training_history": float(history_counts.mean()),
            "max_training_history": int(history_counts.max()),
        }
        if split_name == "chronological":
            split_hash = "f6f5f45d8d4bb8cdebb5e1559537b4e8d32e1f455c650e0f7ad989e693093538"
            split_counts = {
                part: int((split["partition"] == part).sum())
                for part in ("train", "validation", "test")
            }
        else:
            config = load_json(RANDOM_DIR / "experiment_summary.json")
            split_hash = config["split_sha256"]
            split_counts = {
                part: int((split["partition"] == part).sum())
                for part in ("train", "validation", "test")
            }
        split_entry = {
            "description": definition["description"],
            "split_sha256": split_hash,
            "split_counts": split_counts,
            "history_summary": candidate_counts,
            "metrics": global_metrics,
            "history_buckets": load_history(split_name),
            "user_ids": {},
            "chunk_count": {},
        }
        for partition in ("validation", "test"):
            records = targets[partition]
            user_ids = sorted(records, key=int)
            split_entry["user_ids"][partition] = user_ids
            chunks = [
                user_ids[index : index + USERS_PER_CHUNK]
                for index in range(0, len(user_ids), USERS_PER_CHUNK)
            ]
            split_entry["chunk_count"][partition] = len(chunks)
            for chunk_index, user_group in enumerate(chunks):
                compact = {
                    user_id: compact_user_record(
                        records[user_id], model_order, genre_names
                    )
                    for user_id in user_group
                }
                name = f"{split_name}_{partition}_{chunk_index:02d}.json.gz"
                output_path = data_dir / name
                payload = json.dumps(
                    compact, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                with output_path.open("wb") as raw:
                    with gzip.GzipFile(
                        fileobj=raw, mode="wb", compresslevel=9, mtime=0
                    ) as zipped:
                        zipped.write(payload)
                chunk_files.append(output_path)
        splits[split_name] = split_entry

    app_data = {
        "title": "MovieLens Recommendation Lab",
        "dataset": {
            "name": "MovieLens 1M",
            "ratings": 1000209,
            "users": 6040,
            "movies": 3883,
            "rating_scale": [1, 5],
            "relevance_threshold": 4,
            "cutoff": K,
        },
        "model_order": model_order,
        "user_chunk_size": USERS_PER_CHUNK,
        "genre_names": genre_names,
        "models": {
            key: {"name": name, "description": description}
            for key, name, description in MODEL_NAMES
        },
        "movies": movie_lookup,
        "splits": splits,
        "notes": {
            "precision": "Relevant top-10 recommendations divided by a fixed 10.",
            "recall_ndcg": "Users with no relevant held-out ratings are excluded.",
            "ties": "Descending score, then ascending movie ID.",
            "caveat": (
                "Offline held-out ratings are incomplete judgments; unrated "
                "movies are unknown, not necessarily disliked."
            ),
            "hybrid": (
                "Hybrid weights are selected on validation and each frozen hybrid "
                "is evaluated once on test."
            ),
        },
    }
    output = data_dir / "site_index.json"
    output.write_text(
        json.dumps(app_data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    total_size = output.stat().st_size + sum(path.stat().st_size for path in chunk_files)
    print(f"Static index written: {output}")
    print(f"Compressed site data: {total_size / (1024 * 1024):.2f} MiB total")
    print(f"Recommendation chunks: {len(chunk_files)}")
    print(
        "Per-user recommendation panels: "
        + ", ".join(
            f"{name}={len(entry['user_ids']['validation'])} validation/"
            f"{len(entry['user_ids']['test'])} test"
            for name, entry in splits.items()
        )
    )


if __name__ == "__main__":
    build()
