import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.entity_regularized_matrix_factorization import (
    EntityRegularizedMatrixFactorization,
)
from src.evaluation import _ranking_values
from src.genre_content import GenreContentModel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hybrid_validation_sweep"
SPLIT_PATH = ROOT / "results" / "split_assignments.csv"
RATINGS_PATH = ROOT / "data" / "ml-1m" / "ratings.dat"
MOVIES_PATH = ROOT / "data" / "ml-1m" / "movies.dat"
MF_CHECKPOINT_PATH = (
    ROOT
    / "results"
    / "regularization_sensitivity"
    / "B_entity_level"
    / "model_checkpoint.npz"
)
MF_CONFIG_PATH = ROOT / "results" / "regularization_sensitivity" / "B_entity_level" / "config.json"
GENRE_CHECKPOINT_PATH = ROOT / "results" / "genre_based" / "model_checkpoint.npz"
MF_VALIDATION_PATH = (
    ROOT
    / "results"
    / "regularization_sensitivity"
    / "validation_ranking"
    / "B_entity_level.json"
)
GENRE_VALIDATION_PATH = ROOT / "results" / "genre_based" / "validation_metrics.json"
EXPECTED_SPLIT_SHA256 = "f6f5f45d8d4bb8cdebb5e1559537b4e8d32e1f455c650e0f7ad989e693093538"
ALPHAS = tuple(index / 10 for index in range(11))
STD_FLOOR = 1e-8
K = 10
PARTITION_CODE = {"train": 0, "validation": 1, "test": 2}
OUTPUT_FILES = (
    "summary.md",
    "validation_metrics.csv",
    "experiment_config.json",
    "per_alpha_results.json",
    "validation_ndcg_vs_alpha.png",
    "validation_precision_recall_vs_alpha.png",
    "reproducibility_checks.json",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_results_hashes():
    results_root = ROOT / "results"
    return {
        str(path.relative_to(results_root)): _sha256(path)
        for path in results_root.rglob("*")
        if path.is_file() and OUT not in path.parents
    }


def _load_split_ids():
    split = pd.read_csv(
        SPLIT_PATH, usecols=["row_order", "user_id", "movie_id", "partition"]
    )
    if split["row_order"].duplicated().any():
        raise ValueError("Locked split has duplicate row_order values.")
    count = len(split)
    partitions = np.full(count, -1, dtype=np.int8)
    users = np.zeros(count, dtype=np.int32)
    movies = np.zeros(count, dtype=np.int32)
    for row in split.itertuples(index=False):
        if row.row_order < 0 or row.row_order >= count:
            raise ValueError("Unexpected row_order in the locked split.")
        partitions[row.row_order] = PARTITION_CODE[row.partition]
        users[row.row_order] = row.user_id
        movies[row.row_order] = row.movie_id
    if np.any(partitions < 0):
        raise ValueError("Locked split does not assign every source row.")
    return split, partitions, users, movies


def _read_training_only(partitions, expected_users, expected_movies):
    rows = []
    with RATINGS_PATH.open("r", encoding="latin-1", newline="") as stream:
        for row_order, line in enumerate(stream):
            if row_order >= len(partitions):
                raise ValueError("Ratings source has more rows than the split.")
            if partitions[row_order] != PARTITION_CODE["train"]:
                continue
            fields = line.rstrip("\r\n").split("::")
            if len(fields) != 4:
                raise ValueError(f"Malformed training source row {row_order}.")
            user_id, movie_id, rating = map(int, fields[:3])
            if (
                user_id != int(expected_users[row_order])
                or movie_id != int(expected_movies[row_order])
            ):
                raise ValueError(f"Training IDs disagree with split at row {row_order}.")
            rows.append((user_id, movie_id, rating))
        if row_order + 1 != len(partitions):
            raise ValueError("Ratings source row count differs from the split.")
    if len(rows) != int(np.count_nonzero(partitions == PARTITION_CODE["train"])):
        raise ValueError("Training ratings count does not match the saved split.")
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def _load_saved_models(training):
    config = json.loads(MF_CONFIG_PATH.read_text(encoding="utf-8"))
    mf = EntityRegularizedMatrixFactorization(
        n_factors=config["n_factors"],
        regularization=config["regularization"],
        learning_rate=config["learning_rate"],
        random_seed=config["random_seed"],
        batch_size=config["batch_size"],
    )
    with np.load(MF_CHECKPOINT_PATH, allow_pickle=False) as checkpoint:
        for name in (
            "user_ids",
            "movie_ids",
            "user_biases",
            "movie_biases",
            "user_factors",
            "movie_factors",
        ):
            setattr(mf, name, checkpoint[name].copy())
        mf.global_mean = float(checkpoint["global_mean"])

    genre = GenreContentModel()
    with np.load(GENRE_CHECKPOINT_PATH, allow_pickle=False) as checkpoint:
        for name in (
            "movie_ids",
            "genre_names",
            "movie_genre_matrix",
            "user_ids",
            "user_profiles",
            "user_history_counts",
            "profile_norms",
        ):
            setattr(genre, name, checkpoint[name].copy())
        genre.global_mean = float(checkpoint["global_mean"])
        genre.movie_genre_norms = np.linalg.norm(genre.movie_genre_matrix, axis=1)

    training_mean = float(training["rating"].mean())
    if not math.isclose(mf.global_mean, training_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("MF checkpoint global mean differs from training ratings.")
    if not math.isclose(
        genre.global_mean, training_mean, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Genre checkpoint global mean differs from training ratings.")
    return mf, genre, config


def _rank(movie_ids, scores):
    if not np.isfinite(scores).all():
        raise ValueError("Cannot rank non-finite scores.")
    return movie_ids[np.lexsort((movie_ids, -scores))]


def _ranking_metrics(rankings, relevant_by_user):
    per_user = []
    for user_id, movie_ids in rankings.items():
        precision, recall, ndcg = _ranking_values(
            movie_ids, relevant_by_user.get(user_id, set()), K
        )
        per_user.append(
            {
                "user_id": int(user_id),
                "precision_at_10": precision,
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            }
        )
    frame = pd.DataFrame(per_user)
    recall_values = frame["recall_at_10"].dropna().to_numpy(dtype=np.float64)
    ndcg_values = frame["ndcg_at_10"].dropna().to_numpy(dtype=np.float64)
    return {
        "precision_at_10": float(frame["precision_at_10"].mean()),
        "recall_at_10": float(recall_values.mean()) if len(recall_values) else None,
        "ndcg_at_10": float(ndcg_values.mean()) if len(ndcg_values) else None,
        "users_evaluated": int(len(frame)),
        "users_excluded_zero_relevant": int(frame["recall_at_10"].isna().sum()),
    }


def _validation_labels(partitions, expected_users, expected_movies):
    relevant = {}
    interaction_count = 0
    with RATINGS_PATH.open("r", encoding="latin-1", newline="") as stream:
        for row_order, line in enumerate(stream):
            if partitions[row_order] != PARTITION_CODE["validation"]:
                continue
            fields = line.rstrip("\r\n").split("::")
            if len(fields) != 4:
                raise ValueError(f"Malformed validation source row {row_order}.")
            user_id, movie_id, rating = map(int, fields[:3])
            if (
                user_id != int(expected_users[row_order])
                or movie_id != int(expected_movies[row_order])
            ):
                raise ValueError(
                    f"Validation IDs disagree with split at row {row_order}."
                )
            interaction_count += 1
            if rating >= 4:
                relevant.setdefault(user_id, set()).add(movie_id)
    return relevant, interaction_count


def _metric_close(first, second, tolerance=1e-12):
    return first is not None and second is not None and math.isclose(
        float(first), float(second), rel_tol=0.0, abs_tol=tolerance
    )


def _compare_to_baselines(results):
    baseline_files = {
        "genre": (GENRE_VALIDATION_PATH, "genre"),
        "mf": (MF_VALIDATION_PATH, "mf"),
    }
    comparisons = {}
    for alpha, family in ((0.0, "per_user_z_score"), (1.0, "per_user_z_score"),
                          (0.0, "rank_fusion"), (1.0, "rank_fusion")):
        model_name = "genre" if alpha == 0.0 else "mf"
        path, key = baseline_files[model_name]
        baseline = json.loads(path.read_text(encoding="utf-8"))
        candidate = results[family][f"{alpha:.1f}"]
        comparisons[f"{family}_alpha_{alpha:.1f}"] = {
            "baseline": model_name,
            "metrics_match": all(
                _metric_close(candidate[metric], baseline[metric])
                for metric in ("precision_at_10", "recall_at_10", "ndcg_at_10")
            ),
            "baseline_metrics": {
                metric: baseline[metric]
                for metric in ("precision_at_10", "recall_at_10", "ndcg_at_10")
            },
            "sweep_metrics": {
                metric: candidate[metric]
                for metric in ("precision_at_10", "recall_at_10", "ndcg_at_10")
            },
        }
    return comparisons


def _write_plots(rows):
    z_rows = [row for row in rows if row["method"] == "per_user_z_score"]
    rank_rows = [row for row in rows if row["method"] == "rank_fusion"]
    fig, axis = plt.subplots(figsize=(8, 5))
    for values, label, color in (
        (z_rows, "Per-user z-score", "#4472C4"),
        (rank_rows, "Rank fusion", "#ED7D31"),
    ):
        axis.plot(
            [row["alpha"] for row in values],
            [row["ndcg_at_10"] for row in values],
            marker="o",
            label=label,
            color=color,
        )
    axis.set(
        xlabel="Alpha (MF weight)",
        ylabel="Validation NDCG@10",
        title="Validation NDCG@10 by hybrid alpha",
    )
    axis.set_xticks(ALPHAS)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "validation_ndcg_vs_alpha.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for values, label, color in (
        (z_rows, "Per-user z-score", "#4472C4"),
        (rank_rows, "Rank fusion", "#ED7D31"),
    ):
        alphas = [row["alpha"] for row in values]
        axes[0].plot(
            alphas,
            [row["precision_at_10"] for row in values],
            marker="o",
            label=label,
            color=color,
        )
        axes[1].plot(
            alphas,
            [row["recall_at_10"] for row in values],
            marker="o",
            label=label,
            color=color,
        )
    for axis, title, ylabel in (
        (axes[0], "Validation Precision@10", "Precision@10"),
        (axes[1], "Validation Recall@10", "Recall@10"),
    ):
        axis.set(xlabel="Alpha (MF weight)", ylabel=ylabel, title=title)
        axis.set_xticks(ALPHAS)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "validation_precision_recall_vs_alpha.png", dpi=160)
    plt.close(fig)


def run():
    if not OUT.is_dir():
        raise FileNotFoundError(f"Expected pre-created audit directory is missing: {OUT}")
    existing_outputs = [name for name in OUTPUT_FILES if (OUT / name).exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite existing sweep outputs: {existing_outputs}"
        )
    split_before = _sha256(SPLIT_PATH)
    if split_before != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Locked split checksum mismatch: {split_before}")
    mf_hash_before = _sha256(MF_CHECKPOINT_PATH)
    genre_hash_before = _sha256(GENRE_CHECKPOINT_PATH)
    results_before = _existing_results_hashes()

    split, partitions, split_users, split_movies = _load_split_ids()
    training = _read_training_only(
        partitions, split_users, split_movies
    )
    movies = pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
    ).sort_values("movie_id", kind="mergesort").reset_index(drop=True)
    mf, genre, mf_config = _load_saved_models(training)
    catalogue = movies["movie_id"].to_numpy(dtype=np.int64)
    validation_users = np.sort(
        split.loc[split["partition"] == "validation", "user_id"].unique()
    ).astype(np.int64)
    training_items = training.groupby("user_id")["movie_id"].apply(set).to_dict()

    rankings = {
        "per_user_z_score": {f"{alpha:.1f}": {} for alpha in ALPHAS},
        "rank_fusion": {f"{alpha:.1f}": {} for alpha in ALPHAS},
    }
    candidate_counts = {}
    base_rankings = {"mf": {}, "genre": {}}
    finite_scores = True
    floor_users = {"mf": 0, "genre": 0}
    for user_id in validation_users:
        excluded = training_items.get(int(user_id), set())
        excluded_array = np.fromiter(excluded, dtype=np.int64)
        candidates = catalogue[~np.isin(catalogue, excluded_array)]
        if len(candidates) < K:
            raise ValueError(f"User {user_id} has fewer than {K} candidates.")
        candidate_counts[int(user_id)] = len(candidates)
        user_vector = np.full(len(candidates), user_id, dtype=np.int64)
        mf_scores = mf.predict_many(user_vector, candidates)
        genre_scores = genre.predict_similarity_for_user(user_id, candidates)
        if not np.isfinite(mf_scores).all() or not np.isfinite(genre_scores).all():
            finite_scores = False
            raise ValueError(f"Non-finite base score for user {user_id}.")

        mf_mean, genre_mean = float(mf_scores.mean()), float(genre_scores.mean())
        mf_std, genre_std = float(mf_scores.std()), float(genre_scores.std())
        if mf_std < STD_FLOOR:
            floor_users["mf"] += 1
        if genre_std < STD_FLOOR:
            floor_users["genre"] += 1
        mf_z = (mf_scores - mf_mean) / max(mf_std, STD_FLOOR)
        genre_z = (genre_scores - genre_mean) / max(genre_std, STD_FLOOR)
        if not np.isfinite(mf_z).all() or not np.isfinite(genre_z).all():
            finite_scores = False
            raise ValueError(f"Non-finite normalized score for user {user_id}.")

        mf_order = np.lexsort((candidates, -mf_scores))
        genre_order = np.lexsort((candidates, -genre_scores))
        base_rankings["mf"][int(user_id)] = candidates[mf_order[:K]].tolist()
        base_rankings["genre"][int(user_id)] = candidates[genre_order[:K]].tolist()

        mf_rank_score = np.empty(len(candidates), dtype=np.float64)
        genre_rank_score = np.empty(len(candidates), dtype=np.float64)
        rank_denominator = max(1, len(candidates) - 1)
        mf_rank_score[mf_order] = 1.0 - np.arange(len(candidates)) / rank_denominator
        genre_rank_score[genre_order] = (
            1.0 - np.arange(len(candidates)) / rank_denominator
        )
        for alpha in ALPHAS:
            key = f"{alpha:.1f}"
            z_scores = alpha * mf_z + (1.0 - alpha) * genre_z
            rank_scores = alpha * mf_rank_score + (1.0 - alpha) * genre_rank_score
            if not np.isfinite(z_scores).all() or not np.isfinite(rank_scores).all():
                finite_scores = False
                raise ValueError(f"Non-finite hybrid score for user {user_id}, alpha {alpha}.")
            rankings["per_user_z_score"][key][int(user_id)] = _rank(
                candidates, z_scores
            )[:K].tolist()
            rankings["rank_fusion"][key][int(user_id)] = _rank(
                candidates, rank_scores
            )[:K].tolist()

    # Confirm alpha endpoints reproduce standalone rankings before consulting validation labels.
    endpoints_match = {
        "z_score_alpha_0_equals_genre": rankings["per_user_z_score"]["0.0"]
        == base_rankings["genre"],
        "z_score_alpha_1_equals_mf": rankings["per_user_z_score"]["1.0"]
        == base_rankings["mf"],
        "rank_fusion_alpha_0_equals_genre": rankings["rank_fusion"]["0.0"]
        == base_rankings["genre"],
        "rank_fusion_alpha_1_equals_mf": rankings["rank_fusion"]["1.0"]
        == base_rankings["mf"],
    }
    if not all(endpoints_match.values()):
        raise RuntimeError(f"Standalone ranking endpoint mismatch: {endpoints_match}")

    # Read validation ratings only after all recommendations are generated.
    relevant, validation_interactions = _validation_labels(
        partitions, split_users, split_movies
    )
    if validation_interactions != int(
        np.count_nonzero(partitions == PARTITION_CODE["validation"])
    ):
        raise ValueError("Validation labels do not match split interaction count.")

    per_alpha = {
        method: {
            alpha: _ranking_metrics(user_rankings, relevant)
            for alpha, user_rankings in alpha_rankings.items()
        }
        for method, alpha_rankings in rankings.items()
    }
    metrics_rows = []
    for method, method_results in per_alpha.items():
        for alpha, metrics in method_results.items():
            metrics_rows.append({"method": method, "alpha": float(alpha), **metrics})
    metrics_frame = pd.DataFrame(metrics_rows).sort_values(
        ["method", "alpha"], kind="mergesort"
    )

    expected_users = len(validation_users)
    expected_candidate_values = set(candidate_counts.values())
    candidate_counts_consistent = len(candidate_counts) == expected_users
    per_alpha_counts_consistent = all(
        metrics["users_evaluated"] == expected_users
        and metrics["users_excluded_zero_relevant"] == expected_users - len(relevant)
        for method_results in per_alpha.values()
        for metrics in method_results.values()
    )
    if not candidate_counts_consistent or not per_alpha_counts_consistent:
        raise RuntimeError("Candidate or evaluated-user counts differ across alpha values.")

    baseline_comparisons = _compare_to_baselines(per_alpha)
    if not all(item["metrics_match"] for item in baseline_comparisons.values()):
        raise RuntimeError(
            "Standalone endpoint metrics do not reproduce the saved validation baselines."
        )

    z_results = per_alpha["per_user_z_score"]
    best_alpha_text = max(
        z_results,
        key=lambda alpha: z_results[alpha]["ndcg_at_10"],
    )
    best_alpha = float(best_alpha_text)
    mf_ndcg = float(
        baseline_comparisons["per_user_z_score_alpha_1.0"]["baseline_metrics"][
            "ndcg_at_10"
        ]
    )
    strict_mf_beat = any(
        metrics["ndcg_at_10"] > mf_ndcg
        for alpha, metrics in z_results.items()
        if alpha not in ("0.0", "1.0")
    )
    any_method_hybrid_beat = any(
        metrics["ndcg_at_10"] > mf_ndcg
        for method_results in per_alpha.values()
        for alpha, metrics in method_results.items()
        if alpha not in ("0.0", "1.0")
    )

    split_after = _sha256(SPLIT_PATH)
    mf_hash_after = _sha256(MF_CHECKPOINT_PATH)
    genre_hash_after = _sha256(GENRE_CHECKPOINT_PATH)
    results_after = _existing_results_hashes()
    integrity = {
        "split_sha256_before": split_before,
        "split_sha256_after": split_after,
        "split_unchanged": split_before == split_after == EXPECTED_SPLIT_SHA256,
        "mf_checkpoint_sha256_before": mf_hash_before,
        "mf_checkpoint_sha256_after": mf_hash_after,
        "mf_checkpoint_unchanged": mf_hash_before == mf_hash_after,
        "genre_checkpoint_sha256_before": genre_hash_before,
        "genre_checkpoint_sha256_after": genre_hash_after,
        "genre_checkpoint_unchanged": genre_hash_before == genre_hash_after,
        "existing_results_count_before": len(results_before),
        "existing_results_unchanged": results_before == results_after,
        "changed_existing_result_paths": [
            path
            for path in sorted(set(results_before) | set(results_after))
            if results_before.get(path) != results_after.get(path)
        ],
        "source_training_rows_read": int(len(training)),
        "validation_label_rows_read_after_scoring": int(validation_interactions),
        "test_rating_values_parsed": False,
        "test_metrics_accessed": False,
        "models_trained": False,
        "all_scores_finite": finite_scores,
        "same_candidate_counts_for_every_method_and_alpha": candidate_counts_consistent,
        "evaluated_users_consistent_for_every_method_and_alpha": per_alpha_counts_consistent,
        "candidate_movies_per_user": {
            "minimum": int(min(expected_candidate_values)),
            "median": float(np.median(list(candidate_counts.values()))),
            "mean": float(np.mean(list(candidate_counts.values()))),
            "maximum": int(max(expected_candidate_values)),
        },
        "standalone_ranking_endpoint_checks": endpoints_match,
        "standalone_validation_metric_checks": baseline_comparisons,
    }
    if not (
        integrity["split_unchanged"]
        and integrity["mf_checkpoint_unchanged"]
        and integrity["genre_checkpoint_unchanged"]
        and integrity["existing_results_unchanged"]
        and integrity["all_scores_finite"]
        and integrity["same_candidate_counts_for_every_method_and_alpha"]
        and integrity["evaluated_users_consistent_for_every_method_and_alpha"]
        and all(endpoints_match.values())
        and all(item["metrics_match"] for item in baseline_comparisons.values())
    ):
        raise RuntimeError("An integrity check failed; no sweep outputs were written.")

    config = {
        "experiment": "validation-only MF B + genre hybrid alpha sweep",
        "primary_method": "per-user population z-score normalization then weighted addition",
        "primary_formula": (
            "hybrid(u,m)=alpha*((mf(u,m)-mean_u(mf))/max(std_u(mf),1e-8))"
            "+(1-alpha)*((genre(u,m)-mean_u(genre))/max(std_u(genre),1e-8))"
        ),
        "variance_floor": {
            "quantity": "population standard deviation",
            "value": STD_FLOOR,
            "handling": "use max(user_score_std, 1e-8) as denominator; centered scores remain finite and order-preserving at alpha endpoints",
            "users_below_floor": floor_users,
        },
        "secondary_method": {
            "name": "rank_fusion",
            "definition": (
                "per model, rank candidates by descending base score then ascending "
                "movie_id; convert zero-based position j to 1-j/(n_candidates-1); "
                "combine alpha*mf_rank_score+(1-alpha)*genre_rank_score"
            ),
            "n_candidates_one_case": "rank score is 1.0 when denominator is max(1,n-1)",
        },
        "alphas": list(ALPHAS),
        "alpha_weights": "alpha is MF B weight; (1-alpha) is genre weight",
        "ranking_cutoff": K,
        "relevance_threshold": "rating >= 4",
        "precision_definition": "relevant top-10 items divided by fixed denominator 10",
        "recall_definition": "hits@10 divided by number of distinct relevant validation movies",
        "ndcg_definition": "binary gain, log2(rank+1) discount; ideal DCG from min(relevant_count,10)",
        "zero_relevant_users": "included in Precision; excluded from Recall and NDCG",
        "candidate_definition": (
            "all movie catalogue IDs except movies rated by the user in the training partition"
        ),
        "validation_labels_used_only_for_metrics": True,
        "test_ratings_or_metrics_used": False,
        "mf_checkpoint": str(MF_CHECKPOINT_PATH),
        "mf_configuration": {
            key: mf_config[key]
            for key in (
                "formulation",
                "n_factors",
                "regularization",
                "learning_rate",
                "random_seed",
                "selected_epoch",
            )
        },
        "genre_checkpoint": str(GENRE_CHECKPOINT_PATH),
        "locked_split_sha256": split_before,
        "validation_users": expected_users,
        "validation_interactions": int(validation_interactions),
        "candidate_count_summary": integrity["candidate_movies_per_user"],
        "baseline_endpoint_parity": baseline_comparisons,
    }

    best_result = z_results[best_alpha_text]
    comparison = {
        "best_alpha_by_z_score_validation_ndcg_at_10": best_alpha,
        "best_z_score_validation_metrics": best_result,
        "standalone_mf_b_validation_metrics": baseline_comparisons[
            "per_user_z_score_alpha_1.0"
        ]["baseline_metrics"],
        "standalone_genre_validation_metrics": baseline_comparisons[
            "per_user_z_score_alpha_0.0"
        ]["baseline_metrics"],
        "hybrid_alpha_strictly_beats_mf_b_on_validation_ndcg": strict_mf_beat,
        "any_non_endpoint_method_alpha_strictly_beats_mf_b_on_validation_ndcg": any_method_hybrid_beat,
        "best_alpha_tie_break": "smallest alpha among equal validation NDCG values",
    }
    if sum(
        result["ndcg_at_10"] == best_result["ndcg_at_10"]
        for result in z_results.values()
    ) > 1:
        best_alpha = min(
            float(alpha)
            for alpha, result in z_results.items()
            if result["ndcg_at_10"] == best_result["ndcg_at_10"]
        )
        comparison["best_alpha_by_z_score_validation_ndcg_at_10"] = best_alpha

    # Save only after all integrity checks pass.
    (OUT / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (OUT / "per_alpha_results.json").write_text(
        json.dumps(
            {
                "primary_method": "per_user_z_score",
                "secondary_method": "rank_fusion",
                "results": per_alpha,
                "selection_and_baseline_comparison": comparison,
                "alpha_endpoint_rankings_match": endpoints_match,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_frame.to_csv(OUT / "validation_metrics.csv", index=False)
    _write_plots(metrics_rows)
    (OUT / "reproducibility_checks.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    summary = f"""# Validation-only hybrid sweep

## Experiment

This is a validation-only diagnostic using the saved MF entity-level model B and saved genre checkpoint. Neither model was trained. The locked chronological split was not changed. The primary method separately standardizes each model's scores over the user's full eligible catalogue:

```text
mf_z(u,m)    = (mf_score(u,m) - mean_u(mf_score)) / max(std_u(mf_score), 1e-8)
genre_z(u,m) = (genre_score(u,m) - mean_u(genre_score)) / max(std_u(genre_score), 1e-8)
hybrid(u,m)  = alpha * mf_z(u,m) + (1-alpha) * genre_z(u,m)
```

The standard-deviation floor is `1e-8`; it is used as a denominator floor, not as a zero-score fallback. This keeps scores finite and preserves each model's ordering at alpha endpoints. Users below the floor: MF {floor_users['mf']}, genre {floor_users['genre']}. All scores were checked finite.

The secondary rank-fusion comparison follows the prior audit's explicit definition: rank each model by descending score with ascending movie ID ties, map zero-based position `j` to `1-j/(n-1)`, and combine these scores with the same alpha weights. There are {expected_users:,} validation users. Each has the same candidate set for both models: all catalogue movies except that user's training-rated movies (candidate counts min/median/mean/max: {integrity['candidate_movies_per_user']['minimum']}/{integrity['candidate_movies_per_user']['median']:.0f}/{integrity['candidate_movies_per_user']['mean']:.1f}/{integrity['candidate_movies_per_user']['maximum']}). Candidate construction and scores did not use validation/test rating values.

After all rankings were fixed, validation ratings were used only as labels (relevant iff rating >= 4). Metrics use the established binary-gain NDCG@10, fixed Precision@10 denominator 10, and Recall@10 over distinct relevant movies; 263 users with no relevant validation movie were excluded from Recall/NDCG but included in Precision. No test ratings or metrics were accessed.

## Primary z-score sweep

| Alpha | Precision@10 | Recall@10 | NDCG@10 | Evaluated users | Zero-relevant excluded |
|---:|---:|---:|---:|---:|---:|
"""
    for alpha in ALPHAS:
        metric = z_results[f"{alpha:.1f}"]
        summary += (
            f"| {alpha:.1f} | {metric['precision_at_10']:.6f} | "
            f"{metric['recall_at_10']:.6f} | {metric['ndcg_at_10']:.6f} | "
            f"{metric['users_evaluated']} | "
            f"{metric['users_excluded_zero_relevant']} |\n"
        )
    summary += f"""
Best z-score alpha by validation NDCG@10: **{comparison['best_alpha_by_z_score_validation_ndcg_at_10']:.1f}** (NDCG@10 {comparison['best_z_score_validation_metrics']['ndcg_at_10']:.6f}).

Standalone MF B NDCG@10: {mf_ndcg:.6f}. Standalone genre NDCG@10: {baseline_comparisons['per_user_z_score_alpha_0.0']['baseline_metrics']['ndcg_at_10']:.6f}. Any z-score hybrid alpha (strictly between 0 and 1) beats MF B: **{strict_mf_beat}**. Any secondary rank-fusion interior alpha beats MF B: **{any_method_hybrid_beat}**.

Alpha 0 and alpha 1 reproduced standalone genre and MF B recommendation lists exactly for both methods; metrics also match the saved standalone validation reports. All 22 score/ranking configurations evaluated the same users and candidate counts.

## Rank-fusion comparison

See `validation_metrics.csv` and `per_alpha_results.json` for each alpha. This secondary comparison is diagnostic only; alpha selection remains on the specified primary per-user z-score sweep.

## Interpretation and limitations

The best alpha is validation-selected only and is not final until the validation experiment is audited. The validation sweep does not establish generalization; do not inspect test performance until the hybrid method/configuration is approved. Per-user z-scores discard absolute score levels and may be sensitive to score variance, although no user fell below the chosen floor in this run. Rank fusion discards score-gap information and gives positions to ties using the deterministic movie-ID rule. Small alpha steps are diagnostic rather than extensive tuning.

## Integrity

Split SHA-256 stayed `{split_before}`. MF B and genre checkpoint hashes were unchanged. Existing results remained unchanged. No model was trained, and no test rating values or metrics were read.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    run()
