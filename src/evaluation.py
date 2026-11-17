import math

import numpy as np
import pandas as pd


def rating_prediction_metrics(model, test_ratings):
    if test_ratings.empty:
        return {
            "test_interactions": 0,
            "users_evaluated": 0,
            "mae": None,
            "rmse": None,
        }, pd.DataFrame()

    predictions = model.predict_many(
        test_ratings["user_id"].to_numpy(),
        test_ratings["movie_id"].to_numpy(),
    )
    actual = test_ratings["rating"].to_numpy(dtype=np.float64)
    errors = predictions - actual
    per_user = pd.DataFrame(
        {
            "user_id": test_ratings["user_id"].to_numpy(),
            "absolute_error": np.abs(errors),
            "squared_error": errors**2,
        }
    ).groupby("user_id").agg(
        test_ratings=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
    )
    per_user["rmse"] = np.sqrt(per_user["mean_squared_error"])
    per_user = per_user.drop(columns="mean_squared_error").reset_index()
    return {
        "test_interactions": int(len(test_ratings)),
        "users_evaluated": int(test_ratings["user_id"].nunique()),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }, per_user


def _ranking_values(ranked_movie_ids, relevant_movie_ids, k):
    relevant_movie_ids = set(relevant_movie_ids)
    hits = [
        int(movie_id in relevant_movie_ids)
        for movie_id in ranked_movie_ids[:k]
    ]
    hit_count = sum(hits)
    precision = hit_count / k if k else 0.0
    recall = hit_count / len(relevant_movie_ids) if relevant_movie_ids else None
    dcg = sum(
        relevance / math.log2(rank + 2)
        for rank, relevance in enumerate(hits)
    )
    ideal_count = min(len(relevant_movie_ids), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 2) for rank in range(ideal_count)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else None
    return precision, recall, ndcg


def _average_ranking_rows(rows, users_with_candidates, users_without_relevant, k):
    if not rows:
        return {
            f"precision_at_{k}": None,
            f"recall_at_{k}": None,
            f"ndcg_at_{k}": None,
            "users_for_precision": 0,
            "users_for_recall_and_ndcg": 0,
            "users_excluded_no_relevant_test_items": int(users_without_relevant),
        }
    return {
        f"precision_at_{k}": float(np.mean([row["precision"] for row in rows])),
        f"recall_at_{k}": (
            float(np.mean([row["recall"] for row in rows if row["recall"] is not None]))
            if any(row["recall"] is not None for row in rows)
            else None
        ),
        f"ndcg_at_{k}": (
            float(np.mean([row["ndcg"] for row in rows if row["ndcg"] is not None]))
            if any(row["ndcg"] is not None for row in rows)
            else None
        ),
        "users_for_precision": int(len(rows)),
        "users_for_recall_and_ndcg": int(
            sum(row["recall"] is not None for row in rows)
        ),
        "users_excluded_no_relevant_test_items": int(users_without_relevant),
    }


def ranking_metrics(model, movies, split, k=10):
    test = split.loc[split["partition"] == "test"]
    train = split.loc[split["partition"] == "train"]
    test_candidates_by_user = test.groupby("user_id")["movie_id"].apply(
        lambda values: sorted(set(values))
    ).to_dict()
    train_items_by_user = train.groupby("user_id")["movie_id"].apply(set).to_dict()

    restricted_rankings = {}
    for user_id, candidate_ids in test_candidates_by_user.items():
        candidate_array = np.asarray(candidate_ids, dtype=np.int64)
        predictions = model.predict_many(
            np.full(len(candidate_array), user_id, dtype=np.int64),
            candidate_array,
        )
        order = np.lexsort((candidate_array, -predictions))
        restricted_rankings[user_id] = candidate_array[order].tolist()

    catalogue_ids = movies["movie_id"].to_numpy(dtype=np.int64)
    full_rankings = {}
    full_candidate_counts = {}
    users_without_candidates = 0
    for user_id in test_candidates_by_user:
        training_ids = train_items_by_user.get(user_id, set())
        eligible_ids = np.array(
            [movie_id for movie_id in catalogue_ids if movie_id not in training_ids],
            dtype=np.int64,
        )
        if len(eligible_ids) < k:
            users_without_candidates += 1
            continue
        predictions = model.predict_many(
            np.full(len(eligible_ids), user_id, dtype=np.int64), eligible_ids
        )
        top_positions = np.lexsort((eligible_ids, -predictions))[:k]
        full_rankings[user_id] = eligible_ids[top_positions].tolist()
        full_candidate_counts[user_id] = len(eligible_ids)

    relevant_by_user = (
        test.loc[test["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    restricted_rows = []
    restricted_no_relevant = 0
    for user_id, ranked in restricted_rankings.items():
        relevant_ids = relevant_by_user.get(user_id, set())
        if not relevant_ids:
            restricted_no_relevant += 1
        precision, recall, ndcg = _ranking_values(ranked, relevant_ids, k)
        restricted_rows.append(
            {
                "user_id": int(user_id),
                "candidate_movies": len(test_candidates_by_user[user_id]),
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
            }
        )

    full_rows = []
    for user_id, recommended_ids in full_rankings.items():
        relevant_ids = relevant_by_user.get(user_id, set())
        precision, recall, ndcg = _ranking_values(
            recommended_ids,
            relevant_ids,
            k,
        )
        full_rows.append(
            {
                "user_id": int(user_id),
                "candidate_movies": full_candidate_counts[user_id],
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
            }
        )

    full_no_relevant = sum(row["recall"] is None for row in full_rows)
    restricted_metrics = _average_ranking_rows(
        restricted_rows,
        len(restricted_rows),
        restricted_no_relevant,
        k,
    )
    restricted_metrics["precision_at_10_definition"] = (
        "relevant recommendations in top 10 divided by 10; "
        "missing positions count as non-relevant"
    )
    restricted_metrics["users_with_fewer_than_10_candidates"] = int(
        sum(len(candidate_ids) < 10 for candidate_ids in test_candidates_by_user.values())
    )
    catalogue_metrics = _average_ranking_rows(
        full_rows,
        len(full_rows),
        full_no_relevant,
        k,
    )
    catalogue_metrics["precision_at_10_definition"] = (
        "relevant recommendations in top 10 divided by 10"
    )
    catalogue_metrics["users_excluded_insufficient_catalogue_candidates"] = int(
        users_without_candidates
    )
    return {
        "restricted_test_candidate_ranking": restricted_metrics,
        "full_eligible_catalogue_ranking": catalogue_metrics,
    }
