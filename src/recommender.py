import pandas as pd


def recommend_for_user(model, user_id, movies, training_ratings, n=10):
    rated_ids = set(
        training_ratings.loc[
            training_ratings["user_id"] == user_id, "movie_id"
        ].tolist()
    )
    candidates = movies.loc[~movies["movie_id"].isin(rated_ids)].copy()
    if candidates.empty:
        return candidates.assign(predicted_rating=pd.Series(dtype=float))
    candidates["predicted_rating"] = model.predict_many(
        [user_id] * len(candidates), candidates["movie_id"].to_numpy()
    )
    candidates["in_training_history"] = candidates["movie_id"].isin(rated_ids)
    return candidates.sort_values(
        ["predicted_rating", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(n).reset_index(drop=True)


def predict_test_ratings_for_user(model, user_id, test_ratings):
    user_test = test_ratings.loc[test_ratings["user_id"] == user_id].copy()
    if user_test.empty:
        return user_test.assign(predicted_rating=pd.Series(dtype=float))
    user_test["predicted_rating"] = model.predict_many(
        user_test["user_id"].to_numpy(),
        user_test["movie_id"].to_numpy(),
    )
    return user_test.sort_values(
        ["timestamp", "row_order"], kind="mergesort"
    ).reset_index(drop=True)
