import pandas as pd


def build_user_eligibility(users, movies, split):
    """Build per-user evaluation and recommendation eligibility from the split."""
    counts = (
        split.pivot_table(
            index="user_id",
            columns="partition",
            values="movie_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=["train", "validation", "test"], fill_value=0)
        .rename(
            columns={
                "train": "training_ratings",
                "validation": "validation_ratings",
                "test": "test_ratings",
            }
        )
    )
    relevant = (
        split.loc[(split["partition"] == "test") & (split["rating"] >= 4)]
        .groupby("user_id")
        .size()
        .rename("relevant_test_ratings")
    )
    training_movie_counts = (
        split.loc[
            (split["partition"] == "train")
            & split["movie_id"].isin(movies["movie_id"])
        ]
        .groupby("user_id")["movie_id"]
        .nunique()
        .rename("training_movie_count")
    )

    eligibility = users[["user_id"]].drop_duplicates().set_index("user_id")
    eligibility = eligibility.join(counts).join(relevant).join(training_movie_counts)
    count_columns = [
        "training_ratings",
        "validation_ratings",
        "test_ratings",
        "relevant_test_ratings",
        "training_movie_count",
    ]
    eligibility[count_columns] = eligibility[count_columns].fillna(0).astype(int)
    eligibility["eligible_unseen_catalogue_movies"] = (
        len(movies) - eligibility["training_movie_count"]
    ).clip(lower=0)
    eligibility["eligible_for_rating_prediction"] = eligibility["test_ratings"] > 0
    eligibility["eligible_for_ranking_evaluation"] = (
        eligibility["relevant_test_ratings"] > 0
    )
    eligibility["can_receive_top_10_recommendation"] = (
        (eligibility["training_ratings"] > 0)
        & (eligibility["eligible_unseen_catalogue_movies"] >= 10)
    )
    return eligibility.reset_index().drop(columns=["training_movie_count"])
