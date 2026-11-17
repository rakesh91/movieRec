import numpy as np


def chronological_split(ratings):
    """Split chronologically without dividing a user's equal-timestamp ratings."""
    ordered = ratings.sort_values(
        ["user_id", "timestamp", "row_order"], kind="mergesort"
    ).reset_index(drop=True)
    partitions = np.full(len(ordered), "", dtype=object)

    for positions in ordered.groupby("user_id", sort=False).indices.values():
        timestamps = ordered.iloc[positions]["timestamp"].to_numpy()
        group_starts = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1]
        group_ends = np.r_[group_starts[1:], len(positions)]
        group_counts = group_ends - group_starts
        group_count = len(group_counts)
        cumulative_ratings = np.cumsum(group_counts)

        if group_count == 1:
            train_count = len(positions)
            validation_count = 0
        elif group_count == 2:
            train_count = int(group_counts[0])
            validation_count = 0
        else:
            train_boundary = min(
                range(1, group_count - 1),
                key=lambda boundary: (
                    abs(cumulative_ratings[boundary - 1] - 0.8 * len(positions)),
                    boundary,
                ),
            )
            validation_boundary = min(
                range(train_boundary + 1, group_count),
                key=lambda boundary: (
                    abs(cumulative_ratings[boundary - 1] - 0.9 * len(positions)),
                    boundary,
                ),
            )
            train_count = int(cumulative_ratings[train_boundary - 1])
            validation_count = int(
                cumulative_ratings[validation_boundary - 1] - train_count
            )

        test_count = len(positions) - train_count - validation_count
        partitions[positions[:train_count]] = "train"
        partitions[positions[train_count : train_count + validation_count]] = (
            "validation"
        )
        partitions[positions[train_count + validation_count :]] = "test"
        assert train_count >= 1
        assert train_count + validation_count + test_count == len(positions)

    ordered["partition"] = partitions
    assert (ordered["partition"] != "").all()
    assert len(ordered) == len(ratings)
    return ordered
