from pathlib import Path

import pandas as pd


def load_dataset(data_dir):
    """Load MovieLens 1M files without changing or filtering source records."""
    data_dir = Path(data_dir)
    paths = {
        "ratings": data_dir / "ratings.dat",
        "movies": data_dir / "movies.dat",
        "users": data_dir / "users.dat",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "MovieLens 1M files are missing: " + ", ".join(missing)
        )

    ratings = pd.read_csv(
        paths["ratings"],
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": "int32",
            "movie_id": "int32",
            "rating": "int8",
            "timestamp": "int64",
        },
        encoding="latin-1",
    )
    ratings.insert(0, "row_order", range(len(ratings)))

    movies = pd.read_csv(
        paths["movies"],
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
    )
    users = pd.read_csv(
        paths["users"],
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={
            "user_id": "int32",
            "gender": "string",
            "age": "int8",
            "occupation": "int8",
            "zip_code": "string",
        },
        encoding="latin-1",
    )
    return ratings, movies, users
