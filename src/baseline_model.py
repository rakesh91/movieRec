import numpy as np


class UserMovieBiasModel:
    """Regularized additive user/movie rating-bias model.

    Prediction is mu + b_user + b_movie. The biases minimize squared error
    with L2 penalties lambda_user * sum(b_user**2) and
    lambda_movie * sum(b_movie**2), using alternating closed-form updates.
    """

    def __init__(self, lambda_user=10.0, lambda_movie=10.0, iterations=15):
        self.lambda_user = float(lambda_user)
        self.lambda_movie = float(lambda_movie)
        self.iterations = int(iterations)
        self.global_mean = None
        self.user_ids = np.array([], dtype=np.int64)
        self.movie_ids = np.array([], dtype=np.int64)
        self.user_biases = np.array([], dtype=np.float64)
        self.movie_biases = np.array([], dtype=np.float64)

    def fit(self, training_ratings):
        if training_ratings.empty:
            raise ValueError("Cannot fit the model with no training ratings.")

        users = training_ratings["user_id"].to_numpy(dtype=np.int64)
        movies = training_ratings["movie_id"].to_numpy(dtype=np.int64)
        ratings = training_ratings["rating"].to_numpy(dtype=np.float64)
        self.user_ids = np.unique(users)
        self.movie_ids = np.unique(movies)
        user_index = np.searchsorted(self.user_ids, users)
        movie_index = np.searchsorted(self.movie_ids, movies)
        user_counts = np.bincount(user_index, minlength=len(self.user_ids))
        movie_counts = np.bincount(movie_index, minlength=len(self.movie_ids))

        self.global_mean = float(ratings.mean())
        self.user_biases = np.zeros(len(self.user_ids), dtype=np.float64)
        self.movie_biases = np.zeros(len(self.movie_ids), dtype=np.float64)

        for _ in range(self.iterations):
            user_residuals = (
                ratings - self.global_mean - self.movie_biases[movie_index]
            )
            self.user_biases = np.bincount(
                user_index, weights=user_residuals, minlength=len(self.user_ids)
            ) / (self.lambda_user + user_counts)

            movie_residuals = (
                ratings - self.global_mean - self.user_biases[user_index]
            )
            self.movie_biases = np.bincount(
                movie_index, weights=movie_residuals, minlength=len(self.movie_ids)
            ) / (self.lambda_movie + movie_counts)
        return self

    @staticmethod
    def _lookup_biases(ids, known_ids, known_biases):
        positions = np.searchsorted(known_ids, ids)
        found = positions < len(known_ids)
        valid_positions = positions[found]
        found_indices = np.flatnonzero(found)
        found[found_indices] = known_ids[valid_positions] == ids[found_indices]
        result = np.zeros(len(ids), dtype=np.float64)
        result[found] = known_biases[positions[found]]
        return result

    def predict_many(self, user_ids, movie_ids, clip=True):
        if self.global_mean is None:
            raise RuntimeError("Fit the model before requesting predictions.")
        users = np.asarray(user_ids, dtype=np.int64)
        movies = np.asarray(movie_ids, dtype=np.int64)
        if users.shape != movies.shape:
            raise ValueError("user_ids and movie_ids must have matching shapes.")
        predictions = (
            self.global_mean
            + self._lookup_biases(users, self.user_ids, self.user_biases)
            + self._lookup_biases(movies, self.movie_ids, self.movie_biases)
        )
        return np.clip(predictions, 1.0, 5.0) if clip else predictions

    def predict(self, user_id, movie_id, clip=True):
        return float(
            self.predict_many([user_id], [movie_id], clip=clip)[0]
        )
