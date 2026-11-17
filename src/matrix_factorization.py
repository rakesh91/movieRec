import numpy as np
from scipy.sparse import csr_matrix


class RegularizedMatrixFactorization:
    """Biased matrix factorization trained with deterministic mini-batch SGD.

    The prediction is mu + user_bias + movie_bias + user_factors dot
    movie_factors. Each rating contributes squared error and L2 penalties for
    the biases and factors of its user and movie.
    """

    def __init__(
        self,
        n_factors=20,
        regularization=0.05,
        learning_rate=0.01,
        random_seed=42,
        batch_size=8192,
    ):
        self.n_factors = int(n_factors)
        self.regularization = float(regularization)
        self.learning_rate = float(learning_rate)
        self.random_seed = int(random_seed)
        self.batch_size = int(batch_size)
        if self.n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if self.regularization < 0:
            raise ValueError("regularization cannot be negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.global_mean = None
        self.user_ids = np.array([], dtype=np.int64)
        self.movie_ids = np.array([], dtype=np.int64)
        self.user_biases = np.array([], dtype=np.float64)
        self.movie_biases = np.array([], dtype=np.float64)
        self.user_factors = np.empty((0, self.n_factors), dtype=np.float64)
        self.movie_factors = np.empty((0, self.n_factors), dtype=np.float64)
        self.history_ = []
        self.best_epoch_ = None
        self.best_validation_rmse_ = None
        self.best_validation_mae_ = None

    def _raw_predictions_from_indices(self, user_indices, movie_indices):
        return (
            self.global_mean
            + self.user_biases[user_indices]
            + self.movie_biases[movie_indices]
            + np.einsum(
                "ij,ij->i",
                self.user_factors[user_indices],
                self.movie_factors[movie_indices],
            )
        )

    def _batch_update(self, user_indices, movie_indices, ratings):
        predictions = self._raw_predictions_from_indices(
            user_indices, movie_indices
        )
        errors = ratings - predictions
        errors_by_interaction = csr_matrix(
            (errors, (user_indices, movie_indices)),
            shape=(len(self.user_ids), len(self.movie_ids)),
        )

        user_counts = np.bincount(
            user_indices, minlength=len(self.user_ids)
        ).astype(np.float64)
        movie_counts = np.bincount(
            movie_indices, minlength=len(self.movie_ids)
        ).astype(np.float64)
        user_gradient = errors_by_interaction @ self.movie_factors
        movie_gradient = errors_by_interaction.T @ self.user_factors
        user_bias_gradient = np.asarray(errors_by_interaction.sum(axis=1)).ravel()
        movie_bias_gradient = np.asarray(errors_by_interaction.sum(axis=0)).ravel()

        active_users = np.flatnonzero(user_counts)
        active_movies = np.flatnonzero(movie_counts)
        rate = self.learning_rate
        reg = self.regularization

        self.user_biases[active_users] += rate * (
            user_bias_gradient[active_users]
            - reg * user_counts[active_users] * self.user_biases[active_users]
        )
        self.movie_biases[active_movies] += rate * (
            movie_bias_gradient[active_movies]
            - reg * movie_counts[active_movies] * self.movie_biases[active_movies]
        )
        self.user_factors[active_users] += rate * (
            user_gradient[active_users]
            - reg
            * user_counts[active_users, np.newaxis]
            * self.user_factors[active_users]
        )
        self.movie_factors[active_movies] += rate * (
            movie_gradient[active_movies]
            - reg
            * movie_counts[active_movies, np.newaxis]
            * self.movie_factors[active_movies]
        )

    def _metrics(self, ratings, user_indices, movie_indices):
        predictions = np.clip(
            self._raw_predictions_from_indices(user_indices, movie_indices),
            1.0,
            5.0,
        )
        errors = predictions - ratings
        return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))

    def _metrics_by_ids(self, ratings, user_ids, movie_ids):
        predictions = self.predict_many(user_ids, movie_ids, clip=True)
        errors = predictions - ratings
        return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))

    def fit(
        self,
        training_ratings,
        validation_ratings,
        max_epochs=30,
        patience=4,
    ):
        if training_ratings.empty:
            raise ValueError("Cannot fit matrix factorization without training ratings.")
        if validation_ratings.empty:
            raise ValueError(
                "Validation ratings are required for early stopping and model selection."
            )
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive.")

        train_users = training_ratings["user_id"].to_numpy(dtype=np.int64)
        train_movies = training_ratings["movie_id"].to_numpy(dtype=np.int64)
        train_ratings = training_ratings["rating"].to_numpy(dtype=np.float64)
        validation_users = validation_ratings["user_id"].to_numpy(dtype=np.int64)
        validation_movies = validation_ratings["movie_id"].to_numpy(dtype=np.int64)
        validation_values = validation_ratings["rating"].to_numpy(dtype=np.float64)

        self.user_ids = np.unique(train_users)
        self.movie_ids = np.unique(train_movies)
        self.global_mean = float(train_ratings.mean())
        train_user_indices = np.searchsorted(self.user_ids, train_users)
        train_movie_indices = np.searchsorted(self.movie_ids, train_movies)

        rng = np.random.default_rng(self.random_seed)
        self.user_biases = np.zeros(len(self.user_ids), dtype=np.float64)
        self.movie_biases = np.zeros(len(self.movie_ids), dtype=np.float64)
        self.user_factors = rng.normal(
            0.0, 0.1, size=(len(self.user_ids), self.n_factors)
        )
        self.movie_factors = rng.normal(
            0.0, 0.1, size=(len(self.movie_ids), self.n_factors)
        )

        self.history_ = []
        best_state = None
        best_rmse = float("inf")
        best_epoch = 0
        stale_epochs = 0
        interaction_count = len(train_ratings)

        for epoch in range(1, max_epochs + 1):
            order = rng.permutation(interaction_count)
            for start in range(0, interaction_count, self.batch_size):
                batch = order[start : start + self.batch_size]
                self._batch_update(
                    train_user_indices[batch],
                    train_movie_indices[batch],
                    train_ratings[batch],
                )

            train_mae, train_rmse = self._metrics(
                train_ratings, train_user_indices, train_movie_indices
            )
            validation_mae, validation_rmse = self._metrics_by_ids(
                validation_values,
                validation_users,
                validation_movies,
            )
            self.history_.append(
                {
                    "epoch": epoch,
                    "training_mae": train_mae,
                    "training_rmse": train_rmse,
                    "validation_mae": validation_mae,
                    "validation_rmse": validation_rmse,
                }
            )

            if validation_rmse < best_rmse:
                best_rmse = validation_rmse
                best_epoch = epoch
                stale_epochs = 0
                best_state = (
                    self.user_biases.copy(),
                    self.movie_biases.copy(),
                    self.user_factors.copy(),
                    self.movie_factors.copy(),
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("Matrix factorization did not produce a validation score.")
        (
            self.user_biases,
            self.movie_biases,
            self.user_factors,
            self.movie_factors,
        ) = best_state
        self.best_epoch_ = best_epoch
        self.best_validation_rmse_ = best_rmse
        self.best_validation_mae_ = self.history_[best_epoch - 1]["validation_mae"]
        return self

    @staticmethod
    def _lookup(ids, known_ids, known_values):
        positions = np.searchsorted(known_ids, ids)
        in_bounds = positions < len(known_ids)
        found = np.zeros(len(ids), dtype=bool)
        found_indices = np.flatnonzero(in_bounds)
        found[found_indices] = (
            known_ids[positions[found_indices]] == ids[found_indices]
        )
        return positions, found

    def predict_many(self, user_ids, movie_ids, clip=True):
        if self.global_mean is None:
            raise RuntimeError("Fit the model before requesting predictions.")
        users = np.asarray(user_ids, dtype=np.int64)
        movies = np.asarray(movie_ids, dtype=np.int64)
        if users.shape != movies.shape:
            raise ValueError("user_ids and movie_ids must have matching shapes.")

        user_positions, known_users = self._lookup(
            users, self.user_ids, self.user_biases
        )
        movie_positions, known_movies = self._lookup(
            movies, self.movie_ids, self.movie_biases
        )
        predictions = np.full(len(users), self.global_mean, dtype=np.float64)
        predictions[known_users] += self.user_biases[user_positions[known_users]]
        predictions[known_movies] += self.movie_biases[movie_positions[known_movies]]
        known_pairs = known_users & known_movies
        predictions[known_pairs] += np.einsum(
            "ij,ij->i",
            self.user_factors[user_positions[known_pairs]],
            self.movie_factors[movie_positions[known_pairs]],
        )
        return np.clip(predictions, 1.0, 5.0) if clip else predictions

    def predict(self, user_id, movie_id, clip=True):
        return float(self.predict_many([user_id], [movie_id], clip=clip)[0])
