import numpy as np
from scipy.sparse import csr_matrix

from src.matrix_factorization import RegularizedMatrixFactorization


class EntityRegularizedMatrixFactorization(RegularizedMatrixFactorization):
    """MF with once-per-distinct-entity L2 penalties in each mini-batch.

    For batch B, the objective is

        1/2 * sum_(u,i in B) (r_ui - prediction_ui)^2
        + lambda/2 * (
            sum_(distinct u in B) (b_u^2 + ||p_u||^2)
            + sum_(distinct i in B) (b_i^2 + ||q_i||^2)
        ).

    The summed squared-error term and learning-rate scaling match Version 1.
    Therefore the penalty gradient is lambda times each active entity
    parameter, rather than lambda times its interaction count in the batch.
    """

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
        )
        movie_counts = np.bincount(
            movie_indices, minlength=len(self.movie_ids)
        )
        user_gradient = errors_by_interaction @ self.movie_factors
        movie_gradient = errors_by_interaction.T @ self.user_factors
        user_bias_gradient = np.asarray(errors_by_interaction.sum(axis=1)).ravel()
        movie_bias_gradient = np.asarray(errors_by_interaction.sum(axis=0)).ravel()

        active_users = np.flatnonzero(user_counts)
        active_movies = np.flatnonzero(movie_counts)
        rate = self.learning_rate
        reg = self.regularization

        self.user_biases[active_users] += rate * (
            user_bias_gradient[active_users] - reg * self.user_biases[active_users]
        )
        self.movie_biases[active_movies] += rate * (
            movie_bias_gradient[active_movies]
            - reg * self.movie_biases[active_movies]
        )
        self.user_factors[active_users] += rate * (
            user_gradient[active_users] - reg * self.user_factors[active_users]
        )
        self.movie_factors[active_movies] += rate * (
            movie_gradient[active_movies] - reg * self.movie_factors[active_movies]
        )
