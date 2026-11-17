import unittest

import numpy as np
import pandas as pd

from src.baseline_model import UserMovieBiasModel
from src.data_split import chronological_split
from src.entity_regularized_matrix_factorization import (
    EntityRegularizedMatrixFactorization,
)
from src.evaluation import _ranking_values
from src.matrix_factorization import RegularizedMatrixFactorization


class BaselineTests(unittest.TestCase):
    def test_tie_aware_split_keeps_groups_and_handles_small_histories(self):
        ratings = pd.DataFrame(
            {
                "row_order": list(range(12)),
                "user_id": [1] * 8 + [2] * 2 + [3] * 2,
                "movie_id": list(range(12)),
                "rating": [3] * 12,
                "timestamp": [
                    1, 1, 2, 3, 4, 5, 6, 7,
                    1, 2,
                    1, 1,
                ],
            }
        )
        split = chronological_split(ratings)
        repeated = chronological_split(ratings)
        pd.testing.assert_frame_equal(split, repeated)
        self.assertEqual(len(split), len(ratings))
        self.assertEqual(set(split["row_order"]), set(ratings["row_order"]))
        self.assertEqual(split["row_order"].nunique(), len(ratings))

        for _, group in split.groupby(["user_id", "timestamp"]):
            self.assertEqual(group["partition"].nunique(), 1)

        user_one = split.loc[split["user_id"] == 1]
        self.assertEqual(set(user_one["partition"]), {"train", "validation", "test"})
        partition_times = user_one.groupby("partition")["timestamp"].agg(["min", "max"])
        self.assertLess(partition_times.loc["train", "max"], partition_times.loc["validation", "min"])
        self.assertLess(partition_times.loc["validation", "max"], partition_times.loc["test", "min"])

        user_two = split.loc[split["user_id"] == 2]
        self.assertEqual(set(user_two["partition"]), {"train", "test"})
        self.assertEqual(user_two.loc[user_two["partition"] == "train", "timestamp"].max(), 1)
        self.assertEqual(user_two.loc[user_two["partition"] == "test", "timestamp"].min(), 2)

        user_three = split.loc[split["user_id"] == 3]
        self.assertEqual(set(user_three["partition"]), {"train"})

    def test_equal_distance_validation_boundary_chooses_earlier_cut(self):
        ratings = pd.DataFrame(
            {
                "row_order": list(range(15)),
                "user_id": [1] * 15,
                "movie_id": list(range(15)),
                "rating": [3] * 15,
                "timestamp": [1] * 10 + [2, 3, 4, 5, 6],
            }
        )
        split = chronological_split(ratings)
        validation_times = split.loc[
            split["partition"] == "validation", "timestamp"
        ].unique()
        self.assertEqual(validation_times.tolist(), [4])

    def test_matrix_factorization_is_reproducible_and_handles_unknown_ids(self):
        training = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "movie_id": [10, 11, 10, 12, 11, 12],
                "rating": [5, 4, 2, 1, 4, 3],
            }
        )
        validation = pd.DataFrame(
            {
                "user_id": [1, 99],
                "movie_id": [999, 10],
                "rating": [3, 2],
            }
        )
        first = RegularizedMatrixFactorization(
            n_factors=3, learning_rate=0.005, batch_size=2, random_seed=17
        ).fit(training, validation, max_epochs=4, patience=2)
        second = RegularizedMatrixFactorization(
            n_factors=3, learning_rate=0.005, batch_size=2, random_seed=17
        ).fit(training, validation, max_epochs=4, patience=2)

        predictions = first.predict_many(
            [1, 99, 99, 2],
            [10, 10, 999, 999],
            clip=False,
        )
        self.assertTrue(np.isfinite(predictions).all())
        self.assertAlmostEqual(
            predictions[1],
            first.global_mean + first.movie_biases[
                np.searchsorted(first.movie_ids, 10)
            ],
        )
        self.assertAlmostEqual(predictions[2], first.global_mean)
        self.assertGreaterEqual(
            first.predict(1, 999), 1.0
        )
        self.assertLessEqual(first.predict(1, 999), 5.0)
        np.testing.assert_allclose(
            first.predict_many([1, 2], [10, 12], clip=False),
            second.predict_many([1, 2], [10, 12], clip=False),
        )
        self.assertEqual(first.history_, second.history_)
        self.assertGreaterEqual(first.best_epoch_, 1)

    def test_entity_level_batch_update_penalizes_once_per_distinct_entity(self):
        model = EntityRegularizedMatrixFactorization(
            n_factors=1,
            regularization=0.2,
            learning_rate=0.1,
            batch_size=2,
        )
        model.global_mean = 3.0
        model.user_ids = np.array([1])
        model.movie_ids = np.array([10, 11])
        model.user_biases = np.array([0.5])
        model.movie_biases = np.array([0.2, -0.1])
        model.user_factors = np.array([[0.4]])
        model.movie_factors = np.array([[0.3], [0.2]])

        user_indices = np.array([0, 0])
        movie_indices = np.array([0, 1])
        ratings = np.array([4.0, 2.0])
        initial_user_factor = model.user_factors[0, 0]
        initial_movie_factors = model.movie_factors[:, 0].copy()
        errors = ratings - np.array([3.82, 3.48])
        expected_user_factor = initial_user_factor + 0.1 * (
            np.dot(errors, initial_movie_factors) - 0.2 * initial_user_factor
        )
        expected_user_bias = 0.5 + 0.1 * (
            errors.sum() - 0.2 * 0.5
        )
        expected_movie_biases = np.array(
            [
                0.2 + 0.1 * (errors[0] - 0.2 * 0.2),
                -0.1 + 0.1 * (errors[1] - 0.2 * -0.1),
            ]
        )

        model._batch_update(user_indices, movie_indices, ratings)
        self.assertAlmostEqual(model.user_factors[0, 0], expected_user_factor)
        self.assertAlmostEqual(model.user_biases[0], expected_user_bias)
        np.testing.assert_allclose(model.movie_biases, expected_movie_biases)

    def test_model_fits_training_data_and_handles_unknown_ids(self):
        training = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "movie_id": [10, 11, 10, 12],
                "rating": [5, 4, 2, 1],
            }
        )
        model = UserMovieBiasModel(iterations=20).fit(training)
        self.assertGreater(model.predict(1, 10), model.predict(2, 12))
        self.assertGreaterEqual(model.predict(999, 999), 1.0)
        self.assertLessEqual(model.predict(999, 999), 5.0)
        self.assertEqual(len(model.predict_many([1, 2], [10, 12])), 2)

    def test_validation_or_test_rows_are_not_needed_for_model_fit(self):
        training = pd.DataFrame(
            {"user_id": [1, 2], "movie_id": [10, 11], "rating": [5, 1]}
        )
        model = UserMovieBiasModel().fit(training)
        self.assertAlmostEqual(model.global_mean, 3.0)

    def test_short_ranking_uses_fixed_precision_denominator(self):
        precision, recall, ndcg = _ranking_values(
            ranked_movie_ids=[10, 11, 12, 13],
            relevant_movie_ids={10, 14},
            k=10,
        )
        self.assertAlmostEqual(precision, 0.1)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(ndcg, 1.0 / (1.0 + 1.0 / 1.584962500721156))


if __name__ == "__main__":
    unittest.main()
