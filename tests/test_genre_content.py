import inspect
import unittest

import numpy as np
import pandas as pd

from src.genre_content import (
    GenreContentModel,
    _evaluate_partition,
    _rank_candidates,
)


class GenreContentTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame(
            {
                "movie_id": [10, 11, 12, 13],
                "title": ["Alpha", "Beta", "Gamma", "No tags"],
                "genres": ["Action|Comedy", "Action", "Drama", "(no genres listed)"],
            }
        )
        self.training = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "movie_id": [10, 12, 11, 12],
                "rating": [5, 1, 4, 2],
            }
        )
        self.model = GenreContentModel().fit(self.training, self.movies)

    def test_multi_hot_movie_features_and_centered_normalized_profiles(self):
        np.testing.assert_array_equal(
            self.model.movie_genre_matrix,
            np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        self.assertAlmostEqual(self.model.global_mean, 3.0)
        norms = np.linalg.norm(self.model.user_profiles, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0])
        self.assertGreater(self.model.user_profiles[0, 0], 0.0)
        self.assertLess(self.model.user_profiles[0, 2], 0.0)

    def test_cosine_scoring_and_unknown_or_zero_vectors_are_safe(self):
        scores = self.model.predict_similarity(
            [1, 1, 999, 1],
            [10, 13, 10, 999],
        )
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)
        self.assertEqual(scores[3], 0.0)

    def test_recommendations_exclude_training_movies_and_break_ties_by_id(self):
        training_items = {1: {10, 12}}
        recs = _rank_candidates(self.model, 1, self.movies, training_items)
        self.assertFalse(set(recs["movie_id"]) & training_items[1])
        unknown_user_recs = _rank_candidates(self.model, 999, self.movies, {})
        self.assertEqual(unknown_user_recs["movie_id"].tolist(), [10, 11, 12, 13])
        self.assertTrue((unknown_user_recs["content_similarity"] == 0.0).all())

    def test_evaluation_uses_distinct_relevant_movies_and_fixed_precision(self):
        validation = pd.DataFrame(
            {
                "user_id": [1, 1],
                "movie_id": [11, 11],
                "rating": [4, 5],
            }
        )
        metrics, per_user, recommendations = _evaluate_partition(
            self.model,
            self.movies,
            self.training,
            validation,
            "validation",
        )
        self.assertEqual(metrics["users_for_precision"], 1)
        self.assertEqual(metrics["users_for_recall_and_ndcg"], 1)
        self.assertAlmostEqual(metrics["precision_at_10"], 0.1)
        self.assertAlmostEqual(metrics["recall_at_10"], 1.0)
        self.assertAlmostEqual(metrics["ndcg_at_10"], 1.0)
        self.assertNotIn(10, recommendations["movie_id"].tolist())
        self.assertEqual(per_user.loc[0, "relevant_target_movies"], 1)

    def test_profile_fit_accepts_training_ratings_and_movie_metadata_only(self):
        parameters = tuple(inspect.signature(GenreContentModel.fit).parameters)
        self.assertEqual(parameters, ("self", "training_ratings", "movies"))


if __name__ == "__main__":
    unittest.main()
