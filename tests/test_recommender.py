import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from recommender import recommend_movies


class TestRecommender(unittest.TestCase):
    def setUp(self):
        """
        Set up mock data for testing.
        """
        # Mock movies dataset
        self.movies_df = pd.DataFrame({
            'title': ['Movie 1', 'Movie 2', 'Movie 3'],
            'genres': ['Action|Comedy', 'Drama|Fantasy', 'Sci-Fi|Horror']
        })

        # Mock tag relevance matrix
        self.relevance_matrix = np.array([
            [0.0, 0.0, 0.0],  # Movie 1 has no similarity to other movies
            [0.0, 0.0, 0.0],  # Movie 2 has no similarity to other movies
            [0.0, 0.0, 0.0]   # Movie 3 has no similarity to other movies
        ])

    def test_recommend_movies(self):
        """
        Test the recommend_movies function for basic functionality.
        """
        recommendations = recommend_movies(
            movie_title="Movie 1",
            movies_df=self.movies_df,
            relevance_matrix=self.relevance_matrix,
            top_n=2
        )
        # Check the length of the recommendations
        self.assertEqual(len(recommendations), 0, "No recommendations should be made as relevance matrix is zero.")

    def test_invalid_movie_title(self):
        """
        Test behavior when the movie title does not exist in the dataset.
        """
        with self.assertRaises(ValueError):
            recommend_movies(
                movie_title="Nonexistent Movie",
                movies_df=self.movies_df,
                relevance_matrix=self.relevance_matrix
            )

    def test_no_similar_movies(self):
        """
        Test edge case where no movies meet the similarity threshold.
        """
        recommendations = recommend_movies(
            movie_title="Movie 1",
            movies_df=self.movies_df,
            relevance_matrix=self.relevance_matrix,
            top_n=5,
            similarity_threshold=0.1
        )
        # Assert that the recommendations DataFrame is empty
        self.assertTrue(recommendations.empty, "Recommendations should be empty when no similar movies exist.")

    def test_with_similar_movies(self):
        """
        Test case where there are valid recommendations above the similarity threshold.
        """
        relevance_matrix_with_similarity = np.array([
            [1.0, 0.6, 0.2],  # Movie 1 is somewhat similar to Movie 2
            [0.6, 1.0, 0.3],  # Movie 2 is somewhat similar to Movie 1 and Movie 3
            [0.2, 0.3, 1.0]   # Movie 3 is slightly similar to Movie 2
        ])

        recommendations = recommend_movies(
            movie_title="Movie 1",
            movies_df=self.movies_df,
            relevance_matrix=relevance_matrix_with_similarity,
            top_n=2,
            similarity_threshold=0.1
        )
        # Assert that the recommendations DataFrame contains the correct results
        self.assertEqual(len(recommendations), 2, "Two movies should be recommended.")
        self.assertIn("Movie 2", recommendations['title'].values, "Movie 2 should be in recommendations.")
        self.assertNotIn("Movie 1", recommendations['title'].values, "Movie 1 should not recommend itself.")

if __name__ == "__main__":
    unittest.main()