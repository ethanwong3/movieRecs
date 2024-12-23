# File: test_recommender.py
import unittest
import pandas as pd
import numpy as np
from src.recommender import recommend_movies

class TestRecommender(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        # Mock movies dataset
        self.movies_df = pd.DataFrame({
            'title': ['Movie 1', 'Movie 2', 'Movie 3'],
            'genres': ['Action|Comedy', 'Comedy|Drama', 'Action|Drama']
        })
        # Mock tag relevance matrix
        self.relevance_matrix = np.array([
            [0.8, 0.1, 0.2],  # Movie 1
            [0.1, 0.7, 0.3],  # Movie 2
            [0.2, 0.3, 0.9],  # Movie 3
        ])

    def test_recommend_movies(self):
        """Test recommend_movies function."""
        recommendations = recommend_movies("Movie 1", self.movies_df, self.relevance_matrix, top_n=2)
        self.assertEqual(len(recommendations), 2)  # Check number of recommendations
        self.assertNotIn("Movie 1", recommendations['title'].values)  # Exclude input movie

    def test_invalid_movie_title(self):
        """Test invalid movie title."""
        with self.assertRaises(ValueError):
            recommend_movies("Nonexistent Movie", self.movies_df, self.relevance_matrix)

    def test_no_similar_movies(self):
        """Test edge case with no similar movies."""
        # Mock data where no similarity exists
        self.movies_df['genres'] = ['Action', 'Comedy', 'Horror']
        recommendations = recommend_movies("Movie 1", self.movies_df, self.relevance_matrix, top_n=2)
        self.assertTrue(recommendations.empty)  # No recommendations should be found

if __name__ == "__main__":
    unittest.main()
