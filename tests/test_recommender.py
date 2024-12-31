import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from recommender import recommend_movies

class TestRecommender(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        self.movies_df = pd.DataFrame({
            'title': ['Movie 1', 'Movie 2', 'Movie 3'],
            'genres': ['Action|Comedy', 'Comedy|Drama', 'Action|Drama']
        })

        self.relevance_matrix = np.array([
            [0.8, 0.1, 0.2],  # Movie 1
            [0.1, 0.7, 0.3],  # Movie 2
            [0.2, 0.3, 0.9]   # Movie 3
        ])

    def test_recommend_movies(self):
        """Test valid recommendations."""
        recommendations = recommend_movies("Movie 1", self.movies_df, self.relevance_matrix)
        self.assertEqual(len(recommendations), 2)
        self.assertNotIn("Movie 1", recommendations['title'].values)

    def test_invalid_movie_title(self):
        """Test invalid movie title."""
        with self.assertRaises(ValueError):
            recommend_movies("Nonexistent Movie", self.movies_df, self.relevance_matrix)

    def test_no_similar_movies(self):
        """Test edge case with no similar movies."""
        empty_matrix = np.zeros((3, 3))
        recommendations = recommend_movies("Movie 1", self.movies_df, empty_matrix)
        self.assertTrue(recommendations.empty)

if __name__ == "__main__":
    unittest.main()
