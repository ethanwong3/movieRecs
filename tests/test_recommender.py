import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from recommender import recommend_movies

class TestRecommender(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        self.movies_df = pd.DataFrame({
            "title": ["Movie 1", "Movie 2", "Movie 3"],
            "genres": [["Action", "Comedy"], ["Comedy", "Drama"], ["Action", "Drama"]]
        })
        self.genre_similarity = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.2],
            [0.0, 0.2, 1.0]
        ])

    def test_valid_recommendation(self):
        """Test valid movie recommendations."""
        recommendations = recommend_movies("Movie 1", self.movies_df, self.genre_similarity, top_n=2)
        self.assertEqual(len(recommendations), 2)
        self.assertIn("Movie 2", recommendations["title"].values)

    def test_no_recommendations(self):
        """Test case with no similar movies."""
        self.genre_similarity[0, 1] = 0.0  # Remove similarity
        self.genre_similarity[0, 2] = 0.0
        recommendations = recommend_movies("Movie 1", self.movies_df, self.genre_similarity, top_n=2)
        self.assertTrue(recommendations.empty)

    def test_invalid_movie_title(self):
        """Test handling of invalid movie title."""
        with self.assertRaises(ValueError):
            recommend_movies("Nonexistent Movie", self.movies_df, self.genre_similarity)

if __name__ == "__main__":
    unittest.main()
