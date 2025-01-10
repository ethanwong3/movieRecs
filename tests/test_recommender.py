import unittest
import pandas as pd
import numpy as np
from recommender import recommend_movies

class TestRecommender(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        self.movies_df = pd.DataFrame({
            'title': ['Movie 1', 'Movie 2', 'Movie 3'],
            'genres': ['Action|Comedy', 'Comedy|Drama', 'Action|Drama']
        })
        self.genre_matrix = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        self.tag_matrix = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        np.save('test_genre_matrix.npy', self.genre_matrix)
        np.save('test_tag_matrix.npy', self.tag_matrix)

    def test_no_similar_movies(self):
        """Test edge case with no similar movies."""
        empty_matrix = np.zeros((3, 3))
        np.save('test_empty_matrix.npy', empty_matrix)
        recommendations = recommend_movies(
            "Movie 1", self.movies_df, 'test_empty_matrix.npy', 'test_empty_matrix.npy'
        )
        self.assertTrue(recommendations.empty, "Recommendations should be empty when no similar movies exist.")

    def test_recommend_movies(self):
        """Test recommend_movies function with valid data."""
        recommendations = recommend_movies(
            "Movie 1", self.movies_df, 'test_genre_matrix.npy', 'test_tag_matrix.npy'
        )
        self.assertEqual(len(recommendations), 2)
        self.assertNotIn("Movie 1", recommendations['title'].values)

if __name__ == "__main__":
    unittest.main()
