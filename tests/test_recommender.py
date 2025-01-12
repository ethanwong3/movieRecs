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
        self.movies_df = pd.DataFrame({
            'title': ['Movie 1', 'Movie 2', 'Movie 3'],
            'genres': ['Action|Comedy', 'Comedy|Drama', 'Action|Drama']
        })
        self.relevance_matrix = np.array([
            [0.8, 0.1, 0.2],
            [0.1, 0.7, 0.3],
            [0.2, 0.3, 0.9]
        ])

    def test_recommend_movies(self):
        recommendations = recommend_movies("Movie 1", self.movies_df, self.relevance_matrix, top_n=2)
        self.assertEqual(len(recommendations), 2)

    def test_invalid_movie_title(self):
        with self.assertRaises(ValueError):
            recommend_movies("Nonexistent Movie", self.movies_df, self.relevance_matrix)

    def test_no_similar_movies(self):
        empty_matrix = np.zeros((3, 3))
        recommendations = recommend_movies("Movie 1", self.movies_df, empty_matrix)
        self.assertTrue(recommendations.empty)

    def test_edge_case_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['title', 'genres'])
        with self.assertRaises(ValueError):
            recommend_movies("Movie 1", empty_df, self.relevance_matrix)

if __name__ == "__main__":
    unittest.main()
