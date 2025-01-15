import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from recommender import recommend_movies

class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies_df = pd.read_csv("data/cleaned_movies.csv")
        cls.genre_similarity = np.load("data/genre_similarity_matrix.npy")
        cls.tag_similarity = np.load("data/tag_similarity_matrix.npy")
        cls.ratings_similarity = np.load("data/ratings_similarity_matrix.npy")

    def test_default_weights(self):
        recommendations = recommend_movies(
            "Toy Story (1995)", self.movies_df,
            self.genre_similarity, self.tag_similarity, self.ratings_similarity
        )
        self.assertFalse(recommendations.empty, "Recommendations should not be empty.")

    def test_genre_weight_high(self):
        recommendations = recommend_movies(
            "Toy Story (1995)", self.movies_df,
            self.genre_similarity, self.tag_similarity, self.ratings_similarity,
            genre_weight=0.8, tag_weight=0.1, ratings_weight=0.1
        )
        self.assertFalse(recommendations.empty, "Recommendations should not be empty.")

if __name__ == "__main__":
    unittest.main()
