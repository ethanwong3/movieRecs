import unittest
import pandas as pd
from recommender import recommend_movies

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Load preprocessed data."""
        self.movies_df = pd.read_csv('data/processed_movies.csv')
        self.relevance_matrix = pd.read_csv('data/movie_tag_relevance.csv', index_col='movieId')
        self.genre_similarity_file = 'data/genre_similarity_incremental.csv'

    def test_recommendation_workflow(self):
        """Test end-to-end workflow."""
        recommendations = recommend_movies("Toy Story (1995)", self.movies_df, self.relevance_matrix)
        self.assertGreater(len(recommendations), 0)  # Ensure at least one recommendation
        print(recommendations)

    def test_missing_precomputed_file(self):
        """Test system behavior without precomputed genre similarity file."""
        if os.path.exists(self.genre_similarity_file):
            os.remove(self.genre_similarity_file)
        
        with self.assertRaises(FileNotFoundError):
            recommend_movies("Toy Story (1995)", self.movies_df, self.relevance_matrix)

if __name__ == "__main__":
    unittest.main()
