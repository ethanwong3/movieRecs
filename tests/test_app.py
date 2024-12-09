import unittest
import pandas as pd
from src.data_clean import clean_movies, clean_ratings
from src.recommender import recommend_movies_based_on_genres


class TestMovieRecommender(unittest.TestCase):
   
   # set up test data for functions
    
    def setUp(self):

        self.movies = pd.DataFrame({
            'movieId': [1, 2, 3],
            'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)'],
            'genres': ['Adventure|Animation|Children', 'Adventure|Children|Fantasy', 'Action|Crime|Thriller']
        })
        
        self.ratings = pd.DataFrame({
            'userId': [1, 1, 2],
            'movieId': [1, 2, 3],
            'rating': [4.0, 3.5, 5.0],
            'timestamp': [964982703, 964981247, 964982224]
        })

    # Tests for data_clean.py 
    # - testing clean_movies
    # - testing clean_ratings
    
    def test_clean_movies(self):
        cleaned_movies = clean_movies(self.movies)
        # Check that genres are split correctly
        self.assertEqual(cleaned_movies['genres'][0], ['Adventure', 'Animation', 'Children'])
        # Check that no rows were dropped (no missing values in the sample)
        self.assertEqual(len(cleaned_movies), 3)

    def test_clean_ratings(self):
        cleaned_ratings = clean_ratings(self.ratings)
        # Ensure no rows are dropped (no missing values in the sample)
        self.assertEqual(len(cleaned_ratings), 3)

    # Tests for recommender.py 
    # - testing the recommending function
    
    def test_recommend_movies_based_on_genres(self):
        # Clean movies data first
        self.movies['genres'] = self.movies['genres'].str.split('|')
        # Test valid input
        recommendations = recommend_movies_based_on_genres("Toy Story (1995)", self.movies)
        # Check that it returns 2 recommendations
        self.assertEqual(len(recommendations), 2)
        # Verify that "Jumanji (1995)" is one of the recommendations
        self.assertIn("Jumanji (1995)", recommendations['title'].values)
        # Test invalid input
        with self.assertRaises(IndexError):
            recommend_movies_based_on_genres("Nonexistent Movie", self.movies)

    # Tests for data integrity
    # testing that movies with missing genres are handled accurately
    
    def test_missing_genres_handling(self):
        movies_with_missing_genres = self.movies.copy()
        movies_with_missing_genres.loc[1, 'genres'] = None  # Set one row's genres to None
        cleaned_movies = clean_movies(movies_with_missing_genres)
        # Ensure the row with missing genres is dropped
        self.assertEqual(len(cleaned_movies), 2)

if __name__ == "__main__":
    unittest.main()
