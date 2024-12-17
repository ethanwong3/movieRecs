import sys
import os
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import recommend_movies_based_on_genres

# Load the cleaned datasets
movies_df = pd.read_csv('data/cleaned_movies.csv')
ratings_df = pd.read_csv('data/cleaned_ratings.csv')

def test_recommender(movie_title):
    print(f"\nTesting recommendations for: {movie_title}")
    try:
        recommendations = recommend_movies_based_on_genres(movie_title, movies_df, ratings_df)
        print(recommendations)
    except ValueError as e:
        print(e)

# Test the recommender with different movies
test_recommender("Toy Story (1995)")
test_recommender("Jumanji (1995)")
test_recommender("The Lion King (1994)")
test_recommender("Shrek (2001)")
