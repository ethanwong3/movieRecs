import os
import pandas as pd
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import recommend_movies

def test_recommender(movie_title):
    """
    Test the movie recommender with a given movie title.
    """
    print(f"\nTesting recommendations for: {movie_title}")

    # Load the cleaned datasets
    movies_df = pd.read_csv('data/cleaned_movies.csv')

    try:
        recommendations = recommend_movies(movie_title, movies_df)
        print("Recommended Movies:")
        print(recommendations)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test with various movie titles
    test_recommender("Toy Story (1995)")
    test_recommender("Jumanji (1995)")
    test_recommender("The Lion King (2019)")
    test_recommender("Shrek (2001)")
