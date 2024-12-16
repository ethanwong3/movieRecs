import sys
import os
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import recommend_movies_based_on_genres

def test_recommend_movies():
    # Load the cleaned movies dataset
    movies_df = pd.read_csv('data/cleaned_movies.csv')

    # Test the recommender with a known movie title
    movie_title = "Toy Story (1995)"
    print(f"Testing recommendations for: {movie_title}\n")

    recommendations = recommend_movies_based_on_genres(movie_title, movies_df)
    
    # Print the recommendations
    print("Recommended Movies:")
    print(recommendations)

if __name__ == "__main__":
    test_recommend_movies()
