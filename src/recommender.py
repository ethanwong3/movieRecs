import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def recommend_movies(movie_title, movies_df, genre_similarity, top_n=10, similarity_threshold=0.1):
    """Recommend movies based on title and genre similarity."""
    # Find the movie index
    matching_movies = movies_df[movies_df['title'].str.contains(movie_title, case=False)]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")
    movie_index = matching_movies.index[0]

    # Get similarity scores
    similarity_scores = genre_similarity[movie_index]
    similarity_scores[movie_index] = 0  # Exclude self-similarity
    valid_indices = np.where(similarity_scores >= similarity_threshold)[0]

    # Check for no valid recommendations
    if valid_indices.size == 0:
        return pd.DataFrame(columns=["title", "genres"])  # Empty DataFrame for no recommendations

    # Get top N recommendations
    similar_movie_indices = similarity_scores.argsort()[::-1][:top_n]
    recommendations = movies_df.iloc[similar_movie_indices][['title', 'genres']]
    return recommendations

if __name__ == "__main__":
    # Load processed data
    try:
        movies_df = pd.read_csv("data/processed_movies.csv")
        genre_similarity = np.load("data/genre_similarity_matrix.npy")
    except FileNotFoundError as e:
        print(f"Error: Required data file not found. Please run preprocessing first. ({e})")
        exit(1)

    # Prompt user for input
    while True:
        user_input = input("Enter a movie title (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting the recommender. Goodbye!")
            break

        try:
            recommendations = recommend_movies(user_input, movies_df, genre_similarity)
            if recommendations.empty:
                print(f"No similar movies found for '{user_input}'. Try another title.")
            else:
                print(f"Recommendations based on '{user_input}':")
                print(recommendations)
        except ValueError as e:
            print(e)
