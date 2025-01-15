import pandas as pd
import numpy as np

def recommend_movies(movie_title, movies_df, genre_similarity, tag_similarity, ratings_similarity,
                     genre_weight=0.5, tag_weight=0.3, ratings_weight=0.2, top_n=10, similarity_threshold=0.1):
    """Recommend movies based on blended similarity with adjustable weights."""
    # Find the movie index
    matching_movies = movies_df[
        movies_df['title'].str.lower().str.strip().str.contains(
            movie_title.lower().strip(), case=False, regex=False
        )
    ]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")
    movie_index = matching_movies.index[0]

    # Compute blended similarity
    blended_similarity = (
        genre_weight * genre_similarity +
        tag_weight * tag_similarity +
        ratings_weight * ratings_similarity
    )
    similarity_scores = blended_similarity[movie_index]
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
        movies_df = pd.read_csv("data/cleaned_movies.csv")
        genre_similarity = np.load("data/genre_similarity_matrix.npy")
        tag_similarity = np.load("data/tag_similarity_matrix.npy")
        ratings_similarity = np.load("data/ratings_similarity_matrix.npy")
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
            print("Using default weights (0.5 genres, 0.3 tags, 0.2 ratings)...")
            recommendations = recommend_movies(user_input, movies_df, genre_similarity, tag_similarity, ratings_similarity)
            if recommendations.empty:
                print(f"No similar movies found for '{user_input}'. Try another title.")
            else:
                print(f"Recommendations based on '{user_input}':")
                print(recommendations)
        except ValueError as e:
            print(e)
