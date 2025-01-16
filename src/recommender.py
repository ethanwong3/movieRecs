import pandas as pd
import numpy as np

"""
Generate movie recommendations based on blended similarity metrics.
Args:
- movie_title: string, the title of the movie to base recommendations on.
- movies_df: DataFrame, the dataset containing movie information.
- genre_similarity: NumPy array, the precomputed genre similarity matrix.
- tag_similarity: NumPy array, the precomputed tag similarity matrix.
- ratings_similarity: NumPy array, the precomputed ratings similarity matrix.
- genre_weight: float, weight for genre similarity in blended similarity (default: 0.5).
- tag_weight: float, weight for tag similarity in blended similarity (default: 0.3).
- ratings_weight: float, weight for ratings similarity in blended similarity (default: 0.2).
- top_n: int, the number of recommendations to return (default: 10).
- similarity_threshold: float, minimum similarity threshold for recommendations (default: 0.1).
Returns:
- A DataFrame containing the recommended movies and their genres.
Raises:
- ValueError if the movie title is not found in the dataset.
"""

def recommend_movies(movie_title, movies_df, genre_similarity, tag_similarity, ratings_similarity,
                     genre_weight=0.5, tag_weight=0.3, ratings_weight=0.2, top_n=10, similarity_threshold=0.1):
    # Locate the movie in the dataset
    matching_movies = movies_df[
        movies_df['title'].str.lower().str.strip().str.contains(
            movie_title.lower().strip(), case=False, regex=False
        )
    ]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")
    movie_index = matching_movies.index[0]

    # Compute the blended similarity score
    blended_similarity = (
        genre_weight * genre_similarity +
        tag_weight * tag_similarity +
        ratings_weight * ratings_similarity
    )
    similarity_scores = blended_similarity[movie_index]
    similarity_scores[movie_index] = 0  # Avoid recommending the input movie itself

    # Filter recommendations based on the similarity threshold
    valid_indices = np.where(similarity_scores >= similarity_threshold)[0]

    if valid_indices.size == 0:
        # Return an empty DataFrame if no recommendations meet the threshold
        return pd.DataFrame(columns=["title", "genres"])

    # Sort recommendations by similarity score and return the top N
    similar_movie_indices = similarity_scores.argsort()[::-1][:top_n]
    recommendations = movies_df.iloc[similar_movie_indices][['title', 'genres']]

    return recommendations


"""
Load the necessary data and prompt the user for input to generate recommendations.
"""

if __name__ == "__main__":
    try:
        # Load cleaned movies and precomputed similarity matrices
        movies_df = pd.read_csv("data/cleaned_movies.csv")
        genre_similarity = np.load("data/genre_similarity_matrix.npy")
        tag_similarity = np.load("data/tag_similarity_matrix.npy")
        ratings_similarity = np.load("data/ratings_similarity_matrix.npy")
    except FileNotFoundError as e:
        print(f"Error: Required data file not found. Please run preprocessing first. ({e})")
        exit(1)

    # Main recommendation loop
    while True:
        user_input = input("Enter a movie title (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting the recommender. Goodbye!")
            break

        try:
            # Use default similarity weights for recommendations
            print("Using default weights (0.5 genres, 0.3 tags, 0.2 ratings)...")
            recommendations = recommend_movies(
                user_input,
                movies_df,
                genre_similarity,
                tag_similarity,
                ratings_similarity
            )
            if recommendations.empty:
                print(f"No similar movies found for '{user_input}'. Try another title.")
            else:
                print(f"Recommendations based on '{user_input}':")
                print(recommendations)
        except ValueError as e:
            print(e)
