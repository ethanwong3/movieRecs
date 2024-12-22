import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# use genre and tag similarity to recommend movies

def recommend_movies(movie_title, movies_df, relevance_matrix, top_n=10):
    
    # Case-insensitive search for movie title
    matching_movies = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    movie_id = matching_movies.index[0]

    # Compute similarities
    genre_sim = cosine_similarity(
        TfidfVectorizer(stop_words='english').fit_transform(movies_df['genres'])
    )
    tag_sim = cosine_similarity(relevance_matrix)
    blended_sim = 0.5 * genre_sim + 0.5 * tag_sim

    # Get top N recommendations
    similar_indices = blended_sim[movie_id].argsort()[::-1][1:top_n+1]
    if not similar_indices.size:
        print("No similar movies found.")
        return pd.DataFrame()

    recommendations = movies_df.iloc[similar_indices][['title', 'genres']]
    return recommendations


if __name__ == "__main__":
    
    movies_df = pd.read_csv('data/cleaned_movies.csv')
    relevance_matrix = pd.read_csv('data/movie_tag_relevance.csv', index_col='movieId')
    
    try:
        recommendations = recommend_movies("Toy Story (1995)", movies_df, relevance_matrix)
        print("Recommended Movies:")
        print(recommendations)
    except ValueError as e:
        print(e)

    """
possible improvmenets

- testing
- performance optimisation through parallel processing for large dataset ipoerations (esp in data explore and recommender for better edge case handling and caching optimizations)

next steps

- implement collaborative filtering and hybrid recommendation system

    """