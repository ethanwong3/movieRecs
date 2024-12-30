import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# function to recommend movies based on genre and tag similarity
def recommend_movies(movie_title, movies_df, relevance_matrix, top_n=10):
    # case-insensitive search for movie title
    matching_movies = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    movie_id = matching_movies.index[0]

    # insufficient data
    if len(movies_df) == 0 or len(relevance_matrix) == 0:
        print("No valid data available for recommendations.")
        return pd.DataFrame(['title', 'genres'])
    if relevance_matrix.shape[0] != len(movies_df):
        raise ValueError("Relevance matrix dimensions do not match the movies dataset")

    # FIXED (using passed matrix directly) (replaces commented out block below)
    similarity_scores = relevance_matrix[movie_id]
    
    # check for all zero sim scores
    if np.all(similarity_scores == 0):
        print("No similar movies found")
        return pd.DataFrame(columns=['title', 'genres'])

    """
    # compute genre and tag similarity
    genre_tfidf = TfidfVectorizer(stop_words = 'english')
    genre_matrix = genre_tfidf.fit_transform(movies_df['genres'])
    genre_sim = cosine_similarity(genre_matrix)
    if relevance_matrix.shape[0] != len(movies_df):
        raise ValueError("Relevance matrix dimensions do not match the movies dataset")
    tag_sim = cosine_similarity(relevance_matrix)
    
    # blend similarities
    blended_sim = 0.5 * genre_sim + 0.5 * tag_sim
    
    # get sim scores for input movie
    similarity_scores = blended_sim[movie_id]
    """
    
    # get top N recs
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    top_similarities = similarity_scores[similar_indices]

    # check if recs are valid
    if np.all(top_similarities == 0):
        print("No similar movies found.")
        return pd.DataFrame(columns=['title', 'genres'])

    # return valid recs
    recommendations = movies_df.iloc[similar_indices][['title', 'genres']]
    return recommendations

# example execution function
def run_example():
    try:
        # Load data
        movies_df = pd.read_csv('data/cleaned_movies.csv')
        relevance_matrix = pd.read_csv('data/movie_tag_relevance.csv', index_col='movieId')

        # Example: Recommend movies for "Toy Story (1995)"
        recommendations = recommend_movies("Toy Story (1995)", movies_df, relevance_matrix)
        print("Recommended Movies:")
        print(recommendations)
    except ValueError as e:
        print(e)
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")

# ensure this block only runs when executed directly
if __name__ == "__main__":
    run_example()


"""
possible improvmenets

- testing
- performance optimisation through parallel processing for large dataset ipoerations (esp in data explore and recommender for better edge case handling and caching optimizations)

next steps

- implement collaborative filtering and hybrid recommendation system

    """