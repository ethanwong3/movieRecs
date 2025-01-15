import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_genre_column(movies_df):
    """Preprocess the genres column for TF-IDF."""
    movies_df['genres_str'] = movies_df['genres'].apply(lambda genres: " ".join(genres) if isinstance(genres, list) else "")
    return movies_df

def compute_genre_similarity(movies_df):
    """Compute a genre similarity matrix."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])
    return cosine_similarity(tfidf_matrix)

def compute_tag_similarity(tags_df, movies_df):
    """Compute a tag similarity matrix."""
    # Aggregate tags per movie
    tags_per_movie = tags_df.groupby('movieId')['tag'].apply(lambda tags: " ".join(tags)).reindex(movies_df['movieId'], fill_value="")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tags_per_movie)
    return cosine_similarity(tfidf_matrix)

def compute_ratings_similarity(ratings_df, movies_df):
    """Compute a ratings similarity matrix using collaborative filtering."""
    user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return cosine_similarity(user_movie_matrix.T)  # Transpose to get movie-to-movie similarity

def preprocess_data():
    """Preprocess data and compute similarity matrices."""
    # Load cleaned data
    movies_file = "data/cleaned_movies.csv"
    tags_file = "data/cleaned_tags.csv"
    ratings_file = "data/cleaned_ratings.csv"

    movies_df = pd.read_csv(movies_file)
    tags_df = pd.read_csv(tags_file)
    ratings_df = pd.read_csv(ratings_file)

    # Compute similarity matrices
    print("Computing genre similarity matrix...")
    movies_df = preprocess_genre_column(movies_df)
    genre_similarity = compute_genre_similarity(movies_df)
    print("Genre similarity matrix computed.")

    print("Computing tag similarity matrix...")
    tag_similarity = compute_tag_similarity(tags_df, movies_df)
    print("Tag similarity matrix computed.")

    print("Computing ratings similarity matrix...")
    ratings_similarity = compute_ratings_similarity(ratings_df, movies_df)
    print("Ratings similarity matrix computed.")

    # Save results
    print("Saving similarity matrices...")
    np.save("data/genre_similarity_matrix.npy", genre_similarity)
    np.save("data/tag_similarity_matrix.npy", tag_similarity)
    np.save("data/ratings_similarity_matrix.npy", ratings_similarity)
    print("Similarity matrices saved successfully.")

if __name__ == "__main__":
    preprocess_data()
