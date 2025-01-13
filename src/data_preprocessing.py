import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_genre_column(movies_df):
    """Preprocess the genres column for TF-IDF."""
    # Check if 'genres' column exists
    if 'genres' not in movies_df.columns:
        raise ValueError("'genres' column is missing in the input data.")

    # Ensure genres are lists and join them into space-separated strings
    movies_df['genres'] = movies_df['genres'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else '')

    # Replace empty genres with a placeholder
    movies_df['genres'] = movies_df['genres'].replace('', 'unknown')

    return movies_df

def precompute_genre_similarity(movies_df):
    """Precompute genre similarity matrix using TF-IDF and cosine similarity."""
    # Preprocess genres to ensure they are strings
    movies_df = preprocess_genre_column(movies_df)

    # Compute TF-IDF and handle empty vocabulary gracefully
    try:
        tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(movies_df['genres'])
    except ValueError as e:
        print("Error while processing genres for TF-IDF:", e)
        print("Genres column preview:", movies_df['genres'].head())
        raise

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def preprocess_data():
    """Preprocess the data to generate similarity matrices."""
    # Load cleaned data
    movies_df = pd.read_csv('data/cleaned_movies.csv')

    # Precompute genre similarity matrix
    print("Computing genre similarity matrix...")
    genre_similarity = precompute_genre_similarity(movies_df)

    # Save results
    np.save('data/genre_similarity_matrix.npy', genre_similarity)
    print("Genre similarity matrix saved successfully.")

if __name__ == "__main__":
    preprocess_data()
