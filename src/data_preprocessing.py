import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_genre_column(movies_df):
    """Preprocess the genres column to ensure valid inputs for TF-IDF."""
    if 'genres' not in movies_df.columns:
        raise ValueError("The 'genres' column is missing in the dataset.")

    # Debugging: Check initial genre data
    print("Initial genre data sample:", movies_df['genres'].head())

    # Convert genres to a space-separated string
    movies_df['genres_str'] = movies_df['genres'].apply(
        lambda genres: ' '.join(genres) if isinstance(genres, list) else genres
    )

    # Replace empty strings or NaNs with 'unknown'
    movies_df['genres_str'] = movies_df['genres_str'].fillna('unknown')
    movies_df['genres_str'] = movies_df['genres_str'].replace('', 'unknown')

    # Debugging: Check processed genre strings
    print("Processed genre strings sample:", movies_df['genres_str'].head())

    return movies_df


def precompute_genre_similarity(movies_df):
    """Compute the genre similarity matrix using TF-IDF and cosine similarity."""
    # Debugging: Check the unique values in 'genres_str'
    print("Unique genres strings count:", movies_df['genres_str'].nunique())

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])

    # Debugging: Check the TF-IDF vocabulary size and shape of the matrix
    print("TF-IDF Vocabulary Size:", len(tfidf.vocabulary_))
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

    # Compute cosine similarity
    genre_similarity = cosine_similarity(tfidf_matrix)

    # Debugging: Check a sample of the similarity matrix
    print("Sample similarity matrix row (index 0):", genre_similarity[0][:10])

    return genre_similarity


def preprocess_data():
    """Preprocess data and compute the genre similarity matrix."""
    # Load cleaned movies data
    movies_file = "data/cleaned_movies.csv"
    movies_df = pd.read_csv(movies_file)

    # Preprocess genres column
    print("Preprocessing genres column...")
    movies_df = preprocess_genre_column(movies_df)

    # Compute genre similarity matrix
    print("Computing genre similarity matrix...")
    genre_similarity = precompute_genre_similarity(movies_df)

    # Save the genre similarity matrix
    print("Saving genre similarity matrix to file...")
    np.save("data/genre_similarity_matrix.npy", genre_similarity)
    print("Genre similarity matrix saved successfully!")


if __name__ == "__main__":
    preprocess_data()
