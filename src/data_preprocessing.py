import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_genre_column(movies_df):
    """Preprocess the genres column for TF-IDF."""
    movies_df['genres'] = movies_df['genres'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else ''
    )
    movies_df['genres'] = movies_df['genres'].replace('', 'unknown')
    print("Total empty genres after cleaning:", movies_df['genres'].isnull().sum())
    return movies_df


def precompute_genre_similarity(movies_df):
    """Compute a genre similarity matrix using TF-IDF and cosine similarity."""
    movies_df['genres_str'] = movies_df['genres'].apply(
        lambda genres: " ".join(genres) if isinstance(genres, list) else genres
    )

    print("Sample genre strings after processing:", movies_df['genres_str'].head())

    if movies_df['genres_str'].isnull().any() or (movies_df['genres_str'] == "").any():
        raise ValueError("Found empty genre strings after processing. Check the input data.")

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])
    print("TF-IDF Matrix Shape for Genres:", tfidf_matrix.shape)
    return cosine_similarity(tfidf_matrix)


def precompute_tag_similarity(tags_df, movies_df):
    """Compute a tag similarity matrix using grouped tags and cosine similarity."""
    tag_data = tags_df.groupby('movieId')['tag'].apply(
        lambda tags: " ".join(str(tag) for tag in tags if isinstance(tag, str))
    ).reindex(movies_df["movieId"], fill_value="")

    print("Sample tag data after grouping:", tag_data.head())

    tag_data = tag_data.replace('', 'unknown').fillna('unknown')

    if tag_data.str.strip().nunique() == 1 and tag_data.iloc[0] == 'unknown':
        raise ValueError("All tags are empty after processing. Check the tags dataset.")

    tfidf = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = tfidf.fit_transform(tag_data)
    except ValueError as e:
        print("Error during TF-IDF vectorization for tags:", e)
        raise ValueError("Found empty tag strings after processing. Check the tags dataset.") from e

    print("TF-IDF Matrix Shape for Tags:", tfidf_matrix.shape)
    return cosine_similarity(tfidf_matrix)


def precompute_ratings_similarity(ratings_df, movies_df):
    """Compute a ratings similarity matrix based on average ratings."""
    avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reindex(movies_df["movieId"], fill_value=0)
    print("Sample average ratings:", avg_ratings.head())
    return cosine_similarity(avg_ratings.values.reshape(-1, 1))


def preprocess_data():
    """Preprocess data and compute necessary matrices."""
    # Load datasets
    movies_file = "data/cleaned_movies.csv"
    tags_file = "data/cleaned_tags.csv"
    ratings_file = "data/cleaned_ratings.csv"

    movies_df = pd.read_csv(movies_file)
    tags_df = pd.read_csv(tags_file)
    ratings_df = pd.read_csv(ratings_file)

    # Validate input datasets
    if 'movieId' not in movies_df.columns or 'genres' not in movies_df.columns:
        raise ValueError("Missing essential columns in the movies dataset.")
    if 'movieId' not in tags_df.columns or 'tag' not in tags_df.columns:
        raise ValueError("Missing essential columns in the tags dataset.")
    if 'movieId' not in ratings_df.columns or 'rating' not in ratings_df.columns:
        raise ValueError("Missing essential columns in the ratings dataset.")

    # Preprocess genres
    movies_df = preprocess_genre_column(movies_df)

    # Debugging: Check tags_df structure
    print("Sample tags_df before processing:")
    print(tags_df.head())

    # Ensure movieId and tag are correct data types
    tags_df['movieId'] = tags_df['movieId'].astype(int)
    tags_df['tag'] = tags_df['tag'].astype(str)

    movies_df['movieId'] = movies_df['movieId'].astype(int)

    # Group tags and merge with movies_df
    print("Merging tags into movies_df...")
    grouped_tags = tags_df.groupby('movieId')['tag'].apply(
        lambda tags: " ".join(str(tag) for tag in tags)
    ).reset_index()
    print("Sample grouped tags:")
    print(grouped_tags.head())

    # Merge grouped tags with movies_df
    movies_df = movies_df.merge(grouped_tags, on='movieId', how='left')

    # Debugging: Validate merge result
    print("Sample movies_df after merging tags:")
    print(movies_df[['movieId', 'tags']].head())

    # Ensure 'tags' column exists and fill missing values
    if 'tag' in movies_df.columns:
        movies_df.rename(columns={"tag": "tags"}, inplace=True)
    if 'tags' not in movies_df.columns:
        raise ValueError("Merge operation failed to add 'tags' column to movies_df.")
    movies_df['tags'] = movies_df['tags'].fillna('unknown')

    # Compute similarity matrices
    print("Computing genre similarity matrix...")
    genre_similarity = precompute_genre_similarity(movies_df)

    print("Computing tag similarity matrix...")
    tag_similarity = precompute_tag_similarity(tags_df, movies_df)

    print("Computing ratings similarity matrix...")
    ratings_similarity = precompute_ratings_similarity(ratings_df, movies_df)

    # Save results
    print("Saving similarity matrices...")
    np.save("data/genre_similarity_matrix.npy", genre_similarity)
    np.save("data/tag_similarity_matrix.npy", tag_similarity)
    np.save("data/ratings_similarity_matrix.npy", ratings_similarity)
    print("All similarity matrices saved successfully.")

    # Save updated movies_df with tags
    print("Saving updated movies data with tags...")
    movies_df.to_csv("data/cleaned_movies.csv", index=False)
    print("Movies data saved successfully.")


if __name__ == "__main__":
    preprocess_data()
