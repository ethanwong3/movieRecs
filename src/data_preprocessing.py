import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
Compute a genre similarity matrix using TF-IDF and cosine similarity
Args: 
- movies_df: a Pandas DataFrame containing the cleaned movies dataset
Returns: a NumPy array representing the genre similarity matrix
"""

def precompute_genre_similarity(movies_df):
    # Convert genres into space-separated strings
    movies_df["genres_str"] = movies_df["genres"].apply(
        lambda genres: " ".join(genres) if isinstance(genres, list) else ""
    )

    # Replace empty genre strings with a placeholder
    movies_df["genres_str"] = movies_df["genres_str"].replace("", "unknown")

    # Debugging: Validate genres and check for empty strings
    print("Sample genre strings after processing:", movies_df["genres_str"].head())
    if movies_df["genres_str"].str.strip().eq("").any():
        raise ValueError("Found empty genre strings after processing. Check the input data.")

    # Compute TF-IDF matrix and genre similarity
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres_str"])
    genre_similarity = cosine_similarity(tfidf_matrix)

    return genre_similarity

"""
Compute a tag similarity matrix using TF-IDF and cosine similarity
Args: 
- tags_df: a Pandas DataFrame containing the cleaned tags dataset
- movies_df: a Pandas DataFrame containing the cleaned movies dataset
Returns: a NumPy array representing the tag similarity matrix
"""

def precompute_tag_similarity(tags_df, movies_df):
    # Ensure the tags column exists and filter invalid tags
    if "tag" not in tags_df.columns:
        raise KeyError("The 'tag' column is missing in the tags dataframe.")
    tags_df = tags_df[tags_df["tag"] != "unknown"]

    # Group tags by movieId and combine them into single strings
    tag_data = tags_df.groupby("movieId")["tag"].apply(lambda tags: " ".join(tags)).reindex(
        movies_df["movieId"], fill_value=""
    )

    # Debugging: Print sample processed tag data
    print("Sample tag data after processing:", tag_data.head())

    # Compute TF-IDF matrix and tag similarity
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(tag_data)
    tag_similarity = cosine_similarity(tfidf_matrix)

    return tag_similarity

"""
Compute a ratings similarity matrix using user-movie ratings
Args:
- ratings_df: a Pandas DataFrame containing the cleaned ratings dataset
- movies_df: a Pandas DataFrame containing the cleaned movies dataset
Returns: a NumPy array representing the ratings similarity matrix
"""

def precompute_ratings_similarity(ratings_df, movies_df):
    # Create a pivot table with movieId as rows, userId as columns, and ratings as values
    pivot_table = ratings_df.pivot(index="movieId", columns="userId", values="rating").fillna(0)

    # Ensure alignment with the movies dataset
    pivot_table = pivot_table.reindex(index=movies_df["movieId"], fill_value=0)

    # Compute cosine similarity for ratings
    ratings_similarity = cosine_similarity(pivot_table)

    return ratings_similarity

"""
Preprocess the datasets and compute the similarity matrices for genres, tags, and ratings
Raises: an exception with an error message if any issue occurs during preprocessing
"""

def preprocess_data():
    try:
        # Load cleaned datasets
        print("Loading cleaned data...")
        movies_df = pd.read_csv("data/cleaned_movies.csv")
        tags_df = pd.read_csv("data/cleaned_tags.csv")
        ratings_df = pd.read_csv("data/cleaned_ratings.csv")

        # Debugging: Validate genres and tags
        print("Checking for missing genres...")
        if movies_df["genres"].isna().any():
            raise ValueError("Found missing genres in the movies data.")

        print("Checking for missing tags...")
        if tags_df["tag"].isna().any() or tags_df["tag"].str.strip().eq("").any():
            raise ValueError("Found missing tags in the tags data.")

        # Compute similarity matrices
        print("Computing genre similarity matrix...")
        genre_similarity = precompute_genre_similarity(movies_df)

        print("Computing tag similarity matrix...")
        tag_similarity = precompute_tag_similarity(tags_df, movies_df)

        print("Computing ratings similarity matrix...")
        ratings_similarity = precompute_ratings_similarity(ratings_df, movies_df)

        # Save the computed matrices
        print("Saving similarity matrices...")
        np.save("data/genre_similarity_matrix.npy", genre_similarity)
        np.save("data/tag_similarity_matrix.npy", tag_similarity)
        np.save("data/ratings_similarity_matrix.npy", ratings_similarity)

        print("Similarity matrices saved successfully.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    # Execute preprocessing when the script is run directly
    preprocess_data()
