import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def precompute_genre_similarity(movies_df):
    """Compute a genre similarity matrix using TF-IDF and cosine similarity."""
    # Ensure genres are processed correctly
    movies_df["genres_str"] = movies_df["genres"].apply(
        lambda genres: " ".join(genres) if isinstance(genres, list) else ""
    )

    # Handle empty or missing genre strings
    movies_df["genres_str"] = movies_df["genres_str"].replace("", "unknown")

    # Debugging: Check sample genre strings
    print("Sample genre strings after processing:", movies_df["genres_str"].head())

    if movies_df["genres_str"].str.strip().eq("").any():
        raise ValueError("Found empty genre strings after processing. Check the input data.")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres_str"])
    genre_similarity = cosine_similarity(tfidf_matrix)

    return genre_similarity

def precompute_tag_similarity(tags_df, movies_df):
    """
    Compute a tag similarity matrix based on the tags data.
    """
    if "tag" not in tags_df.columns:
        raise KeyError("The 'tag' column is missing in the tags dataframe.")

    # Filter out rows with invalid tags
    tags_df = tags_df[tags_df["tag"] != "unknown"]

    # Group tags by movieId and join them into single strings
    tag_data = tags_df.groupby("movieId")["tag"].apply(lambda tags: " ".join(tags)).reindex(movies_df["movieId"], fill_value="")

    # Debugging: Print sample tag data
    print("Sample tag data after processing:", tag_data.head())

    # Use TF-IDF to vectorize tags
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(tag_data)

    # Compute cosine similarity
    tag_similarity = cosine_similarity(tfidf_matrix)

    # Debugging: Check a sample of the similarity matrix
    print("Sample Tag Similarity Row for Index 0:", tag_similarity[0])

    return tag_similarity

def precompute_ratings_similarity(ratings_df, movies_df):
    """Compute a ratings similarity matrix using user-movie ratings."""
    pivot_table = ratings_df.pivot(index="movieId", columns="userId", values="rating").fillna(0)
    pivot_table = pivot_table.reindex(index=movies_df["movieId"], fill_value=0)
    ratings_similarity = cosine_similarity(pivot_table)

    return ratings_similarity

def preprocess_data():
    """Preprocess data and compute similarity matrices."""
    try:
        # Load cleaned data
        print("Loading cleaned data...")
        movies_df = pd.read_csv("data/cleaned_movies.csv")
        tags_df = pd.read_csv("data/cleaned_tags.csv")
        ratings_df = pd.read_csv("data/cleaned_ratings.csv")

        # Debugging: Validate tags
        print("Checking for missing genres...")
        if movies_df["genres"].isna().any():
            raise ValueError("Found missing genres in the movies data.")

        print("Checking for missing tags...")
        if tags_df["tag"].isna().any() or tags_df["tag"].str.strip().eq("").any():
            raise ValueError("Found missing tags in the tags data.")

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

        print("Similarity matrices saved successfully.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
