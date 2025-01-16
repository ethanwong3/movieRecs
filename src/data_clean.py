import pandas as pd
import os

def clean_movies(file_path):
    """Clean the movies data."""
    movies = pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        header=None,
        names=["movieId", "title", "genres"]
    )
    # Ensure genres are lists and handle missing genres
    movies["genres"] = movies["genres"].fillna("unknown").str.split("|")
    return movies

def clean_ratings(file_path):
    """Clean the ratings data."""
    ratings = pd.read_csv(file_path, sep="::", engine="python", header=None, names=["userId", "movieId", "rating", "timestamp"])
    return ratings

def clean_tags(file_path):
    """Clean the tags data."""
    tags = pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "tag", "timestamp"]
    )
    # Ensure all tags are strings and handle missing/invalid values
    tags["tag"] = tags["tag"].fillna("unknown").astype(str).str.lower().str.strip()

    # Replace tags containing only whitespace or invalid values with "unknown"
    tags["tag"] = tags["tag"].replace("", "unknown")
    tags["tag"] = tags["tag"].replace("unknown", "unknown")  # Reinforce replacements

    # Debugging: Print sample tags and missing counts
    print("Sample cleaned tags:", tags["tag"].head())
    print(f"Total missing tags after cleaning: {tags['tag'].isna().sum()}")
    print(f"Total empty tags after cleaning: {(tags['tag'] == 'unknown').sum()}")

    return tags

def save_cleaned_data(movies, ratings, tags, output_dir="data"):
    """Save cleaned data to CSV."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Debugging: Check for missing tags before saving
    if tags["tag"].isna().any():
        print("Warning: Missing tags detected before saving.")
    if tags["tag"].str.strip().eq("").any():
        print("Warning: Empty tags detected before saving.")

    # Save files
    movies.to_csv(os.path.join(output_dir, "cleaned_movies.csv"), index=False)
    ratings.to_csv(os.path.join(output_dir, "cleaned_ratings.csv"), index=False)
    tags.to_csv(os.path.join(output_dir, "cleaned_tags.csv"), index=False)
    print("Cleaned data saved successfully.")


if __name__ == "__main__":
    # Update the paths to match the subdirectory
    base_dir = "data/ml-10M100K"  # Adjust the path here
    movies_file = os.path.join(base_dir, "movies.dat")
    ratings_file = os.path.join(base_dir, "ratings.dat")
    tags_file = os.path.join(base_dir, "tags.dat")
    
    try:
        print("Cleaning movies data...")
        movies = clean_movies(movies_file)
        print("Cleaning ratings data...")
        ratings = clean_ratings(ratings_file)
        print("Cleaning tags data...")
        tags = clean_tags(tags_file)

        print("Saving cleaned data...")
        save_cleaned_data(movies, ratings, tags)
    except FileNotFoundError as e:
        print(f"Error: {e}")
