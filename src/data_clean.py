import pandas as pd
import os

def clean_movies(file_path):
    """Clean the movies data."""
    movies = pd.read_csv(file_path, sep="::", engine="python", header=None, names=["movieId", "title", "genres"])
    movies["genres"] = movies["genres"].str.split("|")
    return movies

def clean_ratings(file_path):
    """Clean the ratings data."""
    ratings = pd.read_csv(file_path, sep="::", engine="python", header=None, names=["userId", "movieId", "rating", "timestamp"])
    return ratings

def clean_tags(file_path):
    """Clean the tags data."""
    tags = pd.read_csv(file_path, sep="::", engine="python", header=None, names=["userId", "movieId", "tag", "timestamp"])
    tags["tag"] = tags["tag"].str.lower().str.strip()
    return tags

def save_cleaned_data(movies, ratings, tags, output_dir="data"):
    """Save cleaned data to CSV."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
