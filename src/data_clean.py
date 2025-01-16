import pandas as pd
import os

"""
Clean movies data by reading file, handling missing data, and ensuring data is
all formatted correctly and consistently

Args: file_path a string path to the movies data file
Returns: pd.DataFrame is a cleaned movies dataframe.
"""

def clean_movies(file_path):
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

"""
Clean ratings data by turning file into DataFrame
Args: file_path is a string path to the ratings data file
Returns: pd.DataFrame is a cleaned ratings DataFrame
"""

def clean_ratings(file_path):
    ratings = pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"]
    )
    return ratings

"""
Clean tags data by reading file, handling missing data, and ensuring data is
all formatted correctly and consistently
Args: file_path is a string path to the tags data file
Returns: pd.DataFrame is a cleaned tags DataFrame
"""

def clean_tags(file_path):
    tags = pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "tag", "timestamp"]
    )
    # Ensure all tags are strings and handle missing/invalid values
    tags["tag"] = tags["tag"].fillna("unknown").astype(str).str.lower().str.strip()
    tags["tag"] = tags["tag"].replace("", "unknown")  # Handle empty strings explicitly

    # Debugging: Print sample tags and counts of missing/invalid tags
    print("Sample cleaned tags:", tags["tag"].head())
    print(f"Total missing tags after cleaning: {tags['tag'].isna().sum()}")
    print(f"Total empty tags after cleaning: {(tags['tag'] == 'unknown').sum()}")

    return tags

"""
Save cleaned movies, ratings, and tags data files to specified output directory
Args:
movies: pd.DataFrame holding cleaned movies
ratings: pd.DataFrame hodling cleaned ratings
tags: pd.DataFrame holding cleaned tags
output_dir: string path to directory
"""

def save_cleaned_data(movies, ratings, tags, output_dir="data"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Debugging: Check for any issues before saving
    if tags["tag"].isna().any():
        print("Warning: Missing tags detected before saving.")
    if tags["tag"].str.strip().eq("").any():
        print("Warning: Empty tags detected before saving.")

    # Save the cleaned data to CSV files
    movies.to_csv(os.path.join(output_dir, "cleaned_movies.csv"), index=False)
    ratings.to_csv(os.path.join(output_dir, "cleaned_ratings.csv"), index=False)
    tags.to_csv(os.path.join(output_dir, "cleaned_tags.csv"), index=False)
    print("Cleaned data saved successfully.")

if __name__ == "__main__":
    # Define file paths for the input data
    base_dir = "data/ml-10M100K"  # Adjust this path as needed
    movies_file = os.path.join(base_dir, "movies.dat")
    ratings_file = os.path.join(base_dir, "ratings.dat")
    tags_file = os.path.join(base_dir, "tags.dat")
    
    try:
        # Perform data cleaning steps
        print("Cleaning movies data...")
        movies = clean_movies(movies_file)
        print("Cleaning ratings data...")
        ratings = clean_ratings(ratings_file)
        print("Cleaning tags data...")
        tags = clean_tags(tags_file)

        # Save the cleaned data
        print("Saving cleaned data...")
        save_cleaned_data(movies, ratings, tags)
    except FileNotFoundError as e:
        print(f"Error: {e}")
