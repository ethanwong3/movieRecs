import pandas as pd

def clean_movies(movies):
    """
    Cleans the movies dataset by handling missing values and formatting the genres column.
    """
    # Print missing values before cleaning
    print(f"Missing values in movies dataset before cleaning:\n{movies.isnull().sum()}")

    # Remove rows with missing values
    movies = movies.dropna()

    # Ensure the genres column is split into lists
    movies['genres'] = movies['genres'].str.split('|')

    # Print dataset information after cleaning
    print(f"Cleaned movies dataset: {len(movies)} movies remaining.")
    return movies

def clean_ratings(ratings):
    """
    Cleans the ratings dataset by handling missing values.
    """
    # Print missing values before cleaning
    print(f"Missing values in ratings dataset before cleaning:\n{ratings.isnull().sum()}")

    # Remove rows with missing values
    ratings = ratings.dropna()

    # Print dataset information after cleaning
    print(f"Cleaned ratings dataset: {len(ratings)} ratings remaining.")
    return ratings

def save_clean_data(movies, ratings):
    """
    Saves the cleaned movies and ratings datasets to new CSV files.
    """
    path_cleaned_movies = 'data/cleaned_movies.csv'
    path_cleaned_ratings = 'data/cleaned_ratings.csv'

    # Save cleaned datasets to CSV files
    movies.to_csv(path_cleaned_movies, index=False)
    ratings.to_csv(path_cleaned_ratings, index=False)
    print(f"Cleaned datasets saved to {path_cleaned_movies} and {path_cleaned_ratings}.")

if __name__ == "__main__":
    # Load raw data
    from data_load import load
    movies, ratings = load()

    # Clean data
    cleaned_movies = clean_movies(movies)
    cleaned_ratings = clean_ratings(ratings)

    # Save cleaned data
    save_clean_data(cleaned_movies, cleaned_ratings)
