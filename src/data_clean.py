import pandas as pd

def clean_movies(file_path):
    """Clean movies dataset."""
    movies = pd.read_csv(file_path, sep="::", engine="python", header=None,
                         names=["movieId", "title", "genres"])
    movies['genres'] = movies['genres'].str.split('|')
    return movies.dropna()

def clean_tags(file_path):
    """Clean tags dataset."""
    tags = pd.read_csv(file_path, sep="::", engine="python", header=None,
                       names=["userId", "movieId", "tag", "timestamp"])
    tags['tag'] = tags['tag'].str.lower().str.strip()
    return tags.dropna()

def clean_ratings(file_path):
    """Clean ratings dataset."""
    ratings = pd.read_csv(file_path, sep="::", engine="python", header=None,
                          names=["userId", "movieId", "rating", "timestamp"])
    return ratings.dropna()

if __name__ == "__main__":
    movies = clean_movies('data/movies.dat')
    tags = clean_tags('data/tags.dat')
    ratings = clean_ratings('data/ratings.dat')

    # Save cleaned datasets
    movies.to_csv('data/cleaned_movies.csv', index=False)
    tags.to_csv('data/cleaned_tags.csv', index=False)
    ratings.to_csv('data/cleaned_ratings.csv', index=False)
    print("Data cleaned successfully.")
