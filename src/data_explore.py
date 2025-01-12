import pandas as pd

def explore_movies(file_path):
    """Explore movies dataset."""
    movies = pd.read_csv(file_path)
    print(f"Total movies: {len(movies)}")
    print(f"Genres Distribution:\n{movies['genres'].explode().value_counts()}")
    print(f"Sample movies:\n{movies.head()}")

def explore_ratings(file_path):
    """Explore ratings dataset."""
    ratings = pd.read_csv(file_path)
    print(f"Total ratings: {len(ratings)}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    print(f"Sample ratings:\n{ratings.head()}")

if __name__ == "__main__":
    explore_movies('data/cleaned_movies.csv')
    explore_ratings('data/cleaned_ratings.csv')
