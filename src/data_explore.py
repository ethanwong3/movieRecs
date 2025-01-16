import pandas as pd

"""
Explore the cleaned movies dataset
Args: file_path is a string specifying the path to the cleaned movies CSV file
Prints: 
    - Total number of movies
    - Distribution of genres
    - A sample of the movies dataset
"""

def explore_movies(file_path):
    movies = pd.read_csv(file_path)
    print(f"Total movies: {len(movies)}")
    print(f"Genres Distribution:\n{movies['genres'].explode().value_counts()}")
    print(f"Sample movies:\n{movies.head()}")

"""
Explore the cleaned ratings dataset
Args: file_path is a string specifying the path to the cleaned ratings CSV file
Prints: 
    - Total number of ratings
    - Average rating value
    - A sample of the ratings dataset
"""

def explore_ratings(file_path):
    ratings = pd.read_csv(file_path)
    print(f"Total ratings: {len(ratings)}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    print(f"Sample ratings:\n{ratings.head()}")

if __name__ == "__main__":
    # Perform exploration on the cleaned movies and ratings datasets
    explore_movies('data/cleaned_movies.csv')
    explore_ratings('data/cleaned_ratings.csv')
