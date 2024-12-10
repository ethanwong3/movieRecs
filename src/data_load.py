import pandas as pd
import os

# Data Load:
# loads movie and rating data into pandas memory

def load():

    path_ratings = 'data/ml-latest/ratings.csv'
    path_movies = 'data/ml-latest/movies.csv'

    try:
        if not os.path.exists(path_ratings):
            raise FileNotFoundError(f"File not found: {path_ratings}")
        if not os.path.exists(path_movies):
            raise FileNotFoundError(f"File not found: {path_movies}")

        ratings = pd.read_csv(path_ratings)
        movies = pd.read_csv(path_movies)

        # display previews of the datasets
        print("Ratings Data Preview:")
        print(ratings.head())
        print("Movies Data Preview:")
        print(movies.head())

        return movies, ratings

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the files is empty or corrupted: {e}")
        return None, None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

# main ##############################################################

if __name__ == "__main__":
    
    movies, ratings = load()
    
    if movies is not None and ratings is not None:
        print("Datasets loaded successfully.")
    else:
        print("Failed to load datasets. Please check the files and paths.")
