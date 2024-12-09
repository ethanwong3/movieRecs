import pandas as pd

# cleaning:
# - remove incomplete cols
# - format genres col (in movie dataset only)

def clean_movies(movies):
   
    movies = movies.dropna()
    movies['genres'] = movies['genres'].str.split('|')
    return movies

def clean_ratings(ratings):
    
    ratings = ratings.dropna()
    return ratings

# saving: filters then duplicates clean data into new CSV

def save_clean_data(movies, ratings):
    
    path_cleaned_movies = 'data/cleaned_movies.csv'
    path_cleaned_ratings = 'data/cleaned_ratings.csv'

    movies.to_csv(path_cleaned_movies, index=False) # do not include row index
    ratings.to_csv(path_cleaned_ratings, index=False) # do not include row index
    print(f"Cleaned datasets saved to {path_cleaned_movies} and {path_cleaned_ratings}.")

# main ##################################################################################

if __name__ == "__main__":
    # load raw data
    from data_load import load
    movies, ratings = load()

    # clean data
    cleaned_movies = clean_movies(movies)
    cleaned_ratings = clean_ratings(ratings)

    # save cleaned data
    save_clean_data(cleaned_movies, cleaned_ratings)
