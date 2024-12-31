import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# function to recommend movies based on genre and tag similarity
def recommend_movies(movie_title, movies_df, relevance_matrix, genre_weight=0.6, tag_weight=0.4, top_n=10):

    # retrieve movie id from title
    movie = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if movie.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    movie_id = movie.index[0]

    # calculate similarity
    genre_sim = cosine_similarity(
        TfidfVectorizer(stop_words='english').fit_transform(movies_df['genres'])
    )
    tag_sim = cosine_similarity(relevance_matrix)
    blended_sim = (genre_weight * genre_sim) + (tag_weight * tag_sim)

    # get top N recommendations
    similar_movie_ids = blended_sim[movie_id].argsort()[::-1][1:top_n+1] # argsort()[::-1] ==> descending order
    if not similar_movie_ids.size:
        print("No similar movies found.")
        return pd.DataFrame()

    recommendations = movies_df.iloc[similar_movie_ids][['title', 'genres']] # iloc ==> selects rows by indices
    return recommendations

# example execution function
def run_example():
    try:
        # Load data
        movies_df = pd.read_csv('data/cleaned_movies.csv')
        relevance_matrix = pd.read_csv('data/movie_tag_relevance.csv', index_col='movieId')

        print("start")
        
        # Example: Recommend movies for "Toy Story (1995)"
        recommendations = recommend_movies("Scream (1996)", movies_df, relevance_matrix)
        print("Recommended Movies:")
        print(recommendations)
    except ValueError as e:
        print(e)
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")

# ensure this block only runs when executed directly
if __name__ == "__main__":
    run_example()


"""

Method: Hybrid Incremental Recommendation

The current implementation of your movie recommender uses a 
Hybrid Recommendation System combining Content-Based Filtering techniques 
with Genre Similarity and Tag Similarity (via genome scores). 

Pros:

- balanced complexity and performance friendly
- realistic output
- scalable and extensible

How it Works:

Similar movies are found by calculating a similarity score for each movie that blends the movie's aspects below

- genre similarity calculated dynamically using tf-idf vectorisation and cosine similarity
    - tfidf vectoriser (term frequency inverse doc frequency) converts text (genres) into a numerical representation
      this means that each genre is assigned a weight based on how frequent it appears and how unique it is
      genres such as action might have a lower weight since they appear often. Then, fit_transform
      computes the numerical representation and returns a sparse matrix.
    - cosine similarity measures the cosine between the vectors of rows (movies) in a matrix.
      We use this as cosine similarity disregards the magnitude of the angle and strictly captures the presence and weighting of shared genres.
    
- tag similarity is derived from genome data and uses a preprocessed relevance matrix to capture tag-based similarities


POSSIBLE IMPROVEMENTS

- testing
- performance optimisation through parallel processing for large dataset ipoerations (esp in data explore and recommender for better edge case handling and caching optimizations)

next steps

- implement collaborative filtering and hybrid recommendation system

"""