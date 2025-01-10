import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix

def recommend_movies(movie_title, movies_df, genre_matrix_path, tag_matrix_path, top_n=10, similarity_threshold=0.1):
    # Load precomputed matrices
    genre_sim = np.load(genre_matrix_path)
    tag_sim = np.load(tag_matrix_path)

    # Case-insensitive search for movie title
    matching_movies = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    # Get movie index
    movie_index = matching_movies.index[0]

    # Blend similarities
    blended_sim = 0.5 * genre_sim + 0.5 * tag_sim

    # Filter similarities below threshold
    similarity_scores = blended_sim[movie_index]
    similarity_scores[movie_index] = 0  # Exclude self-similarity
    valid_indices = np.where(similarity_scores >= similarity_threshold)[0]

    # Check for no valid recommendations
    if valid_indices.size == 0:
        print("No similar movies found.")
        return pd.DataFrame()

    # Get top N similar movies
    similar_movie_indices = similarity_scores.argsort()[::-1][:top_n]
    recommendations = movies_df.iloc[similar_movie_indices][['title', 'genres']]
    return recommendations

"""

def recommend_movies(movie_title, movies_df, relevance_matrix, genre_weight=0.6, tag_weight=0.4, top_n=10):
    # retrieve movie id from title
    movie = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if movie.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    movie_id = movie.index[0]

    print("movie id retrieved")

    # calculate incremental similarity across the retrieved movie's genres
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movies_df['genres'])
    movie_genre_vector = genre_matrix[movie_id]
    genre_sim = cosine_similarity(movie_genre_vector, genre_matrix).flatten()
    
    # calculate incremental similarity across the retrieved movie's tags
    movie_tag_vector = relevance_matrix.iloc[movie_id].values.reshape(1, -1)
    tag_sim = cosine_similarity(movie_tag_vector, relevance_matrix).flatten()
    
    # blend sims
    blended_sim = (genre_weight * genre_sim) + (tag_weight * tag_sim)

    # REMOVED BLOCK calculate similarity
    genre_sim = cosine_similarity(
        TfidfVectorizer(stop_words='english').fit_transform(movies_df['genres'])
    )
    tag_sim = cosine_similarity(relevance_matrix)
    blended_sim = (genre_weight * genre_sim) + (tag_weight * tag_sim)

    print("similarity calced")

    # get top N recs
    similar_indices = blended_sim.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != movie_id][:top_n]
    
    # REMOVED BLOCKget top N recommendations
    similar_movie_ids = blended_sim[movie_id].argsort()[::-1][1:top_n+1] # argsort()[::-1] ==> descending order
    if not similar_movie_ids.size:
        print("No similar movies found.")
        return pd.DataFrame()

    print("top 10 found")

    recommendations = movies_df.iloc[similar_indices][['title', 'genres']]
    return recommendations
"""

# example execution function
def run_example():
    try:
        movies_df = pd.read_csv('data/processed_movies.csv')
        genre_matrix_path = 'data/genre_similarity_matrix.npy'
        tag_matrix_path = 'data/tag_similarity_matrix.npy'
        recommendations = recommend_movies("Toy Story (1995)", movies_df, genre_matrix_path, tag_matrix_path)
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

The current implementation of the movie recommender uses a Hybrid Recommendation System 
combining Content-Based Filtering techniques with Genre Similarity and Tag Similarity 
(via genome scores).

Pros:

- balanced complexity and performance friendly
- realistic and accurate output
- scalable and extensible

How it Works:

similar movies are determined by blending similarity scores blended from both
genre and tag similarity:

- Genre similarity (Pre computed using tf-idf and cosine similarity)
    - tfidf vectorization
      converts movie genres into numerical representations of assigned weights.
      weights are calculated by its frequency across movies and its originality in the dataset
    - cosine similarity
      measures the cos angle between 2 genre vectors which identifies the weight of shared genres
      while ignoring magnitudes
      
- Tag similarity (Pre computed genome scores)
    - pre processed relevance matrix maps movies to their tag-based relevance scores
    - cos similarity is calculated directly using the precomputed matrix to compare tag vectors of movies
    
- Blended sim = weighted avg of both sims

Suggestions:
- incremental sim computation
- pre computation of matrices
- sparse representations
- parallel processing
"""