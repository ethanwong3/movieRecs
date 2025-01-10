import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix

# Function to recommend movies based on genre and tag similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def recommend_movies(movie_title, movies_df, relevance_matrix, top_n=10, similarity_threshold=0.1):
    """
    Recommend movies based on genre and tag similarity.

    Args:
        movie_title (str): Title of the input movie.
        movies_df (pd.DataFrame): DataFrame containing movies and their genres.
        relevance_matrix (np.ndarray or pd.DataFrame): Precomputed tag relevance matrix.
        top_n (int): Number of top recommendations to return.
        similarity_threshold (float): Minimum similarity score to consider for recommendations.

    Returns:
        pd.DataFrame: DataFrame of recommended movies with titles and genres.
    """
    # Case-insensitive search for movie title
    matching_movies = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    # Get movie index
    movie_index = matching_movies.index[0]

    # Ensure relevance_matrix is a NumPy array
    if isinstance(relevance_matrix, pd.DataFrame):
        relevance_matrix = relevance_matrix.values

    # Compute genre similarity
    genre_vectorizer = TfidfVectorizer(stop_words='english')
    genre_matrix = genre_vectorizer.fit_transform(movies_df['genres'])
    genre_sim = cosine_similarity(genre_matrix)

    # Blend similarities
    blended_sim = 0.5 * genre_sim + 0.5 * relevance_matrix

    # Exclude self-similarity and apply similarity threshold
    similarity_scores = blended_sim[movie_index]
    similarity_scores[movie_index] = 0  # Set self-similarity to 0
    valid_indices = np.where(similarity_scores >= similarity_threshold)[0]

    # Check for no valid recommendations
    if valid_indices.size == 0:
        return pd.DataFrame(columns=['title', 'genres'])  # Return empty DataFrame if no valid recommendations

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