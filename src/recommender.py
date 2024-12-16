import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

# Recommends movies similar to the given movie title based on genres.

def recommend_movies_based_on_genres(movie_title, movies_df, top_n=10):

    # Step 1: Convert genres column from string to list, if necessary
    movies_df['genres'] = movies_df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Step 2: Join genres list into a single string
    movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) and len(x) > 0 else '')

    # Step 3: Drop rows where genres are still empty
    movies_df = movies_df[movies_df['genres'].str.strip() != '']

    # Debug: Confirm rows with valid genres remain
    print(f"Number of rows after filtering: {len(movies_df)}")

    # Check for valid input
    if movies_df.empty:
        raise ValueError("The dataset is empty after cleaning the genres column.")

    # Step 4: Convert genres into numerical values using TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

    # Compute cosine similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the movie that matches the input title
    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    # Retrieve similar movies
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the input movie itself

    # Get movie titles and genres for the recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['title', 'genres']]

if __name__ == "__main__":

    movies_df = pd.read_csv('data/cleaned_movies.csv')

    try:
        recommendations = recommend_movies_based_on_genres("Toy Story (1995)", movies_df)
        print("\nRecommended Movies:")
        print(recommendations)
    except ValueError as e:
        print(f"Error: {e}")
