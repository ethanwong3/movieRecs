import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def recommend_movies_based_on_genres(movie_title, movies_df, ratings_df, top_n=10):
    """
    Recommends movies similar to the given movie title based on genres.
    Incorporates average ratings to refine the recommendations.
    """
    # Clean genres column to remove extra spaces or issues
    movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')

    # Convert genres to numerical values using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

    # Compute cosine similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the movie that matches the input title
    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        raise ValueError(f"The movie '{movie_title}' was not found in the dataset.")

    # Retrieve similar movies based on cosine similarity
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Skip the input movie itself

    # Get recommended movie indices
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df.iloc[movie_indices][['title', 'genres']]

    # Add average rating for sorting
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
    recommendations = recommendations.merge(movies_df[['movieId']], left_index=True, right_index=True)
    recommendations['avg_rating'] = recommendations['movieId'].map(avg_ratings)
    recommendations = recommendations.drop(columns='movieId')

    # Sort by average rating (descending)
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False)

    return recommendations

if __name__ == "__main__":
    # Load cleaned datasets
    movies_df = pd.read_csv('data/cleaned_movies.csv')
    ratings_df = pd.read_csv('data/cleaned_ratings.csv')

    try:
        # Test the recommender with a known movie title
        movie_title = "Toy Story (1995)"
        print(f"Testing recommendations for: {movie_title}\n")
        recommendations = recommend_movies_based_on_genres(movie_title, movies_df, ratings_df)
        print("Recommended Movies:")
        print(recommendations)
    except ValueError as e:
        print(e)
