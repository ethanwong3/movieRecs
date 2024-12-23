def recommend_movies(movie_title, movies_df, relevance_matrix, top_n=10):
    # Case-insensitive search for movie title
    matching_movies = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matching_movies.empty:
        raise ValueError(f"Movie title '{movie_title}' not found in dataset.")

    movie_id = matching_movies.index[0]

    # Compute similarities
    genre_sim = cosine_similarity(
        TfidfVectorizer(stop_words='english').fit_transform(movies_df['genres'])
    )
    tag_sim = cosine_similarity(relevance_matrix)
    blended_sim = 0.5 * genre_sim + 0.5 * tag_sim

    # Filter out the movie itself and any movies with low similarity
    similarity_scores = blended_sim[movie_id]
    similar_indices = similarity_scores.argsort()[::-1]
    similar_indices = [idx for idx in similar_indices if idx != movie_id and similarity_scores[idx] > 0.1]

    # Get top N recommendations
    if not similar_indices:
        return pd.DataFrame()  # No recommendations

    top_indices = similar_indices[:top_n]
    recommendations = movies_df.iloc[top_indices][['title', 'genres']]
    return recommendations
