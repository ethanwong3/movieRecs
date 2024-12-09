import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# recommends movies focusiung on genre similarity

def recommend_movies_based_on_genres(movie_title, movies_df, top_n=10):

    # convert genres into numerical values
    tfidf = TfidfVectorizer(stop_words='english') # removes clutter in english (and, the, ...)
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].apply(lambda x: ' '.join(x))) # joins genre list

    # finds index of movie_title within similarity matrix where [i][j] represents similarity between movie i and j
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]

    # find sim scores with all other movies, sort for highest, skip itself
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][['title', 'genres']]
