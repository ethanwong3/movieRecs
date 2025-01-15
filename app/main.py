import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from recommender import recommend_movies

# Cache the loading of data
@st.cache_data
def load_data():
    movies_df = pd.read_csv("data/cleaned_movies.csv")
    genre_similarity = np.load("data/genre_similarity_matrix.npy")
    tag_similarity = np.load("data/tag_similarity_matrix.npy")
    ratings_similarity = np.load("data/ratings_similarity_matrix.npy")
    return movies_df, genre_similarity, tag_similarity, ratings_similarity

def main():
    # Load the data
    movies_df, genre_similarity, tag_similarity, ratings_similarity = load_data()

    st.title("Movie Recommendation System")

    # Input movie title
    movie_title = st.text_input("Enter a movie title:", value="Toy Story (1995)")

    # Adjustable weights
    st.sidebar.title("Similarity Weights")
    genre_weight = st.sidebar.slider("Genre Weight", 0.0, 1.0, 0.5, 0.1)
    tag_weight = st.sidebar.slider("Tag Weight", 0.0, 1.0, 0.3, 0.1)
    ratings_weight = st.sidebar.slider("Ratings Weight", 0.0, 1.0, 0.2, 0.1)

    # Check if weights add up to 1
    if not round(genre_weight + tag_weight + ratings_weight, 1) == 1.0:
        st.warning("The weights should sum to 1.0.")
        return

    # Number of recommendations
    num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

    if st.button("Get Recommendations"):
        try:
            recommendations = recommend_movies(
                movie_title,
                movies_df,
                genre_similarity,
                tag_similarity,
                ratings_similarity,
                genre_weight=genre_weight,
                tag_weight=tag_weight,
                ratings_weight=ratings_weight,
                top_n=num_recommendations
            )
            if recommendations.empty:
                st.write(f"No similar movies found for '{movie_title}'. Try another title.")
            else:
                st.write("Recommendations:")
                st.dataframe(recommendations)
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
