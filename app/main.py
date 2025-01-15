import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from recommender import recommend_movies

# Load preprocessed data
@st.cache
def load_data():
    movies_df = pd.read_csv('data/cleaned_movies.csv')
    genre_similarity = np.load('data/genre_similarity_matrix.npy')
    return movies_df, genre_similarity

# Main app function
def main():
    st.title("Movie Recommendation System")

    # Load data
    movies_df, genre_similarity = load_data()

    # Input section
    st.subheader("Find Movie Recommendations")
    movie_title = st.text_input("Enter a movie title:")
    st.write(f"You entered: {movie_title}")
    num_recommendations = st.slider("Number of recommendations:", 1, 20)

    # Recommend movies on button click
    if st.button("Get Recommendations"):
        try:
            recommendations = recommend_movies(movie_title, movies_df, genre_similarity, top_n=num_recommendations)
            if recommendations.empty:
                st.write("No similar movies found.")
            else:
                st.write("### Recommendations:")
                st.dataframe(recommendations)
        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
