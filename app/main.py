import streamlit as st
import pandas as pd
import numpy as np
from recommender import recommend_movies

# Load preprocessed data
movies_df = pd.read_csv('data/processed_movies.csv')
genre_similarity = np.load('data/genre_similarity_matrix.npy')

# Streamlit interface
st.title("Movie Recommender System")
st.sidebar.header("Input")
movie_title = st.sidebar.text_input("Enter Movie Title", "Toy Story")
top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

if st.sidebar.button("Get Recommendations"):
    try:
        recommendations = recommend_movies(movie_title, movies_df, genre_similarity, top_n=top_n)
        st.write(f"Recommendations for {movie_title}:")
        st.table(recommendations)
    except ValueError as e:
        st.error(str(e))
