import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from recommender import recommend_movies

# Cached functions for loading data
@st.cache_data
def load_data():
    movies_df = pd.read_csv("data/cleaned_movies.csv")
    genre_similarity = np.load("data/genre_similarity_matrix.npy")
    tag_similarity = np.load("data/tag_similarity_matrix.npy")
    ratings_similarity = np.load("data/ratings_similarity_matrix.npy")
    return movies_df, genre_similarity, tag_similarity, ratings_similarity

def main():
    st.title("Movie Recommendation System")
    st.write("Find movies similar to your favorites!")

    # Load data
    movies_df, genre_similarity, tag_similarity, ratings_similarity = load_data()

    # User input for movie title
    movie_title = st.text_input("Enter a movie title:")
    num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)

    # Sliders for blending weights
    st.write("Set the blending weights (must sum to 1):")
    genre_weight = st.slider("Genre Weight:", 0.0, 1.0, 0.5, step=0.1)
    tag_weight = st.slider("Tag Weight:", 0.0, 1.0, 0.3, step=0.1)
    ratings_weight = st.slider("Ratings Weight:", 0.0, 1.0, 0.2, step=0.1)

    # Ensure weights sum to 1
    total_weight = genre_weight + tag_weight + ratings_weight
    if total_weight != 1.0:
        st.error(f"Blending weights must sum to 1. Current total: {total_weight:.2f}")
        return

    # Toggle options to show tags and ratings
    show_tags = st.checkbox("Show tags with recommendations", value=True)
    show_ratings = st.checkbox("Show ratings with recommendations", value=True)

    if st.button("Get Recommendations"):
        if not movie_title.strip():
            st.warning("Please enter a movie title.")
        else:
            try:
                # Call recommender function with specified weights
                recommendations = recommend_movies(
                    movie_title=movie_title,
                    movies_df=movies_df,
                    genre_similarity=genre_similarity,
                    tag_similarity=tag_similarity,
                    ratings_similarity=ratings_similarity,
                    genre_weight=genre_weight,
                    tag_weight=tag_weight,
                    ratings_weight=ratings_weight,
                    top_n=num_recommendations
                )

                if recommendations.empty:
                    st.error(f"No recommendations found for '{movie_title}'. Please try another title.")
                else:
                    # Add tags and ratings if toggled
                    if show_tags:
                        if 'tags' in movies_df.columns:
                            recommendations = recommendations.merge(
                                movies_df[['title', 'tags']], on='title', how='left'
                            )
                        else:
                            st.warning("Tags information is not available in the dataset.")

                    if show_ratings:
                        if 'avg_rating' in movies_df.columns:
                            recommendations = recommendations.merge(
                                movies_df[['title', 'avg_rating']], on='title', how='left'
                            )
                        else:
                            st.warning("Ratings information is not available in the dataset.")

                    st.write(f"Recommendations based on '{movie_title}':")
                    st.dataframe(recommendations)

            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
