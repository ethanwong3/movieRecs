import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from recommender import recommend_movies

# Mock dataset for movies
@pytest.fixture
def movies_df():
    return pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)"],
        "genres": [["Adventure", "Animation"], ["Adventure", "Children"], ["Comedy", "Romance"]]
    })

# Mock similarity matrices
@pytest.fixture
def genre_similarity():
    return np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0]
    ])

@pytest.fixture
def tag_similarity():
    return np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])

@pytest.fixture
def ratings_similarity():
    return np.array([
        [1.0, 0.7, 0.4],
        [0.7, 1.0, 0.6],
        [0.4, 0.6, 1.0]
    ])

def test_recommend_movies(movies_df, genre_similarity, tag_similarity, ratings_similarity):
    # Test a valid recommendation scenario
    recommendations = recommend_movies(
        movie_title="Toy Story (1995)",
        movies_df=movies_df,
        genre_similarity=genre_similarity,
        tag_similarity=tag_similarity,
        ratings_similarity=ratings_similarity,
        genre_weight=0.5,
        tag_weight=0.3,
        ratings_weight=0.2,
        top_n=2
    )
    assert len(recommendations) == 2
    assert "Jumanji (1995)" in recommendations["title"].values

def test_recommend_movies_not_found(movies_df, genre_similarity, tag_similarity, ratings_similarity):
    # Test when the movie is not found
    with pytest.raises(ValueError, match="Movie title 'Invalid Movie' not found in dataset."):
        recommend_movies(
            movie_title="Invalid Movie",
            movies_df=movies_df,
            genre_similarity=genre_similarity,
            tag_similarity=tag_similarity,
            ratings_similarity=ratings_similarity
        )

def test_recommend_movies_empty_recommendations(movies_df, genre_similarity, tag_similarity, ratings_similarity):
    # Test when no recommendations meet the threshold
    recommendations = recommend_movies(
        movie_title="Toy Story (1995)",
        movies_df=movies_df,
        genre_similarity=genre_similarity,
        tag_similarity=tag_similarity,
        ratings_similarity=ratings_similarity,
        similarity_threshold=0.95  # Set a very high threshold
    )
    assert recommendations.empty
