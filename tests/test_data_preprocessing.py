import pytest
import pandas as pd

import sys
import os

# Add the src directory to the Python path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from data_preprocessing import precompute_genre_similarity, precompute_tag_similarity, precompute_ratings_similarity

def test_precompute_genre_similarity():
    movies_df = pd.DataFrame({
        "movieId": [1, 2],
        "genres": [["Action", "Adventure"], ["Adventure", "Drama"]]
    })
    similarity_matrix = precompute_genre_similarity(movies_df)
    assert similarity_matrix.shape == (2, 2)
    assert similarity_matrix[0, 1] > 0  # Non-zero similarity

def test_precompute_tag_similarity():
    tags_df = pd.DataFrame({
        "movieId": [1, 2],
        "tag": ["fun action", "drama adventure"]
    })
    movies_df = pd.DataFrame({"movieId": [1, 2]})
    similarity_matrix = precompute_tag_similarity(tags_df, movies_df)
    assert similarity_matrix.shape == (2, 2)
    assert similarity_matrix[0, 1] > 0

def test_precompute_ratings_similarity():
    ratings_df = pd.DataFrame({
        "userId": [1, 2, 1],
        "movieId": [1, 1, 2],
        "rating": [5.0, 4.0, 3.0]
    })
    movies_df = pd.DataFrame({"movieId": [1, 2]})
    similarity_matrix = precompute_ratings_similarity(ratings_df, movies_df)
    assert similarity_matrix.shape == (2, 2)
    assert similarity_matrix[0, 1] > 0
