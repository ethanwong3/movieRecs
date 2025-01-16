import pytest
import pandas as pd
import os
import tempfile

import sys
import os

# Add the src directory to the Python path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from data_clean import clean_movies, clean_ratings, clean_tags, save_cleaned_data

def test_clean_movies():
    # Mock movies dataset
    data = "1::Toy Story (1995)::Adventure|Animation\n2::Jumanji (1995)::Adventure|Children"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.seek(0)
        movies = clean_movies(f.name)
    assert len(movies) == 2
    assert movies["genres"].iloc[0] == ["Adventure", "Animation"]

def test_clean_ratings():
    # Mock ratings dataset
    data = "1::1::5.0::1234567890\n2::1::4.0::1234567891"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.seek(0)
        ratings = clean_ratings(f.name)
    assert len(ratings) == 2
    assert ratings["rating"].iloc[0] == 5.0

def test_clean_tags():
    # Mock tags dataset
    data = "1::1::funny::1234567890\n2::1::action::1234567891"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(data)
        f.seek(0)
        tags = clean_tags(f.name)
    assert len(tags) == 2
    assert tags["tag"].iloc[0] == "funny"
